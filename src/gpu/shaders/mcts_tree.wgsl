// =============================================================================
// GPU-Native MCTS Tree - All Four Phases on GPU
// =============================================================================
// This shader implements a fully GPU-based Monte Carlo Tree Search:
// 1. Selection: Descend tree using PUCT with atomic virtual losses
// 2. Expansion: Atomic node allocation from pre-allocated pool
// 3. Simulation: Random rollout (game-specific)
// 4. Backpropagation: Walk up parent chain with atomic updates
//
// Key design decisions:
// - Fixed-size node pool with atomic allocation counter
// - Index-based tree (no pointers, just u32 indices into node array)
// - Children stored as fixed-size array per node
// - Virtual losses via atomics prevent path convergence
// - No CPU-GPU sync during MCTS iterations
// =============================================================================

// =============================================================================
// Constants
// =============================================================================

// Maximum children per node (Othello can have up to ~30 moves, others less)
const MAX_CHILDREN: u32 = 64u;

// Maximum depth for path storage
const MAX_PATH_LENGTH: u32 = 128u;

// Sentinel value for invalid/empty index
const INVALID_INDEX: u32 = 0xFFFFFFFFu;

// Virtual loss weight
const VIRTUAL_LOSS_WEIGHT: f32 = 1.0;

// Node states
const NODE_STATE_EMPTY: u32 = 0u;
const NODE_STATE_EXPANDING: u32 = 1u;
const NODE_STATE_READY: u32 = 2u;
const NODE_STATE_TERMINAL: u32 = 3u;

// Work item states
const WORK_SELECTING: u32 = 0u;
const WORK_EXPANDING: u32 = 1u;
const WORK_SIMULATING: u32 = 2u;
const WORK_BACKPROP: u32 = 3u;
const WORK_DONE: u32 = 4u;

// =============================================================================
// Data Structures
// =============================================================================

// MCTS execution parameters
struct MctsParams {
    num_iterations: u32,
    max_nodes: u32,
    exploration: f32,
    root_idx: u32,
    seed: u32,
    board_width: u32,
    board_height: u32,
    game_type: u32,
}

// Non-atomic node data (read-mostly after init)
struct NodeInfo {
    parent_idx: u32,
    move_id: u32,
    num_children: u32,
    player_at_node: i32,
}

// Per-iteration work tracking
struct WorkItem {
    current_node: u32,
    leaf_node: u32,
    path_length: u32,
    status: u32,
    sim_result: i32,
    leaf_player: i32,
    _pad0: u32,
    _pad1: u32,
}

// =============================================================================
// Buffer Bindings - Group 0: Node Pool
// =============================================================================

@group(0) @binding(0) var<storage, read_write> node_info: array<NodeInfo>;
@group(0) @binding(1) var<storage, read_write> node_visits: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> node_wins: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> node_vl: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> node_state: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> children_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> children_priors: array<f32>;

// =============================================================================
// Buffer Bindings - Group 1: Execution State
// =============================================================================

@group(1) @binding(0) var<uniform> params: MctsParams;
@group(1) @binding(1) var<storage, read_write> work_items: array<WorkItem>;
@group(1) @binding(2) var<storage, read_write> paths: array<u32>;
@group(1) @binding(3) var<storage, read_write> alloc_counter: atomic<u32>;

// =============================================================================
// Buffer Bindings - Group 2: Game Boards for Simulation
// =============================================================================

@group(2) @binding(0) var<storage, read_write> sim_boards: array<i32>;

// =============================================================================
// RNG
// =============================================================================

var<private> rng_state: u32;

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand() -> f32 {
    rng_state ^= rng_state << 13u;
    rng_state ^= rng_state >> 17u;
    rng_state ^= rng_state << 5u;
    rng_state = pcg_hash(rng_state);
    return f32(rng_state) / 4294967296.0;
}

fn rand_int(max_exclusive: u32) -> u32 {
    return u32(rand() * f32(max_exclusive));
}

fn init_rng(thread_id: u32, base_seed: u32) {
    rng_state = pcg_hash(base_seed + thread_id * 1337u + 12345u);
    for (var i = 0u; i < 4u; i++) {
        rng_state = pcg_hash(rng_state);
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn get_child_idx(node_idx: u32, slot: u32) -> u32 {
    return children_indices[node_idx * MAX_CHILDREN + slot];
}

fn set_child_idx(node_idx: u32, slot: u32, child_idx: u32) {
    children_indices[node_idx * MAX_CHILDREN + slot] = child_idx;
}

fn get_child_prior(node_idx: u32, slot: u32) -> f32 {
    return children_priors[node_idx * MAX_CHILDREN + slot];
}

fn set_child_prior(node_idx: u32, slot: u32, prior: f32) {
    children_priors[node_idx * MAX_CHILDREN + slot] = prior;
}

fn get_path_node(iter_idx: u32, depth: u32) -> u32 {
    return paths[iter_idx * MAX_PATH_LENGTH + depth];
}

fn set_path_node(iter_idx: u32, depth: u32, node_idx: u32) {
    paths[iter_idx * MAX_PATH_LENGTH + depth] = node_idx;
}

// Calculate PUCT score for a child node
fn calculate_puct(node_idx: u32, child_slot: u32, parent_visits: i32) -> f32 {
    let child_idx = get_child_idx(node_idx, child_slot);
    
    if (child_idx == INVALID_INDEX) {
        return -1000000.0;
    }
    
    let visits = atomicLoad(&node_visits[child_idx]);
    let wins = atomicLoad(&node_wins[child_idx]);
    let vl = atomicLoad(&node_vl[child_idx]);
    let prior = get_child_prior(node_idx, child_slot);
    
    let effective_visits = f32(visits) + f32(vl) * VIRTUAL_LOSS_WEIGHT;
    let parent_sqrt = sqrt(f32(max(parent_visits, 1)));
    
    if (effective_visits < 0.5) {
        // Unvisited: exploration term only
        return params.exploration * prior * parent_sqrt;
    }
    
    // Q value: average reward in [0, 1]
    let q = f32(wins) / (f32(max(visits, 1)) * 2.0);
    
    // U value: exploration bonus
    let u = params.exploration * prior * parent_sqrt / (1.0 + effective_visits);
    
    return q + u;
}

// Select best child using PUCT
fn select_best_child(node_idx: u32) -> u32 {
    let info = node_info[node_idx];
    let num_children = info.num_children;
    
    if (num_children == 0u) {
        return INVALID_INDEX;
    }
    
    let parent_visits = atomicLoad(&node_visits[node_idx]);
    var best_score = -1000000.0;
    var best_slot = 0u;
    
    for (var i = 0u; i < num_children; i++) {
        let score = calculate_puct(node_idx, i, parent_visits);
        if (score > best_score) {
            best_score = score;
            best_slot = i;
        }
    }
    
    return get_child_idx(node_idx, best_slot);
}

// Allocate a new node
fn allocate_node() -> u32 {
    let idx = atomicAdd(&alloc_counter, 1u);
    if (idx >= params.max_nodes) {
        atomicSub(&alloc_counter, 1u);
        return INVALID_INDEX;
    }
    return idx;
}

// Initialize a node
fn init_node(idx: u32, parent: u32, move_id: u32, player: i32) {
    node_info[idx] = NodeInfo(parent, move_id, 0u, player);
    atomicStore(&node_visits[idx], 0);
    atomicStore(&node_wins[idx], 0);
    atomicStore(&node_vl[idx], 0);
    atomicStore(&node_state[idx], NODE_STATE_READY);
    
    for (var i = 0u; i < MAX_CHILDREN; i++) {
        set_child_idx(idx, i, INVALID_INDEX);
        set_child_prior(idx, i, 0.0);
    }
}

// =============================================================================
// Selection Kernel
// =============================================================================

@compute @workgroup_size(64)
fn mcts_select(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.num_iterations) {
        return;
    }
    
    init_rng(tid, params.seed);
    
    var work = WorkItem(
        params.root_idx,
        INVALID_INDEX,
        0u,
        WORK_SELECTING,
        0,
        0,
        0u, 0u
    );
    
    var current = params.root_idx;
    var depth = 0u;
    
    for (var iter = 0u; iter < MAX_PATH_LENGTH; iter++) {
        set_path_node(tid, depth, current);
        depth++;
        
        // Apply virtual loss
        atomicAdd(&node_vl[current], 1);
        
        let state = atomicLoad(&node_state[current]);
        
        if (state == NODE_STATE_TERMINAL) {
            work.leaf_node = current;
            work.status = WORK_BACKPROP;
            break;
        }
        
        let info = node_info[current];
        
        if (info.num_children == 0u || state == NODE_STATE_EMPTY) {
            work.leaf_node = current;
            work.leaf_player = info.player_at_node;
            work.status = WORK_EXPANDING;
            break;
        }
        
        let child = select_best_child(current);
        if (child == INVALID_INDEX) {
            work.leaf_node = current;
            work.leaf_player = info.player_at_node;
            work.status = WORK_SIMULATING;
            break;
        }
        
        current = child;
    }
    
    work.path_length = depth;
    work.current_node = current;
    if (work.leaf_node == INVALID_INDEX) {
        work.leaf_node = current;
        work.leaf_player = node_info[current].player_at_node;
        work.status = WORK_SIMULATING;
    }
    
    work_items[tid] = work;
}

// =============================================================================
// Backpropagation Kernel
// =============================================================================

@compute @workgroup_size(64)
fn mcts_backprop(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.num_iterations) {
        return;
    }
    
    var work = work_items[tid];
    
    if (work.status != WORK_BACKPROP) {
        return;
    }
    
    let result = work.sim_result;
    let leaf_player = work.leaf_player;
    
    for (var i = 0u; i < work.path_length; i++) {
        let node_idx = get_path_node(tid, i);
        
        // Remove virtual loss
        atomicAdd(&node_vl[node_idx], -1);
        
        // Increment visits
        atomicAdd(&node_visits[node_idx], 1);
        
        // Calculate reward from node's perspective
        let node_player = node_info[node_idx].player_at_node;
        var reward = result;
        if (node_player != leaf_player) {
            reward = 2 - result;
        }
        
        atomicAdd(&node_wins[node_idx], reward);
    }
    
    work.status = WORK_DONE;
    work_items[tid] = work;
}

// =============================================================================
// Statistics (for debugging)
// =============================================================================

struct TreeStats {
    total_nodes: u32,
    root_visits: i32,
    root_wins: i32,
    _pad: u32,
}

@group(3) @binding(0) var<storage, read_write> tree_stats: TreeStats;

@compute @workgroup_size(1)
fn mcts_get_stats(@builtin(global_invocation_id) global_id: vec3<u32>) {
    tree_stats.total_nodes = atomicLoad(&alloc_counter);
    tree_stats.root_visits = atomicLoad(&node_visits[params.root_idx]);
    tree_stats.root_wins = atomicLoad(&node_wins[params.root_idx]);
}
