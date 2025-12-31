// =============================================================================
// Top-Down Dynamic Pruning Kernel (Design Doc Algorithm)
// =============================================================================
// Inputs:
//   - unreachable_roots: array<u32, MAX_UNREACHABLE_ROOTS> (indices of old root's children except new root)
//   - work_queue: array<u32, MAX_WORK_QUEUE> (global work queue for dynamic partitioning)
//   - work_head: atomic<u32> (global atomic counter for work queue)
//
// Each worker pops a node index from the work queue, traverses its subtree (BFS/DFS),
// atomically sets the deleted bit, and enqueues children. The atomic deleted bit ensures
// each node is only processed once.

const MAX_UNREACHABLE_ROOTS: u32 = 64u;
const MAX_WORK_QUEUE: u32 = 8192u;

@group(4) @binding(0) var<storage, read> unreachable_roots: array<u32, MAX_UNREACHABLE_ROOTS>;
@group(4) @binding(1) var<storage, read_write> work_queue: array<u32, MAX_WORK_QUEUE>;
@group(4) @binding(2) var<storage, read_write> work_head: atomic<u32>;

// Atomic set-and-check for deleted bit (bit 0)
fn atomic_set_deleted(node_idx: u32) -> bool {
    let old = atomicOr(&node_info[node_idx].flags, 1u);
    return (old & 1u) == 0u; // true if not previously deleted
}

// Shared atomics for PRUNING_START/END coordination
var<workgroup> pruning_started: atomic<u32>;
var<workgroup> pruning_threads_remaining: atomic<u32>;

@compute @workgroup_size(64)
fn prune_unreachable_topdown(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32, @builtin(num_workgroups) num_wg: vec3<u32>, @builtin(workgroup_id) wg_id: vec3<u32>) {
    // Only one workgroup for now; if multi-workgroup, use global atomics in a buffer
    if (local_idx == 0u) {
        atomicStore(&pruning_started, 0u);
        atomicStore(&pruning_threads_remaining, 64u); // workgroup_size
    }
    workgroupBarrier();

    // First thread to set pruning_started = 1 logs PRUNING_START
    let was_started = atomicExchange(&pruning_started, 1u);
    if (was_started == 0u) {
        var payload: array<u32, 255>;
        payload[0] = global_id.x;
        for (var i = 1u; i < 255u; i++) { payload[i] = 0u; }
        write_urgent_event(URGENT_EVENT_PRUNING_START, &payload);
    }

    // ...existing code...
    let tid = global_id.x;
    // Each worker pops work from the global queue
    loop {
        let work_idx = atomicAdd(&work_head, 1u);
        if (work_idx >= MAX_WORK_QUEUE) {
            break;
        }
        let node_idx = work_queue[work_idx];
        if (node_idx == INVALID_INDEX) {
            continue;
        }
        // Atomically set deleted bit; skip if already deleted
        if (!atomic_set_deleted(node_idx)) {
            continue;
        }
        // Free node (add to free list, clear state, etc.)
        atomicStore(&node_visits[node_idx], 0);
        atomicStore(&node_wins[node_idx], 0);
        atomicStore(&node_vl[node_idx], 0);
        atomicStore(&node_state[node_idx], NODE_STATE_EMPTY);
        // Mark all children for processing
        let info = node_info[node_idx];
        for (var i = 0u; i < info.num_children && i < MAX_CHILDREN; i++) {
            let child_idx = get_child_idx(node_idx, i);
            if (child_idx != INVALID_INDEX) {
                // Atomically push to work queue
                let qidx = atomicAdd(&work_head, 1u);
                if (qidx < MAX_WORK_QUEUE) {
                    work_queue[qidx] = child_idx;
                }
            }
        }
        // Optionally: add node to free list (not shown here)
    }

    // Last thread to decrement pruning_threads_remaining to 0 logs PRUNING_END
    let threads_left = atomicSub(&pruning_threads_remaining, 1u);
    if (threads_left == 1u) {
        var payload: array<u32, 255>;
        payload[0] = global_id.x;
        for (var i = 1u; i < 255u; i++) { payload[i] = 0u; }
        write_urgent_event(URGENT_EVENT_PRUNING_END, &payload);
    }
    workgroupBarrier();
    // Optionally: log RESUME_WORKERS event here if you want to resume MCTS workers after pruning
}
// =============================================================================
// Urgent Event Logging Buffer (Ring Buffer)
// =============================================================================
// 256 events x 1024 bytes = 256 KiB
struct UrgentEvent {
    timestamp: u32,
    event_type: u32,
    _pad: u32,
    payload: array<u32, 255>, // 1020 bytes (255*4) for payload, padded to 1024B
};

@group(3) @binding(0) var<storage, read_write> urgent_event_buffer: array<UrgentEvent, 256>;
@group(3) @binding(1) var<storage, read_write> urgent_event_write_head: atomic<u32>;

// =============================================================================
// Urgent Event Types
// =============================================================================

const URGENT_EVENT_START: u32 = 1u;
const URGENT_EVENT_HALT: u32 = 2u;
const URGENT_EVENT_REROOT_START: u32 = 10u;
const URGENT_EVENT_REROOT_END: u32 = 11u;
const URGENT_EVENT_PRUNING_START: u32 = 12u;
const URGENT_EVENT_PRUNING_END: u32 = 13u;
const URGENT_EVENT_MEMORY_PRESSURE: u32 = 14u;

// =============================================================================
// Helper: Write an urgent event to the ring buffer
// =============================================================================
fn write_urgent_event(event_type: u32, payload: ptr<function, array<u32, 255>>) {
    // Atomically reserve a slot in the ring buffer
    let idx = atomicAdd(&urgent_event_write_head, 1u) % 256u;
    let now = 0u; // TODO: Replace with a real timestamp if available
    urgent_event_buffer[idx].timestamp = now;
    urgent_event_buffer[idx].event_type = event_type;
    urgent_event_buffer[idx]._pad = 0u;
    // Copy payload (if any)
    for (var i = 0u; i < 255u; i++) {
        urgent_event_buffer[idx].payload[i] = (*payload)[i];
    }
}

// =============================================================================
// Main MCTS Kernel Implementation (restored minimal version)
// =============================================================================
// Shared atomics for REROOT_START/END coordination
var<workgroup> reroot_started: atomic<u32>;
var<workgroup> reroot_threads_remaining: atomic<u32>;

fn mcts_othello_iteration(global_id: vec3<u32>, local_idx: u32) {
    // Only one workgroup for now; if multi-workgroup, use global atomics in a buffer
    if (local_idx == 0u) {
        atomicStore(&reroot_started, 0u);
        atomicStore(&reroot_threads_remaining, 64u); // workgroup_size
    }
    workgroupBarrier();

    // First thread to set reroot_started = 1 logs REROOT_START
    let was_started = atomicExchange(&reroot_started, 1u);
    if (was_started == 0u) {
        var payload_reroot: array<u32, 255>;
        payload_reroot[0] = global_id.x;
        for (var i = 1u; i < 255u; i++) { payload_reroot[i] = 0u; }
        write_urgent_event(URGENT_EVENT_REROOT_START, &payload_reroot);
    }

    // ...existing code...
    atomicAdd(&diagnostics._pad0, 1u); // Use _pad0 as "kernel_entries" counter
    let my_workgroup = global_id.x % 256u;
    let thread_id = global_id.x;
    init_rng(thread_id, params.seed);

    // --- Log a START event at the beginning of each iteration ---
    var payload: array<u32, 255>;
    payload[0] = thread_id;
    payload[1] = params.root_idx;
    for (var i = 2u; i < 255u; i++) { payload[i] = 0u; }
    write_urgent_event(URGENT_EVENT_START, &payload);

    // Selection phase: start at root
    var path: array<u32, MAX_PATH_LENGTH>;
    var path_len: u32 = 0u;
    var current = params.root_idx;
    path[path_len] = current;
    path_len++;

    // Traverse down the tree until a leaf or terminal node
    loop {
        let info = node_info[current];
        let state = atomicLoad(&node_state[current]);
        if (state != NODE_STATE_READY) {
            atomicAdd(&diagnostics.selection_terminal, 1u);
            break;
        }
        if (info.num_children == 0u) {
            atomicAdd(&diagnostics.selection_no_children, 1u);
            break;
        }
        if (path_len >= MAX_PATH_LENGTH) {
            atomicAdd(&diagnostics.selection_path_cap, 1u);
            break;
        }
        // Select child
        let child = select_best_child(current);
        if (child == INVALID_INDEX) {
            atomicAdd(&diagnostics.selection_invalid_child, 1u);
            break;
        }
        path[path_len] = child;
        path_len++;
        current = child;
    }

    // Expansion phase
    atomicAdd(&diagnostics.expansion_attempts, 1u);
    var board = reconstruct_board(&path, path_len);
    let expand_success = expand_node(current, &board, my_workgroup);
    if (expand_success) {
        atomicAdd(&diagnostics.expansion_success, 1u);
    }
    else {
        // Log MEMORY_PRESSURE event if expansion failed (likely due to OOM)
        if (global_id.x == 0u) {
            var payload_mem: array<u32, 255>;
            payload_mem[0] = current;
            for (var i = 1u; i < 255u; i++) { payload_mem[i] = 0u; }
            write_urgent_event(URGENT_EVENT_MEMORY_PRESSURE, &payload_mem);
        }
    }

    // Rollout phase
    atomicAdd(&diagnostics.rollouts, 1u);
    var rollout_board = reconstruct_board(&path, path_len);
    let leaf_player = node_info[current].player_at_node;
    let rollout_result = simulate_game(&rollout_board, leaf_player);

    // Backpropagation phase
    for (var i = path_len; i > 0u; i--) {
        let node_idx = path[i - 1u];
        atomicAdd(&node_vl[node_idx], -1);
        atomicAdd(&node_visits[node_idx], 1);
        let node_player = node_info[node_idx].player_at_node;
        var reward = rollout_result;
        if (node_player != leaf_player) {
            reward = 2 - rollout_result;
        }
        atomicAdd(&node_wins[node_idx], reward);
    }

    // Last thread to decrement reroot_threads_remaining to 0 logs REROOT_END
    let threads_left = atomicSub(&reroot_threads_remaining, 1u);
    if (threads_left == 1u) {
        var payload_reroot_end: array<u32, 255>;
        payload_reroot_end[0] = global_id.x;
        for (var i = 1u; i < 255u; i++) { payload_reroot_end[i] = 0u; }
        write_urgent_event(URGENT_EVENT_REROOT_END, &payload_reroot_end);
    }
    workgroupBarrier();
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    mcts_othello_iteration(global_id, local_idx);
}
// =============================================================================
// GPU-Native MCTS for Othello - Clean Implementation
// =============================================================================
// Architecture:
// - Root board buffer: holds current game state
// - Root node (index 0): standard node with parent=INVALID, move=INVALID
// - All nodes represent game states via path from root
// - State reconstruction: root_board + apply moves along path
// - No transposition table: same move from different parents = different nodes
// =============================================================================

// =============================================================================
// Constants
// =============================================================================


const MAX_CHILDREN: u32 = 64u;
const MAX_PATH_LENGTH: u32 = 128u;
const INVALID_INDEX: u32 = 0xFFFFFFFFu;

// Explicit error codes for select_best_child
const SELECT_BEST_CHILD_NO_CHILDREN: u32 = 0xFFFFFFFEu;
const SELECT_BEST_CHILD_NO_VALID: u32 = 0xFFFFFFFDu;
const SELECT_BEST_CHILD_SOFTMAX_PANIC: u32 = 0xFFFFFFFCu;

const NODE_STATE_EMPTY: u32 = 0u;
const NODE_STATE_EXPANDING: u32 = 1u;
const NODE_STATE_READY: u32 = 2u;
const NODE_STATE_TERMINAL: u32 = 3u;

// Othello directions (8 directions)
const DIR_X: array<i32, 8> = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
const DIR_Y: array<i32, 8> = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);

const MAX_SIM_MOVES: i32 = 60;

// =============================================================================
// Data Structures
// =============================================================================

struct MctsParams {
    num_iterations: u32,
    max_nodes: u32,
    exploration: f32,
    virtual_loss_weight: f32,
    root_idx: u32,
    seed: u32,
    board_width: u32,
    board_height: u32,
    game_type: u32,
    temperature: f32,
    _pad0: u32,
    _pad1: u32,
}

struct NodeInfo {
    parent_idx: u32,
    move_id: u32,       // Encoded as y * width + x, or INVALID for root
    num_children: u32,
    player_at_node: i32,
    flags: atomic<u32>, // bit 0: deleted, bit 1: zero, bit 2: dirty
    _pad: u32,          // for alignment (optional, for 32-byte struct)
}

struct Diagnostics {
    selection_terminal: atomic<u32>,
    selection_no_children: atomic<u32>,
    selection_invalid_child: atomic<u32>,
    selection_path_cap: atomic<u32>,
    expansion_attempts: atomic<u32>,
    expansion_success: atomic<u32>,
    expansion_locked: atomic<u32>,
    exp_lock_rollout: atomic<u32>,
    exp_lock_sibling: atomic<u32>,
    exp_lock_retry: atomic<u32>,
    expansion_terminal: atomic<u32>,
    alloc_failures: atomic<u32>,
    recycling_events: atomic<u32>, // NEW: count value-based recycling
    rollouts: atomic<u32>,
    _pad0: atomic<u32>,
    _pad1: atomic<u32>,
}

// =============================================================================
// Buffer Bindings
// =============================================================================

// Group 0: Node Pool
@group(0) @binding(0) var<storage, read_write> node_info: array<NodeInfo>;
@group(0) @binding(1) var<storage, read_write> node_visits: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> node_wins: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> node_vl: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> node_state: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> children_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> children_priors: array<f32>;
// Per-workgroup free lists
@group(0) @binding(7) var<storage, read_write> free_lists: array<array<u32, 8192>, 256>;
@group(0) @binding(8) var<storage, read_write> free_tops: array<atomic<u32>, 256>;

// Group 1: Execution State
@group(1) @binding(0) var<uniform> params: MctsParams;
@group(1) @binding(1) var<storage, read_write> work_items: array<u32>;  // Not used but kept for compatibility
@group(1) @binding(2) var<storage, read_write> paths: array<u32>;
@group(1) @binding(3) var<storage, read_write> alloc_counter: atomic<u32>;
@group(1) @binding(4) var<storage, read_write> diagnostics: Diagnostics;

// Group 2: Root Board State
@group(2) @binding(0) var<storage, read> root_board: array<i32>;

// =============================================================================
// RNG
// =============================================================================

var<private> rng_state: u32;

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_u32() -> u32 {
    rng_state = pcg_hash(rng_state);
    return rng_state;
}

fn rand_f32() -> f32 {
    return f32(rand_u32()) / 4294967296.0;
}

fn init_rng(thread_id: u32, base_seed: u32) {
    rng_state = pcg_hash(base_seed + thread_id * 1337u + 12345u);
    // Warm up
    for (var i = 0u; i < 4u; i++) {
        _ = rand_u32();
    }
}

// =============================================================================
// Othello Game Logic
// =============================================================================

fn decode_move(move_id: u32) -> vec2<i32> {
    let w = i32(params.board_width);
    return vec2<i32>(i32(move_id) % w, i32(move_id) / w);
}

fn encode_move(x: i32, y: i32) -> u32 {
    return u32(y * i32(params.board_width) + x);
}

fn get_cell(board: ptr<function, array<i32, 64>>, x: i32, y: i32) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    if (x < 0 || x >= w || y < 0 || y >= h) {
        return 0;
    }
    return (*board)[y * w + x];
}

fn set_cell(board: ptr<function, array<i32, 64>>, x: i32, y: i32, value: i32) {
    let w = i32(params.board_width);
    (*board)[y * w + x] = value;
}

// Count flips in one direction for a move
fn count_flips_in_direction(board: ptr<function, array<i32, 64>>, x: i32, y: i32, player: i32, dir: i32) -> i32 {
    let dx = DIR_X[dir];
    let dy = DIR_Y[dir];
    let opponent = -player;
    
    var cx = x + dx;
    var cy = y + dy;
    var count = 0;
    
    loop {
        let cell = get_cell(board, cx, cy);
        if (cell == 0) {
            return 0;  // Empty cell, no flips
        }
        if (cell == opponent) {
            count++;
            cx += dx;
            cy += dy;
        } else if (cell == player) {
            return count;  // Found our piece, return flip count
        } else {
            return 0;  // Should not happen
        }
    }
    return 0;
}

// Check if a move is valid
fn is_valid_move(board: ptr<function, array<i32, 64>>, x: i32, y: i32, player: i32) -> bool {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    
    if (x < 0 || x >= w || y < 0 || y >= h) {
        return false;
    }
    if (get_cell(board, x, y) != 0) {
        return false;  // Cell occupied
    }
    // Check if it flips any pieces
    for (var d = 0; d < 8; d++) {
        if (count_flips_in_direction(board, x, y, player, d) > 0) {
            return true;
        }
    }
    return false;
}

// Apply a move to the board
fn apply_move(board: ptr<function, array<i32, 64>>, x: i32, y: i32, player: i32) {
    set_cell(board, x, y, player);
    
    // Flip pieces in all directions
    for (var d = 0; d < 8; d++) {
        let flip_count = count_flips_in_direction(board, x, y, player, d);
        if (flip_count > 0) {
            let dx = DIR_X[d];
            let dy = DIR_Y[d];
            var cx = x + dx;
            var cy = y + dy;
            for (var i = 0; i < flip_count; i++) {
                set_cell(board, cx, cy, player);
                cx += dx;
                cy += dy;
            }
        }
    }
}

// Count valid moves for a player
fn count_valid_moves(board: ptr<function, array<i32, 64>>, player: i32) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    var count = 0;
    
    for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
            if (is_valid_move(board, x, y, player)) {
                count++;
            }
        }
    }
    
    return count;
}

// Run a random simulation from a board state
// Returns: 2 for player win, 1 for draw, 0 for loss
fn simulate_game(board: ptr<function, array<i32, 64>>, start_player: i32) -> i32 {
    var current_player = start_player;
    var consecutive_passes = 0;
    
    for (var move_count = 0; move_count < MAX_SIM_MOVES; move_count++) {
        let num_moves = count_valid_moves(board, current_player);
        
        if (num_moves == 0) {
            consecutive_passes++;
            if (consecutive_passes >= 2) {
                break;  // Game over
            }
            current_player = -current_player;
            continue;
        }
        consecutive_passes = 0;
        
        // Pick a random valid move
        let pick = i32(rand_f32() * f32(num_moves));
        var found = 0;
        var made_move = false;
        
        let w = i32(params.board_width);
        let h = i32(params.board_height);
        for (var y = 0; y < h; y++) {
            for (var x = 0; x < w; x++) {
                if (is_valid_move(board, x, y, current_player)) {
                    if (found == pick) {
                        apply_move(board, x, y, current_player);
                        made_move = true;
                        break;
                    }
                    found++;
                }
            }
            if (made_move) {
                break;
            }
        }
        
        current_player = -current_player;
    }
    
    // Count pieces
    var player_count = 0;
    var opponent_count = 0;
    for (var i = 0; i < 64; i++) {
        if ((*board)[i] == start_player) {
            player_count++;
        } else if ((*board)[i] == -start_player) {
            opponent_count++;
        }
    }
    
    if (player_count > opponent_count) {
        return 2;
    } else if (player_count < opponent_count) {
        return 0;
    } else {
        return 1;
    }
}

// =============================================================================
// Tree Helper Functions
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

// Calculate PUCT score for a child
fn calculate_puct(parent_idx: u32, child_slot: u32) -> f32 {
    let child_idx = get_child_idx(parent_idx, child_slot);
    if (child_idx == INVALID_INDEX) {
        return -1000000.0;
    }
    
    let parent_visits = atomicLoad(&node_visits[parent_idx]);
    let child_visits = atomicLoad(&node_visits[child_idx]);
    let child_wins = atomicLoad(&node_wins[child_idx]);
    let child_vl = atomicLoad(&node_vl[child_idx]);
    let prior = get_child_prior(parent_idx, child_slot);
    
    // Virtual loss adjustment
    let vl_weight = max(params.virtual_loss_weight, 0.001);
    let effective_visits = f32(child_visits) + f32(child_vl) * vl_weight;
    
    if (effective_visits < 0.5) {
        // Unvisited node - use prior only
        let parent_sqrt = sqrt(f32(max(parent_visits, 1)));
        return params.exploration * prior * parent_sqrt;
    }
    
    // Q-value: child_wins / (2 * effective_visits)
    // The wins are stored from the perspective of the player who made the move TO this child
    // This represents how good this move was for the parent (who made the move)
    let q = f32(child_wins) / (2.0 * effective_visits);
    
    // Exploration term
    let parent_sqrt = sqrt(f32(max(parent_visits, 1)));
    let u = params.exploration * prior * parent_sqrt / (1.0 + effective_visits);
    
    return q + u;
}

// Select child by sampling from probability distribution based on PUCT scores
fn select_best_child(parent_idx: u32) -> u32 {
    let info = node_info[parent_idx];
    let num_children = info.num_children;
    if (num_children == 0u) {
        return SELECT_BEST_CHILD_NO_CHILDREN;
    }

    // Collect valid children (child_idx != INVALID_INDEX)
    var valid_slots: array<u32, 64>;
    var valid_count: u32 = 0u;
    for (var i = 0u; i < num_children; i++) {
        let child_idx = get_child_idx(parent_idx, i);
        if (child_idx != INVALID_INDEX) {
            valid_slots[valid_count] = i;
            valid_count++;
        }
    }
    if (valid_count == 0u) {
        return SELECT_BEST_CHILD_NO_VALID;
    }

    // Calculate PUCT scores for valid children only
    var scores: array<f32, 64>;
    var max_score = -1e9;
    for (var j = 0u; j < valid_count; j++) {
        let slot = valid_slots[j];
        scores[j] = calculate_puct(parent_idx, slot);
        max_score = max(max_score, scores[j]);
    }

    // Convert to probabilities using softmax with temperature (subtract max for numerical stability)
    var probs: array<f32, 64>;
    var sum_exp = 0.0;
    let temp = max(params.temperature, 0.00001);
    for (var j = 0u; j < valid_count; j++) {
        probs[j] = exp((scores[j] - max_score) / temp);
        sum_exp += probs[j];
    }

    // Normalize probabilities
    for (var j = 0u; j < valid_count; j++) {
        probs[j] /= sum_exp;
    }

    // Sample from cumulative distribution
    let rand_val = rand_f32();
    var cumulative = 0.0;
    for (var j = 0u; j < valid_count; j++) {
        cumulative += probs[j];
        if (rand_val <= cumulative) {
            return get_child_idx(parent_idx, valid_slots[j]);
        }
    }

    // Fallback (should rarely happen due to floating point precision)
    // PANIC: This should never happen! Print diagnostics and trap.
    // Print parent_idx, valid_count, scores, probs, rand_val
    // (WGSL has no printf, so use atomic counters for diagnostics)
    atomicAdd(&diagnostics.selection_invalid_child, 1000000u); // Mark as panic
    // Return explicit panic code; host must handle as panic
    return SELECT_BEST_CHILD_SOFTMAX_PANIC;
}

// Try to allocate a new node
fn try_allocate_node(my_workgroup: u32) -> u32 {
    // Try per-workgroup free list first
    let local_top = atomicSub(&free_tops[my_workgroup], 1u);
    if (local_top > 0u && local_top <= 8192u) {
        let idx = free_lists[my_workgroup][local_top - 1u];
        if (idx != INVALID_INDEX) {
            // Clear deleted and dirty bits, set zero bit if node is zeroed
            atomicAnd(&node_info[idx].flags, ~1u); // clear deleted
            atomicAnd(&node_info[idx].flags, ~(1u << 2)); // clear dirty
            // zero bit is set if node is zeroed, otherwise must be cleared by user
            return idx;
        }
    }
    // Fallback: global allocation
    let alloc_idx = atomicAdd(&alloc_counter, 1u);
    if (alloc_idx < params.max_nodes) {
        // New node: set zero bit, clear deleted and dirty
        atomicStore(&node_info[alloc_idx].flags, (1u << 1)); // zero bit
        return alloc_idx;
    }

    // Value-based node recycling: scan for low-value leaves
    // Only attempt a small window to avoid stalls; randomize start
    let scan_window: u32 = 128u;
    let start_idx = rand_u32() % (params.max_nodes - scan_window);
    var best_idx: u32 = INVALID_INDEX;
    var best_visits: i32 = 1000000000;
    for (var i = 0u; i < scan_window; i++) {
        let node_idx = start_idx + i;
        if (node_idx == params.root_idx) { continue; }
        let info = node_info[node_idx];
        // Only recycle leaf nodes that are not root, not already empty, and not in use
        let state = atomicLoad(&node_state[node_idx]);
        if (state != NODE_STATE_READY) { continue; }
        if (info.num_children != 0u) { continue; }
        let visits = atomicLoad(&node_visits[node_idx]);
        // Only consider nodes with very low visits (e.g., <= 1)
        if (visits < best_visits && visits <= 1) {
            best_visits = visits;
            best_idx = node_idx;
        }
    }
    if (best_idx != INVALID_INDEX) {
        // Recycle this node
        atomicStore(&node_visits[best_idx], 0);
        atomicStore(&node_wins[best_idx], 0);
        atomicStore(&node_vl[best_idx], 0);
        atomicStore(&node_state[best_idx], NODE_STATE_EMPTY);
        node_info[best_idx].parent_idx = INVALID_INDEX;
        node_info[best_idx].move_id = INVALID_INDEX;
        node_info[best_idx].num_children = 0u;
        node_info[best_idx].player_at_node = 0;
        atomicStore(&node_info[best_idx].flags, 1u); // set deleted bit
        node_info[best_idx]._pad = 0u;
        for (var i = 0u; i < MAX_CHILDREN; i++) {
            set_child_idx(best_idx, i, INVALID_INDEX);
            set_child_prior(best_idx, i, 0.0);
        }
        // Add to free list and return
        let local_top2 = atomicAdd(&free_tops[my_workgroup], 1u);
        if (local_top2 < 8192u) {
            free_lists[my_workgroup][local_top2] = best_idx;
        }
        // Diagnostics: count recycling event and value
        atomicAdd(&diagnostics.recycling_events, 1u);
        return best_idx;
    }

    // No recyclable node found: log memory pressure and fail
    atomicAdd(&diagnostics.alloc_failures, 1u);
    // Optionally, add a separate memory pressure counter here
    return INVALID_INDEX;
}

// Free a node by adding it to the free list
fn free_node(node_idx: u32, my_workgroup: u32) {
    if (node_idx == INVALID_INDEX || node_idx >= params.max_nodes) {
        return;
    }
    // Set deleted bit, clear dirty and zero bits
    atomicOr(&node_info[node_idx].flags, 1u); // set deleted
    atomicAnd(&node_info[node_idx].flags, ~(1u << 1)); // clear zero
    atomicAnd(&node_info[node_idx].flags, ~(1u << 2)); // clear dirty
    let local_top = atomicAdd(&free_tops[my_workgroup], 1u);
    if (local_top < 8192u) {
        free_lists[my_workgroup][local_top] = node_idx;
    }
}

// Mark a subtree as reachable (used during advance_root)
// This uses a simple iterative BFS approach
fn mark_subtree_reachable(root_idx: u32, reachable: ptr<function, array<u32, 256>>, reachable_count: ptr<function, u32>) {
    if (root_idx == INVALID_INDEX) {
        return;
    }
    
    // Simple queue for BFS (limited size)
    var queue: array<u32, 256>;
    var queue_start = 0u;
    var queue_end = 0u;
    
    // Add root to queue
    queue[queue_end] = root_idx;
    queue_end++;
    
    while (queue_start < queue_end && queue_start < 256u) {
        let node_idx = queue[queue_start];
        queue_start++;
        
        // Mark this node as reachable
        if (*reachable_count < 256u) {
            (*reachable)[*reachable_count] = node_idx;
            (*reachable_count)++;
        }
        
        // Add children to queue
        let info = node_info[node_idx];
        for (var i = 0u; i < info.num_children && i < MAX_CHILDREN; i++) {
            let child_idx = get_child_idx(node_idx, i);
            if (child_idx != INVALID_INDEX && queue_end < 256u) {
                queue[queue_end] = child_idx;
                queue_end++;
            }
        }
    }
}

// Reconstruct board at a node by applying moves along path
fn reconstruct_board(path: ptr<function, array<u32, 128>>, path_length: u32) -> array<i32, 64> {
    var board: array<i32, 64>;
    
    // Copy root board
    for (var i = 0; i < 64; i++) {
        board[i] = root_board[i];
    }
    
    // Apply moves along path (skip root at index 0)
    for (var i = 1u; i < path_length; i++) {
        let node_idx = (*path)[i];
        let info = node_info[node_idx];
        let move_id = info.move_id;
        
        if (move_id != INVALID_INDEX) {
            let pos = decode_move(move_id);
            apply_move(&board, pos.x, pos.y, -info.player_at_node);
        }
    }
    
    return board;
}

// Expand a node by generating legal moves and creating children
fn expand_node(node_idx: u32, board: ptr<function, array<i32, 64>>, my_workgroup: u32) -> bool {
    // Try to acquire expanding lock
    let old_state = atomicCompareExchangeWeak(&node_state[node_idx], NODE_STATE_READY, NODE_STATE_EXPANDING);
    if (old_state.exchanged == false || old_state.old_value != NODE_STATE_READY) {
        atomicAdd(&diagnostics.expansion_locked, 1u);
        return false;  // Someone else is expanding or already expanded
    }

    // Minimal expansion: generate all valid moves for the current player
    let info = node_info[node_idx];
    let player = info.player_at_node;
    var num_children: u32 = 0u;
    for (var y = 0; y < i32(params.board_height); y++) {
        for (var x = 0; x < i32(params.board_width); x++) {
            if (is_valid_move(board, x, y, player)) {
                // Allocate child node
                let child_idx = try_allocate_node(my_workgroup);
                if (child_idx == INVALID_INDEX) {
                    continue;
                }
                // Set up child node info
                node_info[child_idx].parent_idx = node_idx;
                node_info[child_idx].move_id = encode_move(x, y);
                node_info[child_idx].num_children = 0u;
                node_info[child_idx].player_at_node = -player;
                atomicStore(&node_info[child_idx].flags, 0u); // not deleted
                node_info[child_idx]._pad = 0u;
                atomicStore(&node_state[child_idx], NODE_STATE_READY);
                set_child_idx(node_idx, num_children, child_idx);
                set_child_prior(node_idx, num_children, 1.0); // Uniform prior for now
                num_children++;
                if (num_children >= MAX_CHILDREN) {
                    break;
                }
            }
        }
        if (num_children >= MAX_CHILDREN) {
            break;
        }
    }
    // Update parent node info
    node_info[node_idx].num_children = num_children;
    atomicStore(&node_state[node_idx], NODE_STATE_READY);
    return true;
}


// =============================================================================
// Pruning Kernel - Dynamic Partitioning Only
// This kernel uses dynamic partitioning: each thread checks if its node is reachable from the root by traversing parent pointers.
// No static or generation-based logic is used.

@compute @workgroup_size(256)
fn prune_unreachable(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let node_idx = global_id.x;
    if (node_idx >= params.max_nodes) {
        return;
    }
    
    // Root is always reachable
    if (node_idx == params.root_idx) {
        return;
    }
    
    // Check if this node is reachable from root via parent pointers
    var current = node_idx;
    var depth = 0u;
    var found_root = false;
    
    // Traverse up to root (max 128 steps to prevent infinite loops)
    while (depth < 128u) {
        let info = node_info[current];
        
        if (current == params.root_idx) {
            found_root = true;
            break;
        }
        
        if (info.parent_idx == INVALID_INDEX) {
            // Reached a disconnected node
            break;
        }
        
        current = info.parent_idx;
        depth++;
    }
    
    // If not reachable from root, free this node
    if (!found_root) {
        atomicStore(&node_visits[node_idx], 0);
        atomicStore(&node_wins[node_idx], 0);
        atomicStore(&node_vl[node_idx], 0);
        atomicStore(&node_state[node_idx], NODE_STATE_EMPTY);
        node_info[node_idx].parent_idx = INVALID_INDEX;
        node_info[node_idx].move_id = INVALID_INDEX;
        node_info[node_idx].num_children = 0u;
        node_info[node_idx].player_at_node = 0;
        atomicStore(&node_info[node_idx].flags, 1u); // set deleted bit
        node_info[node_idx]._pad = 0u;
        for (var i = 0u; i < MAX_CHILDREN; i++) {
            set_child_idx(node_idx, i, INVALID_INDEX);
            set_child_prior(node_idx, i, 0.0);
        }
        let my_workgroup = workgroup_id.x;
        free_node(node_idx, my_workgroup);
    }
}
