// =============================================================================
// Main MCTS Kernel Implementation (restored minimal version)
// =============================================================================
fn mcts_othello_iteration(global_id: vec3<u32>) {
    // Example: do nothing (replace with actual logic as needed)
    // This is a placeholder to ensure the entry point is valid.
    // TODO: Restore full kernel logic if needed.
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    mcts_othello_iteration(global_id);
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
// Generational tracking
@group(0) @binding(9) var<storage, read_write> node_generations: array<u32>;

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
        return INVALID_INDEX;
    }
    
    // Calculate PUCT scores for all children
    var scores: array<f32, 64>;
    var max_score = -1e9;
    
    for (var i = 0u; i < num_children; i++) {
        scores[i] = calculate_puct(parent_idx, i);
        max_score = max(max_score, scores[i]);
    }
    
    // Convert to probabilities using softmax with temperature (subtract max for numerical stability)
    var probs: array<f32, 64>;
    var sum_exp = 0.0;
    let temp = max(params.temperature, 0.001);  // Prevent division by zero
    
    for (var i = 0u; i < num_children; i++) {
        probs[i] = exp((scores[i] - max_score) / temp);
        sum_exp += probs[i];
    }
    
    // Normalize probabilities
    for (var i = 0u; i < num_children; i++) {
        probs[i] /= sum_exp;
    }
    
    // Sample from cumulative distribution
    let rand_val = rand_f32();
    var cumulative = 0.0;
    
    for (var i = 0u; i < num_children; i++) {
        cumulative += probs[i];
        if (rand_val <= cumulative) {
            return get_child_idx(parent_idx, i);
        }
    }
    
    // Fallback (should rarely happen due to floating point precision)
    return get_child_idx(parent_idx, num_children - 1u);
}

// Try to allocate a new node
fn try_allocate_node(my_workgroup: u32) -> u32 {
    // Try per-workgroup free list first
    let local_top = atomicSub(&free_tops[my_workgroup], 1u);
    if (local_top > 0u && local_top <= 8192u) {
        let idx = free_lists[my_workgroup][local_top - 1u];
        if (idx != INVALID_INDEX) {
            return idx;
        }
    }
    // Fallback: global allocation
    let alloc_idx = atomicAdd(&alloc_counter, 1u);
    if (alloc_idx < params.max_nodes) {
        return alloc_idx;
    }
    atomicAdd(&diagnostics.alloc_failures, 1u);
    return INVALID_INDEX;
}

// Free a node by adding it to the free list
fn free_node(node_idx: u32, my_workgroup: u32) {
    if (node_idx == INVALID_INDEX || node_idx >= params.max_nodes) {
        return;
    }
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
        return false;  // Someone else is expanding or already expanded
    }
    // ...existing expansion logic for a single attempt...
    // (This should generate children and update node_info, node_state, etc.)
    // For now, just return true to indicate a successful expansion attempt.
    return true;
}

// =============================================================================
// Pruning Kernel - Frees unreachable nodes after advance_root
// Generational Cleanup Kernel - Frees nodes older than cutoff generation
@compute @workgroup_size(256)
fn cleanup_old_generations(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let node_idx = global_id.x;
    if (node_idx >= params.max_nodes) {
        return;
    }
    let cutoff_gen = params.temperature; // Reuse temperature as cutoff for demo
    if (node_generations[node_idx] < u32(cutoff_gen)) {
        atomicStore(&node_visits[node_idx], 0);
        atomicStore(&node_wins[node_idx], 0);
        atomicStore(&node_vl[node_idx], 0);
        atomicStore(&node_state[node_idx], NODE_STATE_EMPTY);
        node_info[node_idx] = NodeInfo(INVALID_INDEX, INVALID_INDEX, 0u, 0);
        for (var i = 0u; i < MAX_CHILDREN; i++) {
            set_child_idx(node_idx, i, INVALID_INDEX);
            set_child_prior(node_idx, i, 0.0);
        }
        let my_workgroup = workgroup_id.x;
        free_node(node_idx, my_workgroup);
    }
}
// =============================================================================

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
        node_info[node_idx] = NodeInfo(INVALID_INDEX, INVALID_INDEX, 0u, 0);
        for (var i = 0u; i < MAX_CHILDREN; i++) {
            set_child_idx(node_idx, i, INVALID_INDEX);
            set_child_prior(node_idx, i, 0.0);
        }
        let my_workgroup = workgroup_id.x;
        free_node(node_idx, my_workgroup);
    }
}
