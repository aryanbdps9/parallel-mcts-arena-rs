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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
@group(0) @binding(7) var<storage, read_write> free_list: array<u32>;
@group(0) @binding(8) var<storage, read_write> free_top: atomic<u32>;

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
    
    // Q-value (parent perspective): child_wins / (2 * effective_visits)
    // The wins are stored from parent's perspective (the player who made the move TO this child)
    let q = f32(child_wins) / (2.0 * effective_visits);
    
    // Exploration term
    let parent_sqrt = sqrt(f32(max(parent_visits, 1)));
    let u = params.exploration * prior * parent_sqrt / (1.0 + effective_visits);
    
    return q + u;
}

// Select best child using PUCT
fn select_best_child(parent_idx: u32) -> u32 {
    let info = node_info[parent_idx];
    let num_children = info.num_children;
    
    if (num_children == 0u) {
        return INVALID_INDEX;
    }
    
    var best_score = -1e9;
    var best_child = INVALID_INDEX;
    
    for (var i = 0u; i < num_children; i++) {
        let score = calculate_puct(parent_idx, i);
        if (score > best_score) {
            best_score = score;
            best_child = get_child_idx(parent_idx, i);
        }
    }
    
    return best_child;
}

// Try to allocate a new node
fn try_allocate_node() -> u32 {
    // Try free list first
    let ft = atomicLoad(&free_top);
    if (ft > 0u) {
        let new_ft = atomicSub(&free_top, 1u);
        if (new_ft > 0u && new_ft <= ft) {
            return free_list[new_ft - 1u];
        } else {
            // Failed to pop, restore
            atomicAdd(&free_top, 1u);
        }
    }
    
    // Allocate from pool
    let alloc = atomicAdd(&alloc_counter, 1u);
    if (alloc >= params.max_nodes) {
        atomicAdd(&diagnostics.alloc_failures, 1u);
        return INVALID_INDEX;
    }
    
    return alloc;
}

// Free a node by adding it to the free list
fn free_node(node_idx: u32) {
    if (node_idx == INVALID_INDEX || node_idx >= params.max_nodes) {
        return;
    }
    
    let ft = atomicAdd(&free_top, 1u);
    if (ft < params.max_nodes) {
        free_list[ft] = node_idx;
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
fn expand_node(node_idx: u32, board: ptr<function, array<i32, 64>>) -> bool {
    // Try to acquire expanding lock
    let old_state = atomicCompareExchangeWeak(&node_state[node_idx], NODE_STATE_READY, NODE_STATE_EXPANDING);
    if (old_state.exchanged == false || old_state.old_value != NODE_STATE_READY) {
        return false;  // Someone else is expanding or already expanded
    }
    
    atomicAdd(&diagnostics.expansion_attempts, 1u);
    
    let info = node_info[node_idx];
    let player = info.player_at_node;
    
    // Find all valid moves
    var valid_moves: array<u32, 64>;
    var num_valid = 0u;
    
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
            if (is_valid_move(board, x, y, player)) {
                if (num_valid < 64u) {
                    valid_moves[num_valid] = encode_move(x, y);
                    num_valid++;
                }
            }
        }
    }
    
    // Check for terminal state
    if (num_valid == 0u) {
        let opponent_moves = count_valid_moves(board, -player);
        if (opponent_moves == 0) {
            // Game over - both players have no moves
            atomicStore(&node_state[node_idx], NODE_STATE_TERMINAL);
            atomicAdd(&diagnostics.expansion_terminal, 1u);
            
            // Update node info
            var new_info = info;
            new_info.num_children = 0u;
            node_info[node_idx] = new_info;
            
            return true;
        } else {
            // Pass situation - current player has no moves but opponent does
            // Create a single "pass" child with INVALID_INDEX as move_id
            let child_idx = try_allocate_node();
            if (child_idx == INVALID_INDEX) {
                // Failed to allocate, mark as ready with 0 children (will retry later)
                atomicStore(&node_state[node_idx], NODE_STATE_READY);
                return false;
            }
            
            // Initialize pass child - player stays the same (opponent's turn)
            let child_info = NodeInfo(
                node_idx,           // parent
                INVALID_INDEX,      // move_id (pass)
                0u,                 // num_children
                -player             // player_at_node (opponent)
            );
            
            node_info[child_idx] = child_info;
            atomicStore(&node_visits[child_idx], 0);
            atomicStore(&node_wins[child_idx], 0);
            atomicStore(&node_vl[child_idx], 0);
            atomicStore(&node_state[child_idx], NODE_STATE_READY);
            
            // Set child in parent
            set_child_idx(node_idx, 0u, child_idx);
            set_child_prior(node_idx, 0u, 1.0);  // Only one move, prior = 1.0
            
            // Update parent with child count
            var new_info = info;
            new_info.num_children = 1u;
            node_info[node_idx] = new_info;
            
            // Mark as ready
            atomicStore(&node_state[node_idx], NODE_STATE_READY);
            atomicAdd(&diagnostics.expansion_success, 1u);
            
            return true;
        }
    }
    
    // Allocate child nodes for normal moves
    let uniform_prior = 1.0 / f32(max(num_valid, 1u));
    let opposite_player = -player;
    
    var actual_children = 0u;
    for (var i = 0u; i < num_valid; i++) {
        let child_idx = try_allocate_node();
        if (child_idx == INVALID_INDEX) {
            break;  // Out of nodes
        }
        
        // Initialize child
        let child_info = NodeInfo(
            node_idx,           // parent
            valid_moves[i],     // move_id
            0u,                 // num_children
            opposite_player     // player_at_node
        );
        
        node_info[child_idx] = child_info;
        atomicStore(&node_visits[child_idx], 0);
        atomicStore(&node_wins[child_idx], 0);
        atomicStore(&node_vl[child_idx], 0);
        atomicStore(&node_state[child_idx], NODE_STATE_READY);
        
        // Set child in parent
        set_child_idx(node_idx, actual_children, child_idx);
        set_child_prior(node_idx, actual_children, uniform_prior);
        actual_children++;
    }
    
    // Update parent with actual child count
    var new_info = info;
    new_info.num_children = actual_children;
    node_info[node_idx] = new_info;
    
    // Mark as ready
    atomicStore(&node_state[node_idx], NODE_STATE_READY);
    atomicAdd(&diagnostics.expansion_success, 1u);
    
    return true;
}

// =============================================================================
// Main MCTS Kernel
// =============================================================================

@compute @workgroup_size(64)
fn mcts_othello_iteration(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let iter_idx = global_id.x;
    if (iter_idx >= params.num_iterations) {
        return;
    }
    
    // Initialize RNG
    init_rng(iter_idx, params.seed);
    
    // Path storage
    var path: array<u32, 128>;
    var path_length = 0u;
    
    // === PHASE 1: SELECTION ===
    // Descend from root using PUCT, adding virtual losses
    var current_idx = params.root_idx;
    path[path_length] = current_idx;
    path_length++;
    atomicAdd(&node_vl[current_idx], 1);
    
    loop {
        if (path_length >= MAX_PATH_LENGTH) {
            atomicAdd(&diagnostics.selection_path_cap, 1u);
            break;
        }
        
        let state = atomicLoad(&node_state[current_idx]);
        if (state == NODE_STATE_TERMINAL) {
            atomicAdd(&diagnostics.selection_terminal, 1u);
            break;
        }
        
        let info = node_info[current_idx];
        
        // If node has no children, it needs expansion - stop here (leaf node)
        if (info.num_children == 0u) {
            atomicAdd(&diagnostics.selection_no_children, 1u);
            break;
        }
        
        let child = select_best_child(current_idx);
        if (child == INVALID_INDEX) {
            atomicAdd(&diagnostics.selection_invalid_child, 1u);
            break;
        }
        
        current_idx = child;
        path[path_length] = current_idx;
        path_length++;
        atomicAdd(&node_vl[current_idx], 1);
    }
    
    var leaf_idx = current_idx;
    var leaf_player = node_info[leaf_idx].player_at_node;
    
    // === PHASE 2: EXPANSION ===
    // Reconstruct board at leaf
    var board = reconstruct_board(&path, path_length);
    
    // Try to expand if not terminal and has no children
    var leaf_state = atomicLoad(&node_state[leaf_idx]);
    var leaf_info = node_info[leaf_idx];
    
    // Retry expansion and descend deeper when contention occurs
    var expansion_retries = 0u;
    var backoff_delay = 0u;
    while (leaf_state != NODE_STATE_TERMINAL && expansion_retries < 10u) {
        // Check if this node needs expansion
        if (leaf_info.num_children == 0u) {
            let expanded = expand_node(leaf_idx, &board);
            
            if (expanded) {
                // Successfully expanded - done
                break;
            }
            
            // Failed to expand (another thread is expanding)
            expansion_retries++;
            
            // Exponential backoff with jitter (max 256 iterations)
            backoff_delay = min(1u << expansion_retries, 256u);
            // Add jitter: 0 to backoff_delay-1
            let jitter = rand_u32() % max(backoff_delay, 1u);
            for (var spin = 0u; spin < jitter; spin++) {
                // Just spin to add delay
            }
            
            // Reload to see if another thread finished expanding
            leaf_info = node_info[leaf_idx];
            leaf_state = atomicLoad(&node_state[leaf_idx]);
        }
        
        // If node now has children (either we expanded or someone else did),
        // descend to one of them to try expanding deeper
        if (leaf_info.num_children > 0u) {
            let child = select_best_child(leaf_idx);
            if (child == INVALID_INDEX) {
                break;  // No valid child found
            }
            
            // Remove vl from current, add to child
            atomicAdd(&node_vl[leaf_idx], -1);
            
            leaf_idx = child;
            path[path_length] = leaf_idx;
            path_length++;
            atomicAdd(&node_vl[leaf_idx], 1);
            
            // Update state for next iteration
            leaf_player = node_info[leaf_idx].player_at_node;
            leaf_state = atomicLoad(&node_state[leaf_idx]);
            leaf_info = node_info[leaf_idx];
            
            // Reconstruct board with new path
            board = reconstruct_board(&path, path_length);
            
            // Continue loop to try expanding this child
            expansion_retries = 0u;  // Reset retry counter for new node
        } else {
            // Node still has no children and we failed to expand - give up
            break;
        }
    }
    
    // === PHASE 3: SIMULATION ===
    let sim_result = simulate_game(&board, leaf_player);
    atomicAdd(&diagnostics.rollouts, 1u);
    
    // === PHASE 4: BACKPROPAGATION ===
    // Update all nodes along path
    for (var i = 0u; i < path_length; i++) {
        let node_idx = path[i];
        
        // Remove virtual loss
        atomicAdd(&node_vl[node_idx], -1);
        
        // Increment visits
        atomicAdd(&node_visits[node_idx], 1);
        
        // Calculate reward from parent's perspective
        // The player who moved TO this node is -node_player
        // (because player_at_node is the player who plays FROM this node)
        let node_player = node_info[node_idx].player_at_node;
        let parent_player = -node_player;
        
        var reward = sim_result;
        // If parent_player is not the simulation winner (leaf_player), flip the result
        if (parent_player != leaf_player) {
            reward = 2 - sim_result;
        }
        
        atomicAdd(&node_wins[node_idx], reward);
    }
}
