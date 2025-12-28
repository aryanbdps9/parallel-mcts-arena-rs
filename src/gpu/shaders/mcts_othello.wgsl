// =============================================================================
// GPU-Native MCTS for Othello - All Four Phases in One Kernel
// =============================================================================
// This shader performs complete MCTS iterations for Othello on GPU:
// 1. Selection: Descend tree using PUCT with virtual losses
// 2. Expansion: Generate legal Othello moves, allocate child nodes
// 3. Simulation: Random rollout from leaf position
// 4. Backpropagation: Update statistics along path
//
// Key optimization: Board state is reconstructed by replaying moves from root
// =============================================================================

// =============================================================================
// Constants
// =============================================================================

const MAX_CHILDREN: u32 = 64u;
const MAX_PATH_LENGTH: u32 = 128u;
const INVALID_INDEX: u32 = 0xFFFFFFFFu;
const MAX_SELECTION_RETRIES: u32 = 16u;

const NODE_STATE_EMPTY: u32 = 0u;
const NODE_STATE_EXPANDING: u32 = 1u;
const NODE_STATE_READY: u32 = 2u;
const NODE_STATE_TERMINAL: u32 = 3u;

// Direction arrays for Othello
const DIR_X: array<i32, 8> = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
const DIR_Y: array<i32, 8> = array<i32, 8>(-1, -1, 0, 1, 1, 1, 0, -1);

const MAX_SIM_MOVES: i32 = 64;

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
    move_id: u32,  // Encoded as y * width + x
    num_children: u32,
    player_at_node: i32,
}

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

struct Diagnostics {
    selection_terminal: atomic<u32>,
    selection_no_children: atomic<u32>,
    selection_invalid_child: atomic<u32>,
    selection_path_cap: atomic<u32>,
    expansion_attempts: atomic<u32>,
    expansion_success: atomic<u32>,
    expansion_locked: atomic<u32>,
    expansion_terminal: atomic<u32>,
    alloc_failures: atomic<u32>,
    rollouts: atomic<u32>,
    _pad0: atomic<u32>,
    _pad1: atomic<u32>,
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
@group(1) @binding(4) var<storage, read_write> diagnostics: Diagnostics;

// =============================================================================
// Buffer Bindings - Group 2: Root Board State
// =============================================================================

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

fn rand() -> f32 {
    rng_state ^= rng_state << 13u;
    rng_state ^= rng_state >> 17u;
    rng_state ^= rng_state << 5u;
    rng_state = pcg_hash(rng_state);
    return f32(rng_state) / 4294967296.0;
}

fn init_rng(thread_id: u32, base_seed: u32) {
    rng_state = pcg_hash(base_seed + thread_id * 1337u + 12345u);
    for (var i = 0u; i < 4u; i++) {
        rng_state = pcg_hash(rng_state);
    }
}

// =============================================================================
// Othello Logic
// =============================================================================

fn othello_count_flips_dir(board: ptr<function, array<i32, 64>>, x: i32, y: i32, player: i32, d: i32) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    let dx = DIR_X[d];
    let dy = DIR_Y[d];
    let opponent = -player;
    
    var cx = x + dx;
    var cy = y + dy;
    var count = 0;
    
    while (cx >= 0 && cx < w && cy >= 0 && cy < h) {
        let cell = (*board)[cy * w + cx];
        if (cell == opponent) {
            count++;
            cx += dx;
            cy += dy;
        } else if (cell == player && count > 0) {
            return count;
        } else {
            return 0;
        }
    }
    return 0;
}

fn othello_is_valid_move(board: ptr<function, array<i32, 64>>, x: i32, y: i32, player: i32) -> bool {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    
    if (x < 0 || x >= w || y < 0 || y >= h) { return false; }
    if ((*board)[y * w + x] != 0) { return false; }
    
    for (var d = 0; d < 8; d++) {
        if (othello_count_flips_dir(board, x, y, player, d) > 0) { return true; }
    }
    return false;
}

fn othello_make_move(board: ptr<function, array<i32, 64>>, x: i32, y: i32, player: i32) {
    let w = i32(params.board_width);
    
    (*board)[y * w + x] = player;
    
    for (var d = 0; d < 8; d++) {
        let flip_count = othello_count_flips_dir(board, x, y, player, d);
        if (flip_count > 0) {
            let dx = DIR_X[d];
            let dy = DIR_Y[d];
            var cx = x + dx;
            var cy = y + dy;
            for (var i = 0; i < flip_count; i++) {
                (*board)[cy * w + cx] = player;
                cx += dx;
                cy += dy;
            }
        }
    }
}

fn othello_count_valid_moves(board: ptr<function, array<i32, 64>>, player: i32) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    var count = 0;
    
    for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
            if (othello_is_valid_move(board, x, y, player)) { count++; }
        }
    }
    return count;
}

// Decode move_id to (x, y)
fn decode_move(move_id: u32) -> vec2<i32> {
    let w = i32(params.board_width);
    return vec2<i32>(i32(move_id) % w, i32(move_id) / w);
}

// Encode (x, y) to move_id
fn encode_move(x: i32, y: i32) -> u32 {
    return u32(y * i32(params.board_width) + x);
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

fn calculate_puct(node_idx: u32, child_slot: u32, parent_visits: i32) -> f32 {
    let child_idx = get_child_idx(node_idx, child_slot);
    
    if (child_idx == INVALID_INDEX) {
        return -1000000.0;
    }
    
    let visits = atomicLoad(&node_visits[child_idx]);
    let wins = atomicLoad(&node_wins[child_idx]);
    let vl = atomicLoad(&node_vl[child_idx]);
    let prior = get_child_prior(node_idx, child_slot);
    
    let vl_weight = max(params.virtual_loss_weight, 0.001);
    let effective_visits = f32(visits) + f32(vl) * vl_weight;
    let parent_sqrt = sqrt(f32(max(parent_visits, 1)));
    
    if (effective_visits < 0.5) {
        return params.exploration * prior * parent_sqrt;
    }
    
    // Q-value: wins / (2 * visits)
    // With "Parent Perspective" storage, 'wins' tracks the win count for the 
    // player who made the move to this node (i.e., the parent).
    // So we simply maximize this value.
    // CRITICAL FIX: Include virtual loss in Q-value calculation to prevent "stampedes"
    // We treat virtual visits as losses (0 wins) to temporarily lower the Q-value
    // of nodes currently being explored by other threads.
    let q = f32(wins) / (f32(max(effective_visits, 1.0)) * 2.0);
    let u = params.exploration * prior * parent_sqrt / (1.0 + effective_visits);
    
    return q + u;
}

fn select_best_child(node_idx: u32) -> u32 {
    let info = node_info[node_idx];
    let num_children = info.num_children;
    
    if (num_children == 0u) {
        return INVALID_INDEX;
    }
    
    // CRITICAL FIX: Use max(parent_visits, 1) to ensure exploration even at root
    // Without this, when parent_visits=0, sqrt(0)=0 and exploration term vanishes!
    let parent_visits = max(atomicLoad(&node_visits[node_idx]), 1);
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

// Allocate a new node atomically
fn allocate_node() -> u32 {
    let idx = atomicAdd(&alloc_counter, 1u);
    if (idx >= params.max_nodes) {
        atomicSub(&alloc_counter, 1u);
        return INVALID_INDEX;
    }
    return idx;
}

// =============================================================================
// Expansion
// =============================================================================

fn expand_node(node_idx: u32, board: ptr<function, array<i32, 64>>) {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    let current_player = node_info[node_idx].player_at_node;
    let next_player = -current_player;
    
    // Count valid moves
    var valid_count = 0u;
    for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
            if (othello_is_valid_move(board, x, y, current_player)) {
                valid_count++;
            }
        }
    }
    
    // Handle Pass or Terminal
    if (valid_count == 0u) {
        // Check if opponent has moves
        let opp_moves = othello_count_valid_moves(board, next_player);
        if (opp_moves == 0) {
            // Terminal
            atomicStore(&node_state[node_idx], NODE_STATE_TERMINAL);
            return;
        } else {
            // Pass move
            let child_idx = allocate_node();
            if (child_idx != INVALID_INDEX) {
                // Initialize child
                var child_info: NodeInfo;
                child_info.parent_idx = node_idx;
                child_info.move_id = INVALID_INDEX; // Pass
                child_info.num_children = 0u;
                child_info.player_at_node = next_player;
                node_info[child_idx] = child_info;
                
                // Initialize stats to 0
                atomicStore(&node_visits[child_idx], 0);
                atomicStore(&node_wins[child_idx], 0);
                atomicStore(&node_vl[child_idx], 0);
                
                atomicStore(&node_state[child_idx], NODE_STATE_READY);
                
                // Link to parent
                set_child_idx(node_idx, 0u, child_idx);
                set_child_prior(node_idx, 0u, 1.0);
                
                // Update parent
                node_info[node_idx].num_children = 1u;
            }
            if (child_idx == INVALID_INDEX) {
                atomicAdd(&diagnostics.alloc_failures, 1u);
            }
            return;
        }
    }
    
    // Allocate children
    var allocated_count = 0u;
    let prior = 1.0 / f32(valid_count);
    
    for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
            if (othello_is_valid_move(board, x, y, current_player)) {
                if (allocated_count >= MAX_CHILDREN) { break; }
                
                let child_idx = allocate_node();
                if (child_idx == INVALID_INDEX) {
                    atomicAdd(&diagnostics.alloc_failures, 1u);
                    break;
                }
                
                // Initialize child
                var child_info: NodeInfo;
                child_info.parent_idx = node_idx;
                child_info.move_id = encode_move(x, y);
                child_info.num_children = 0u;
                child_info.player_at_node = next_player;
                node_info[child_idx] = child_info;
                
                // Initialize stats to 0
                atomicStore(&node_visits[child_idx], 0);
                atomicStore(&node_wins[child_idx], 0);
                atomicStore(&node_vl[child_idx], 0);
                
                atomicStore(&node_state[child_idx], NODE_STATE_READY);
                
                // Link to parent
                set_child_idx(node_idx, allocated_count, child_idx);
                set_child_prior(node_idx, allocated_count, prior);
                
                allocated_count++;
            }
        }
    }
    
    // Update parent
    if (allocated_count > 0u) {
        node_info[node_idx].num_children = allocated_count;
    } else {
        // Should not happen if valid_count > 0, unless OOM
        // If OOM, we treat as leaf?
    }
}

// =============================================================================
// Reconstruct Board State from Root by Replaying Moves
// =============================================================================

fn reconstruct_board_at_node(
    board: ptr<function, array<i32, 64>>,
    path: ptr<function, array<u32, 128>>,
    path_length: u32
) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    
    // Copy root board
    for (var i = 0; i < 64; i++) {
        (*board)[i] = root_board[i];
    }
    
    // Get root player (stored in root node)
    var current_player = node_info[params.root_idx].player_at_node;
    
    // Replay each move in the path (skip root, start from first actual move)
    for (var i = 1u; i < path_length; i++) {
        let node_idx = (*path)[i];
        let info = node_info[node_idx];
        let move_id = info.move_id;
        
        if (move_id != INVALID_INDEX) {
            let pos = decode_move(move_id);
            let move_player = -info.player_at_node;  // Player who made the move TO this node
            othello_make_move(board, pos.x, pos.y, move_player);
        }
        
        current_player = info.player_at_node;
    }
    
    return current_player;
}

// =============================================================================
// Random Rollout Simulation
// =============================================================================

fn othello_simulate(board: ptr<function, array<i32, 64>>, starting_player: i32) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    
    var sim_player = starting_player;
    var consecutive_passes = 0;
    var moves_made = 0;
    
    while (consecutive_passes < 2 && moves_made < MAX_SIM_MOVES) {
        // Count valid moves and pick one randomly
        var valid_count = 0;
        for (var y = 0; y < h; y++) {
            for (var x = 0; x < w; x++) {
                if (othello_is_valid_move(board, x, y, sim_player)) {
                    valid_count++;
                }
            }
        }
        
        if (valid_count == 0) {
            consecutive_passes++;
            sim_player = -sim_player;
            continue;
        }
        
        consecutive_passes = 0;
        
        // Pick random move (TODO: allow host-provided move weights to bias rollouts like CPU get_move_weight)
        var pick_index = i32(rand() * f32(valid_count));
        var picked_x = -1;
        var picked_y = -1;
        var count = 0;
        
        for (var y = 0; y < h; y++) {
            for (var x = 0; x < w; x++) {
                if (othello_is_valid_move(board, x, y, sim_player)) {
                    if (count == pick_index) {
                        picked_x = x;
                        picked_y = y;
                        y = h;
                        break;
                    }
                    count++;
                }
            }
        }
        
        if (picked_x >= 0) {
            othello_make_move(board, picked_x, picked_y, sim_player);
            moves_made++;
        }
        
        sim_player = -sim_player;
    }
    
    // Count pieces
    var player_count = 0;
    var opp_count = 0;
    for (var i = 0; i < 64; i++) {
        if ((*board)[i] == starting_player) { player_count++; }
        else if ((*board)[i] == -starting_player) { opp_count++; }
    }
    
    if (player_count > opp_count) { return 2; }  // Win
    else if (opp_count > player_count) { return 0; }  // Loss
    else { return 1; }  // Draw
}

// =============================================================================
// Main MCTS Kernel - Full Iteration
// =============================================================================

@compute @workgroup_size(64)
fn mcts_othello_iteration(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.num_iterations) {
        return;
    }
    
    init_rng(tid, params.seed);
    
    var leaf_idx = params.root_idx;
    var leaf_player: i32 = 0;
    var path: array<u32, 128>;
    var path_length = 0u;
    var selection_reason = 0u; // 0=normal,1=terminal,2=no_children,3=invalid_child,4=path_cap
    var board: array<i32, 64>;
    var sim_player: i32 = 0;

    // Allow a small number of retries if we hit an expanding leaf, so work can move to other ready leaves.
    for (var attempt = 0u; attempt < MAX_SELECTION_RETRIES; attempt++) {
        path_length = 0u;
        selection_reason = 0u;
        var current = params.root_idx;
        var expanded_stop = false;
        var lock_hits = 0u;

        // === PHASE 1: SELECTION (with inline expansion/backoff) ===
        var retry_selection = false;
        for (var iter = 0u; iter < MAX_PATH_LENGTH; iter++) {
            path[path_length] = current;
            path_length++;

            // Apply virtual loss
            atomicAdd(&node_vl[current], 1);

            let state = atomicLoad(&node_state[current]);
            let info = node_info[current];

            if (state == NODE_STATE_TERMINAL) {
                selection_reason = 1u;
                break;
            }

            if (info.num_children == 0u) {
                // Try to expand if ready; otherwise back off or stop
                var expanded = false;

                // Ensure board is reconstructed for expansion/simulation from this path
                sim_player = reconstruct_board_at_node(&board, &path, path_length);

                if (state == NODE_STATE_READY) {
                    let old_state = atomicExchange(&node_state[current], NODE_STATE_EXPANDING);
                    if (old_state == NODE_STATE_READY) {
                        atomicAdd(&diagnostics.expansion_attempts, 1u);
                        expand_node(current, &board);

                        let final_state = atomicLoad(&node_state[current]);
                        if (final_state != NODE_STATE_TERMINAL) {
                            atomicStore(&node_state[current], NODE_STATE_READY);
                        } else {
                            atomicAdd(&diagnostics.expansion_terminal, 1u);
                        }
                        if (node_info[current].num_children > 0u || final_state == NODE_STATE_TERMINAL) {
                            atomicAdd(&diagnostics.expansion_success, 1u);
                        }

                        if (final_state == NODE_STATE_TERMINAL) {
                            selection_reason = 1u;
                            break;
                        }

                        if (node_info[current].num_children > 0u) {
                            expanded = true;
                            expanded_stop = true;
                            // Stop selection after first expansion; rollout from this node
                            break;
                        }
                    } else if (old_state == NODE_STATE_EXPANDING) {
                        atomicAdd(&diagnostics.expansion_locked, 1u);
                        lock_hits += 1u;
                        // Mixed strategy to reduce hyperparams:
                        // 50/50 coin: heads -> rollout now from this node; tails -> try sibling/random path.
                        if (rand() < 0.5) {
                            // Rollout from current node despite lock; keep VLs as is.
                            sim_player = reconstruct_board_at_node(&board, &path, path_length);
                            expanded_stop = true;
                            break;
                        } else {
                            // Try a random READY sibling first; fallback to full retry.
                            let span = info.num_children;
                            var found = false;
                            for (var r = 0u; r < 4u; r++) {
                                let j = u32(rand() * f32(span));
                                let alt = get_child_idx(current, j);
                                if (alt != INVALID_INDEX && atomicLoad(&node_state[alt]) == NODE_STATE_READY) {
                                    current = alt;
                                    found = true;
                                    break;
                                }
                            }
                            if (found) {
                                continue;
                            }
                            // No ready sibling: retry selection from root; remove VLs along this path
                            retry_selection = true;
                        }
                    } else {
                        // Terminal or other state
                        if (old_state == NODE_STATE_TERMINAL) {
                            atomicAdd(&diagnostics.expansion_terminal, 1u);
                            atomicStore(&node_state[current], NODE_STATE_TERMINAL);
                            selection_reason = 1u;
                            break;
                        } else {
                            atomicStore(&node_state[current], old_state);
                        }
                    }
                }

                if (retry_selection) {
                    break;
                }

                if (expanded) {
                    // Already broke if expanded_stop; safety
                    break;
                }

                // Still no children: genuine no-children leaf
                selection_reason = 2u;
                break;
            }

            let child = select_best_child(current);
            if (child == INVALID_INDEX) {
                selection_reason = 3u;
                break;
            }

            // Jitter: if child is expanding, try a random ready child to reduce lock contention
            let child_state = atomicLoad(&node_state[child]);
            if (child_state == NODE_STATE_EXPANDING && info.num_children > 1u) {
                let span = info.num_children;
                var picked = child;
                // bounded retries
                for (var r = 0u; r < 4u; r++) {
                    let j = u32(rand() * f32(span));
                    let alt = get_child_idx(current, j);
                    if (alt != INVALID_INDEX && atomicLoad(&node_state[alt]) == NODE_STATE_READY) {
                        picked = alt;
                        break;
                    }
                }
                current = picked;
            } else {
                current = child;
            }
        }

        if (selection_reason == 0u && path_length >= MAX_PATH_LENGTH) {
            selection_reason = 4u;
        }

        leaf_idx = current;
        leaf_player = node_info[leaf_idx].player_at_node;

        if (retry_selection) {
            // Remove virtual losses we added along this path before retrying
            for (var i = 0u; i < path_length; i++) {
                atomicAdd(&node_vl[path[i]], -1);
            }
            continue;
        }

        if (expanded_stop) {
            // We expanded once; proceed directly to simulation from this node
            sim_player = reconstruct_board_at_node(&board, &path, path_length);
            break;
        }

        if (selection_reason == 1u) { atomicAdd(&diagnostics.selection_terminal, 1u); }
        if (selection_reason == 2u) { atomicAdd(&diagnostics.selection_no_children, 1u); }
        if (selection_reason == 3u) { atomicAdd(&diagnostics.selection_invalid_child, 1u); }
        if (selection_reason == 4u) { atomicAdd(&diagnostics.selection_path_cap, 1u); }

        // === PHASE 2: RECONSTRUCTION ===
        sim_player = reconstruct_board_at_node(&board, &path, path_length);

        break;
    }

    // If selection never succeeded (all retries hit expanding), we still have leaf_idx/path as last attempt.
    // No extra handling needed; we fall through to simulation.

    // Check if terminal
    let my_moves = othello_count_valid_moves(&board, sim_player);
    let opp_moves = othello_count_valid_moves(&board, -sim_player);
    
    var result: i32;
    if (my_moves == 0 && opp_moves == 0) {
        // Game over - count pieces
        var player_count = 0;
        var opp_count = 0;
        for (var i = 0; i < 64; i++) {
            if (board[i] == leaf_player) { player_count++; }
            else if (board[i] == -leaf_player) { opp_count++; }
        }
        if (player_count > opp_count) { result = 2; }
        else if (opp_count > player_count) { result = 0; }
        else { result = 1; }
    } else {
        // Run simulation
        result = othello_simulate(&board, sim_player);
        atomicAdd(&diagnostics.rollouts, 1u);
        // Adjust result if sim_player != leaf_player
        if (sim_player != leaf_player) {
            result = 2 - result;
        }
    }
    
    // === PHASE 3: BACKPROPAGATION ===
    for (var i = 0u; i < path_length; i++) {
        let node_idx = path[i];
        
        // Remove virtual loss
        atomicAdd(&node_vl[node_idx], -1);
        
        // Increment visits
        atomicAdd(&node_visits[node_idx], 1);
        
        // Calculate reward from Parent's perspective (Action Value)
        // We want to store wins for the player who made the move TO this node.
        // That player is -node_player.
        let node_player = node_info[node_idx].player_at_node;
        var reward = result;
        
        // If the player who moved to this node (-node_player) is NOT the winner (leaf_player),
        // then they lost (reward = 2 - result).
        // Note: result is 2 for leaf_player win, 0 for loss, 1 for draw.
        if (-node_player != leaf_player) {
            reward = 2 - result;
        }
        
        atomicAdd(&node_wins[node_idx], reward);
    }
}
