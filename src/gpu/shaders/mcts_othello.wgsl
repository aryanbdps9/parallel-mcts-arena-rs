// =============================================================================
// Two-Phase Pruning (Design Doc Algorithm)
// =============================================================================

struct RerootParams {
    move_x: u32,
    move_y: u32,
    current_root: u32,
    max_nodes: u32,  // Work queue capacity
};

// =============================================================================
// Pre-Computation Optimization Structures (Group 6)
// =============================================================================

struct LeafCandidate {
    node_id: u32,      // Index of the leaf node
    score: u32,         // Parent's visit count (heuristic for selection probability)
};

struct PrecomputedMoves {
    leaf_id: u32,       // Which leaf this cache entry is for
    num_moves: u32,     // How many legal moves
    moves: array<i32, 60>,  // Pre-computed legal moves (max 60 for Othello)
};

struct PrecomputeDiagnostics {
    candidates_found: atomic<u32>,    // Total leaves discovered
    cache_entries_created: atomic<u32>, // How many we pre-computed
    cache_hits: atomic<u32>,          // Expansions that used cache
    cache_misses: atomic<u32>,        // Expansions that computed on-the-fly
};

@group(6) @binding(0) var<storage, read_write> leaf_candidates: array<LeafCandidate>;
@group(6) @binding(1) var<storage, read_write> candidate_count: atomic<u32>;
@group(6) @binding(2) var<storage, read_write> precomputed_cache: array<PrecomputedMoves>;
@group(6) @binding(3) var<storage, read_write> cache_size: u32;
@group(6) @binding(4) var<storage, read_write> precompute_diag: PrecomputeDiagnostics;

// =============================================================================
// Incremental Execution Thread State (Group 7)
// =============================================================================

// Phase constants
const PHASE_SELECTION: u32 = 0u;
const PHASE_EXPANSION: u32 = 1u;
const PHASE_ROLLOUT_ACTIVE: u32 = 2u;
const PHASE_BACKPROP: u32 = 3u;
const PHASE_FINISHED: u32 = 4u;
const PHASE_IDLE: u32 = 5u;

struct ThreadState {
    phase: u32,                         // Current MCTS phase (SELECTION/EXPANSION/ROLLOUT_ACTIVE/BACKPROP/FINISHED)
    current_node: u32,                  // Node we're currently at (during selection)
    path: array<u32, 128>,              // Path from root to current position
    path_len: u32,                      // Number of nodes in path
    rng_seed: u32,                      // Random seed for this thread
    leaf_player: i32,                   // Player at leaf (for backprop perspective)
    rollout_result: u32,                // Rollout outcome (0=loss, 1=draw, 2=win for leaf_player)
    backprop_index: u32,                // Which node we're backpropping (counts down from path_len)
    
    // Rollout state (for chunked rollout execution in separate kernel)
    rollout_board: array<i32, 64>,      // Current board state during rollout
    rollout_player: i32,                // Current player during rollout
    rollout_moves_remaining: u32,       // Moves until terminal (or chunk limit, max 60)
    
    _pad0: u32,                         // Padding for alignment
};

@group(7) @binding(0) var<storage, read_write> thread_states: array<ThreadState>;

// =============================================================================
// Decoupled Rollout Queue Structures (Group 7 - Additional Bindings)
// =============================================================================

// Rollout job: tree workers queue these after expansion
struct RolloutJob {
    position: array<i32, 64>,  // Board state at leaf
    leaf_node_idx: u32,        // Which node this rollout is for
    leaf_player: i32,          // Player at leaf (for backprop perspective)
    _pad0: u32,                // Padding for alignment (total: 68*4 = 272 bytes)
};

// Backprop job: rollout workers queue these after completing rollouts
struct BackpropJob {
    leaf_node_idx: u32,  // Start backprop from this node
    result: u32,         // Rollout outcome (0=loss, 1=draw, 2=win for leaf_player)
    leaf_player: i32,    // Player at leaf (for perspective calculation)
    _pad0: u32,          // Padding for alignment (total: 16 bytes)
};

@group(7) @binding(1) var<storage, read_write> rollout_queue: array<RolloutJob>;
@group(7) @binding(2) var<storage, read_write> rollout_head: atomic<u32>;
@group(7) @binding(3) var<storage, read_write> backprop_queue: array<BackpropJob>;
@group(7) @binding(4) var<storage, read_write> backprop_head: atomic<u32>;

// Queue helper functions
fn enqueue_rollout(position: array<i32, 64>, leaf_idx: u32, leaf_player: i32) -> bool {
    let idx = atomicAdd(&rollout_head, 1u);
    // Check for overflow (queue size is max_threads * 2)
    if (idx >= arrayLength(&rollout_queue)) {
        // Queue full - decrement and fail
        atomicSub(&rollout_head, 1u);
        return false;
    }
    
    rollout_queue[idx].position = position;
    rollout_queue[idx].leaf_node_idx = leaf_idx;
    rollout_queue[idx].leaf_player = leaf_player;
    return true;
}

fn dequeue_rollout() -> RolloutJob {
    let idx = atomicSub(&rollout_head, 1u);
    // Check for underflow
    if (idx == 0u || idx > arrayLength(&rollout_queue)) {
        // Queue empty - restore and return invalid job
        atomicAdd(&rollout_head, 1u);
        var invalid: RolloutJob;
        invalid.leaf_node_idx = 0xFFFFFFFFu;  // INVALID_INDEX marker
        return invalid;
    }
    
    return rollout_queue[idx - 1u];
}

fn enqueue_backprop(leaf_idx: u32, result: u32, leaf_player: i32) -> bool {
    let idx = atomicAdd(&backprop_head, 1u);
    if (idx >= arrayLength(&backprop_queue)) {
        atomicSub(&backprop_head, 1u);
        return false;
    }
    
    backprop_queue[idx].leaf_node_idx = leaf_idx;
    backprop_queue[idx].result = result;
    backprop_queue[idx].leaf_player = leaf_player;
    return true;
}

fn dequeue_backprop() -> BackpropJob {
    let idx = atomicSub(&backprop_head, 1u);
    if (idx == 0u || idx > arrayLength(&backprop_queue)) {
        atomicAdd(&backprop_head, 1u);
        var invalid: BackpropJob;
        invalid.leaf_node_idx = 0xFFFFFFFFu;
        return invalid;
    }
    
    return backprop_queue[idx - 1u];
}

/// Phase 5: Dynamic Load Balancing
/// Decide if thread should become a rollout worker based on queue pressure.
/// Returns true if rollout_queue is fuller than backprop_queue (needs rollout workers).
fn should_be_rollout_worker() -> bool {
    let rollout_size = atomicLoad(&rollout_head);
    let backprop_size = atomicLoad(&backprop_head);
    
    // If rollout queue has work and backprop queue is not overwhelming, do rollouts
    // Threshold: if rollout_queue >= backprop_queue, prioritize rollouts
    return rollout_size >= backprop_size;
}

// =============================================================================
// Pruning Structures (Group 4)
// =============================================================================

@group(4) @binding(0) var<uniform> reroot_params: RerootParams;
@group(4) @binding(1) var<storage, read_write> new_root_output: u32;
@group(4) @binding(2) var<storage, read_write> global_free_queue: array<u32>;
@group(4) @binding(3) var<storage, read_write> global_free_head: atomic<u32>;
@group(4) @binding(4) var<storage, read_write> work_queue: array<u32>;
@group(4) @binding(5) var<storage, read_write> work_head: atomic<u32>;
@group(4) @binding(6) var<storage, read_write> work_claimed: atomic<u32>;
@group(4) @binding(7) var<storage, read_write> work_completed: atomic<u32>;

// Atomic set-and-check for deleted bit (bit 0)
fn atomic_set_deleted(node_idx: u32) -> bool {
    let old = atomicOr(&node_info[node_idx].flags, 1u);
    return (old & 1u) == 0u; // true if not previously deleted
}

@compute @workgroup_size(1)
fn identify_garbage() {
    let current_root = reroot_params.current_root;
    let move_x = reroot_params.move_x;
    let move_y = reroot_params.move_y;
    let target_move_id = encode_move(i32(move_x), i32(move_y));

    // DEBUG: Write sentinel to verify shader execution
    new_root_output = 0xBEEFCAFEu;
    
    // Initialize work queues
    atomicStore(&work_head, 0u);
    atomicStore(&work_claimed, 0u);
    atomicStore(&work_completed, 0u);
    
    let info = node_info[current_root];
    var found_new_root = false;

    let num_children = atomicLoad(&node_info[current_root].num_children);
    if (num_children == 0u) {
        new_root_output = 0xFFFFFFF0u; // DEBUG: No children
        // Also mark the old root as garbage
        let qidx = atomicAdd(&work_head, 1u);
        if (qidx < reroot_params.max_nodes) {
            work_queue[qidx] = current_root;
        }
        return;
    }

    var new_root_idx_temp = INVALID_INDEX;
    for (var i = 0u; i < num_children && i < MAX_CHILDREN; i++) {
        let child_idx = get_child_idx(current_root, i);
        if (child_idx == INVALID_INDEX) { continue; }
        
        let child_info = node_info[child_idx];
        if (child_info.move_id == target_move_id) {
            // This is the survivor
            new_root_idx_temp = child_idx;
            new_root_output = child_idx;
            found_new_root = true;
        } else {
            // DEBUG: If we fail to find the move, let's see what moves ARE there.
            // We can't print from shader, but we can maybe return the first mismatch as an error code?
            // For now, just mark as garbage.
            
            // This is garbage
            let qidx = atomicAdd(&work_head, 1u);
            if (qidx < reroot_params.max_nodes) {
                work_queue[qidx] = child_idx;
            }
        }
    }

    if (!found_new_root) {
        new_root_output = 0xFFFFFFFFu;
    } else {
        // New root and its subtree are preserved - do NOT modify them
        new_root_output = new_root_idx_temp;
        
        // CRITICAL: The new root is now the tree root, so it has no parent
        // We MUST clear its parent pointer, otherwise backprop will walk past the root
        // into the old (freed) root node, corrupting VL accounting
        node_info[new_root_idx_temp].parent_idx = INVALID_INDEX;
    }
    
    // Manually free the old root (current_root) WITHOUT adding its children to the work queue.
    // We have already added the siblings (garbage) to the work queue above.
    // The survivor (new_root) was NOT added to the queue, so it is preserved.
    
    // Clear data
    atomicStore(&node_visits[current_root], 0);
    atomicStore(&node_wins[current_root], 0);
    atomicStore(&node_vl[current_root], 0);
    atomicStore(&node_state[current_root], NODE_STATE_EMPTY);
    node_info[current_root].parent_idx = INVALID_INDEX;
    node_info[current_root].move_id = INVALID_INDEX;
    atomicStore(&node_info[current_root].num_children, 0u);
    node_info[current_root].player_at_node = 0;
    
    // Clear children indices and priors
    for (var i = 0u; i < MAX_CHILDREN; i++) {
        set_child_idx(current_root, i, INVALID_INDEX);
        set_child_prior(current_root, i, 0.0);
    }
    
    // Mark as deleted and add to free list (using workgroup 0)
    // We use free_node which handles flags and free list insertion
    free_node(current_root, 0u);
}

@compute @workgroup_size(64)
fn prune_unreachable_topdown(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = global_id.x;
    
    // RECURSIVE: Process garbage nodes and their entire subtrees
    // Each thread claims work items until the queue is empty
    loop {
        let head = atomicLoad(&work_head);
        let claimed = atomicLoad(&work_claimed);
        
        if (claimed < head) {
            // Try to claim next work item
            let original = atomicCompareExchangeWeak(&work_claimed, claimed, claimed + 1u);
            if (original.exchanged) {
                // Successfully claimed work item at index `claimed`
                atomicAdd(&diagnostics.prune_work_claimed, 1u);
                let qidx = claimed;
                
                if (qidx < reroot_params.max_nodes) {
                    let node_idx = work_queue[qidx];
                    
                    if (node_idx != INVALID_INDEX && node_idx < params.max_nodes) {
                        // CRITICAL: Add this node's children to work queue BEFORE freeing
                        // This implements recursive traversal of the garbage subtree
                        let num_children = atomicLoad(&node_info[node_idx].num_children);
                        for (var i = 0u; i < num_children && i < MAX_CHILDREN; i++) {
                            let child_idx = get_child_idx(node_idx, i);
                            if (child_idx != INVALID_INDEX && child_idx < params.max_nodes) {
                                // Add child to work queue
                                let new_head = atomicAdd(&work_head, 1u);
                                if (new_head < reroot_params.max_nodes) {
                                    work_queue[new_head] = child_idx;
                                }
                            }
                        }
                        
                        // Free this node - clear all data and add to free list
                        atomicStore(&node_visits[node_idx], 0);
                        atomicStore(&node_wins[node_idx], 0);
                        atomicStore(&node_vl[node_idx], 0);
                        atomicStore(&node_state[node_idx], NODE_STATE_EMPTY);
                        
                        node_info[node_idx].parent_idx = INVALID_INDEX;
                        node_info[node_idx].move_id = INVALID_INDEX;
                        atomicStore(&node_info[node_idx].num_children, 0u);
                        node_info[node_idx].player_at_node = 0;
                        node_info[node_idx]._pad = 0u;
                        
                        // Clear children indices and priors
                        for (var i = 0u; i < MAX_CHILDREN; i++) {
                            set_child_idx(node_idx, i, INVALID_INDEX);
                            set_child_prior(node_idx, i, 0.0);
                        }
                        
                        // Set deleted flag
                        atomicOr(&node_info[node_idx].flags, 1u);
                        
                        // Distribute freed nodes across ALL 256 workgroups
                        let target_wg = node_idx % 256u;
                        
                        // Atomically reserve a slot in the free list
                        var pushed = false;
                        loop {
                            atomicAdd(&diagnostics.prune_push_attempts, 1u);
                            let current_top = atomicLoad(&free_tops[target_wg]);
                            if (current_top >= params.free_list_capacity) {
                                // List is full, try global free list
                                break;
                            }
                            // Try to claim this slot
                            let result = atomicCompareExchangeWeak(&free_tops[target_wg], current_top, current_top + 1u);
                            if (result.exchanged) {
                                // Successfully claimed slot
                                let flat_idx = target_wg * params.free_list_capacity + current_top;
                                free_lists_flat[flat_idx] = node_idx;
                                atomicAdd(&diagnostics.nodes_freed, 1u);
                                pushed = true;
                                break;
                            }
                            // Else retry (another thread beat us to it)
                        }
                        
                        if (!pushed) {
                            // Try global free list (overflow)
                            let global_idx = atomicAdd(&global_free_head_alloc, 1u);
                            if (global_idx < params.max_nodes) {
                                global_free_queue_alloc[global_idx] = node_idx;
                                atomicAdd(&diagnostics.nodes_freed, 1u);
                            }
                        }
                    }
                }
                
                atomicAdd(&work_completed, 1u);
                // DON'T break - continue processing more work items from the growing queue
            }
            // If compare-exchange failed, another thread claimed this work - try again
        } else {
            // No more work available (claimed >= head)
            break;
        }
    }
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
const URGENT_EVENT_BATCH_START: u32 = 10u;
const URGENT_EVENT_BATCH_END: u32 = 11u;
const URGENT_EVENT_PRUNING_START: u32 = 12u;
const URGENT_EVENT_PRUNING_END: u32 = 13u;
const URGENT_EVENT_MEMORY_PRESSURE: u32 = 14u;
const URGENT_EVENT_REROOT_OP_START: u32 = 15u;
const URGENT_EVENT_REROOT_OP_END: u32 = 16u;
const URGENT_EVENT_DEBUG: u32 = 99u;

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

// Global atomics for REROOT_START/END coordination (in storage buffer)
@group(5) @binding(0) var<storage, read_write> global_reroot_threads_remaining: atomic<u32>;
@group(5) @binding(1) var<storage, read_write> global_reroot_start_threads_remaining: atomic<u32>;

fn mcts_othello_iteration(global_id: vec3<u32>, local_idx: u32, workgroup_id: vec3<u32>, num_workgroups: vec3<u32>) {
    // No workgroup-local initialization needed for global coordination
    // if (local_idx == 0u) { ... }
    // workgroupBarrier();

    // DISABLED: Old REROOT_START coordination - replaced by pruning-based tree reuse
    // // REROOT_START Coordination:
    // // We use a global atomic counter initialized to total_threads.
    // // Each thread decrements it. The FIRST thread (which sees the value drop to total_threads - 1)
    // // is responsible for logging the event.
    // // Note: atomicSub returns the ORIGINAL value. So if we decrement and get total_threads,
    // // it means we were the first one.
    // let total_threads = num_workgroups.x * num_workgroups.y * 64u;
    // let prev_start = atomicSub(&global_reroot_start_threads_remaining, 1u);
    // if (prev_start == total_threads) {
    //     var payload_reroot: array<u32, 255>;
    //     payload_reroot[0] = global_id.x;
    //     // We can also log the total threads count for debugging if needed, but for now just the thread ID
    //     for (var i = 1u; i < 255u; i++) { payload_reroot[i] = 0u; }
    //     write_urgent_event(URGENT_EVENT_BATCH_START, &payload_reroot);
    // }
    
    let total_threads = num_workgroups.x * num_workgroups.y * 64u;

    // ...existing code...
    // atomicAdd(&diagnostics._pad0, 1u); // Use _pad0 as "kernel_entries" counter
    let stride = num_workgroups.x * 64u;
    let thread_id = global_id.x + global_id.y * stride;
    let flat_workgroup_id = workgroup_id.x + workgroup_id.y * num_workgroups.x;
    let my_workgroup = flat_workgroup_id % 256u;
    
    init_rng(thread_id, params.seed);

    // --- Log a START event at the beginning of each iteration ---
    // var payload: array<u32, 255>;
    // payload[0] = thread_id;
    // payload[1] = params.root_idx;
    // for (var i = 2u; i < 255u; i++) { payload[i] = 0u; }
    // write_urgent_event(URGENT_EVENT_START, &payload);

    // Selection phase: start at root
    var path: array<u32, MAX_PATH_LENGTH>;
    var path_len: u32 = 0u;
    var current = params.root_idx;
    path[path_len] = current;
    path_len++;
    // Add virtual loss to root
    atomicAdd(&node_vl[current], 1);

    // Traverse down the tree until a leaf or terminal node
    loop {
        // Fix Read-After-Write Hazard: Check state BEFORE reading info
        // If we read info first, we might get stale data (num_children=0), then read state=READY,
        // and incorrectly conclude it's a leaf.
        let state = atomicLoad(&node_state[current]);
        if (state != NODE_STATE_READY) {
            atomicAdd(&diagnostics.selection_terminal, 1u);
            break;
        }
        
        // Check if node is deleted before reading its data
        if (is_node_deleted(current)) {
            atomicAdd(&diagnostics.selection_terminal, 1u);
            break;
        }
        
        let info = node_info[current];
        if (atomicLoad(&node_info[current].num_children) == 0u) {
            atomicAdd(&diagnostics.selection_no_children, 1u);
            break;
        }
        if (path_len >= MAX_PATH_LENGTH) {
            atomicAdd(&diagnostics.selection_path_cap, 1u);
            break;
        }
        // Select child
        let child = select_best_child(current);
        // Check for all error codes (INVALID_INDEX, NO_CHILDREN, NO_VALID, SOFTMAX_PANIC)
        // All error codes are >= 0xFFFFFFFB, so check if child is in that range
        if (child >= SELECT_BEST_CHILD_SOFTMAX_PANIC) {
            atomicAdd(&diagnostics.selection_invalid_child, 1u);
            break;
        }
        // Add virtual loss to selected child to discourage other threads from selecting it
        atomicAdd(&node_vl[child], 1);
        path[path_len] = child;
        path_len++;
        current = child;
    }

    // Expansion phase
    // Check if expansion is paused due to memory exhaustion
    let is_paused = atomicLoad(&expansion_paused) != 0u;
    var expand_success = false;
    if (!is_paused) {
        atomicAdd(&diagnostics.expansion_attempts, 1u);
        var board = reconstruct_board(&path, path_len);
        expand_success = expand_node(current, &board, my_workgroup);
        if (expand_success) {
            atomicAdd(&diagnostics.expansion_success, 1u);
        }
        else {
            // Log MEMORY_PRESSURE event if expansion failed (likely due to OOM)
            if (thread_id == 0u) {
                var payload_mem: array<u32, 255>;
                payload_mem[0] = current;
                for (var i = 1u; i < 255u; i++) { payload_mem[i] = 0u; }
                write_urgent_event(URGENT_EVENT_MEMORY_PRESSURE, &payload_mem);
            }
        }
    }

    // Rollout/Evaluation phase
    atomicAdd(&diagnostics.rollouts, 1u);
    var rollout_board = reconstruct_board(&path, path_len);
    let leaf_player = node_info[current].player_at_node;
    
    // Check if current position is terminal:
    // After expansion, if current node has 0 children, current player has no moves.
    // We need to check if opponent also has no moves to determine if game is over.
    let current_has_moves = atomicLoad(&node_info[current].num_children) > 0u;
    var is_terminal = false;
    if (!current_has_moves) {
        // Current player has no moves. Check if opponent has moves.
        let opponent_moves = count_valid_moves(&rollout_board, -leaf_player);
        if (opponent_moves == 0) {
            // Both players have no moves - game is terminal
            is_terminal = true;
            atomicAdd(&diagnostics.expansion_terminal, 1u);
        }
    }
    
    // Determine game winner (actual player who won: 1, -1, or 0 for draw)
    var winner: i32;
    if (is_terminal) {
        // Terminal position - evaluate directly by counting pieces
        var p1_count = 0;
        var p2_count = 0;
        for (var i = 0; i < 64; i++) {
            if (rollout_board[i] == 1) {
                p1_count++;
            } else if (rollout_board[i] == -1) {
                p2_count++;
            }
        }
        if (p1_count > p2_count) {
            winner = 1;  // Player 1 wins
        } else if (p1_count < p2_count) {
            winner = -1;  // Player -1 wins
        } else {
            winner = 0;  // Draw
        }
    } else {
        // Non-terminal - simulate game to completion
        winner = simulate_game(&rollout_board, leaf_player);
    }

    // Backpropagation phase
    for (var i = path_len; i > 0u; i--) {
        let node_idx = path[i - 1u];
        
        // Check if node was deleted during traversal
        if (is_node_deleted(node_idx)) {
            atomicAdd(&diagnostics.selection_terminal, 1u);
            break;
        }
        
        atomicAdd(&node_vl[node_idx], -1);
        let v = atomicAdd(&node_visits[node_idx], 1);
        mark_node_dirty(node_idx);
        
        // CRITICAL: Reward calculation must match CPU MCTS convention
        let player_at_node = node_info[node_idx].player_at_node;
        var reward: i32;
        
        // Special case: Root node (has no parent)
        // CPU stores root stats from perspective of player whose turn it is at root
        if (node_idx == params.root_idx) {
            if (winner == player_at_node) {
                reward = 2;  // Win for player at root
            } else if (winner == 0) {
                reward = 1;  // Draw
            } else {
                reward = 0;  // Loss
            }
        } else {
            // Non-root nodes: use parent's perspective
            // player_who_moved = opponent of player_at_node
            let player_who_moved = -player_at_node;
            if (winner == player_who_moved) {
                reward = 2;  // Win for player_who_moved
            } else if (winner == 0) {
                reward = 1;  // Draw
            } else {
                reward = 0;  // Loss
            }
        }
        atomicAdd(&node_wins[node_idx], reward);
    }

    // DISABLED: Old REROOT_END coordination - replaced by pruning-based tree reuse
    // // ...existing code...
    // workgroupBarrier();
    // // let total_threads = 64u * num_workgroups.x; // Already defined above
    // // Use a global atomic counter to ensure only one thread emits
    // var is_last_thread = false;
    // var prev = 0u;
    // if (total_threads > 0u && thread_id < total_threads) {
    //     prev = atomicSub(&global_reroot_threads_remaining, 1u);
    //     if (prev == 1u) {
    //         is_last_thread = true;
    //     }
    // }
    // if (is_last_thread) {
    //     var payload_reroot_end: array<u32, 255>;
    //     payload_reroot_end[0] = thread_id;
    //     payload_reroot_end[1] = params.turn_number;
    //     payload_reroot_end[2] = prev; // DIAG: include prev value
    //     payload_reroot_end[3] = total_threads;
    //     payload_reroot_end[4] = atomicLoad(&global_reroot_threads_remaining); // Instrument: value after emission
    //     for (var i = 5u; i < 255u; i++) { payload_reroot_end[i] = 0u; }
    //     write_urgent_event(URGENT_EVENT_BATCH_END, &payload_reroot_end);
    // }
}
@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    // HEARTBEAT: Count total threads launched
    atomicAdd(&diagnostics.exp_lock_rollout, 1u);
    
    // DEBUG: Check if num_workgroups is zero
    if (num_workgroups.x == 0u) {
        atomicAdd(&diagnostics.exp_lock_sibling, 1u);
    }

    // Global atomic is initialized by host before dispatch
    
    // DEBUG: Compute root_board hash on thread 0
    if (global_id.x == 0u && global_id.y == 0u) {
        // Compute FNV-1a hash of the board (32-bit)
        var hash: u32 = 0x811c9dc5u;
        for (var i = 0u; i < 64u; i++) {
            let val = u32(root_board.cells[i]);
            hash = hash ^ val;
            hash = hash * 0x01000193u;
        }
        atomicExchange(&diagnostics.root_board_hash, hash);
    }
    
    mcts_othello_iteration(global_id, local_idx, workgroup_id, num_workgroups);
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
const PASS_MOVE_ID: u32 = 0xFFFFFFFEu;  // Special ID for pass moves

// Explicit error codes for select_best_child
const SELECT_BEST_CHILD_NO_CHILDREN: u32 = 0xFFFFFFFDu;
const SELECT_BEST_CHILD_NO_VALID: u32 = 0xFFFFFFFCu;
const SELECT_BEST_CHILD_SOFTMAX_PANIC: u32 = 0xFFFFFFFBu;

const NODE_STATE_EMPTY: u32 = 0u;
const NODE_STATE_EXPANDING: u32 = 1u;
const NODE_STATE_READY: u32 = 2u;
const NODE_STATE_TERMINAL: u32 = 3u;

// Othello directions (8 directions)
const DIR_X: array<i32, 8> = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
const DIR_Y: array<i32, 8> = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);

const MAX_SIM_MOVES: i32 = 60;

// Flag bit positions
const FLAG_DELETED: u32 = 0x1u;    // Bit 0: node is deleted/freed
const FLAG_ZERO: u32 = 0x2u;       // Bit 1: node is freshly allocated
const FLAG_DIRTY: u32 = 0x4u;      // Bit 2: node has been modified

// Helper functions for flag management
fn is_node_deleted(node_idx: u32) -> bool {
    return (atomicLoad(&node_info[node_idx].flags) & FLAG_DELETED) != 0u;
}

fn mark_node_dirty(node_idx: u32) {
    atomicOr(&node_info[node_idx].flags, FLAG_DIRTY);
}

fn mark_node_deleted(node_idx: u32) {
    atomicOr(&node_info[node_idx].flags, FLAG_DELETED);
}

fn mark_node_clean(node_idx: u32) {
    atomicAnd(&node_info[node_idx].flags, ~FLAG_DELETED);
    atomicAnd(&node_info[node_idx].flags, ~FLAG_DIRTY);
}

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
    turn_number: u32, // NEW: unique per-turn identifier
    free_list_capacity: u32, // Capacity per free list (max_nodes / 256 rounded up)
    vl_temp_scale: f32, // Scaling factor for VL-based temperature boost
    num_threads: u32, // NEW: Number of threads for incremental execution
    root_node: u32, // NEW: Current root node index for incremental execution
    use_vl_preincrement: u32, // 1 = use pre-increment VL, 0 = increment only selected child
    use_random_rollouts: u32, // 1 = use random test rollouts, 0 = real game simulation
}

struct NodeInfo {
    parent_idx: u32,
    move_id: u32,       // Encoded as y * width + x, or INVALID for root
    num_children: atomic<u32>,  // Made atomic for memory coherency
    player_at_node: i32,
    flags: atomic<u32>, // bit 0: deleted, bit 1: zero, bit 2: dirty
    _pad: u32,          // for alignment (optional, for 32-byte struct)
    _pad2: u32,
    _pad3: u32,
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
    nodes_freed: atomic<u32>, // Count nodes freed during pruning
    rollouts: atomic<u32>,
    root_board_hash: atomic<u32>, // Replaces _pad0
    init_nodes_count: atomic<u32>,
    total_children_gen: atomic<u32>,
    prune_work_claimed: atomic<u32>, // DEBUG: How many work items successfully claimed
    prune_push_attempts: atomic<u32>, // DEBUG: Total attempts in free list push loop
    max_temp_boost: atomic<u32>, // Maximum temperature boost observed (stored as u32, divide by 1000 for f32)
    random_rollout_wins: atomic<u32>, // Count of result=2 (win) from random rollouts (from p1 perspective)
    random_rollout_draws: atomic<u32>, // Count of result=1 (draw) from random rollouts
    random_rollout_losses: atomic<u32>, // Count of result=0 (loss) from random rollouts (from p1 perspective)
    random_rollout_from_p1: atomic<u32>, // Count of rollouts where leaf_player == 1
    random_rollout_from_p2: atomic<u32>, // Count of rollouts where leaf_player == -1
    random_rollout_p1_wins_raw: atomic<u32>, // Count where p1_score > 32 in game simulation
    random_rollout_p1_losses_raw: atomic<u32>, // Count where p1_score < 32 in game simulation
    random_rollout_p1_score_sum: atomic<u32>, // Sum of all p1_score values for distribution checking
    random_rollout_min_tid: atomic<u32>, // Minimum thread ID that did a rollout (initialized to 0xFFFFFFFF)
    random_rollout_max_tid: atomic<u32>, // Maximum thread ID that did a rollout
    global_rollout_counter: atomic<u32>, // Global counter for independent RNG seeding
    phase_selection_count: atomic<u32>, // Threads in PHASE_SELECTION
    phase_expansion_count: atomic<u32>, // Threads in PHASE_EXPANSION
    phase_rollout_count: atomic<u32>, // Threads in PHASE_ROLLOUT_ACTIVE
    phase_backprop_count: atomic<u32>, // Threads in PHASE_BACKPROP
    phase_idle_count: atomic<u32>, // Threads in PHASE_IDLE
    phase_finished_count: atomic<u32>, // Threads in PHASE_FINISHED
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
// Per-workgroup free lists (runtime-sized inner array)
@group(0) @binding(7) var<storage, read_write> free_lists_flat: array<u32>;
@group(0) @binding(8) var<storage, read_write> free_tops: array<atomic<u32>, 256>;
@group(0) @binding(9) var<storage, read_write> free_list_ownership: array<atomic<u32>, 256>;  // Which workgroup owns each free list (0-255 or 0xFFFF=unowned)
@group(0) @binding(10) var<storage, read_write> global_free_queue_alloc: array<u32>;
@group(0) @binding(11) var<storage, read_write> global_free_head_alloc: atomic<u32>;
@group(0) @binding(12) var<storage, read_write> expansion_paused: atomic<u32>;

// Group 1: Execution State
@group(1) @binding(0) var<uniform> params: MctsParams;
@group(1) @binding(1) var<storage, read_write> work_items: array<u32>;  // Not used but kept for compatibility
@group(1) @binding(2) var<storage, read_write> paths: array<u32>;
@group(1) @binding(3) var<storage, read_write> alloc_counter: atomic<u32>;
@group(1) @binding(4) var<storage, read_write> diagnostics: Diagnostics;

struct RootChildStats {
    move_id: u32,
    visits: i32,
    wins: i32,
    _pad: u32,
}
@group(1) @binding(5) var<storage, read_write> root_stats: array<RootChildStats>;

// Group 2: Root Board State
struct Board {
    cells: array<i32, 64>,
};
@group(2) @binding(0) var<storage, read> root_board: Board;

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
    let rng_val = atomicAdd(&diagnostics.global_rollout_counter, 1u);
    return pcg_hash(rng_val);
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
    
    // Count pieces to determine actual winner
    var p1_count = 0;
    var p2_count = 0;
    for (var i = 0; i < 64; i++) {
        if ((*board)[i] == 1) {
            p1_count++;
        } else if ((*board)[i] == -1) {
            p2_count++;
        }
    }
    
    // Return actual winner: 1, -1, or 0 (draw)
    if (p1_count > p2_count) {
        return 1;
    } else if (p1_count < p2_count) {
        return -1;
    } else {
        return 0;
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
    
    // Check if child was deleted
    if (is_node_deleted(child_idx)) {
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
    
    // Q-value: child_wins stores wins from the parent's perspective
    // (the player who made the move to reach the child node)
    let q = f32(child_wins) / (2.0 * effective_visits);
    
    // Exploration term
    let parent_sqrt = sqrt(f32(max(parent_visits, 1)));
    let u = params.exploration * prior * parent_sqrt / (1.0 + effective_visits);
    
    return q + u;
}

// Select child by sampling from probability distribution based on PUCT scores
fn select_best_child(parent_idx: u32) -> u32 {
    // Check if parent node was deleted
    if (is_node_deleted(parent_idx)) {
        return SELECT_BEST_CHILD_NO_VALID;
    }
    
    let info = node_info[parent_idx];
    let num_children = atomicLoad(&node_info[parent_idx].num_children);
    if (num_children == 0u) {
        return SELECT_BEST_CHILD_NO_CHILDREN;
    }

    // Collect valid children (child_idx != INVALID_INDEX and not deleted)
    var valid_slots: array<u32, 64>;
    var valid_count: u32 = 0u;
    for (var i = 0u; i < num_children; i++) {
        let child_idx = get_child_idx(parent_idx, i);
        if (child_idx != INVALID_INDEX && !is_node_deleted(child_idx)) {
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

    // DEBUG: If this is root node (parent_idx==0 or root_idx), print PUCT scores
    // (We can't actually print from shader, but we can write to diagnostics or check values)
    
    // Convert to probabilities using softmax with temperature (subtract max for numerical stability)
    // Higher virtual loss at parent -> higher effective temperature to spread out threads more
    var probs: array<f32, 64>;
    var sum_exp = 0.0;
    let parent_vl = atomicLoad(&node_vl[parent_idx]);
    let vl_boost = 1.0 + f32(parent_vl) * params.virtual_loss_weight * params.vl_temp_scale;
    let temp = max(params.temperature * vl_boost, 0.00001);
    
    // Track maximum temperature boost for diagnostics
    let boost_as_u32 = u32(vl_boost * 1000.0);
    atomicMax(&diagnostics.max_temp_boost, boost_as_u32);
    
    for (var j = 0u; j < valid_count; j++) {
        let exponent = (scores[j] - max_score) / temp;
        probs[j] = exp(exponent);
        sum_exp += probs[j];
    }

    // Check for NaN/Inf in sum_exp (defensive)
    if (sum_exp <= 0.0) {
        // Fallback to uniform if softmax fails
        return get_child_idx(parent_idx, valid_slots[0u]);
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
    return get_child_idx(parent_idx, valid_slots[valid_count - 1u]);
}

// Try to allocate a new node
fn try_allocate_node(my_workgroup: u32) -> u32 {
    const UNOWNED: u32 = 0xFFFFu;
    
    // Try all free lists owned by my workgroup
    for (var list_idx = 0u; list_idx < 256u; list_idx++) {
        let owner = atomicLoad(&free_list_ownership[list_idx]);
        if (owner == my_workgroup) {
            // This list is owned by my workgroup, try to pop from it
            let local_top = atomicSub(&free_tops[list_idx], 1u);
            if (local_top > 0u && local_top <= params.free_list_capacity) {
                let flat_idx = list_idx * params.free_list_capacity + (local_top - 1u);
                let idx = free_lists_flat[flat_idx];
                if (idx != INVALID_INDEX) {
                    // Use helper to mark clean (clears deleted and dirty)
                    mark_node_clean(idx);
                    // Initialize to prevent garbage data causing stampede
                    atomicStore(&node_info[idx].num_children, 0u);
                    node_info[idx].parent_idx = INVALID_INDEX;
                    node_info[idx].move_id = INVALID_INDEX;
                    // Clear children indices (defense in depth - should already be clear from free_node)
                    for (var i = 0u; i < MAX_CHILDREN; i++) {
                        set_child_idx(idx, i, INVALID_INDEX);
                    }
                    return idx;
                }
            } else {
                // List is empty - restore counter and release ownership
                atomicAdd(&free_tops[list_idx], 1u);
                atomicStore(&free_list_ownership[list_idx], UNOWNED);
            }
        }
    }

    // No owned free lists have nodes - try to claim an unowned free list
    for (var list_idx = 0u; list_idx < 256u; list_idx++) {
        let old_owner = atomicCompareExchangeWeak(&free_list_ownership[list_idx], UNOWNED, my_workgroup);
        if (old_owner.exchanged) {
            // Successfully claimed this free list! Try to pop from it
            let local_top = atomicSub(&free_tops[list_idx], 1u);
            if (local_top > 0u && local_top <= params.free_list_capacity) {
                let flat_idx = list_idx * params.free_list_capacity + (local_top - 1u);
                let idx = free_lists_flat[flat_idx];
                if (idx != INVALID_INDEX) {
                    // Use helper to mark clean (clears deleted and dirty)
                    mark_node_clean(idx);
                    // Initialize to prevent garbage data causing stampede
                    atomicStore(&node_info[idx].num_children, 0u);
                    node_info[idx].parent_idx = INVALID_INDEX;
                    node_info[idx].move_id = INVALID_INDEX;
                    // Clear children indices (defense in depth - should already be clear from free_node)
                    for (var i = 0u; i < MAX_CHILDREN; i++) {
                        set_child_idx(idx, i, INVALID_INDEX);
                    }
                    return idx;
                }
            } else {
                // Claimed an empty list - restore counter and release ownership
                atomicAdd(&free_tops[list_idx], 1u);
                atomicStore(&free_list_ownership[list_idx], UNOWNED);
            }
        }
    }

    // Try global free list
    let global_top = atomicLoad(&global_free_head_alloc);
    if (global_top > 0u) {
        let claimed = atomicSub(&global_free_head_alloc, 1u);
        if (claimed > 0u) {
            let idx = global_free_queue_alloc[claimed - 1u];
            if (idx != INVALID_INDEX && idx != 0u) {
                 // Use helper to mark clean (clears deleted and dirty)
                 mark_node_clean(idx);
                 // Initialize to prevent garbage data causing stampede
                 atomicStore(&node_info[idx].num_children, 0u);
                 node_info[idx].parent_idx = INVALID_INDEX;
                 node_info[idx].move_id = INVALID_INDEX;
                 // Clear children indices (defense in depth - should already be clear from free_node)
                 for (var i = 0u; i < MAX_CHILDREN; i++) {
                     set_child_idx(idx, i, INVALID_INDEX);
                 }
                 return idx;
            }
        } else {
            atomicAdd(&global_free_head_alloc, 1u);
        }
    }

    // Fallback: global allocation
    // Only use alloc_counter during initial tree creation (when it's < max_nodes/2)
    // After that, rely solely on free lists to prevent alloc_counter overflow
    let current_alloc = atomicLoad(&alloc_counter);
    if (current_alloc < params.max_nodes / 2u) {
        let alloc_idx = atomicAdd(&alloc_counter, 1u);
        if (alloc_idx < params.max_nodes) {
            if (alloc_idx == 0u) { return INVALID_INDEX; }
            atomicStore(&node_info[alloc_idx].flags, (1u << 1)); // zero bit
            // Initialize fresh nodes to prevent garbage data
            atomicStore(&node_info[alloc_idx].num_children, 0u);
            node_info[alloc_idx].parent_idx = INVALID_INDEX;
            node_info[alloc_idx].move_id = INVALID_INDEX;
            node_info[alloc_idx].player_at_node = 0;
            atomicStore(&node_visits[alloc_idx], 0);
            atomicStore(&node_wins[alloc_idx], 0);
            atomicStore(&node_vl[alloc_idx], 0);
            return alloc_idx;
        }
    }

    // All allocation sources exhausted
    atomicStore(&expansion_paused, 1u);
    return INVALID_INDEX;
}


// Free a node by adding it to the free list
fn free_node(node_idx: u32, my_workgroup: u32) {
    if (node_idx == INVALID_INDEX || node_idx >= params.max_nodes) {
        return;
    }
    // Clear node data to prevent garbage on reallocation
    atomicStore(&node_info[node_idx].num_children, 0u);
    node_info[node_idx].parent_idx = INVALID_INDEX;
    node_info[node_idx].move_id = INVALID_INDEX;
    atomicStore(&node_visits[node_idx], 0);
    atomicStore(&node_wins[node_idx], 0);
    atomicStore(&node_vl[node_idx], 0);
    // Clear all children indices to prevent stale pointers
    for (var i = 0u; i < MAX_CHILDREN; i++) {
        set_child_idx(node_idx, i, INVALID_INDEX);
        set_child_prior(node_idx, i, 0.0);
    }
    // Use helper to mark deleted (sets deleted, clears zero/dirty)
    mark_node_deleted(node_idx);
    
    let local_top = atomicAdd(&free_tops[my_workgroup], 1u);
    if (local_top < params.free_list_capacity) {
        let flat_idx = my_workgroup * params.free_list_capacity + local_top;
        free_lists_flat[flat_idx] = node_idx;
    } else {
        // Local list full, try global free list
        atomicSub(&free_tops[my_workgroup], 1u); // Restore local counter
        
        let global_idx = atomicAdd(&global_free_head_alloc, 1u);
        if (global_idx < params.max_nodes) {
            global_free_queue_alloc[global_idx] = node_idx;
        }
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
        let num_children = atomicLoad(&node_info[node_idx].num_children);
        for (var i = 0u; i < num_children && i < MAX_CHILDREN; i++) {
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
        board[i] = root_board.cells[i];
    }
    
    // Apply moves along path (skip root at index 0)
    for (var i = 1u; i < path_length; i++) {
        let node_idx = (*path)[i];
        let info = node_info[node_idx];
        let move_id = info.move_id;
        
        if (move_id != INVALID_INDEX && move_id != PASS_MOVE_ID) {
            // Regular move: apply it to the board
            let pos = decode_move(move_id);
            apply_move(&board, pos.x, pos.y, -info.player_at_node);
        }
        // Pass moves (PASS_MOVE_ID): don't modify board, just switches player
        // Player switch is already handled by player_at_node in the tree
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

    // Double-check: did we race with someone who just finished expansion?
    // If we read num_children=0 (stale) before acquiring the lock, but now it's > 0,
    // it means someone else expanded it and set it back to READY.
    // We just transitioned it to EXPANDING, so we own it.
    // We should check if it's already expanded.
    let nc = atomicLoad(&node_info[node_idx].num_children);
    if (nc > 0u) {
        // Check if children are valid (not just garbage num_children)
        let first_child = get_child_idx(node_idx, 0u);
        if (first_child != INVALID_INDEX) {
            // Already expanded with valid children!
            // Release lock (set back to READY).
            atomicStore(&node_state[node_idx], NODE_STATE_READY);
            atomicAdd(&diagnostics.exp_lock_retry, 1u); // Log this specific race
            return false;
        }
        // num_children > 0 but first child is INVALID - this is garbage data
        // Proceed with expansion and reset num_children
        atomicStore(&node_info[node_idx].num_children, 0u);
        // Clear all children_indices to prevent reading garbage
        for (var clear_i = 0u; clear_i < MAX_CHILDREN; clear_i++) {
            set_child_idx(node_idx, clear_i, INVALID_INDEX);
        }
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
                    // Memory pressure! Rollback: free already allocated children
                    for (var k = 0u; k < num_children; k++) {
                        let allocated_child = get_child_idx(node_idx, k);
                        free_node(allocated_child, my_workgroup);
                    }
                    atomicStore(&node_state[node_idx], NODE_STATE_READY);
                    return false;
                }
                // Set up child node info and mark as dirty
                mark_node_dirty(child_idx);
                node_info[child_idx].parent_idx = node_idx;
                node_info[child_idx].move_id = encode_move(x, y);
                atomicStore(&node_info[child_idx].num_children, 0u);
                node_info[child_idx].player_at_node = -player;
                atomicStore(&node_info[child_idx].flags, FLAG_DIRTY); // mark dirty, not deleted
                node_info[child_idx]._pad = 0u;
                node_info[child_idx]._pad2 = 0u;
                node_info[child_idx]._pad3 = 0u;
                atomicStore(&node_state[child_idx], NODE_STATE_READY);
                set_child_idx(node_idx, num_children, child_idx);
                // Temporarily set prior to 1.0, will normalize after expansion
                set_child_prior(node_idx, num_children, 1.0);
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
    
    // PASS MOVE HANDLING: If no regular moves found, check if opponent has moves
    // If opponent has moves, this is a pass situation (not terminal)
    if (num_children == 0u) {
        let opponent_moves = count_valid_moves(board, -player);
        if (opponent_moves > 0) {
            // Create a pass move child
            let child_idx = try_allocate_node(my_workgroup);
            if (child_idx == INVALID_INDEX) {
                // Memory pressure! Can't create pass move
                atomicStore(&node_state[node_idx], NODE_STATE_READY);
                return false;
            }
            // Set up pass move child
            node_info[child_idx].parent_idx = node_idx;
            node_info[child_idx].move_id = PASS_MOVE_ID;  // Special pass move ID
            atomicStore(&node_info[child_idx].num_children, 0u);
            node_info[child_idx].player_at_node = -player;  // Switch player
            atomicStore(&node_info[child_idx].flags, 0u);
            node_info[child_idx]._pad = 0u;
            node_info[child_idx]._pad2 = 0u;
            node_info[child_idx]._pad3 = 0u;
            atomicStore(&node_state[child_idx], NODE_STATE_READY);
            set_child_idx(node_idx, 0u, child_idx);
            set_child_prior(node_idx, 0u, 1.0);  // 100% probability (only move)
            num_children = 1u;
        }
        // else: num_children stays 0, node will be treated as terminal
    }
    
    // Normalize priors to sum to 1.0 (uniform distribution)
    if (num_children > 0u) {
        let uniform_prior = 1.0 / f32(num_children);
        for (var i = 0u; i < num_children; i++) {
            set_child_prior(node_idx, i, uniform_prior);
        }
    }
    
    // Update parent node info
    atomicStore(&node_info[node_idx].num_children, num_children);
    
    // DEBUG: Count total children generated to verify expansion is working
    atomicAdd(&diagnostics.total_children_gen, num_children);
    
    // DEBUG: Log root expansion
    if (node_idx == 0u) {
        var payload: array<u32, 255>;
        payload[0] = num_children;  // Number of children
        payload[1] = node_idx;      // Node index (0)
        for (var i = 2u; i < 255u; i++) { payload[i] = 0u; }
        write_urgent_event(URGENT_EVENT_DEBUG, &payload);
    }
    
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

@compute @workgroup_size(64)
fn init_allocator(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Support 2D dispatch for large node counts (> 65535 workgroups)
    // We assume the X dimension is filled to 65535 workgroups if Y > 0
    // Stride = 65535 * 64 = 4194240
    let node_idx = global_id.x + global_id.y * 4194240u;
    
    // Skip node 0 (root) and out of bounds
    if (node_idx == 0u || node_idx >= params.max_nodes) {
        return;
    }

    // Initialize node state and clear all data to prevent garbage
    atomicStore(&node_state[node_idx], NODE_STATE_EMPTY);
    atomicStore(&node_info[node_idx].num_children, 0u);
    node_info[node_idx].parent_idx = INVALID_INDEX;
    node_info[node_idx].move_id = INVALID_INDEX;
    node_info[node_idx].player_at_node = 0;
    atomicStore(&node_info[node_idx].flags, 0u);
    atomicStore(&node_visits[node_idx], 0);
    atomicStore(&node_wins[node_idx], 0);
    atomicStore(&node_vl[node_idx], 0);
    
    // Clear children_indices (critical to prevent garbage pointers)
    for (var i = 0u; i < MAX_CHILDREN; i++) {
        set_child_idx(node_idx, i, INVALID_INDEX);
    }
    
    atomicAdd(&diagnostics.init_nodes_count, 1u);
    
    // Add to free list
    // Distribute nodes round-robin across 256 lists
    let list_idx = node_idx % 256u;
    let top = atomicAdd(&free_tops[list_idx], 1u);
    if (top < params.free_list_capacity) { // Check capacity of free list
        let flat_idx = list_idx * params.free_list_capacity + top;
        free_lists_flat[flat_idx] = node_idx;
    }
}

@compute @workgroup_size(64)
fn gather_root_stats(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot = global_id.x;
    if (slot >= 64u) { return; }
    
    // Use the current root index from params
    // children_indices is flat array: node_idx * MAX_CHILDREN + slot
    let root_idx = params.root_idx;
    
    // Check if root was deleted
    if (is_node_deleted(root_idx)) {
        root_stats[slot].move_id = INVALID_INDEX;
        root_stats[slot].visits = 0;
        root_stats[slot].wins = 0;
        return;
    }
    
    let child_idx = children_indices[root_idx * MAX_CHILDREN + slot];
    
    if (child_idx == INVALID_INDEX || is_node_deleted(child_idx)) {
        root_stats[slot].move_id = INVALID_INDEX;
        root_stats[slot].visits = 0;
        root_stats[slot].wins = 0;
    } else {
        root_stats[slot].move_id = node_info[child_idx].move_id;
        root_stats[slot].visits = atomicLoad(&node_visits[child_idx]);
        root_stats[slot].wins = atomicLoad(&node_wins[child_idx]);
    }
}
// Defragmentation kernel: Consolidate free lists and release unused ownership
// This runs after pruning to prevent free list fragmentation
@compute @workgroup_size(256)
fn defragment_free_lists(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let list_idx = global_id.x;
    if (list_idx >= 256u) { return; }
    
    let owner = atomicLoad(&free_list_ownership[list_idx]);
    let top = atomicLoad(&free_tops[list_idx]);
    
    // If this list is owned but empty, release ownership back to UNOWNED
    const UNOWNED: u32 = 0xFFFFu;
    if (owner != UNOWNED && top == 0u) {
        atomicStore(&free_list_ownership[list_idx], UNOWNED);
    }
    
    // If this list has many nodes, transfer some to global free list
    // This helps redistribute nodes for better workgroup access
    if (top > params.free_list_capacity / 2u) {
        // Transfer half of the nodes to global free list
        let transfer_count = top / 2u;
        for (var i = 0u; i < transfer_count; i++) {
            let local_top = atomicSub(&free_tops[list_idx], 1u);
            if (local_top > 0u && local_top <= params.free_list_capacity) {
                let flat_idx = list_idx * params.free_list_capacity + (local_top - 1u);
                let node_idx = free_lists_flat[flat_idx];
                if (node_idx != INVALID_INDEX) {
                    // Add to global free list
                    let global_idx = atomicAdd(&global_free_head_alloc, 1u);
                    if (global_idx < params.max_nodes) {
                        global_free_queue_alloc[global_idx] = node_idx;
                    } else {
                        // Global list full, restore local counter
                        atomicAdd(&free_tops[list_idx], 1u);
                        break;
                    }
                }
            } else {
                // Restore counter if underflow
                atomicAdd(&free_tops[list_idx], 1u);
                break;
            }
        }
    }
}// =============================================================================
// Incremental MCTS Execution (Fast Phases Only)
// =============================================================================
// This file contains kernels for incremental MCTS execution and chunked rollouts.
// Designed to work with the thread state buffer (Group 7).

/// Incremental MCTS step: ONE phase per thread per dispatch.
/// Handles SELECTION, EXPANSION, and BACKPROP phases.
/// ROLLOUT_ACTIVE threads are skipped (handled by separate rollout_chunk_kernel).
@compute @workgroup_size(64)
fn incremental_mcts_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let thread_id = global_id.x;
    
    // Bounds check
    if (thread_id >= params.num_threads) {
        return;
    }
    
    var state = thread_states[thread_id];
    
    // Track phase distribution for diagnostics
    if (state.phase == PHASE_SELECTION) {
        atomicAdd(&diagnostics.phase_selection_count, 1u);
    } else if (state.phase == PHASE_EXPANSION) {
        atomicAdd(&diagnostics.phase_expansion_count, 1u);
    } else if (state.phase == PHASE_ROLLOUT_ACTIVE) {
        atomicAdd(&diagnostics.phase_rollout_count, 1u);
    } else if (state.phase == PHASE_BACKPROP) {
        atomicAdd(&diagnostics.phase_backprop_count, 1u);
    } else if (state.phase == PHASE_IDLE) {
        atomicAdd(&diagnostics.phase_idle_count, 1u);
    } else if (state.phase == PHASE_FINISHED) {
        atomicAdd(&diagnostics.phase_finished_count, 1u);
    }
    
    // Skip if finished or in rollout (handled by separate kernel)
    if (state.phase == PHASE_FINISHED || state.phase == PHASE_ROLLOUT_ACTIVE) {
        return;
    }
    
    // Execute ONE step based on current phase
    if (state.phase == PHASE_SELECTION) {
        // Traverse ONE node down the tree
        let current = state.current_node;
        
        // Add virtual loss to current node before selecting child
        // This balances the VL decrement in backprop, which walks the entire path
        atomicAdd(&node_vl[current], 1);
        
        // Check if current node is terminal or has no children
        let num_children = atomicLoad(&node_info[current].num_children);
        if (num_children == 0u) {
            // Reached a leaf - move to expansion
            state.phase = PHASE_EXPANSION;
            state.leaf_player = node_info[current].player_at_node;
        } else {
            // Calculate PUCT scores for all children
            var puct_scores: array<f32, MAX_CHILDREN>;
            // Initialize to zero to avoid undefined behavior
            for (var init_i = 0u; init_i < MAX_CHILDREN; init_i++) {
                puct_scores[init_i] = 0.0;
            }
            var valid_count = 0u;
            
            let parent_visits = f32(atomicLoad(&node_visits[current]));
            let parent_vl = f32(atomicLoad(&node_vl[current]));
            // Include parent VL in sqrt calculation to account for in-flight selections
            let sqrt_parent = sqrt(parent_visits + parent_vl + 1.0);
            
            // Conditionally pre-increment VL for all children and capture old values
            var old_vl: array<i32, MAX_CHILDREN>;
            if (params.use_vl_preincrement != 0u) {
                for (var i = 0u; i < num_children && i < MAX_CHILDREN; i++) {
                    let child_idx = get_child_idx(current, i);
                    if (child_idx != INVALID_INDEX) {
                        old_vl[i] = atomicAdd(&node_vl[child_idx], 1);
                    } else {
                        old_vl[i] = 0;
                    }
                }
            }
            
            for (var i = 0u; i < num_children && i < MAX_CHILDREN; i++) {
                let child_idx = get_child_idx(current, i);
                if (child_idx == INVALID_INDEX) { continue; }
                
                let visits = f32(atomicLoad(&node_visits[child_idx]));
                let wins = f32(atomicLoad(&node_wins[child_idx]));
                let vl = f32(select(atomicLoad(&node_vl[child_idx]), old_vl[i], params.use_vl_preincrement != 0u));
                let prior = children_priors[current * MAX_CHILDREN + i];
                
                // Q-value (parent's perspective)
                // Wins are in 0-2 range, so normalize to 0-1
                var q = 0.5;
                if (visits + vl > 0.0) {
                    q = wins / (2.0 * (visits + vl));
                }
                
                // U-value (exploration bonus)
                let u = params.exploration * prior * sqrt_parent / (1.0 + visits + vl);
                
                puct_scores[i] = q + u;
                valid_count++;
            }
            
            if (valid_count == 0u) {
                // No valid children - treat as leaf
                state.phase = PHASE_EXPANSION;
                state.leaf_player = node_info[current].player_at_node;
            } else {
                var selected_child = INVALID_INDEX;
                var selected_idx = 0u;
                
                // Temperature = 0 or very small: argmax selection with tie-breaking
                if (params.temperature < 0.01) {
                    // Find max PUCT score
                    var max_puct = -1000.0;
                    for (var i = 0u; i < num_children && i < MAX_CHILDREN; i++) {
                        let child_idx = get_child_idx(current, i);
                        if (child_idx == INVALID_INDEX) { continue; }
                        max_puct = max(max_puct, puct_scores[i]);
                    }
                    
                    // Collect all children with max score
                    var max_children: array<u32, MAX_CHILDREN>;
                    var max_count = 0u;
                    for (var i = 0u; i < num_children && i < MAX_CHILDREN; i++) {
                        let child_idx = get_child_idx(current, i);
                        if (child_idx == INVALID_INDEX) { continue; }
                        
                        if (abs(puct_scores[i] - max_puct) < 0.0001) {
                            max_children[max_count] = i;
                            max_count++;
                        }
                    }
                    
                    // Randomly select among tied children
                    if (max_count > 0u) {
                        let rng_val = pcg_hash(atomicAdd(&diagnostics.global_rollout_counter, 1u));
                        // For small max_count, use high bits to avoid modulo bias
                        // High 8 bits give range [0, 256), scale to [0, max_count)
                        let rand_idx = ((rng_val >> 24u) * max_count) >> 8u;
                        selected_idx = max_children[rand_idx];
                        selected_child = get_child_idx(current, selected_idx);
                    }
                } else {
                    // Apply softmax with temperature and sample
                    // Find max PUCT for numerical stability
                    var max_puct = -1000.0;
                    for (var i = 0u; i < num_children && i < MAX_CHILDREN; i++) {
                        let child_idx = get_child_idx(current, i);
                        if (child_idx == INVALID_INDEX) { continue; }
                        max_puct = max(max_puct, puct_scores[i]);
                    }
                    
                    var exp_sum = 0.0;
                    var exp_scores: array<f32, MAX_CHILDREN>;
                    // Initialize to zero to avoid undefined behavior
                    for (var init_i = 0u; init_i < MAX_CHILDREN; init_i++) {
                        exp_scores[init_i] = 0.0;
                    }
                    
                    for (var i = 0u; i < num_children && i < MAX_CHILDREN; i++) {
                        let child_idx = get_child_idx(current, i);
                        if (child_idx == INVALID_INDEX) { continue; }
                        
                        // Standard softmax: exp((score - max) / temp)
                        // Higher scores → exponent closer to 0 → exp closer to 1 → higher probability
                        let exp_score = exp((puct_scores[i] - max_puct) / params.temperature);
                        exp_scores[i] = exp_score;
                        exp_sum += exp_score;
                    }
                    
                    // Sample from softmax distribution using independent RNG
                    let rng_val = pcg_hash(atomicAdd(&diagnostics.global_rollout_counter, 1u));
                    // Use power-of-2 masking: 2^20 = 1048576 for good precision
                    let rand_val = rng_val & 1048575u;
                    let sample = f32(rand_val) / 1048576.0;
                    
                    var cumulative = 0.0;
                    
                    for (var i = 0u; i < num_children && i < MAX_CHILDREN; i++) {
                        let child_idx = get_child_idx(current, i);
                        if (child_idx == INVALID_INDEX) { continue; }
                        
                        cumulative += exp_scores[i] / exp_sum;
                        if (sample <= cumulative) {
                            selected_child = child_idx;
                            selected_idx = i;
                            break;
                        }
                    }
                    
                    // Fallback to last valid child if sampling failed (e.g., NaN/Inf from extreme temperatures)
                    if (selected_child == INVALID_INDEX) {
                        for (var i = 0u; i < num_children && i < MAX_CHILDREN; i++) {
                            let child_idx = get_child_idx(current, i);
                            if (child_idx != INVALID_INDEX) {
                                selected_child = child_idx;
                                selected_idx = i;
                            }
                        }
                    }
                }
                
                if (selected_child != INVALID_INDEX) {
                    // Conditionally decrement VL for non-selected children
                    if (params.use_vl_preincrement != 0u) {
                        for (var i = 0u; i < num_children && i < MAX_CHILDREN; i++) {
                            let child_idx = get_child_idx(current, i);
                            if (child_idx != INVALID_INDEX && child_idx != selected_child) {
                                atomicAdd(&node_vl[child_idx], -1);
                            }
                        }
                    }
                    // Note: VL for selected_child will be added when it becomes 'current' in next iteration
                    
                    // Add to path
                    state.path[state.path_len] = selected_child;
                    state.path_len++;
                    state.current_node = selected_child;
                } else {
                    // Should never happen, but treat as leaf
                    state.phase = PHASE_EXPANSION;
                    state.leaf_player = node_info[current].player_at_node;
                }
            }
        }
    } else if (state.phase == PHASE_EXPANSION) {
        // Try to expand current node (the leaf) using existing expand_node logic
        let leaf_idx = state.current_node;
        
        // Reconstruct board (use existing function)
        var temp_path = state.path;
        var board = reconstruct_board(&temp_path, state.path_len);
        
        // Try to expand using existing logic
        let my_workgroup = workgroup_id.x;
        let expanded = expand_node(leaf_idx, &board, my_workgroup);
        
        if (expanded) {
            atomicAdd(&diagnostics.expansion_success, 1u);
        }
        
        // DO NOT remove VL here! VL must stay until backprop completes.
        // Backprop will remove VL as it walks up the tree.
        
        // PHASE 2: Enqueue rollout instead of blocking
        // Tree worker queues rollout and becomes available for other work
        let enqueued = enqueue_rollout(board, leaf_idx, state.leaf_player);
        
        // Transition to IDLE (available for backprop or new MCTS cycle)
        state.phase = PHASE_IDLE;
        
    } else if (state.phase == PHASE_BACKPROP) {
        // PHASE 4: Parent pointer backprop (walk up tree ONE NODE AT A TIME)
        let node_idx = state.current_node;
        
        // Decrement virtual loss (added during selection)
        atomicAdd(&node_vl[node_idx], -1);
        
        // Update this node's stats
        atomicAdd(&node_visits[node_idx], 1);
        
        // Calculate reward from parent's perspective
        let player_at_node = node_info[node_idx].player_at_node;
        let player_who_moved = -player_at_node; // Parent is opponent
        var reward = i32(state.rollout_result);
        if (player_who_moved != state.leaf_player) {
            reward = 2 - reward; // Flip perspective
        }
        atomicAdd(&node_wins[node_idx], reward);
        
        // Move to parent (ONE STEP - will continue in next dispatch)
        let parent_idx = node_info[node_idx].parent_idx;
        if (parent_idx == INVALID_INDEX) {
            // Reached root - backprop complete, transition to IDLE
            state.phase = PHASE_IDLE;
        } else {
            // Continue backprop at parent in NEXT dispatch
            state.current_node = parent_idx;
            // Stay in PHASE_BACKPROP - will process parent next time
        }
    } else if (state.phase == PHASE_IDLE) {
        // PHASE 2 + 5: Tree worker looking for work with dynamic load balancing
        // Priority 1: Process completed rollouts (backprop queue)
        let backprop_job = dequeue_backprop();
        
        if (backprop_job.leaf_node_idx != INVALID_INDEX) {
            // Got backprop work - start backpropping via parent pointers (Phase 4)
            state.current_node = backprop_job.leaf_node_idx;
            state.rollout_result = backprop_job.result;
            state.leaf_player = backprop_job.leaf_player;
            state.phase = PHASE_BACKPROP;
        } else {
            // No backprop work - check queue balance before starting new MCTS cycle
            // If rollout queue is overloaded, skip new work to avoid overwhelming it
            let rollout_size = atomicLoad(&rollout_head);
            let queue_capacity = arrayLength(&rollout_queue);
            let queue_usage = (rollout_size * 100u) / queue_capacity;
            
            // Only start new MCTS cycles if rollout queue isn't near capacity
            if (queue_usage < 90u) {
                // Start new MCTS cycle (will eventually enqueue more rollouts)
                state.phase = PHASE_SELECTION;
                state.current_node = params.root_node;
                state.path_len = 1u;
                state.path[0] = params.root_node;
            }
            // else: Stay IDLE if queue is overloaded - rollout workers need to catch up
        }
    }
    
    // Save state for next dispatch
    thread_states[thread_id] = state;
}

// =============================================================================
// Chunked Rollout Kernel (Separate from Fast Phases)
// =============================================================================

/// Chunked rollout: Execute N moves per dispatch for rollout jobs from queue.
/// This prevents fast phases (selection/backprop) from waiting for slow rollouts.
@compute @workgroup_size(64)
fn rollout_chunk_kernel(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let thread_id = global_id.x;
    
    // Bounds check
    if (thread_id >= params.num_threads) {
        return;
    }
    
    // PHASE 3: Dequeue rollout job from queue
    let job = dequeue_rollout();
    
    // If no work, exit early
    if (job.leaf_node_idx == INVALID_INDEX) {
        return;
    }
    
    // Track that we're executing a rollout
    atomicAdd(&diagnostics.rollouts, 1u);
    
    // Initialize rollout state from job
    var rollout_board = job.position;
    var rollout_player = job.leaf_player;
    var rollout_moves_remaining = 60u;
    var rollout_result = 1u; // Default to draw
    
    // Get thread's RNG state
    var state = thread_states[thread_id];
    rng_state = state.rng_seed;
    
    // Test mode: generate random rollout result
    if (params.use_random_rollouts != 0u) {
        // Track thread ID range
        atomicMin(&diagnostics.random_rollout_min_tid, thread_id);
        atomicMax(&diagnostics.random_rollout_max_tid, thread_id);
        
        // Track which player perspective this rollout is from
        if (job.leaf_player == 1) {
            atomicAdd(&diagnostics.random_rollout_from_p1, 1u);
        } else {
            atomicAdd(&diagnostics.random_rollout_from_p2, 1u);
        }
        
        // NEW APPROACH: Use independent global counter for seeding each rollout
        // This completely bypasses selection-phase RNG state modifications
        let rollout_id = atomicAdd(&diagnostics.global_rollout_counter, 1u);
        rng_state = pcg_hash(rollout_id);
        
        // Generate random score
        rng_state = pcg_hash(rng_state);
        let p1_score = rng_state % 65u;
        atomicAdd(&diagnostics.random_rollout_p1_score_sum, p1_score);
        let p2_score = 64u - p1_score;
        
        // Determine winner
        var winner = 0;
        if (p1_score > p2_score) {
            winner = 1;
            atomicAdd(&diagnostics.random_rollout_p1_wins_raw, 1u);
        } else if (p1_score < p2_score) {
            winner = -1;
            atomicAdd(&diagnostics.random_rollout_p1_losses_raw, 1u);
        }
        // else draw (p1_score == p2_score == 32)
        
        // Convert to leaf_player's perspective
        if (winner == job.leaf_player) {
            rollout_result = 2u;  // Win
        } else if (winner == 0) {
            rollout_result = 1u;  // Draw
        } else {
            rollout_result = 0u;  // Loss
        }
        
        // Count results for diagnostics
        // Update diagnostics - track from player 1's perspective for consistency
        var result_from_p1_perspective = rollout_result;
        if (job.leaf_player == -1) {
            // Flip perspective: leaf_player's win = p1's loss
            if (rollout_result == 2u) {
                result_from_p1_perspective = 0u; // Win for -1 = loss for 1
            } else if (rollout_result == 0u) {
                result_from_p1_perspective = 2u; // Loss for -1 = win for 1
            }
        }
        
        if (result_from_p1_perspective == 2u) {
            atomicAdd(&diagnostics.random_rollout_wins, 1u);
        } else if (result_from_p1_perspective == 1u) {
            atomicAdd(&diagnostics.random_rollout_draws, 1u);
        } else {
            atomicAdd(&diagnostics.random_rollout_losses, 1u);
        }
        
        // Update thread's RNG state
        thread_states[thread_id].rng_seed = rng_state;
        
        // Enqueue result and return early
        let enqueued = enqueue_backprop(job.leaf_node_idx, rollout_result, job.leaf_player);
        return;
    }
    
    // Real rollout mode: run full game simulation
    
    // Run rollout to completion (not chunked anymore for simplicity)
    var consecutive_passes = 0u;
    var moves_remaining = 60u; // Safety limit
    
    // Simulate moves until terminal
    loop {
        // Check if game is terminal
        if (consecutive_passes >= 2u || moves_remaining == 0u) {
            // Game over - count pieces to determine winner
            var p1_count = 0;
            var p2_count = 0;
            for (var j = 0; j < 64; j++) {
                if (rollout_board[j] == 1) {
                    p1_count++;
                } else if (rollout_board[j] == -1) {
                    p2_count++;
                }
            }
            
            // Determine winner
            var winner = 0;
            if (p1_count > p2_count) {
                winner = 1;
            } else if (p1_count < p2_count) {
                winner = -1;
            }
            
            // Convert to reward from leaf_player's perspective
            if (winner == job.leaf_player) {
                rollout_result = 2u;  // Win
            } else if (winner == 0) {
                rollout_result = 1u;  // Draw
            } else {
                rollout_result = 0u;  // Loss
            }
            
            // PHASE 3: Enqueue result to backprop queue
            let enqueued = enqueue_backprop(job.leaf_node_idx, rollout_result, job.leaf_player);
            break;
        }
        
        // Collect valid moves inline (avoid helper function to dodge Naga bug)
        var valid_moves: array<i32, 64>;
        var num_moves = 0;
        
        for (var sq = 0; sq < 64; sq++) {
            if (rollout_board[sq] != 0) {
                continue;  // Square occupied
            }
            
            let row = sq / 8;
            let col = sq % 8;
            var is_valid = false;
            
            // Check all 8 directions for valid captures
            for (var dir = 0; dir < 8; dir++) {
                let dx = array<i32, 8>(-1, -1, -1, 0, 0, 1, 1, 1)[dir];
                let dy = array<i32, 8>(-1, 0, 1, -1, 1, -1, 0, 1)[dir];
                
                var found_opponent = false;
                var steps = 1;
                
                // Walk in this direction
                loop {
                    let nx = i32(col) + dx * steps;
                    let ny = i32(row) + dy * steps;
                    
                    if (nx < 0 || nx >= 8 || ny < 0 || ny >= 8) {
                        break;  // Out of bounds
                    }
                    
                    let check_sq = ny * 8 + nx;
                    let piece = rollout_board[check_sq];
                    
                    if (piece == -rollout_player) {
                        found_opponent = true;
                        steps++;
                    } else if (piece == rollout_player && found_opponent) {
                        is_valid = true;  // Valid capture sequence
                        break;
                    } else {
                        break;  // Empty or our piece without opponent between
                    }
                    
                    if (steps > 8) { break; }  // Safety limit
                }
                
                if (is_valid) { break; }
            }
            
            if (is_valid) {
                valid_moves[num_moves] = sq;
                num_moves++;
            }
        }
        
        // If no moves, pass
        if (num_moves == 0) {
            consecutive_passes++;
            rollout_player = -rollout_player;
            continue;
        }
        consecutive_passes = 0u;
        
        // Select random move
        let move_idx = i32(rand_f32() * f32(num_moves));
        let selected_sq = valid_moves[move_idx];
        let sel_row = selected_sq / 8;
        let sel_col = selected_sq % 8;
        
        // Place piece
        rollout_board[selected_sq] = rollout_player;
        
        // Flip pieces in all valid directions
        for (var dir = 0; dir < 8; dir++) {
            let dx = array<i32, 8>(-1, -1, -1, 0, 0, 1, 1, 1)[dir];
            let dy = array<i32, 8>(-1, 0, 1, -1, 1, -1, 0, 1)[dir];
            
            var found_opponent = false;
            var steps = 1;
            var flip_count = 0;
            
            // Check if this direction has valid captures
            loop {
                let nx = i32(sel_col) + dx * steps;
                let ny = i32(sel_row) + dy * steps;
                
                if (nx < 0 || nx >= 8 || ny < 0 || ny >= 8) {
                    break;
                }
                
                let check_sq = ny * 8 + nx;
                let piece = rollout_board[check_sq];
                
                if (piece == -rollout_player) {
                    found_opponent = true;
                    flip_count++;
                    steps++;
                } else if (piece == rollout_player && found_opponent) {
                    // Valid capture - flip all pieces in between
                    for (var f = 1; f <= flip_count; f++) {
                        let flip_x = i32(sel_col) + dx * f;
                        let flip_y = i32(sel_row) + dy * f;
                        let flip_sq = flip_y * 8 + flip_x;
                        rollout_board[flip_sq] = rollout_player;
                    }
                    break;
                } else {
                    break;
                }
                
                if (steps > 8) { break; }
            }
        }
        
        // Update state
        rollout_player = -rollout_player;
        rollout_moves_remaining--;
        moves_remaining--;
    }
    
    // No need to save RNG state - we use independent global counter now
    thread_states[thread_id] = state;
}
// ============================================================================
// Kernel: collect_leaf_candidates
// 
// Purpose: Scan all allocated nodes to find leaves suitable for pre-computation.
// A leaf is suitable if:
//   - It has num_children == 0 (unexpanded)
//   - It has visits > 0 (has been encountered during selection)
//   - Its parent has high visit count (prioritize popular subtrees)
//
// Each workgroup scans 64 nodes. Threads write candidates
// to a local buffer, then the workgroup writes them to the global candidates
// buffer using an atomic counter.
//
// Buffer Layout:
//   - candidate_count: atomic u32 (number of candidates written)
//   - leaf_candidates: array of LeafCandidate structs (node_id, score)
//
// This kernel is called before each batch of incremental steps to identify
// which leaves need their legal moves pre-computed.
// ============================================================================

// Shared memory for workgroup-level aggregation
var<workgroup> wg_candidates: array<LeafCandidate, 64>;
var<workgroup> wg_candidate_count: atomic<u32>;

@compute
@workgroup_size(64)
fn collect_leaf_candidates(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let thread_idx = local_id.x;
    let node_idx = global_id.x;
    
    // Initialize workgroup-shared atomic counter (first thread only)
    if (thread_idx == 0u) {
        atomicStore(&wg_candidate_count, 0u);
    }
    workgroupBarrier();
    
    // Each thread examines one node
    var is_candidate = false;
    var candidate_score: u32 = 0u;
    
    if (node_idx < params.max_nodes) {
        let visits = atomicLoad(&node_visits[node_idx]);
        
        // Load num_children atomically
        let num_children = atomicLoad(&node_info[node_idx].num_children);
        let parent_idx = node_info[node_idx].parent_idx;
        
        // Check if this is a suitable leaf:
        // 1. Has been visited (visits > 0)
        // 2. Is unexpanded (num_children == 0)
        // 3. Is not the root (parent_idx != INVALID_INDEX)
        if (visits > 0 && num_children == 0u && parent_idx != INVALID_INDEX) {
            is_candidate = true;
            
            // Score by parent's visit count (for prioritization)
            let parent_visits = atomicLoad(&node_visits[parent_idx]);
            candidate_score = u32(parent_visits);
        }
    }
    
    // Thread writes its candidate to workgroup-local buffer
    if (is_candidate) {
        let local_slot = atomicAdd(&wg_candidate_count, 1u);
        if (local_slot < 64u) {
            wg_candidates[local_slot] = LeafCandidate(node_idx, candidate_score);
        }
    }
    
    // Wait for all threads in workgroup to finish
    workgroupBarrier();
    
    // First thread in workgroup writes all candidates to global buffer
    if (thread_idx == 0u) {
        let count = atomicLoad(&wg_candidate_count);
        if (count > 0u) {
            // Reserve space in global buffer using atomic counter
            let global_offset = atomicAdd(&candidate_count, count);
            
            // Write candidates to global buffer
            for (var i = 0u; i < count && i < 64u; i++) {
                let global_slot = global_offset + i;
                // Note: Buffer might overflow if too many leaves exist
                // In practice, we'll sort and take top K, so overflow is acceptable
                if (global_slot < arrayLength(&leaf_candidates)) {
                    leaf_candidates[global_slot] = wg_candidates[i];
                }
            }
        }
    }
}

// ============================================================================
// Kernel: bitonic_sort_candidates
// 
// Purpose: Sort candidates by score (descending) using parallel bitonic sort.
// Each candidate is 2 u32s: [score, node_idx]
//
// Bitonic sort is a comparison-based parallel sorting algorithm that works
// in stages. For N elements, it takes log2(N) stages, each with multiple steps.
//
// Parameters passed via push constants or uniform:
//   - stage: Current stage (0 to log2(N)-1)
//   - step: Current step within stage (0 to stage)
//
// This kernel is called multiple times with different stage/step values.
// After all stages complete, candidates are sorted by score descending.
// ============================================================================

@compute
@workgroup_size(64)
fn bitonic_sort_candidates(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let thread_id = global_id.x;
    
    // Get total number of candidates from counter
    let num_candidates = atomicLoad(&candidate_count);
    
    // Round up to next power of 2 for bitonic sort
    var n = 1u;
    while (n < num_candidates) {
        n = n << 1u;
    }
    
    // Each thread handles one pair of elements
    let i = thread_id;
    
    if (i >= n / 2u) {
        return; // Thread has no work
    }
    
    // Bitonic sort: log2(n) stages, each stage has multiple steps
    // For simplicity, we'll do this in multiple kernel dispatches
    // Each dispatch handles one (stage, step) pair
    
    // We'll use params.turn_number to pass stage and params.game_type to pass step
    // (abusing existing fields to avoid adding new params struct)
    let stage = params.turn_number;
    let step = params.game_type;
    
    // Calculate partner index for compare-and-swap
    let block_size = 2u << stage;
    let block_start = (i / (block_size / 2u)) * block_size;
    let offset_in_block = i % (block_size / 2u);
    
    var partner: u32;
    let step_size = 1u << step;
    
    if (offset_in_block < step_size) {
        partner = block_start + offset_in_block + step_size;
    } else {
        return; // Already handled by partner
    }
    
    // Skip if partner is out of bounds
    if (partner >= n || i >= arrayLength(&leaf_candidates) || partner >= arrayLength(&leaf_candidates)) {
        return;
    }
    
    // Load candidates and their scores
    let cand_a = leaf_candidates[i];
    let cand_b = leaf_candidates[partner];
    
    let score_a = cand_a.score;
    let score_b = cand_b.score;
    
    // Determine sort direction (ascending or descending for this block)
    // For descending overall sort, we want largest scores first
    let ascending = ((i >> stage) & 1u) == 0u;
    
    // Compare and swap if needed
    var should_swap = false;
    if (ascending) {
        should_swap = score_a > score_b;
    } else {
        should_swap = score_a < score_b;
    }
    
    if (should_swap) {
        // Swap the two LeafCandidate structs
        leaf_candidates[i] = cand_b;
        leaf_candidates[partner] = cand_a;
    }
}
