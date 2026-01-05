// =============================================================================
// Two-Phase Pruning (Design Doc Algorithm)
// =============================================================================

struct RerootParams {
    move_x: u32,
    move_y: u32,
    current_root: u32,
    max_nodes: u32,  // Work queue capacity
};

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

    for (var i = 0u; i < num_children && i < MAX_CHILDREN; i++) {
        let child_idx = get_child_idx(current_root, i);
        if (child_idx == INVALID_INDEX) { continue; }
        
        let child_info = node_info[child_idx];
        if (child_info.move_id == target_move_id) {
            // This is the survivor
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
        if (child == INVALID_INDEX) {
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

    // Rollout phase
    atomicAdd(&diagnostics.rollouts, 1u);
    var rollout_board = reconstruct_board(&path, path_len);
    let leaf_player = node_info[current].player_at_node;
    let rollout_result = simulate_game(&rollout_board, leaf_player);

    // Backpropagation phase
    for (var i = path_len; i > 0u; i--) {
        let node_idx = path[i - 1u];
        atomicAdd(&node_vl[node_idx], -1);
        let v = atomicAdd(&node_visits[node_idx], 1);
        
        // CRITICAL: Node stats use PARENT'S perspective (player who moved TO this node)
        // This matches CPU MCTS convention where wins[node] = wins from parent's POV
        // parent = the player who chose this node (i.e., opponent of player_at_node)
        let player_at_node = node_info[node_idx].player_at_node;
        let player_who_moved = -player_at_node;  // Parent is opponent
        var reward = rollout_result;
        if (player_who_moved != leaf_player) {
            reward = 2 - rollout_result;
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
    turn_number: u32, // NEW: unique per-turn identifier
    free_list_capacity: u32, // Capacity per free list (max_nodes / 256 rounded up)
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
@group(0) @binding(9) var<storage, read_write> global_free_queue_alloc: array<u32>;
@group(0) @binding(10) var<storage, read_write> global_free_head_alloc: atomic<u32>;
@group(0) @binding(11) var<storage, read_write> expansion_paused: atomic<u32>;

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
    let info = node_info[parent_idx];
    let num_children = atomicLoad(&node_info[parent_idx].num_children);
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

    // DEBUG: If this is root node (parent_idx==0 or root_idx), print PUCT scores
    // (We can't actually print from shader, but we can write to diagnostics or check values)
    
    // Convert to probabilities using softmax with temperature (subtract max for numerical stability)
    var probs: array<f32, 64>;
    var sum_exp = 0.0;
    let temp = max(params.temperature, 0.00001);
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
    // Try per-workgroup free list first
    let local_top = atomicSub(&free_tops[my_workgroup], 1u);
    if (local_top > 0u && local_top <= params.free_list_capacity) {
        let flat_idx = my_workgroup * params.free_list_capacity + (local_top - 1u);
        let idx = free_lists_flat[flat_idx];
        if (idx != INVALID_INDEX) {
            // Clear deleted and dirty bits, set zero bit if node is zeroed
            atomicAnd(&node_info[idx].flags, ~1u); // clear deleted
            atomicAnd(&node_info[idx].flags, ~(1u << 2)); // clear dirty
            // zero bit is set if node is zeroed, otherwise must be cleared by user
            return idx;
        }
    } else {
        // Restore counter if we failed to pop (empty or underflow)
        atomicAdd(&free_tops[my_workgroup], 1u);
    }

    // Try global free list
    let global_top = atomicLoad(&global_free_head_alloc);
    if (global_top > 0u) {
        // Try to claim a slot via compare-exchange to avoid underflow races
        let claimed = atomicSub(&global_free_head_alloc, 1u);
        if (claimed > 0u) {
            // Successfully claimed a node from the global free list
            let idx = global_free_queue_alloc[claimed - 1u];
            if (idx != INVALID_INDEX && idx != 0u) { // Extra safety: never return node 0
                 atomicAnd(&node_info[idx].flags, ~1u); // clear deleted
                 atomicAnd(&node_info[idx].flags, ~(1u << 2)); // clear dirty
                 return idx;
            }
            // Invalid node in queue, try again (fall through to allocator)
        } else {
            // Another thread claimed the last slot, restore counter
            atomicAdd(&global_free_head_alloc, 1u);
        }
    }

    // Fallback: global allocation
    let alloc_idx = atomicAdd(&alloc_counter, 1u);
    if (alloc_idx < params.max_nodes) {
        // Safety check: never allocate root (0)
        if (alloc_idx == 0u) { return INVALID_INDEX; }
        
        // New node: set zero bit, clear deleted and dirty
        atomicStore(&node_info[alloc_idx].flags, (1u << 1)); // zero bit
        return alloc_idx;
    }

    // All allocation sources exhausted - set expansion pause flag
    atomicStore(&expansion_paused, 1u);
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

    // Double-check: did we race with someone who just finished expansion?
    // If we read num_children=0 (stale) before acquiring the lock, but now it's > 0,
    // it means someone else expanded it and set it back to READY.
    // We just transitioned it to EXPANDING, so we own it.
    // We should check if it's already expanded.
    if (atomicLoad(&node_info[node_idx].num_children) > 0u) {
        // Already expanded!
        // Release lock (set back to READY).
        atomicStore(&node_state[node_idx], NODE_STATE_READY);
        atomicAdd(&diagnostics.exp_lock_retry, 1u); // Log this specific race
        return false; 
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
                // Set up child node info
                node_info[child_idx].parent_idx = node_idx;
                node_info[child_idx].move_id = encode_move(x, y);
                atomicStore(&node_info[child_idx].num_children, 0u);
                node_info[child_idx].player_at_node = -player;
                atomicStore(&node_info[child_idx].flags, 0u); // not deleted
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

    // Initialize node state
    atomicStore(&node_state[node_idx], NODE_STATE_EMPTY);
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
    let child_idx = children_indices[root_idx * MAX_CHILDREN + slot];
    
    if (child_idx == INVALID_INDEX) {
        root_stats[slot].move_id = INVALID_INDEX;
        root_stats[slot].visits = 0;
        root_stats[slot].wins = 0;
    } else {
        root_stats[slot].move_id = node_info[child_idx].move_id;
        root_stats[slot].visits = atomicLoad(&node_visits[child_idx]);
        root_stats[slot].wins = atomicLoad(&node_wins[child_idx]);
    }
}
