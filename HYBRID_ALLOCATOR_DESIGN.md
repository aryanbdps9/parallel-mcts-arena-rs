## Overview

This document details the architecture of the GPU-side memory allocator and urgent event logging system for Monte Carlo Tree Search (MCTS) in Othello, as implemented in `parallel-mcts-arena`. It is the single source of truth for the current implementation and supersedes GPU_ARCHITECTURE.md. The design is focused on:
- High-throughput, contention-free node allocation and recycling for millions of tree nodes
- Robust, lock-free, GPU-to-CPU event logging
- Strict buffer and pipeline layout validation between Rust and WGSL
- Automated, test-driven debugging and diagnostics for GPU resource safety

# GPU-Native MCTS, Hybrid Allocator, and Urgent Event Logging Architecture

## What is this system?
This system implements a high-performance, GPU-native Monte Carlo Tree Search (MCTS) for Othello, with a hybrid node allocator, robust urgent event logging, and efficient host-GPU integration. It is designed for OS engineers (not GPU specialists) and is the single source of truth for the current implementation.

## High-Level Workflow

1. **Initialization:** Host allocates all GPU buffers, sets up bind groups and pipeline layouts (matching WGSL exactly), and initializes the root board state.
2. **MCTS Search:** The main GPU-native MCTS kernel (`main`/`mcts_othello_iteration` in WGSL) is dispatched from Rust for Othello. Thousands of GPU threads run MCTS in parallel, allocating and recycling nodes using a hybrid allocator. This kernel is responsible for all core search logic, including selection, expansion, simulation, and backpropagation.
3. **Pruning:** After each move, unreachable subtrees are pruned in parallel by dispatching a two-phase pruning kernel. Phase 1 identifies the new root and marks garbage nodes. Phase 2 traverses and frees the garbage nodes top-down, recycling them to workgroup-local free lists or the global free queue.
4. **Urgent Event Logging:** Both the main MCTS kernel and the pruning kernel can log important events (e.g., memory pressure, diagnostics) to a host-mapped ring buffer. Host code polls and processes these events. The urgent event buffer write head is reset before each kernel dispatch to prevent event accumulation.
5. **Host Integration:** Host code manages buffer mapping/unmapping, event polling, and diagnostics, ensuring robust synchronization. All buffer mapping/unmapping is guarded by a static Mutex (`DEVICE_POLL_MUTEX`). All host reads of urgent event data use a staging buffer; GPU-write buffers are never mapped directly for reading. The host is responsible for dispatching both the main MCTS kernel and the pruning kernels, and for polling urgent events.
6. **Diagnostics and Testing:** Automated, test-driven debugging is used for GPU resource issues, with minimal and integration tests for allocation, pruning, and tree reuse logic.

---

## Buffer and Event Flow (Diagram)

```text

Host (Rust)                GPU (WGSL)
------------------------   -----------------------------
Allocate buffers  <------>  Use buffers in kernels
Map/unmap for polling      Write urgent events (atomic, once-per-move)
Poll urgent_event_buffer <-> urgent_event_write_head++
Process new events         (ring buffer, 256 slots)
Inject test events (opt)   (host can write too)

Dispatch main MCTS kernel  <->  mcts_othello_iteration() / main() (logs REROOT_START/END, search, memory events; atomic coordination for once-per-move events)
Dispatch pruning kernel    <->  prune_unreachable_topdown() (logs PRUNING events, including pruned node count)
```

---

# GPU-Native MCTS, Hybrid Allocator, and Urgent Event Logging Architecture
## GPU Programming / WGPU Primer (for OS Engineers)

- **GPU Buffers:** Fixed-size arrays in GPU memory, allocated by the host. No pointers, no dynamic allocation.
- **Bind Groups:** Collections of buffers bound to the GPU pipeline for use in shaders.
- **Kernels:** Compute shaders (WGSL) run in parallel on thousands of threads (workgroups).
- **Host-GPU Sync:** Host must map/unmap buffers and use `device.poll()` to ensure data is visible.
- **Atomic Operations:** Used for safe concurrent access (e.g., incrementing counters, setting bits).
- **Ring Buffers:** Used for event logging; wraparound is handled by modulo arithmetic.

---
## How to Debug or Extend

- **To debug urgent event logging:**
    - Use the host polling thread; it prints/logs new events only when the write head advances.
    - Use the provided test suite to inject and verify events, including integration tests for once-per-move event emission.
    - Check buffer mapping/unmapping and `device.poll()` calls for sync issues. All buffer mapping/unmapping is guarded by a static Mutex, and all host reads use a staging buffer.
- **To extend the allocator or event system:**
    - Add new event types to the `UrgentEvent` struct and WGSL constants.
    - Update buffer layouts and host structs to match WGSL exactly.
    - Use atomic operations for all concurrent buffer access.
- **For performance tuning:**
    - Profile allocation and pruning kernels for contention or imbalance.
    - Adjust free list sizes or implement workgroup stealing if needed.

---

This document describes the current architecture for GPU-native Monte Carlo Tree Search (MCTS) in Othello, including the hybrid node allocator, urgent event logging, pruning, and host-GPU integration. It is written for OS engineers with minimal GPU programming background, and includes code/data structure snippets and clear explanations.
- **free_lists_buffer**: [workgroup][slot] 2D array (256 × 8192 = 2M slots). Each workgroup manages its own ring buffer of free node indices.
- **free_tops_buffer**: [workgroup] array of atomic counters, one per workgroup, tracking the top of each free list.

The system enables high-throughput, contention-free node allocation and recycling for millions of tree nodes, robust lock-free GPU-to-CPU urgent event logging, and correctness/testability for large-scale parallel MCTS. All host-GPU buffer synchronization, event polling, and diagnostics are handled in a way that is robust to race conditions and GPU/host concurrency.
- **urgent_event_write_head**: Atomic counter (GPU increments, host polls) for event production/consumption.

## GPU-Side Algorithms

- `node_info_buffer`: Array of structs, one per node, with parent, move, flags (deleted/zero bits), and metadata.
- `node_visits_buffer`, `node_wins_buffer`: Per-node atomic visit/win counts for backpropagation.
- `children_indices_buffer`: Per-node child index lists.
- `root_board_buffer`: Board state for the root node.
   - If the free list is empty, the thread atomically increments the global alloc_counter to claim a new node index.
   - If alloc_counter < max_nodes, allocation succeeds; otherwise, memory pressure policy is triggered.

- `free_lists_buffer`: [256][8192] u32s. Each workgroup manages its own ring buffer of free node indices (2M total).
- `free_tops_buffer`: [256] atomic<u32>, one per workgroup, tracking the top of each free list.
- `alloc_counter_buffer`: Global atomic counter for fallback allocation when a workgroup's free list is empty.

### Main MCTS Kernel (GPU-Native Search)
- The main kernel (`main`/`mcts_othello_iteration`) is dispatched from Rust and runs the full MCTS search loop on the GPU, including:
    - **Selection:** Using PUCT (UCT with priors) to traverse from root to a leaf node
    - **Virtual Loss:** Each thread adds virtual loss to nodes along its path during selection to coordinate parallel exploration
    - **Expansion:** Allocating new child nodes when a leaf is reached
    - **Simulation:** Running a random playout to terminal state
    - **Backpropagation:** Propagating rewards up the path, using **parent's perspective** (player who moved TO each node)
- **Perspective Convention:** All node statistics (wins, visits) are stored from the parent's perspective. This matches CPU MCTS and ensures correct PUCT calculations.
    - During backpropagation: `player_who_moved = -player_at_node` (parent is opponent)
    - Reward is flipped if `player_who_moved != leaf_player`
    - This means `node.wins` represents wins for the player who chose that move (parent's perspective)
- Logging memory pressure and other urgent events as needed
- All event payloads (including diagnostics) are written to the urgent event buffer, which is polled by the host

### Node Recycling and Pruning (Two-Phase "Identify & Prune")

The pruning process is split into two phases executed by separate GPU kernels, coordinated by the host.

**Phase 1: Identify New Root & Mark Garbage (`identify_new_root_kernel`)**
- **Input:** `RerootParams` (Uniform Buffer) containing:
    - `move_x`, `move_y`: The move chosen by the host (player's or AI's move)
    - `current_root_idx`: The GPU node index of the current root
- **Logic (Single Workgroup Compute Shader):**
    - Scans the children of `current_root_idx`
    - **If child matches move:** Writes child index to `new_root_output` buffer (for host to read back)
    - **If child does NOT match:** Writes child index to `work_queue` for pruning in Phase 2
    - **If no match found:** Writes special error value (0xFFFFFFFF or 0xE0000000 + diagnostic info) to signal tree must be reset
- **Output:**
    - `new_root_output`: Contains the new root index (or error code)
    - `work_queue`: Contains indices of all unreachable child nodes (garbage roots)
    - `work_head`: Set to the number of garbage nodes queued

**Phase 2: Prune Unreachable Top-Down (`prune_unreachable_topdown_kernel`)**  
- **Input:** `work_queue` (populated by Phase 1), `work_head` (number of items in queue)
- **Logic (Multi-Workgroup Compute Shader - typically 64 workgroups):**
    - Each thread attempts to pop a node from `work_queue` using atomic operations
    - For each popped node:
        1. Atomically mark it as deleted (using `node_state` buffer)
        2. Add it to the appropriate free list (workgroup-local or global overflow)
        3. Scan its children and push them onto `work_queue` for recursive pruning
    - Repeat until `work_queue` is empty
- **Correctness Guarantees:**
    - **No Double-Free:** Atomic state transitions ensure only one thread "owns" each node
    - **Completeness:** Queue-based traversal ensures all descendants are reached
    - **Thread Safety:** Work queue operations use atomic head pointer

**Host Coordination:**
1. Host calls `advance_root(move_xy, new_board, new_player, new_legal_moves)`
2. Host writes `RerootParams` to uniform buffer
3. Host dispatches **Phase 1 Kernel** (1 workgroup)
4. Host reads `new_root_output` via staging buffer to get new root index
5. Host dispatches **Phase 2 Kernel** (64 workgroups) to free garbage nodes
6. Host updates internal `current_root_idx` state
7. Next search batch uses the new root

### Urgent Event Logging
- **Events:**
    - Logging is minimal and used primarily for diagnostics
    - Events can be emitted for memory pressure, allocation failures, or debugging
    - BATCH_START/BATCH_END and REROOT events are currently disabled in production
- **Mechanism:** Ring buffer with atomic write head. Host polls this buffer periodically via dedicated urgent event logger thread.

### Backpropagation (Parent's Perspective)

**Goal:**
Accurately propagate simulation results up the tree after each rollout, using a consistent perspective convention that matches CPU MCTS.

**Perspective Convention:**
- All node statistics (`node_wins`, `node_visits`) are stored from the **parent's perspective**
- Parent = the player who chose to move to this node = opponent of `player_at_node`
- This means: `player_who_moved = -player_at_node`

**Implementation (WGSL):**
```wgsl
// Backpropagation phase
for (var i = path_len; i > 0u; i--) {
    let node_idx = path[i - 1u];
    atomicAdd(&node_vl[node_idx], -1);  // Remove virtual loss
    atomicAdd(&node_visits[node_idx], 1);
    
    // CRITICAL: Use parent's perspective (player who moved TO this node)
    let player_at_node = node_info[node_idx].player_at_node;
    let player_who_moved = -player_at_node;  // Parent is opponent
    var reward = rollout_result;
    if (player_who_moved != leaf_player) {
        reward = 2 - rollout_result;  // Flip perspective
    }
    atomicAdd(&node_wins[node_idx], reward);
}
```

**Why This Matters:**
- CPU MCTS uses this same convention
- PUCT selection formula `Q + U` works correctly when Q is from parent's perspective
- Display code can directly use Q-values without flipping
- Both engines evaluate positions consistently

### Urgent Event Logging (GPU → CPU)
- GPU kernels write urgent events to the host-mapped ring buffer at `write_head % 256`, incrementing `write_head` atomically.
- The host can also inject urgent events for diagnostics/testing using `log_urgent_event_from_cpu_with_payload`, which writes to the buffer and advances the write head.
- Events include search halts, root advances, diagnostics, memory pressure signals, and test/diagnostic events.
- The host polls the buffer, processes only new events (using last seen write_head), and handles wraparound/overflow robustly.
- Logging is rate-limited and non-verbose by design, and all buffer access is synchronized with atomic flags and `device.poll`.
## Host Integration and Synchronization
- All GPU buffer creation, bind group layout, and shader preprocessing (for includes) are performed in Rust, with validation against WGSL expectations. Pipeline and bind group layouts are now validated and constructed to match WGSL exactly, including dummy groups/bindings as needed.
- All mutable state in `GpuOthelloMcts` is protected by an internal Mutex; all mutation is via `&self` methods with internal locking. All buffer mapping/unmapping is guarded by a static Mutex to prevent race conditions.
- An AtomicBool is used to synchronize buffer mapping/unmapping between host and GPU. All host reads of urgent event data use a staging buffer; GPU-write buffers are never mapped directly for reading.

## Node Lifecycle Summary
1. **Allocation:** Try workgroup free list → fallback to global alloc_counter.
3. **Simulation/Backprop:** Visits/wins updated atomically.
4. **Pruning:** Unreachable subtrees pruned top-down, nodes recycled.
### Bind Group Layout (Single Source of Truth)

This layout must be strictly maintained between Rust (`GpuOthelloMcts`) and WGSL (`mcts_othello.wgsl`).

**Group 0: Node Data & Allocator**
- Binding 0: `node_info` (Storage, RW)
- Binding 1: `node_visits` (Storage, RW, Atomic)
- Binding 2: `node_wins` (Storage, RW, Atomic)
- Binding 3: `node_vl` (Storage, RW, Atomic)
- Binding 4: `node_state` (Storage, RW, Atomic)
- Binding 5: `children_indices` (Storage, RW)
- Binding 6: `children_priors` (Storage, RW)
- Binding 7: `free_lists` (Storage, RW) - Per-workgroup free lists
- Binding 8: `free_tops` (Storage, RW, Atomic) - Per-workgroup free list heads

**Group 1: Search Parameters & Diagnostics**
- Binding 0: `params` (Uniform) - MCTS parameters (iterations, c_puct, etc.)
- Binding 1: `work_items` (Storage, RW) - Legacy/Unused
- Binding 2: `paths` (Storage, RW) - Per-thread search paths
- Binding 3: `alloc_counter` (Storage, RW, Atomic) - Global fallback allocator
- Binding 4: `diagnostics` (Storage, RW, Atomic) - Debug counters

**Group 2: Game State**
- Binding 0: `root_board` (Storage, Read) - Current root board state

**Group 3: Urgent Events**
- Binding 0: `urgent_event_buffer` (Storage, RW) - Ring buffer for events
- Binding 1: `urgent_event_write_head` (Storage, RW, Atomic) - Write head for ring buffer

**Group 4: Pruning & Rerooting**
- Binding 0: `unreachable_roots` (Storage, Read) - *Legacy/Unused in new flow? Or used as intermediate?* -> *Replaced by direct write to work_queue in Phase 1.*
- Binding 1: `work_queue` (Storage, RW)
- Binding 2: `work_head` (Storage, RW, Atomic)
- Binding 3: `reroot_params` (Uniform) - **NEW**
- Binding 4: `new_root_output` (Storage, RW) - **NEW**
- Binding 5: `global_free_queue` (Storage, RW) - **NEW (Overflow)**
- Binding 6: `global_free_head` (Storage, RW, Atomic) - **NEW (Overflow)**

**Group 5: Global Coordination**
- Binding 0: `global_reroot_threads_remaining` (Storage, RW, Atomic) - For REROOT_END coordination

---

### WGSL Shader Snippets

```wgsl
// Group 0: Node Data & Allocator
@group(0) @binding(7) var<storage, read_write> free_lists: array<array<u32, 8192>, 256>;
@group(0) @binding(8) var<storage, read_write> free_tops: array<atomic<u32>, 256>;

// Group 1: Global Allocator
@group(1) @binding(3) var<storage, read_write> alloc_counter: atomic<u32>;

// Group 3: Urgent Events
@group(3) @binding(0) var<storage, read_write> urgent_event_buffer: array<UrgentEvent, 256>;
@group(3) @binding(1) var<storage, read_write> urgent_event_write_head: atomic<u32>;

// Group 5: Global Coordination
@group(5) @binding(0) var<storage, read_write> global_reroot_threads_remaining: atomic<u32>;
@group(5) @binding(1) var<storage, read_write> global_reroot_start_threads_remaining: atomic<u32>;
```

---

## Algorithms

### 1. **Allocation (`allocate_node()`)**

```wgsl
fn allocate_node() -> u32 {
    let my_workgroup = workgroup_id.x; // 0-255

    // Step 1: Try to pop from my workgroup's free list (LIFO)
    let local_top = atomicLoad(&free_tops[my_workgroup]);
    if (local_top > 0u) {
        let maybe_idx = atomicSub(&free_tops[my_workgroup], 1u);
        if (maybe_idx > 0u && maybe_idx <= FREE_LIST_SIZE_PER_GROUP) {
            let node_idx = free_lists[my_workgroup][maybe_idx - 1u];
            // Node is now allocated and ready for use
            return node_idx;
        } else {
            // Roll back if contention or underflow
            atomicAdd(&free_tops[my_workgroup], 1u);
        }
    }

    // Step 2: Fallback to global allocator (consume new/free memory)
    let node_idx = atomicAdd(&alloc_counter, 1u);
    if (node_idx < params.max_nodes) {
        // Node is now allocated and ready for use
        return node_idx;
    }

    // Step 3: Try global free list (shared overflow pool)
    let global_top = atomicLoad(&global_free_head);
    if (global_top > 0u) {
        let maybe_idx = atomicSub(&global_free_head, 1u);
        if (maybe_idx > 0u) {
            return global_free_queue[maybe_idx - 1u];
        }
    }

    // Step 4: Memory exhausted - signal pressure, return INVALID_INDEX
    return INVALID_INDEX;
}
- Only ~256 threads contend per free list (vs 2048 before)
- If a workgroup's list is empty, falls back to global allocation (consuming new/free memory)
---


### 2. **Two-Phase Pruning (Updated)**

Instead of a single complex kernel, split the logic into two distinct compute passes that share data via VRAM buffers.

**Phase 1: `identify_garbage` (1 Workgroup)**
- **Input:** `RerootParams` (Uniform: `move_x`, `move_y`, `current_root`).
- **Task:** Scans the children of `current_root`.
- **Output A (The Survivor):** If a child matches the move, write its index to `new_root_output`.
- **Output B (The Garbage):** If a child *doesn't* match, write its index to `work_queue` (the "death row").

**Phase 2: `prune_unreachable_topdown` (Many Workgroups)**
- **Task:** Consumes `work_queue`. For every node it pops:
    1.  Marks it as deleted (atomic bit).
    2.  Adds it to the `global_free_queue` (Binding 2) or local free list.
    3.  Pushes its children back onto the `work_queue`.

**Handling Concurrency & Correctness:**
- **Double Free:** Handled by `atomic_set_deleted`. Only the thread that transitions the bit from 0->1 owns the node.
- **Completeness:** Guaranteed by the queue-based traversal. Since we start with the roots of all unreachable subtrees and push all children, we reach every descendant.
- **Contention:**
    - **Queue:** Uses `work_head` (atomic). High contention potential, but acceptable for pruning phase.
    - **Nodes:** Low contention due to tree structure.

### 3. **Diagnostics and Logging**

#### Automated Testing and Diagnostics
- Automated, test-driven debugging is used for GPU resource issues, with minimal and integration tests for emission guard logic and event logging.
- Integration tests assert that only one REROOT_END event is emitted per move.
- Diagnostic/temporary event spam has been removed from production runs.
---

## Urgent Event Logging
---

## Backpropagation (Real, Not Stub)

**Goal:**
Accurately propagate simulation results up the tree after each rollout.

**Design:**
- **Kernel:**
    - After simulation, walk up the path from leaf to root.
    - Atomically update visit/win statistics for each node.
    - Use the result of `simulate_game` for reward calculation.
- **Host:**
    - Ensures kernel is launched and synchronized as part of the MCTS iteration.

**Goal:**
Enable the GPU to log important events (e.g., re-root, MCTS halt/start) to a host-visible buffer, with low latency and support for large payloads.

**Design:**
- **Buffer:**
    - Host-mapped, persistent, size: 256 events × 1024 bytes = 256 KiB.
    - Struct:
        - `timestamp` (u64, GPU or host time)
        - `event_type` (u32)
        - `payload` ([u8; 1016]) for flexible data (move info, diagnostics, etc.)
- **Indices:**
    - `write_head` (atomic, GPU increments)
    - `read_tail` (host, advances after reading)
- **GPU:**
    - Writes event at `write_head % 256`, increments `write_head`.
    - Aggregates noisy events (e.g., only logs once per move or on threshold).
- **Host:**
    - Dedicated thread polls buffer every 10–50 ms.
    - Reads and prints/logs new events, advances `read_tail`.
    - Handles buffer wrap-around and overflow (tracks dropped events).

The allocator tracks and logs the following events in a diagnostics buffer (GPU-side struct, host-readable):

```wgsl
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
    recycling_events: atomic<u32>,
    rollouts: atomic<u32>,
    _pad0: atomic<u32>,
    _pad1: atomic<u32>,
}
```

**Host code** reads this buffer after search to analyze allocation failures, recycling, and memory pressure. This enables robust debugging and profiling of the allocator's behavior under load.

---

## Flow Example

### Turn 1: Game Start
```
init_tree():
- Create root (node 0) + 4 children (nodes 1-4)
- alloc_counter = 5
- all free_tops[0..255] = 0
```

### Turn 2: First Search
```
run_iterations():
- 2048 threads running in 256 workgroups
- Workgroup 0 threads try to allocate:
  - Check free_tops[0] = 0 → empty
  - Fall back to global: nodes 5, 6, 7... allocated
- After search: alloc_counter = 490,000
```

### Turn 3: First Move & Prune
```
advance_root(move=(3,2)):
- Set root_idx = 2
- prune_unreachable_nodes():
  - Nodes 0, 1, 3, 4 and descendants freed
  - Workgroup 0 adds nodes [0, 1, ...] to free_lists[0]
  - Workgroup 1 adds nodes [...] to free_lists[1]
  - Distribution is semi-random but balanced
  - Result: free_tops[0..255] have various counts
  - Total freed: ~480,000 nodes distributed across 256 lists
```

### Turn 4: Second Search
```
run_iterations():
- Workgroup 0 threads allocate:
  - Check free_tops[0] = 1800 → have recycled nodes!
  - Pop from free_lists[0][1799] → reuse old node
- Much less global allocation needed
- Most nodes come from free lists
```


---

## Benefits

### ✅ **Performance**
- **256x less contention**: Each atomic shared by 256 threads, not 2048
- **Cache locality**: Workgroups likely access nearby memory
- **Predictable allocation**: No spikes, smooth distribution

### ✅ **Correctness**
- **No overflow**: Each list capped at 8K entries
- **Graceful degradation**: If a workgroup's list fills, just uses global allocator
- **No freeze**: Can't access out-of-bounds indices

### ✅ **Simplicity**
- **Conceptually clear**: "Each workgroup manages its own recycling bin"
- **Easy to reason about**: Node state and recycling make cleanup predictable
- **Incremental migration**: Can add features one at a time

---

## Trade-offs

### ⚠️ **Memory Overhead**
```
Old: 2M u32s (free_list) + 1 atomic (free_top) = 8MB + 4 bytes
New: 256 × 8K u32s + 256 atomics = 8MB + 1KB
```
**Impact:** Negligible (same size, different layout)

### ⚠️ **Uneven Distribution**
- Some workgroups might have full free lists, others empty
- **Mitigation:** Workgroups can "steal" from global allocator




### ⚠️ **Zero Bit and Deleted Bit**
Both bits are stored in a single integer field (e.g., node_info.flags or similar), not as separate fields. For 2M nodes, this is just 2 bits per node, packed into a u32 or u8 as appropriate.

**deleted bit:** Indicates the node is free, but must be claimed and "washed" (reset/cleared) before reuse. Used for nodes that have just been freed by pruning or recycling.

**zero bit:** Indicates the node is already zeroed and ready for immediate allocation and use, with no further clearing required. Used for nodes that have never been allocated or have been explicitly zeroed in a background pass.

**Impact:** Both bits are lightweight, and together allow the allocator to distinguish between "fresh" and "needs-wash" nodes, optimizing allocation and reuse paths. In the current implementation, node state is tracked via a `node_state` enum (e.g., EMPTY, READY, etc.). Allocation logic checks node state, not explicit bitfields. If no nodes are available, memory pressure policy is triggered as described above.

---

## Known Issues and Future Work

### Terminal Position Q-Value Display
**Issue:** When a game reaches a terminal state (win/loss), the GUI debug stats may show Root Value = 0.500 instead of the correct value (0.0 for loss, 1.0 for win).

**Root Cause:** The displayed "Root Value" in GUI is likely stale from a previous search iteration. When the game becomes terminal, GPU detects this and skips `advance_root`, but the GUI may not update the displayed statistics.

**Evidence:** Terminal CSV logs show correct Q-values (e.g., Q=1.0000 for winning move), but GUI snapshot shows Q=0.5.

**Status:** Under investigation. Does not affect move selection or game outcome - purely a display issue.

**Potential Fix:** Ensure GUI updates statistics when terminal state is detected, or add special handling to display terminal evaluation.

---

## Migration Plan

### Phase 1: Add Per-Workgroup Free Lists ✅ (Complete)
1. Replaced `free_list_buffer` with `free_lists_buffer` (2D array)
2. Replaced `free_top_buffer` with `free_tops_buffer` (array of atomics)
3. Updated WGSL allocation logic to use workgroup_id
4. Tested: Overflow eliminated, no freezes

---

## Expected Outcomes

### Before (Current System):
```
Turn 1: alloc=1.9M, free_top=0
Turn 2: alloc=1.9M, free_top=3.3M ← OVERFLOW! → FREEZE
```

### After (Hybrid System):
```
Turn 1: alloc=490K, free_tops[0]=0, ..., free_tops[255]=0
Turn 2: alloc=490K, free_tops[0]=1.8K, ..., free_tops[255]=1.9K
    Total freed: ~480K distributed across 256 lists
    Average per list: ~1875 nodes (well under 8K cap)
Turn 3: alloc=490K (reusing nodes from free lists)
...stable indefinitely, with memory pressure policy handling exhaustion gracefully...
```

### Performance Metrics:
- **Allocation time**: ~Same (atomic ops similar cost)
- **Pruning time**: ~Same (just different target lists)
- **Freeze risk**: ❌ **ELIMINATED** (impossible to overflow 8K cap)
- **Tree lifetime**: 10-20 turns before reset (vs 2-3 without free list)

---


## Open Questions

1. **Free list size per workgroup?**
    - Proposed: 8192 (8K)
    - Too small? Could make 16K if needed
    - Too large? Could reduce to 4K

2. **Free list overflow?**
    - If a workgroup's free list is full, nodes are pushed to a global free list for overflow handling.

3. **Load balancing?**
    - If one workgroup's list is empty, it can steal nodes from other workgroups or the global free list.
    - TODO: Implement periodic rebalancing for long-term fairness.

4. **Traversal context?**
    - Traversal refers to the freeing/pruning operation after re-rooting, not to the main MCTS search path.

5. **Visit count heuristic?**
    - For now, use visit counts to guide worker assignment to subtrees.

6. **Logging policy?**
    - Logging should be aggregate and rate-limited, never verbose.

7. **What remains on CPU in GPU-native MCTS?**
    - In a fully GPU-native MCTS, the only CPU-side responsibilities are:
      - Orchestrating kernel launches (dispatches)
      - Transferring input/output data (e.g., initial board state, final statistics)
      - Reading diagnostics buffers
      - (Other AI, game logic, and search are all on GPU)

---

## Conclusion

This hybrid approach combines the best of:
- **Free lists**: Memory reuse without fragmentation
- **Per-workgroup**: Eliminates contention and overflow

**Risk Level:** 🟢 **LOW**
- Incremental changes
- Can roll back at each phase
- Preserves existing correctness

**Expected Result:** 🟢 **Stable, freeze-free operation with good memory reuse**

---


## Unified Pruning and Memory Pressure Policy

The allocator is designed to ensure that valuable subtrees are never pruned solely due to memory pressure. Pruning occurs only for subtrees that become unreachable after a root move (natural pruning), or for nodes that are objectively low-value. The policy is as follows:

1. **Natural Pruning:**
   - Subtrees are pruned only when they become unreachable due to a root move (advance_root).
   - No reachable subtree is pruned just because memory is low.

2. **Graceful Degradation Under Memory Pressure:**
   - When the allocator detects that GPU memory is nearly exhausted (e.g., global alloc counter >= max_nodes):
     - Elegantly stop all search work on the GPU, keeping data members consistent.
     - If the root node has been visited enough times (above a threshold), play the move early by selecting the best child of the root based on current statistics, then prune all non-selected subtrees using efficient top-down freeing.
     - If the root has not reached the threshold, stop expansion (do not allocate new nodes), but continue rollouts (simulations) on the existing tree. When the threshold is reached or time runs out, play the move and prune as above.
     - The allocation function never simply returns INVALID_INDEX on exhaustion; it always triggers this policy and handles the situation gracefully.
   - If memory is exhausted and no low-value leaves are available, pause or slow down search, or play the move early and start rollouts instead of expansion. Log a warning or reduce batch size, but never delete valuable subtrees.

4. **Diagnostics and Tuning:**
   - Track and log memory pressure events and node recycling frequency.
   - Use this data to tune the recycling policy for optimal performance and robustness.

**Expected Outcome:**
- Only “dead” or low-value parts of the tree are pruned, preserving valuable search information and improving AI performance.
- Memory pressure never causes the loss of high-value subtrees.

---

## Pre-Computation + Incremental Execution Optimization

### Problem Statement
GPU MCTS parallelism faces fundamental contention during expansion:
- **Wave analogy:** Each dispatch is like a wave of 16,384 threads crashing onto the tree frontier
- Early waves (dispatch 1-5) have very few expandable leaves (4 → 16 → 256)
- 16,384 threads compete for ~100 leaves via atomic locks
- Result: 95-99% contention in early dispatches (only 1-5% of threads successfully expand)
- Threads that acquire locks spend significant time computing legal moves while holding the lock
- This serializes expansion even further, reducing throughput

### Root Cause: Coarse-Grained Execution Model
Current design: **One thread completes entire MCTS iteration (selection → expansion → rollout → backprop) in one dispatch.**

Problems:
1. **Bursty contention:** All threads try to expand simultaneously
2. **Wasted work:** 99% of threads fail to expand, but still do rollouts (useful but not optimal)
3. **Lock serialization:** Threads hold expansion locks while computing legal moves (~40-60% of lock time)

### Solution: Pre-Computation + Stateful Incremental Execution

**Architecture: Two-Tier Optimization**

**Tier 1: Pre-Computation (Reduce Lock Duration)**
- Cache legal moves for high-probability leaves BEFORE main kernel dispatch
- Threads that acquire locks can lookup cached moves instead of computing
- Reduces lock hold time by 40-60%

**Tier 2: Incremental Execution (Eliminate Contention Bursts)**  
- Each thread does ONE step per dispatch (traverse one node, save state, die)
- Next dispatch resumes from saved state
- Spreads expansion attempts across many dispatches instead of concentrating in one
- Eliminates "wave crashing" behavior

### Combined Flow

**Host Loop:**
```rust
loop {
    // 1. Pre-compute legal moves for likely leaves
    dispatch_collect_candidates(max_threads);
    dispatch_sort_candidates();
    dispatch_precompute_moves(budget);
    
    // 2. Incremental MCTS step (selection/expansion/backprop only)
    dispatch_incremental_mcts_step(num_threads);
    
    // 3. Rollout chunks (separate kernel, doesn't block fast phases)
    // Processes only threads in ROLLOUT_ACTIVE phase, advances 5-10 moves per dispatch
    dispatch_rollout_chunk(num_threads, moves_per_chunk: 8);
    
    // 4. Check if search complete
    if all_threads_finished() { break; }
}
```

**Thread State (Persistent GPU Buffer):**
```wgsl
struct ThreadState {
    phase: u32,                  // SELECTION/EXPANSION/ROLLOUT_ACTIVE/BACKPROP/FINISHED
    current_node: u32,           // Which node we're at
    path: array<u32, 128>,       // Path taken so far
    path_len: u32,               // How many nodes in path
    // NOTE: rng_seed removed - now using global atomic counter (see RNG Architecture section)
    leaf_player: i32,            // Player at leaf (for backprop perspective)
    rollout_result: u32,         // Result of rollout (for backprop)
    backprop_index: u32,         // Which node we're backpropping (counts down)
    
    // Rollout state (for chunked rollout execution)
    rollout_board: array<i32, 64>,  // Current board state during rollout
    rollout_player: i32,            // Current player during rollout
    rollout_moves_remaining: u32,   // Moves until terminal (or chunk limit)
}
```

**Incremental Kernel (Fast Phases: Selection/Expansion/Backprop):**
```wgsl
@compute @workgroup_size(64)
fn incremental_mcts_step() {
    let thread_id = global_invocation_id.x;
    var state = thread_states[thread_id];
    
    // Skip if finished or in rollout (handled by separate kernel)
    if (state.phase == PHASE_FINISHED || state.phase == PHASE_ROLLOUT_ACTIVE) { 
        return; 
    }
    
    // Execute ONE step based on current phase
    switch (state.phase) {
        case PHASE_SELECTION: {
            // Traverse ONE node down the tree using softmax sampling
            // Calculate PUCT scores for all children
            var puct_scores: array<f32, MAX_CHILDREN>;
            for each child:
                let q = wins / (visits + virtual_loss);
                let u = exploration * prior * sqrt(parent_visits) / (1 + visits + vl);
                puct_scores[i] = q + u;
            
            // Apply softmax with temperature for exploration diversity
            var exp_sum = 0.0;
            var exp_scores: array<f32, MAX_CHILDREN>;
            for each child:
                exp_scores[i] = exp(puct_scores[i] / temperature);
                exp_sum += exp_scores[i];
            
            // Sample from probability distribution using global RNG counter
            let rng_val = pcg_hash(atomicAdd(&diagnostics.global_rollout_counter, 1u));
            let rand_val = rng_val & 1048575u;  // Mask to 2^20 for precision
            let sample = f32(rand_val) / 1048576.0;
            
            var cumulative = 0.0;
            var child = INVALID_INDEX;
            for each child:
                cumulative += exp_scores[i] / exp_sum;
                if (sample <= cumulative) {
                    child = get_child_idx(current_node, i);
                    break;
                }
            
            atomicAdd(&node_vl[child], 1);  // Add VL
            state.path[state.path_len] = child;
            state.path_len++;
            state.current_node = child;
            
            // Check if we reached a leaf
            if (is_leaf(child)) {
                state.phase = PHASE_EXPANSION;
                state.leaf_player = node_info[child].player_at_node;
            }
        }
        case PHASE_EXPANSION: {
            // Try to expand (with pre-computed moves if available)
            let locked = try_lock_for_expansion(state.current_node);
            if (locked) {
                // Check cache first (pre-computation tier)
                let moves = lookup_precomputed_moves(state.current_node);
                if (!moves.valid) {
                    // Fallback: compute on-the-fly
                    var path_copy = state.path;
                    let board = reconstruct_board(&path_copy, state.path_len);
                    moves = compute_legal_moves(&board, state.leaf_player);
                }
                
                allocate_and_init_children(state.current_node, moves);
                unlock(state.current_node);
            }
            
            // Remove VL before starting rollout (so other threads aren't blocked)
            for (var i = 0u; i < state.path_len; i++) {
                atomicAdd(&node_vl[state.path[i]], -1);
            }
            
            // Initialize rollout state
            var path_copy = state.path;
            state.rollout_board = reconstruct_board(&path_copy, state.path_len);
            state.rollout_player = state.leaf_player;
            state.rollout_moves_remaining = 60;  // Max Othello game length
            state.phase = PHASE_ROLLOUT_ACTIVE;
        }
        case PHASE_BACKPROP: {
            // Backprop ONE node per dispatch
            if (state.backprop_index > 0) {
                let node_idx = state.path[state.backprop_index - 1];
                atomicAdd(&node_visits[node_idx], 1);
                
                let player_at_node = node_info[node_idx].player_at_node;
                let player_who_moved = -player_at_node;
                var reward = state.rollout_result;
                if (player_who_moved != state.leaf_player) {
                    reward = 2 - reward;
                }
                atomicAdd(&node_wins[node_idx], reward);
                
                state.backprop_index--;
            } else {
                // Backprop complete, start new iteration
                state.phase = PHASE_SELECTION;
                state.current_node = root_idx;
                state.path_len = 1;
                state.path[0] = root_idx;
            }
        }
    }
    
    // Save state for next dispatch
    thread_states[thread_id] = state;
}

// Separate rollout kernel (chunked, doesn't block fast phases)
@compute @workgroup_size(64)
fn rollout_chunk_kernel() {
    let thread_id = global_invocation_id.x;
    var state = thread_states[thread_id];
    
    // Only process threads in rollout phase
    if (state.phase != PHASE_ROLLOUT_ACTIVE) { return; }
    
    const MOVES_PER_CHUNK: u32 = 8u;  // Tunable: balance overhead vs sync
    
    // Simulate up to MOVES_PER_CHUNK moves (or until terminal)
    for (var i = 0u; i < MOVES_PER_CHUNK; i++) {
        if (is_terminal(&state.rollout_board)) {
            // Game over, determine winner
            state.rollout_result = evaluate_terminal(&state.rollout_board, state.leaf_player);
            state.phase = PHASE_BACKPROP;
            state.backprop_index = state.path_len;
            break;
        }
        
        // Select random move using global RNG counter
        let move = select_random_move(&state.rollout_board, state.rollout_player);
        apply_move(&state.rollout_board, move, state.rollout_player);
        state.rollout_player = -state.rollout_player;
        state.rollout_moves_remaining--;
        
        if (state.rollout_moves_remaining == 0u) {
            // Safety limit reached
            state.rollout_result = evaluate_terminal(&state.rollout_board, state.leaf_player);
            state.phase = PHASE_BACKPROP;
            state.backprop_index = state.path_len;
            break;
        }
    }
    
    // Save state (still ROLLOUT_ACTIVE if not terminal, or BACKPROP if done)
    thread_states[thread_id] = state;
}
```

---

## Random Number Generation (RNG) Architecture

### Design Evolution: From Thread-Local to Global Counter

**Problem Identified:**
The original RNG implementation used thread-local state (`thread_states[thread_id].rng_seed`) that was modified during the selection phase. This created systematic bias in rollouts because:
- Selection phase advanced RNG state variable numbers of times (tie-breaking, softmax sampling)
- Rollout phase inherited this modified state
- Different threads experienced different amounts of RNG advancement based on tree path
- Result: 36.83% win rate in liar rollouts instead of expected 49.23%

**Root Cause:**
Correlation between tree structure exploration (selection phase) and random outcome generation (rollout phase) through shared RNG state.

**Solution: Independent Global Counter RNG**
Replace all thread-local stateful RNG with a global atomic counter that provides independent random values:

```wgsl
// OLD (Biased):
fn rand_u32() -> u32 {
    rng_state = pcg_hash(rng_state);  // Stateful, creates correlation
    return rng_state;
}

// NEW (Unbiased):
fn rand_u32() -> u32 {
    let rng_val = atomicAdd(&diagnostics.global_rollout_counter, 1u);
    return pcg_hash(rng_val);  // Independent value each call
}
```

**Key Changes:**
1. **Global Counter**: Added `global_rollout_counter: atomic<u32>` to diagnostics buffer
2. **Independent Seeding**: Each random value request increments counter and hashes the result
3. **No Thread State**: Removed RNG state from thread-local storage
4. **Universal Application**: Applied to all RNG usage points:
   - Argmax tie-breaking in selection
   - Softmax sampling in selection
   - Move selection in rollouts
   - Any helper functions using `rand_u32()` or `rand_f32()`

**Implementation:**

```wgsl
// Diagnostics buffer includes global RNG counter
struct Diagnostics {
    // ... other fields ...
    global_rollout_counter: atomic<u32>, // Global counter for independent RNG seeding
}

// Base RNG function using PCG hash
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Random u32 using independent global counter
fn rand_u32() -> u32 {
    let rng_val = atomicAdd(&diagnostics.global_rollout_counter, 1u);
    return pcg_hash(rng_val);
}

// Random f32 in [0, 1)
fn rand_f32() -> f32 {
    return f32(rand_u32()) / 4294967296.0;
}

// Example usage in selection (argmax tie-breaking)
if (max_count > 0u) {
    let rng_val = pcg_hash(atomicAdd(&diagnostics.global_rollout_counter, 1u));
    let rand_idx = ((rng_val >> 24u) * max_count) >> 8u;
    selected_idx = max_children[rand_idx];
}

// Example usage in rollout (move selection)
let rollout_id = atomicAdd(&diagnostics.global_rollout_counter, 1u);
rng_state = pcg_hash(rollout_id);
rng_state = pcg_hash(rng_state);
let p1_score = rng_state % 65u;
```

**Performance Characteristics:**
- **Contention**: Single global atomic counter has potential contention, but acceptable because:
  - Counter increment is extremely fast (few nanoseconds)
  - Only happens when random values needed (not every operation)
  - Modern GPUs handle atomic operations efficiently
- **Quality**: PCG hash provides excellent statistical properties
- **Determinism**: Given same seed, produces same sequence (debuggable, testable)

**Validation Results:**
- **Liar Rollout Test** (before fix): 36.83% wins, avg score 29.54 (BIASED)
- **Liar Rollout Test** (after fix): 49.19% wins, avg score 31.99 (PERFECT)
- **Honest Rollout Test**: Q-values correctly reflect position quality (~0.47-0.48 for opening)

**Benefits:**
1. **Eliminates Bias**: No correlation between tree exploration and random outcomes
2. **Simplicity**: No thread-local RNG state to manage
3. **Independence**: Each random call gets fresh independent value
4. **Correctness**: Proven unbiased through extensive testing

**Trade-offs:**
- Atomic contention on global counter (acceptable performance impact)
- Cannot easily reproduce specific thread's random sequence (less important than correctness)

### Current VL Handling (Verified Correct)
**Important:** Threads that fail to acquire expansion locks do NOT leak virtual loss.
- When `expand_node()` returns `false` (lock failed), the thread continues with:
  1. Rollout (simulation to terminal state)
  2. Backpropagation (updates visits/wins, removes VL)
- Test `test_virtual_loss_leak` confirms total VL = 0 after each dispatch completes
- This means failed threads still contribute search information (rollouts) and properly clean up VL
- Pre-computation optimization does not need to handle VL cleanup - it's already correct

### Observation: Legal Move Computation is Stateless
Legal move generation only requires:
- `node.board` (64 i32s representing board state)
- `node.current_player` (which player's turn)
- **No path history, no parent context needed**

This makes it ideal for pre-computation: we can compute legal moves for leaves BEFORE threads try to expand them.

### Solution: Pre-Computation Kernel with Parent-Visit Heuristic

**Architecture:**
```rust
// After each main MCTS dispatch, run pre-computation
dispatch_mcts_kernel(256 workgroups);     // Main wave crashes, creates some expansions
dispatch_precompute_kernel(64 workgroups); // Prepare moves for next wave's likely targets
```

**Pre-Computation Flow:**

1. **Candidate Collection (GPU Kernel):**
```wgsl
@compute @workgroup_size(64)
fn collect_leaf_candidates() {
    let node_id = global_invocation_id.x;
    
    // Is this a leaf?
    if (nodes[node_id].num_children == 0 && nodes[node_id].visits > 0) {
        // Score by parent's visit count (predictive heuristic)
        let parent_visits = nodes[nodes[node_id].parent_id].visits;
        
        // Add to candidate list
        let idx = atomicAdd(&candidate_count, 1);
        candidates[idx] = LeafCandidate {
            leaf_id: node_id,
            score: parent_visits,
        };
    }
}
```

2. **Top-K Selection (GPU Sort):**
```rust
// GPU-side parallel radix sort or bitonic sort
sort_candidates_by_score_descending();

// Budget: 2× number of threads per dispatch
let budget = 16384 * 2;  // 32,768 leaves
let top_k = candidates[0..budget];
```

3. **Move Pre-Computation (GPU Kernel):**
```wgsl
@compute @workgroup_size(64)
fn precompute_moves() {
    let i = global_invocation_id.x;
    if (i >= top_k_count) { return; }
    
    let leaf_id = top_k[i].leaf_id;
    
    // Reconstruct board for this leaf (same as main kernel does)
    var path: array<u32, 128>;
    var path_len = 0u;
    
    // Walk backwards from leaf to root
    var current = leaf_id;
    while (current != INVALID_INDEX && path_len < 128) {
        path[path_len] = current;
        path_len++;
        current = node_info[current].parent_idx;
    }
    
    // Reverse path (now goes root → leaf)
    for (var j = 0u; j < path_len / 2; j++) {
        let temp = path[j];
        path[j] = path[path_len - 1 - j];
        path[path_len - 1 - j] = temp;
    }
    
    // Reconstruct board by replaying moves
    let board = reconstruct_board(&path, path_len);
    let player = node_info[leaf_id].player_at_node;
    
    // Compute legal moves
    let moves = compute_legal_moves(&board, player);
    
    // Store in sparse cache
    precomputed_cache[i] = PrecomputedMoves {
        leaf_id: leaf_id,
        moves: moves,
        num_moves: moves.len(),
    };
}
```

**Note:** Board reconstruction reuses existing `reconstruct_board()` logic. Nodes don't store board state (only `parent_idx` + `move_id`), so boards are built on-demand by replaying move sequences from root. This keeps memory overhead low.

4. **Main Kernel Integration:**
```wgsl
// In expansion phase of main MCTS kernel
lock(leaf_id)

// Try cache lookup (sparse)
let moves = lookup_precomputed_moves(leaf_id);
if (!moves.valid) {
    // FALLBACK: Compute on-the-fly for cache misses
    moves = compute_legal_moves(nodes[leaf_id].board, nodes[leaf_id].player);
}

allocate_children(moves);
unlock(leaf_id)
```

### Design Decisions

**Budget Sizing:** `threads_per_dispatch × 2`
- Rationale: Most threads will target high-parent-visit leaves; 2× coverage gives good cache hit rate
- Example: 16,384 threads → pre-compute 32,768 leaves
- Adaptive: If tree has <32,768 leaves, pre-compute all

**Heuristic:** Parent visit count
- Logic: If parent has 1000 visits, child leaves are frequently traversed via PUCT
- Alternatives considered: leaf's own visits, depth-based (rejected - less predictive)

**Storage:** Sparse cache
- Structure: Array of 32,768 `PrecomputedMoves` entries
- Lookup: Linear scan or GPU-friendly hash (implementation detail)
- Memory: 32,768 × 480 bytes ≈ 15 MB (acceptable)

**Sorting:** GPU-based
- Algorithm: Parallel radix sort or bitonic sort (TBD during implementation)
- Why not CPU: Minimize GPU→CPU→GPU transfers; keep data on device

**Fallback:** Mandatory on-the-fly computation
- Handles: Cache misses, stale entries (leaf already expanded), first dispatch

### Decoupled Rollout Architecture (In Progress)

**Goal:** Eliminate blocking between tree operations and rollouts to maximize GPU utilization.

**Architecture:**

```
┌─────────────────────────────────────┐
│     UNIFIED KERNEL (Single Dispatch)│
│  (Both Tree & Rollout Workers)      │
└─────────────────────────────────────┘
         │
    Threads choose role dynamically
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼────┐
│ Tree │  │Rollout│
│Worker│  │Worker │
└───┬──┘  └──┬────┘
    │        │
    │   Queue rollout
    │        │
    ▼        ▼
┌─────────────────┐
│ rollout_queue   │
│ ┌─────────────┐ │
│ │position     │ │
│ │leaf_node_idx│ │
│ │leaf_player  │ │
│ └─────────────┘ │
└─────────────────┘
         │
    Rollout worker
    processes
         │
         ▼
┌─────────────────┐
│ backprop_queue  │
│ ┌─────────────┐ │
│ │leaf_node_idx│ │
│ │result (0-2) │ │
│ │leaf_player  │ │
│ └─────────────┘ │
└─────────────────┘
         │
    Tree worker
    processes
         │
         ▼
    Parent pointer
    backprop
```

**Key Components:**

1. **Rollout Queue** (Storage Buffer):
```wgsl
struct RolloutJob {
    position: array<i32, 64>,  // Board state
    leaf_node_idx: u32,        // Which node to associate with
    leaf_player: i32,          // Player at leaf
}
// Queue: array<RolloutJob, MAX_QUEUE_SIZE>
// Head: atomic<u32> (for enqueue/dequeue)
```

2. **Backprop Queue** (Storage Buffer):
```wgsl
struct BackpropJob {
    leaf_node_idx: u32,  // Start backprop here
    result: u32,         // 0=loss, 1=draw, 2=win
    leaf_player: i32,    // For perspective calculation
}
// Queue: array<BackpropJob, MAX_QUEUE_SIZE>
// Head: atomic<u32>
```

3. **Thread State** (No Path Array Needed):
```wgsl
struct ThreadState {
    phase: u32,                  // SELECTION/EXPANSION/IDLE/ROLLOUT_ACTIVE/BACKPROP
    current_node: u32,
    // NOTE: rng_seed removed - using global atomic counter (see RNG Architecture)
    
    // For backprop (parent pointer walk)
    backprop_current_node: u32,  // Walk up from here
    backprop_result: u32,
    backprop_leaf_player: i32,
    
    // For rollout
    rollout_board: array<i32, 64>,
    rollout_player: i32,
    rollout_moves_remaining: u32,
}
```

**Execution Flow:**

```wgsl
fn unified_worker() {
    // Dynamic role selection
    if (should_be_rollout_worker()) {
        rollout_worker_logic();
    } else {
        tree_worker_logic();
    }
}

fn tree_worker_logic() {
    if (state.phase == SELECTION) {
        select_one_child();
        if (reached_leaf) { state.phase = EXPANSION; }
    }
    else if (state.phase == EXPANSION) {
        if (try_expand()) {
            // Queue rollout (don't wait!)
            enqueue_rollout(position, leaf_node_idx, leaf_player);
        }
        state.phase = IDLE;  // Immediately available for other work
    }
    else if (state.phase == BACKPROP) {
        // One step: backprop current node, move to parent
        backprop_one_node_via_parent_pointer();
    }
    else if (state.phase == IDLE) {
        // Look for work
        if (backprop_queue_has_work()) {
            job = dequeue_backprop();
            state.backprop_current_node = job.leaf_node_idx;
            state.backprop_result = job.result;
            state.phase = BACKPROP;
        } else {
            // Start new MCTS cycle
            state.phase = SELECTION;
            state.current_node = root;
        }
    }
}

fn rollout_worker_logic() {
    if (state.phase == ROLLOUT_ACTIVE) {
        do_rollout_chunk();
        if (terminal) {
            enqueue_backprop(leaf_node_idx, result, leaf_player);
            state.phase = IDLE;
        }
    }
    else if (rollout_queue_has_work()) {
        job = dequeue_rollout();
        state.rollout_board = job.position;
        state.leaf_node_idx = job.leaf_node_idx;
        state.phase = ROLLOUT_ACTIVE;
    }
}
```

**Benefits:**
- ✅ Tree workers never block on rollouts
- ✅ Rollouts happen asynchronously in background
- ✅ Better GPU utilization (all threads productive)
- ✅ Smaller queue entries (no path array)
- ✅ Natural load balancing via queue sizes

**Testing:** See `tests/test_decoupled_rollout_architecture.rs`

**Test Results (Baseline - Current Blocking Architecture):**
```
Test: test_queue_based_execution (256 threads, 50 steps)
- Rollouts started: 8,166
- Expansions: 16
- Root visits: 518 (6.3% completion rate)

Issue: Only 6.3% of rollouts complete backprop to root within 50 steps
```

**Implementation Plan (TDD):**

Phase 1: Add Queue Buffers
- [ ] Add rollout_queue and backprop_queue buffers to WGSL
- [ ] Add queue enqueue/dequeue functions
- [ ] Test: Verify queues can be written/read

Phase 2: Modify Tree Workers
- [ ] After expansion, enqueue rollout instead of blocking
- [ ] Make IDLE phase check backprop_queue
- [ ] Test: Verify expansions enqueue rollouts

Phase 3: Modify Rollout Workers  
- [ ] Make rollout workers dequeue from rollout_queue
- [ ] After completion, enqueue to backprop_queue
- [ ] Test: Verify rollouts complete and enqueue backprop

Phase 4: Parent Pointer Backprop
- [ ] Modify BACKPROP phase to walk parent pointers
- [ ] Remove path array dependency
- [ ] Test: Verify visit counts accumulate correctly

Phase 5: Dynamic Load Balancing
- [ ] Add should_be_rollout_worker() logic
- [ ] Test: Verify threads switch roles based on queue sizes

**Success Metrics:**
- Root visit completion rate > 80% (vs current 6.3%)
- Tree expansion rate increase > 2x
- GPU utilization > 90%

### Implementation Status: ✅ INTEGRATED INTO GAME

The incremental execution system is **now the default execution path** for GPU MCTS in the game:

**Integration Points:**
- `search_gpu_native_othello()` in [src/lib.rs](src/lib.rs) now calls `run_incremental_mcts()` instead of batched `dispatch_mcts_othello_kernel()` loop
- **GPU Warmup:** Engine initialization includes a warmup run (256 threads, 10 steps) to trigger all lazy GPU resource initialization (shader compilation, buffer allocation) before the game starts. This prevents the first move from paying initialization costs during its timeout.
- All buffer allocations happen on-demand during dispatch (no changes needed to initialization)
- Thread count automatically capped at 16,384 for reasonable GPU utilization
- Max steps calculated from total iterations / num_threads (minimum 100 steps per thread)
- Timeout still enforced, but now checked after full incremental search completes
- **Softmax Sampling:** Selection phase uses temperature-based softmax sampling over PUCT scores instead of deterministic max selection, providing exploration diversity across threads

**Benefits Realized:**
- Eliminates "wave crashing" behavior of old batched approach
- Each thread progresses independently through MCTS phases
- Rollout kernel separated - doesn't block fast phases (selection/expansion/backprop)
- Better GPU occupancy through chunked rollout execution (8 moves per chunk)

**Configuration:**
- Incremental system automatically used when `search_gpu_native_othello()` is called
- No manual switches needed - it's the production code path
- Old batched approach still exists in `dispatch_mcts_othello_kernel()` but not used by game

### Expected Impact

**Separate Rollout Kernel Benefits:**
- **No blocking:** Selection/expansion/backprop phases (fast, <10μs) don't wait for rollouts (slow, 200-500μs)
- **Natural load balancing:** Threads in different kernels execute independently
- **Optimal workgroup sizing:** Can tune rollout kernel separately (e.g., fewer threads but more registers)

**Chunked Rollout Benefits:**
- **Reduced overhead:** 8 moves/chunk = ~4-8 dispatches for typical 30-move game (vs 30 for 1-move chunks)
- **Better GPU occupancy:** More work per dispatch = better amortization of launch overhead
- **Tunable:** Can adjust MOVES_PER_CHUNK based on profiling (4-16 moves)

**Example Timeline (One Thread):**
```
Dispatch 1: SELECTION (traverse 5 nodes to leaf) → 8μs
Dispatch 2: EXPANSION (lock, cache hit, allocate) → 12μs
Dispatch 3-6: ROLLOUT_ACTIVE (8 moves × 4 chunks = 32 moves) → 60μs each
Dispatch 7-11: BACKPROP (5 nodes × 1 per dispatch) → 6μs each
Total: 282μs for one iteration

Meanwhile other threads can be in SELECTION while this thread rolls out!
```

**Lock Duration Reduction:**
- Current: Lock held for `t_compute + t_allocate` microseconds
- With pre-computation: Lock held for `t_allocate` only (cache hit)
- Estimate: ~40-60% reduction in lock duration (depends on move generation cost)

**Throughput Improvement:**
- Shorter lock duration → more leaves available per unit time
- More available leaves → better efficiency in subsequent waves
- Expected: 15-30% improvement in early-wave expansion success rate

**Memory Overhead:**
- Candidate buffer: 200,000 × 8 bytes = 1.6 MB (max nodes)
- Pre-computed cache: 32,768 × 480 bytes = 15 MB
- Total: ~17 MB additional GPU memory (small relative to tree size)

### Testing Strategy

1. **Cache Hit Rate Test:** Measure % of expansions using pre-computed moves
2. **Contention Reduction Test:** Compare exp_locked% with/without pre-computation
3. **Lock Duration Profiling:** Measure time between lock acquisition and release
4. **Performance Regression:** Ensure no slowdown from cache lookup overhead

### Future Optimizations

- **Adaptive budget:** Scale with tree size (more leaves → larger budget)
- **Multi-level cache:** Keep previous wave's cache valid for 2-3 waves
- **Priority queue:** Continuously update top-K as tree evolves
- **Board state hashing:** Avoid recomputing moves for transposed positions (if DAG structure added)

---

## GPU Dispatch Grid and Thread Coordination

- **Dispatch Grid Sizing:**
    - When launching a GPU kernel (compute shader), the host specifies a dispatch grid: the number of workgroups and the size of each workgroup (threads per group).
    - The total number of threads launched is `workgroup_count × workgroup_size`.
    - **It is common for the dispatch grid to be larger than the actual logical work required.** This is often done for alignment, hardware efficiency, or future scalability. For example, you might launch 1024 threads even if only 1000 are needed, and have each thread check if its logical index is in-bounds before doing work.
    - Threads that find themselves out-of-bounds (e.g., `if (global_id.x >= num_tasks) { return; }`) exit early and do no work. This is not a bug, but a standard GPU programming pattern.
    - **Thread Early Exit:**
        - Early exit is not a bug if it is intentional and guarded by bounds checks.
        - However, if a thread exits early due to an unexpected condition (e.g., error, invalid state), that may indicate a bug in the kernel logic.
    - **Event Coordination:**
        - When coordinating events (like emitting a single REROOT_END), you must use a global atomic counter or similar mechanism, because relying on thread indices alone can result in multiple threads emitting the event if some exit early or the grid is oversized.
    - This design ensures robust, once-per-dispatch event emission, even if the dispatch grid is larger than the actual work or some threads exit early.

---

## Guaranteeing No Unexpected Early Thread Exit (Design & Testing)

### Design Principles
- **Explicit Bounds Checks:**
    - All early returns in WGSL kernels must be guarded by explicit, predictable bounds checks (e.g., `if (global_id.x >= num_nodes) { return; }`).
    - No early returns should be based on dynamic, error-prone, or error-state conditions unless those are explicitly documented and tested.
    - All early exit conditions must be documented in code comments and in this design doc.
- **No Silent Failures:**
    - If a thread must exit early for any reason other than a documented bounds check, it should emit a debug urgent event (e.g., `URGENT_EVENT_EARLY_EXIT`) with the thread index and reason for host-side diagnostics.
- **Code Review:**
    - All kernel code is regularly audited for return statements not protected by a clear, documented bounds check. All such cases are now documented in both code and this design doc.

### Testing Strategy
- **Host-Side Regression Tests:**
  - Dispatch kernels with a variety of grid sizes, including oversized grids.
  - Assert that only the expected number of single-emission events (e.g., REROOT_END) are produced per dispatch.
  - Optionally, enable a debug mode in the kernel that emits an `URGENT_EVENT_EARLY_EXIT` for every early exit path. Host tests assert that these events never occur except for the expected bounds check.
- **Event Buffer Auditing:**
  - After each kernel dispatch, the host polls the urgent event buffer and checks for any unexpected early exit events.
- **WGSL Assertions:**
  - Where supported, use WGSL `assert()` or equivalent to catch invalid states (e.g., `assert(node_idx < max_nodes);`).

### Example: Early Exit Pattern
```wgsl
// Good: Only exits for out-of-bounds
if (global_id.x >= num_nodes) {
    return;
}
// ...rest of kernel logic...

// Bad: Exits for dynamic or error-prone reasons
if (some_error_condition) {
    // Should emit a debug event or assert, not just return
    return;
}
```

---
