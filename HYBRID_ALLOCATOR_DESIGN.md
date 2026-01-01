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
2. **MCTS Search:** The main GPU-native MCTS kernel (`main`/`mcts_othello_iteration` in WGSL) is dispatched from Rust for Othello. Thousands of GPU threads run MCTS in parallel, allocating and recycling nodes using a hybrid allocator. This kernel is responsible for all core search logic, including re-rooting, expansion, simulation, backpropagation, and robust urgent event logging (including REROOT events, with atomic coordination to ensure only one REROOT_END per move).
3. **Pruning:** After each move, unreachable subtrees are pruned in parallel by dispatching the pruning kernel, and nodes are recycled. PRUNING_END events now include the pruned node count in their payload.
4. **Urgent Event Logging:** Both the main MCTS kernel and the pruning kernel log important events (e.g., REROOT, PRUNING, memory pressure) to a host-mapped ring buffer. Host code polls and processes these events, ensuring that REROOT events are visible in real gameplay. The urgent event buffer write head is reset before each kernel dispatch to prevent event spam.
5. **Host Integration:** Host code manages buffer mapping/unmapping, event polling, and diagnostics, ensuring robust synchronization. All buffer mapping/unmapping is guarded by a static Mutex. All host reads of urgent event data use a staging buffer; GPU-write buffers are never mapped directly for reading. The host is responsible for dispatching both the main MCTS kernel and the pruning kernel, and for polling/logging urgent events.
6. **Diagnostics and Testing:** Automated, test-driven debugging is used for GPU resource issues, with minimal and integration tests for emission guard logic and event logging. Integration tests assert that only one REROOT_END event is emitted per move. Diagnostic/temporary event spam has been removed from production runs.

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

### Main MCTS Kernel (GPU-Native Search, BATCH Logging)
- The main kernel (`main`/`mcts_othello_iteration`) is dispatched from Rust and runs the full MCTS search loop on the GPU, including:
    - Selection, expansion, simulation, backpropagation
    - Re-rooting logic (after a move)
    - Logging BATCH_START and BATCH_END events to the urgent event buffer using atomic coordination.
    - **BATCH_START Coordination:** A global atomic counter (`global_reroot_start_threads_remaining`) is used to ensure only the very last thread of the entire dispatch logs the BATCH_START event. This is initialized by the host before dispatch.
    - **BATCH_END Coordination:** A global atomic counter (`global_reroot_threads_remaining`) is used to ensure only the very last thread of the entire dispatch logs the BATCH_END event. **Crucially, this counter is initialized by the host (Rust) before dispatch**, not by the shader, to prevent race conditions where multiple workgroups might try to initialize it simultaneously.
    - Logging memory pressure and other urgent events as needed
    - All event payloads (including turn number and atomic state) are written to the urgent event buffer, which is polled by the host

### Node Recycling and Pruning (Updated for "Reroot & Identify")

The pruning process is split into two phases to allow the GPU to identify which nodes to prune based on the Host's move choice.

**Phase 1: Identify Garbage & Next Root**
- **Input:** `RerootParams` (Uniform Buffer) containing:
    - `move_x`, `move_y`: The move chosen by the Host.
    - `current_root_idx`: The GPU node index of the current root.
- **Logic (Compute Shader):**
    - Scans the children of `current_root_idx`.
    - **If child matches move:** Writes child index to `NewRootOutput` buffer (for Host to read).
    - **If child does NOT match:** Writes child index to `work_queue` for pruning.
    - **If no match found:** (Error case) Signals failure or resets tree.

**Phase 2: Prune Unreachable Top-Down**
- **Input:** `work_queue` (populated by Phase 1).
- **Logic (Compute Shader):**
    - Pops a node from `work_queue`.
    - Marks it as deleted (adds to free list).
    - Pushes all its children to `work_queue`.
    - Repeats until queue is empty.

**Host Coordination:**
1.  Host calls `advance_root(move)`.
2.  Host writes `RerootParams`.
3.  Host dispatches **Phase 1 Kernel**.
4.  Host dispatches **Phase 2 Kernel**.
5.  Host reads `NewRootOutput` to get the new `root_idx`.
6.  Host updates `MctsParams.root_idx` for the next search batch.

### Urgent Event Logging
- **Events:**
    - `BATCH_START` / `BATCH_END`: Mark the start/end of a search batch (formerly REROOT_START/END).
    - `REROOT_OP_START` / `REROOT_OP_END`: Mark the start/end of the pruning/rerooting operation.
    - `PRUNING_START` / `PRUNING_END`: Specific to the pruning kernel execution.
- **Mechanism:** Ring buffer with atomic write head. Host polls this buffer.

### Backpropagation
- After each simulation, the kernel walks up the path from leaf to root, atomically updating visit/win stats for each node.

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

    // Step 3: Memory pressure fallback (not just INVALID_INDEX)
    return INVALID_INDEX; // Actual handling is policy-dependent
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

**Impact:** Both bits are lightweight, and together allow the allocator to distinguish between "fresh" and "needs-wash" nodes, optimizing allocation and reuse paths. In the current implementation, node state is tracked via a `node_state` enum (e.g., EMPTY, READY, etc.), and value-based recycling is used to reclaim memory. Allocation logic checks node state, not explicit bitfields. If no nodes are available, memory pressure policy is triggered as described above.

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

2. **Selective Node Recycling:**
   - As memory capacity is approached, instead of pruning entire subtrees, only recycle (prune) nodes with the lowest PUCT value (least promising/visited/valuable leaves).
   - Maintain a small pool or priority queue of recyclable nodes, and only recycle these when absolutely necessary.
   - High-value, high-visit, or high-PUCT nodes are never deleted.

3. **Graceful Degradation Under Memory Pressure:**
   - When the allocator detects that GPU memory is nearly exhausted (e.g., global alloc counter >= max_nodes):
     - Elegantly stop all search work on the GPU, keeping data members consistent.
     - If the root node has been visited enough times (above a threshold), play the move early by selecting the best child of the root based on current statistics, then prune all non-selected subtrees using efficient top-down freeing.
     - If the root has not reached the threshold, stop expansion (do not allocate new nodes), but continue rollouts (simulations) on the existing tree. When the threshold is reached or time runs out, play the move and prune as above.
     - The allocation function never simply returns INVALID_INDEX on exhaustion; it always triggers this policy and handles the situation gracefully.
   - If memory is exhausted and no low-value leaves are available, pause or slow down search, or play the move early and start rollouts instead of expansion. Log a warning or reduce batch size, but never delete valuable subtrees.

4. **Diagnostics and Tuning:**
   - Track and log memory pressure events, node recycling frequency, and the value distribution of recycled nodes.
   - Use this data to tune the recycling policy for optimal performance and robustness.

**Expected Outcome:**
- Only “dead” or low-value parts of the tree are pruned, preserving valuable search information and improving AI performance.
- Memory pressure never causes the loss of high-value subtrees.

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
