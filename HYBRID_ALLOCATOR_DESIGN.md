# December 2025 Update: Robust Urgent Event Logging and Host Injection

Recent changes:
- The urgent event logging pipeline is now robustly tested for both GPU-to-CPU and CPU-to-CPU (host-injected) events.
- The Rust struct `UrgentEvent` now exactly matches the WGSL struct layout (`timestamp: u32`, not `u64`), ensuring correct field alignment and event detection.
- The host can inject urgent events with arbitrary payloads for diagnostics and testing, via `log_urgent_event_from_cpu_with_payload`.
- The urgent event polling thread tracks the write head and only processes new events, with correct wraparound handling and minimal terminal output.
- All event logging and buffer access is synchronized and tested, with buffer mapping/unmapping and `device.poll` for correctness.
- The design is validated by passing both GPU and CPU event logging tests, ensuring reliability for both production and diagnostics.


# GPU-Side Hybrid Allocator and Event Logging Architecture (December 2025)

## Overview
This document details the architecture of the GPU-side memory allocator and urgent event logging system for Monte Carlo Tree Search (MCTS) in Othello, as implemented in `parallel-mcts-arena`. The design is focused on:
- High-throughput, contention-free node allocation and recycling for millions of tree nodes
- Robust, lock-free GPU-to-CPU urgent event logging
- Correctness and testability for large-scale, parallel MCTS

## Core GPU Data Structures

### Node Pool Buffers
- **node_info_buffer**: Array of structs, one per node, containing parent, move, flags (including deleted/zero bits), and other metadata.
- **node_visits_buffer**: Per-node visit counts (atomic, for backpropagation).
- **node_wins_buffer**: Per-node win counts (atomic, for backpropagation).
- **children_indices_buffer**: Per-node child index lists.
- **root_board_buffer**: Board state for the root node.

### Hybrid Allocator Buffers
- **free_lists_buffer**: [workgroup][slot] 2D array (256 × 8192 = 2M slots). Each workgroup manages its own ring buffer of free node indices.
- **free_tops_buffer**: [workgroup] array of atomic counters, one per workgroup, tracking the top of each free list.
- **alloc_counter_buffer**: Global atomic counter for fallback allocation when a workgroup's free list is empty.

### Urgent Event Logging Buffers
- **urgent_event_buffer**: Host-mapped ring buffer (256 × 1024 bytes). Each event is a struct with `timestamp: u32`, `event_type: u32`, `_pad: u32`, and `payload: [u32; 255]` (matches WGSL layout exactly).
- **urgent_event_write_head**: Atomic counter (GPU increments, host polls) for event production/consumption.

## GPU-Side Algorithms

### Node Allocation (WGSL)
1. **Workgroup-Local Fast Path:**
   - Each thread first tries to pop a node index from its workgroup's free list (LIFO, atomic decrement of free_tops[wg]).
   - If successful, the node is reused and ready for initialization.
2. **Global Fallback:**
   - If the free list is empty, the thread atomically increments the global alloc_counter to claim a new node index.
   - If alloc_counter < max_nodes, allocation succeeds; otherwise, memory pressure policy is triggered.
3. **Memory Pressure Handling:**
   - If all nodes are exhausted, expansion stops. Rollouts may continue, or the host may be signaled to play a move early and prune.

### Node Recycling and Pruning
- **Top-Down Pruning:**
  - After a root move, all unreachable subtrees are pruned in parallel.
  - Each worker claims a subtree root and traverses descendants (BFS/DFS), atomically setting a deleted bit per node.
  - Freed nodes are pushed to the workgroup's free list (if space), or to a global overflow if needed.
  - The deleted bit ensures each node is only processed once, even if reached by multiple workers.
- **Value-Based Recycling:**
  - When memory is nearly exhausted, only low-value leaves (lowest PUCT/visit) are recycled, never high-value subtrees.

### Backpropagation
- After each simulation, the kernel walks up the path from leaf to root, atomically updating visit/win stats for each node.

### Urgent Event Logging (GPU → CPU)
- GPU kernels write urgent events to the host-mapped ring buffer at `write_head % 256`, incrementing `write_head` atomically.
- The host can also inject urgent events for diagnostics/testing using `log_urgent_event_from_cpu_with_payload`, which writes to the buffer and advances the write head.
- Events include search halts, root advances, diagnostics, memory pressure signals, and test/diagnostic events.
- The host polls the buffer, processes only new events (using last seen write_head), and handles wraparound/overflow robustly.
- Logging is rate-limited and non-verbose by design, and all buffer access is synchronized with atomic flags and `device.poll`.

## Host Integration and Synchronization
- All GPU buffer creation, bind group layout, and shader preprocessing (for includes) are performed in Rust, with validation against WGSL expectations.
- All mutable state in `GpuOthelloMcts` is protected by an internal Mutex; all mutation is via `&self` methods with internal locking.
- An AtomicBool is used to synchronize buffer mapping/unmapping between host and GPU.
- Tests drain pre-existing urgent events, use `device.poll(wgpu::Maintain::Wait)` for synchronization, and validate all buffer layouts and event flows.

## Node Lifecycle Summary
1. **Allocation:** Try workgroup free list → fallback to global alloc_counter.
2. **Initialization:** Node fields set, children created as needed.
3. **Simulation/Backprop:** Visits/wins updated atomically.
4. **Pruning:** Unreachable subtrees pruned top-down, nodes recycled.
5. **Recycling:** Node index returned to workgroup free list.
6. **Urgent Events:** GPU logs events for host to process (e.g., memory pressure, search halt).

## Data Structures

### GPU Buffers (Modified)

```rust
pub struct GpuOthelloMcts {
    // ===== EXISTING (unchanged) =====
    max_nodes: u32,  // Still 2,000,000
    node_info_buffer: Buffer,
    node_visits_buffer: Buffer,
    node_wins_buffer: Buffer,
    children_indices_buffer: Buffer,
    root_board_buffer: Buffer,
    // ... etc
    
    // ===== Per-Workgroup Free Lists =====
    free_lists_buffer: Buffer,  // [256 workgroups][8192 slots] = 2M u32s
    free_tops_buffer: Buffer,   // [256] atomic<u32> counters
    // cutoff_generation: u32, // REMOVED: generational cleanup no longer used
    // ===== Global Allocator =====
    alloc_counter_buffer: Buffer,  // Global atomic counter for fallback allocation
}
```

### WGSL Shader Changes

```wgsl
// Constants
const WORKGROUPS: u32 = 256u;
const FREE_LIST_SIZE_PER_GROUP: u32 = 8192u;  // 8K slots per workgroup

// Per-workgroup free lists (2D array)
@group(1) @binding(5) var<storage, read_write> free_lists: array<array<u32, FREE_LIST_SIZE_PER_GROUP>, WORKGROUPS>;
@group(1) @binding(6) var<storage, read_write> free_tops: array<atomic<u32>, WORKGROUPS>;

// Global allocator (fallback)
@group(1) @binding(8) var<storage, read_write> alloc_counter: atomic<u32>;

// Params
struct MctsParams {
    // ... existing fields ...
}
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
    // Instead of returning INVALID_INDEX, trigger memory pressure handling:
    // - Stop expansion, only allow rollouts
    // - Optionally, signal to host or log event
    // - Return a sentinel or handle gracefully
    return INVALID_INDEX; // Actual handling is policy-dependent
}
```

**Key Points:**
- Only ~256 threads contend per free list (vs 2048 before)
- If a workgroup's list is empty, falls back to global allocation (consuming new/free memory)
- Node state is managed via node_state and value-based recycling
- If all memory is exhausted, memory pressure policy is triggered (see below)

---

### 2. **Pruning Unreachable Subtrees (Efficient Top-Down Freeing)**



After re-rooting (i.e., after a move is made and the root node is changed), we know exactly which subtrees are no longer reachable: all children of the previous root except the selected child. Instead of scanning the entire node pool or doing bottom-up reachability checks, we can efficiently free these subtrees using a top-down traversal with a deleted bit, and distribute the work efficiently across parallel workers:

**TODO:** Implement periodic rebalancing of per-workgroup free lists to address long-term imbalance. For now, focus on workgroup stealing (if a workgroup's free list is empty, it can steal nodes from other workgroups or the global free list).

**Single-Pass Deleted-Bit Algorithm:**


**Dynamic Partitioning for Pruning:**

1. Identify all children of the outgoing root except the selected child (these are the roots of unreachable subtrees).
2. Launch a parallel pruning kernel where each worker dynamically claims node indices from a global atomic counter or work queue. No static partitioning is used.
3. Each worker performs a top-down traversal (BFS or DFS) of the subtree it claims, visiting all descendants. The atomic deleted bit ensures each node is only processed once, even if multiple workers reach it via different paths.
4. For each visited node:
    - Atomically set a `deleted` bit (or field) in the node. If the bit was already set, skip further processing for this node (deduplication).
    - Optionally, add the node to the appropriate workgroup's free list for recycling. If the workgroup's free list is full, push the node to a global free list for overflow handling.
    - Do not clear other node data immediately; allocation and traversal logic will respect the deleted bit. The deleted bit should be checked only in allocation and pruning contexts, not in the main MCTS search path for performance.

**Why this works:**

- The atomic set-and-check of the deleted bit ensures that each node is only processed (and freed) once, even if multiple workers reach it via different paths.
- No node will be missed, as all descendants are visited from the known subtree roots.
- No ancestor will be deleted before its descendants in a way that causes missed nodes, because the deleted bit prevents double-processing and the traversal is exhaustive from each root.
- Dynamic partitioning (atomic work queue/counter) ensures efficient load balancing and prevents bottlenecks from large subtrees. Static partitioning is not used.

**Key Points:**

- Only a single pass is needed; no need for a second clearing pass.
- The atomic deleted bit guarantees no duplicates and no missed nodes, even with parallel workers.
- Allocation and traversal logic must check the deleted bit to avoid using freed nodes, but this check should be as infrequent as possible for performance.
- Optionally, a background or periodic pass can clear node data for debugging or memory hygiene, but this is not required for correctness.
- Logging should never be verbose; aggregate fast events and log only at a slower rate to avoid buffer saturation and performance impact.

----


### 3. **Diagnostics and Logging**
---

## Urgent Event Logging (2025 Feature)
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

**Ready to implement?** Review this design and give the go-ahead! 🚀

---

## Selective Node Recycling and Pruning Policy (December 2025 Update)

### Motivation

The allocator should never prune large, valuable subtrees solely due to memory pressure. Pruning should only occur for subtrees that become unreachable after a move (natural pruning), or for nodes that are objectively low-value. 


### Memory Pressure Policy (2025 Update)

When the allocator detects that GPU memory is nearly exhausted (e.g., global alloc counter >= max_nodes):

1. Elegantly stop all search work on the GPU (while keeping data members consistent).

2. **If root visits exceed a threshold (e.g., > X):**
    - Play the move early (select the best child of the root based on current statistics).
    - Immediately prune all non-selected subtrees using the efficient top-down freeing method.
    - Log a terminal message indicating that an early move was played due to memory pressure.

3. **If root visits are below the threshold:**
    - Stop expansion (do not allocate new nodes), but continue rollouts (simulations) on the existing tree.
    - When the threshold to play early move is reached or time runs out, play the move and prune as above.

This policy ensures the system never overruns memory, maintains robust operation, and degrades gracefully under pressure. It also prioritizes search quality by only playing early when enough information has been gathered. The allocation function should not simply return INVALID_INDEX on exhaustion, but should trigger this policy and handle the situation gracefully.


---

### New Policy

1. **Natural Pruning Only:**  
    - Subtrees are pruned only when they become unreachable due to a root move (advance_root).  
    - No reachable subtree is pruned just because memory is low.

2. **Selective Node Recycling:**  
    - When nearing memory capacity, instead of pruning entire subtrees, only recycle (prune) nodes with the lowest PUCT value (i.e., least promising/visited/valuable leaves).
    - Maintain a small pool or priority queue of recyclable nodes, and only recycle these when absolutely necessary.
    - Never delete high-value, high-visit, or high-PUCT nodes.

3. **Graceful Degradation:**  
    - If memory is exhausted and no low-value leaves are available, pause or slow down search, or play the move early and start rollouts instead of expansion.
    - Log a warning or reduce batch size, but never delete valuable subtrees.

4. **Diagnostics:**  
    - Track and log memory pressure events, node recycling frequency, and the value distribution of recycled nodes.
    - Use this data to tune the recycling policy.

### Expected Outcome

- Only “dead” or low-value parts of the tree are pruned, preserving valuable search information and improving AI performance.
- Memory pressure never causes the loss of high-value subtrees.
