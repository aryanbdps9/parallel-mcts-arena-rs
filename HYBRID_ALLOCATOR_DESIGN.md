# Hybrid Allocator Design (Updated as of December 2025)


## Overview
This document describes the current design of the hybrid memory allocator for GPU-native Monte Carlo Tree Search (MCTS) in Othello, as implemented in the `parallel-mcts-arena` project. The allocator is designed for high performance, robustness, and correctness, supporting large-scale tree search on the GPU with per-workgroup free lists and diagnostics.

## Key Features
- **Per-Workgroup Free Lists:**
    - Each GPU workgroup maintains its own free list for fast, contention-free allocation and deallocation of tree nodes.
    - Free lists are implemented as ring buffers in GPU memory, sized for the maximum expected concurrency.
    - When a workgroup's free list is exhausted, it falls back to a global allocator.

- **Global Allocator Fallback:**
    - A global atomic counter is used to allocate new nodes when all local free lists are full.
    - Ensures that allocation never fails as long as there is global capacity.

- **Diagnostics and Logging:**
    - Allocation, deallocation, and pruning events are logged to a diagnostics buffer for host-side analysis.
    - Diagnostic counters track allocation failures, free list usage, and pruning statistics.

- **Buffer and Bind Group Layouts:**
    - All GPU buffers are created with correct usage flags (STORAGE, UNIFORM, COPY_SRC, COPY_DST) to match their use in compute shaders and bind group layouts.
    - Bind group layouts are explicitly defined and validated in Rust tests to ensure compatibility with WGSL shaders and wgpu validation rules.

- **Test Coverage and Validation Loop:**
    - Rust unit tests validate that all bind group layouts, bind group descriptors, and buffer usage flags are correct and that wgpu validation passes at runtime.
    - Tests now include direct validation of the root node's children after every tree initialization, and a multi-turn Othello test that simulates several moves and root advances, checking root children after each. This ensures that GPU state corruption or root mismatch bugs are caught immediately.
    - Tests are run in a continuous loop until all pass with no warnings or errors.
    - The development process enforces: "Do not return to the user until all tests pass and there are no build errors."

## Implementation Notes
- The allocator logic is implemented in WGSL (see `src/gpu/shaders/mcts_othello.wgsl`) and invoked from Rust (`src/gpu/mcts_othello.rs`).
- All buffer and bind group creation is performed in Rust, with layouts matching the shader expectations.
- The system is designed to be robust against panics, validation errors, and resource leaks.
- Diagnostics can be extended for further profiling and debugging as needed.
- Some fields and methods in the Rust implementation are intentionally marked with `#[allow(dead_code)]` to suppress warnings about unused code. This is by design, to support extensibility, future features, and robust diagnostics without cluttering the build output with warnings.

## Recent Changes
- Bind group layouts in Rust are now created with explicit entries matching the shader and bind group descriptors (9 for node pool, 5 for execution, 1 for board).
- The `params_buffer` now includes the `UNIFORM` usage flag to satisfy wgpu validation.
- All tests pass with no errors; only warnings about unused fields/methods remain.

---
# Hybrid Memory Allocator Design
## Per-Workgroup Free Lists

---

## Problem Statement

**Current System Issues:**
1. ❌ **Race condition overflow**: 2048 threads → `free_top` exceeds capacity (3.3M > 2M)
2. ❌ **Freeze on allocation**: Reading `free_list[3.3M]` → out of bounds → hang
3. ❌ **Contention**: All 2048 threads fighting over one `free_top` atomic
4. ❌ **Unpredictable**: Can't guarantee how many nodes we'll actually reuse

---

## New Design: Hybrid Approach

### Core Concepts

**1. Per-Workgroup Free Lists** (Reduces Contention)
- GPU dispatches 256 workgroups of 256 threads each
- Each workgroup gets its own small free list (8K entries)
- 256 lists × 8K = 2,048,000 total free list capacity
- **Benefit**: Only 256 threads per list instead of 2048!


**3. Fallback Global Allocator** (Safety Net)
- If workgroup's free list is empty, allocate new node
- **Benefit**: Never run out of nodes until truly full

---

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


**Efficient Parallel Work Distribution (Dynamic Partitioning):**

1. Identify all children of the outgoing root except the selected child (these are the roots of unreachable subtrees).
2. Launch a parallel pruning kernel where each worker dynamically claims work from a global atomic counter or work queue.
3. Each worker performs a top-down traversal (BFS or DFS) of the subtree it claims, visiting all descendants. The atomic deleted bit ensures each node is only processed once, even if multiple workers reach it via different paths.
4. For each visited node:
    - Atomically set a `deleted` bit (or field) in the node. If the bit was already set, skip further processing for this node (deduplication).
    - Optionally, add the node to the appropriate workgroup's free list for recycling. If the workgroup's free list is full, push the node to a global free list for overflow handling.
    - Do not clear other node data immediately; allocation and traversal logic will respect the deleted bit. The deleted bit should be checked only in allocation and pruning contexts, not in the main MCTS search path for performance.

**Why this works:**

- The atomic set-and-check of the deleted bit ensures that each node is only processed (and freed) once, even if multiple workers reach it via different paths.
- No node will be missed, as all descendants are visited from the known subtree roots.
- No ancestor will be deleted before its descendants in a way that causes missed nodes, because the deleted bit prevents double-processing and the traversal is exhaustive from each root.
- Dynamic partitioning (atomic work queue/counter) ensures efficient load balancing and prevents bottlenecks from large subtrees.

**Key Points:**

- Only a single pass is needed; no need for a second clearing pass.
- The atomic deleted bit guarantees no duplicates and no missed nodes, even with parallel workers.
- Allocation and traversal logic must check the deleted bit to avoid using freed nodes, but this check should be as infrequent as possible for performance.
- Optionally, a background or periodic pass can clear node data for debugging or memory hygiene, but this is not required for correctness.
- Logging should never be verbose; aggregate fast events and log only at a slower rate to avoid buffer saturation and performance impact.

----


### 3. **Diagnostics and Logging**

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
