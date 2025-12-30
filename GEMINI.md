### Goal
Make GPU-native MCTS clearly outperform CPU-only MCTS in strength and throughput while staying correct, stable, and reusable across turns and games.  Main idea is this: We'll run MCTS on CPU and GPU almost identically (except the async thing maybe), but identical in terms of identical reward shaping (if any), identical rollout (including rollout parameters if any), etc. Since with GPU we can throw more compute at MCTS, it should outcompete an almost identical MCTS with fewer compute (CPU). The following is written by the AI who worked on it with me. So, please take it with a grain of salt.

### What we've done
- Complete rewrite of GPU-native Othello MCTS (December 2025): Clean 4-phase kernel (selection/expansion/rollout/backprop) with root board buffer + path-based state reconstruction.
- Files: `src/gpu/shaders/mcts_othello.wgsl`, `src/gpu/mcts_othello.rs`.
- Subtree reuse now working 100% - advance_root tracks `root_idx`, validates child was expanded, reparents child to index 0, preserves entire subtree.
- Fixed offset calculations: all buffer reads (node info, children) use `root_idx` offset instead of hardcoded 0.
- Performance: 1.4M-2.1M iterations per turn (vs CPU ~600k), growing trees from 200k to 1.8M nodes with continuous reuse.
- Diagnostics: selection/expansion counters, node usage tracking, children validation - all clean.

### Hiccups we faced (kept for history)

**From old implementation (pre-December 2025):**
- CPU hang after GPU move traced to per-node subtree invalidation; fixed with bulk staging traversal.
- Same-buffer copies and missing COPY_SRC triggered wgpu validation errors; resolved with staging and usage fixes.
- Free-list duplicate entry panic when reusing trees; added host-side dedup before pushing freed indices.

**From new implementation (December 2025 rewrite):**
- Tree not expanding beyond 10k nodes: selection logic stopped at unexpanded nodes (`num_children == 0`) before expansion phase could run. Fixed by treating `num_children == 0` as leaf condition, letting expansion phase handle it. Result: 150k-250k nodes per position.
- Thread clustering (only 3-10 expansion attempts per batch despite 4096 threads): threads converging on same nodes, 97% sitting idle. Fixed with retry-and-descend logic - failed expansion attempts retry selection to spread through tree. Result: 60-96 successful expansions per batch. TODO: Make it better! Consider sampling using the Q + U values as a distribution, and following the sampled route
- Expansion lock contention: many threads trying to expand same node simultaneously. Added exponential backoff with random jitter (delay = 2^retry_count iterations, max 256), increased max retries from 5 to 10.
- Subtree reuse validation failures: `get_children_stats()` hardcoded to read from offset 0, but after `advance_root` the root moved to different node index (e.g., 100). Validation read wrong children → garbage data like `(7, 536870911)` → "root children mismatch" warnings → unnecessary rebuilds. Fixed by calculating offset as `root_idx * MAX_CHILDREN * sizeof(u32)`. Now zero validation failures.

### Learnings (Do / Don't)
- Correctness first: surface divergence from invariants rather than papering over them.
- Do validate GPU root board and legal moves against host-computed legals before reuse; Don't rebuild on any discrepancy - fail instead (correctness first).
- Do ensure selected child is expanded (or rebuilt) before next search; zero-child roots are unacceptable.
- Do use `root_idx` offset for ALL buffer reads after advance_root - never assume root is at index 0.
- Do implement retry-and-descend for expansion failures to spread threads through tree instead of clustering.
- Do use exponential backoff with jitter to reduce lock contention during expansion.
- Don't stop selection at unexpanded nodes - let expansion phase handle them.
- Don't rebuild tree unnecessarily - if child was expanded, preserve its entire subtree.
- Don't perform same-buffer copies for stats/priors; use direct writes or staging buffers.

### Recent observations (December 29, 2025)
- **Subtree reuse working perfectly** - trees growing 200k → 483k → 792k → 1.08M → 1.30M → 1.53M → 1.80M nodes across turns.
- **Pruning system fully functional** - after advance_root, pruning frees 1.1M-2.7M unreachable nodes per turn.
- **Free list working correctly** - freed nodes are reused in next search, keeping alloc_count stable around 6.8M.
- **Capacity warnings expected** - with batch_size=2048 and 300+ batches, tree legitimately grows to ~6.8M nodes (90% of 7.5M capacity).
- Zero validation failures, zero crashes, zero buffer overflows.
- Expansion success rate excellent: 60-96 expansions per 4096-thread batch (95%+ utilization).
- Diagnostics clean: `sel_noch=2048` (all threads finding unexpanded nodes), `exp_locked=0` (no contention), `alloc_fail=0`.
- **GPU losing to CPU despite 2-3x more visits** - GPU gets 3.9M root visits (Q=0.517), CPU gets 600k visits (Q=0.699), yet CPU's position evaluation much better. Hypothesis: exploration factor too low (using 0.1 now, was 5.0), or virtual loss weight wrong, or PUCT formula differs from CPU, or rollouts too noisy.

### GPU-Native MCTS Architecture (December 28, 2025)

**Core design:**
- **Root board buffer:** Fixed storage holding current game board state (separate from node pool)
- **Root node tracking:** `root_idx` field tracks which node is current root (was 0 initially, changes with `advance_root` - can be any node!)
- **All other nodes:** Each node represents a game state reached via a specific path from the initial game state => which implies that each node represents a game state reached via a spacific path from the current root (as root is in that path between initial game state and the current state represented by the root).
- **Path identity:** Node's state = `root_board + sequence of moves from root to node`
- **No transposition:** Same move from different parents = different nodes (different game states)

**State reconstruction during search:**
```
Worker selects path: [root, child1, child2, ...]
Reconstructs board:
  board = root_board_buffer.clone()
  for each node in path[1..]:
    apply node.move_id to board
  Result: board at leaf node
```

Node: During a search in the selection phase, instead of selecting a move with the highest PUCT, use the PUCT as a PDF and sample a move. This will help with the problem of all workers picking up the same path.

**Expansion:**
- Worker at leaf node either calls `expand_node(node_idx, &board)` (or does a rollout)
- Board was reconstructed from current `root_board_buffer` + path
- Generates legal moves for that board state
- Creates child nodes with those moves

**Move selection and advance_root:**
1. **Search phase:** GPU searches from current root, builds tree
2. **Move selection:** Pick root's child with highest visits, extract its `move_id`
3. **Advance root (same player or opponent):**
   - Compute new board state (apply the move to current root board)
   - Update `root_board_buffer` to new board
   - Find child node corresponding to the move
   - Validate child was expanded (`num_children > 0`)
   - If not expanded: rebuild tree from scratch ; Print while doing so, as not expanding a valid move at root level is very very suspicious. 
   - If expanded: reparent child to index 0, update `root_idx` to point to new root, keep entire subtree
   - Free all other nodes (siblings and their subtrees) - currently not implemented, just overwrites

**Critical constraint for subtree reuse:**
- When child becomes new root, its children were expanded from state: `old_root_board + child.move_id`
- After `advance_root`, `new_root_board` must equal `old_root_board + child.move_id`
- **Grandchildren's children** were expanded from: `old_root_board + child.move_id + grandchild.move_id`
- After `advance_root`, they represent: `new_root_board + grandchild.move_id` (and new_root_board is same as old_root_board + child.move_id) 
- If we don't find a bijection between (old) grandchild's move id's and new root's moves, then there is some implementation bug, as this is an invariant.
- Note that when a move happens, root will change, and so will all paths, this still won't make the nodes of selected subtree invalid, as they represent the state, not the path, and the state they are representing is going to be the same irrespective of perspective (state1 -> move1 -> move2 -> move3) is same as (state2 -> move2 -> move3) where state 2 is the state derived from state1 when move1 was played at state1.

**Important implementation details from December 2025 rewrite:**
```rust
// Reading root node info - MUST use root_idx offset!
let root_offset = self.root_idx as u64 * std::mem::size_of::<OthelloNodeInfo>() as u64;

// Reading root's children - MUST use root_idx offset!
let children_offset = self.root_idx as u64 * MAX_CHILDREN as u64 * std::mem::size_of::<u32>() as u64;
```

```wgsl
// Selection logic - keep descending until num_children == 0
while num_children > 0 {
    best_child = select_best_child_puct(current);
    path[path_len] = best_child;
    path_len++;
    current = best_child;
    num_children = nodes[current].num_children;
}
// Exit when num_children == 0, let expansion phase handle it

// Expansion logic - use atomic CAS to acquire lock
if atomicLoad(&nodes[node_idx].num_children) == 0 {
    let old_children = atomicCompareExchangeWeak(&nodes[node_idx].num_children, 0, EXPANSION_LOCK);
    if old_children.exchanged {
        // Won the lock, expand node
        expand_node(node_idx, board);
    } else {
        // Lost the lock, retry selection with exponential backoff
        let delay = (1u << min(retry_count, 8u)) + (rng.next() & 0xFFu);
        for (var i = 0u; i < delay; i++) { /* busy wait */ }
        continue;  // Retry selection
    }
}
```

### Next steps (objectives)

Fix move quality - GPU not consistently winning despite more compute
GPU is getting more compute and yet it doesn't win. 


The command used by AI to build should be: cargo build
Command used by user would be: cargo build --release
Command used by user to run currently is: $env:RUST_BACKTRACE="full"; .\target\release\play.exe -n 24 -b 8 -l 4 --stats-interval-secs 1 --gpu-use-heuristic false --timeout-secs 3 --gpu-exploration-factor 5 -m 1000000 -s 1000000000 --gpu-native-batch-size 4096  --gpu-virtual-loss-weight 1
