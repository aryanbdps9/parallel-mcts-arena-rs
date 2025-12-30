# GPU-Native MCTS Architecture (ELI5 Version)

## The Big Picture: Why GPU?

**CPU AI (traditional):**
- Has a tree in regular RAM (pointers, dynamic allocation)
- Does MCTS one path at a time
- Like a single person exploring a maze

**GPU AI (yours):**
- Has ALL the tree data in GPU memory (fixed arrays, no pointers)
- Runs 2048 MCTS searches **simultaneously** (one per GPU thread)
- Like 2048 people exploring the maze at the same time

## The Core Data Structure: The Node Pool

Think of it like a **hotel with 2 million rooms**:

```rust
pub struct GpuOthelloMcts {
    max_nodes: u32,  // 2,000,000 rooms in the hotel
    
    // Every node has a room number (0 to 1,999,999)
    // Each room stores info in PARALLEL ARRAYS:
    
    node_info_buffer: Buffer,      // Room 5 → who lives here, who's their parent?
    node_visits_buffer: Buffer,    // Room 5 → how many times visited?
    node_wins_buffer: Buffer,      // Room 5 → how many wins?
    children_indices_buffer: Buffer, // Room 5 → which rooms are my children?
    
    // The "front desk" that tracks room usage:
    alloc_counter_buffer: Buffer,  // "We've rented out 1,234,567 rooms so far"
    free_top_buffer: Buffer,       // "We have 500,000 rooms ready to re-rent"
    free_list_buffer: Buffer,      // "List of room numbers that are clean and empty"
    
    root_idx: u32,  // "The VIP suite" - where the current game position is
}
```

## How It Works: The Full Flow

### 1. **Game Start - `init_tree()`**
```
Hotel opens for business!
- Room 0 becomes the VIP suite (root)
- Create 4 children (legal moves) → give them rooms 1, 2, 3, 4
- alloc_counter = 5 (we've used 5 rooms)
- free_top = 0 (no recycled rooms yet)
```

### 2. **MCTS Search - `run_iterations()`**
```
2048 guests arrive simultaneously!

Each guest (GPU thread):
1. Start at room 0 (root)
2. Pick best child → go to that room
3. Keep going down until you find unexpanded room
4. Need a new room for a child? 
   - Check free list first (any recycled rooms?)
   - If yes: pop from free_list, use that room number
   - If no: use alloc_counter++, get a brand new room
5. Simulate game to end
6. Walk back up, update visit counts in each room
```

### 3. **Make a Move - `advance_root()`**
```
Player picks move (3,2)
- Old VIP suite was room 0
- Child representing (3,2) is room 2
- Make room 2 the new VIP suite!
- root_idx = 2
```

### 4. **Cleanup - `prune_unreachable_nodes()`**
```
Problem: Rooms 0, 1, 3, 4 are now "old game positions"
         We'll never visit them again!

Solution: PRUNING
- Start at room 2 (new VIP suite)
- Mark all rooms reachable from room 2 (walk up parent pointers)
- Every OTHER room → EVICT THEM!
  - Clear their data (visits=0, children cleared)
  - Add their room number to the free_list
  - free_top++ (we have one more recycled room)

Example:
- Before: free_list = [], free_top = 0
- After:  free_list = [0, 1, 3, 4, 5, 6, ..., 1481263], free_top = 1,481,264
```

## The Free List: The Recycling System

**Without free list (naive approach):**
```
Turn 1: Use rooms 0-500,000 (alloc=500,000)
Turn 2: Use rooms 500,000-1,000,000 (alloc=1,000,000)
Turn 3: Use rooms 1,000,000-1,500,000 (alloc=1,500,000)
Turn 4: Use rooms 1,500,000-2,000,000 (alloc=2,000,000)
Turn 5: OUT OF MEMORY! → Reset everything → init_tree()
```

**With free list (smart approach):**
```
Turn 1: Use rooms 0-500,000 (alloc=500,000, free=0)
Turn 2: Prune → 400K rooms freed → free_top=400,000
        Reuse rooms from free_list! (alloc stays ~500K)
Turn 3: Prune → 350K more freed → free_top=750,000
        Reuse rooms from free_list! (alloc stays ~500K)
...tree never resets!
```

## The Data Flow (Concrete Example)

**Starting position (root = room 0):**
```
Room 0: [parent=NONE, move=NONE, children=[1,2,3,4]]
Room 1: [parent=0, move=(2,3), children=[]]
Room 2: [parent=0, move=(3,2), children=[]]
Room 3: [parent=0, move=(4,5), children=[]]
Room 4: [parent=0, move=(5,4), children=[]]
alloc_counter = 5
free_list = []
free_top = 0
```

**After MCTS (200K iterations, expanded many nodes):**
```
Room 0: children=[1,2,3,4]
Room 1: children=[5,6,7] (expanded!)
Room 2: children=[8,9,10]
...
Room 490,000: some leaf node
alloc_counter = 490,001
free_list = []
free_top = 0
```

**Player makes move (3,2) → advance to room 2:**
```
root_idx = 2  (room 2 is now VIP)

PRUNE:
- Room 2 is VIP → KEEP
- Room 8,9,10 are children of 2 → KEEP (reachable via parent pointers)
- Room 0,1,3,4 and all their descendants → EVICT

free_list = [0, 1, 3, 4, 5, 6, 7, 11, 12, ..., 489999]
free_top = 489,000 (we freed 489K rooms!)
alloc_counter = 490,001 (unchanged - we used this many total)
```

**Next MCTS search:**
```
Need a new room for a child?
1. Check: free_top > 0? YES (489,000)
2. Pop from free_list: room_number = free_list[489,000-1]
3. free_top = 488,999
4. Use that room! (recycle old room instead of allocating new)
```

## The Problem You're Hitting

**Free list overflow:**
```
free_top = 3,311,271  ← BIGGER than max_nodes (2,000,000)!
```

**Why it happens:**
- Pruning shader runs 2,048 threads in parallel
- All trying to do `free_top++` at the same time
- Race condition: multiple threads increment past the limit

**Why it freezes:**
- The allocation shader tries to pop from `free_list[free_top-1]`
- But `free_list` only has 2,000,000 slots!
- Accessing `free_list[3,311,270]` → **out of bounds** → crash/freeze

**The fix (clamping):**
```rust
if free_top > max_nodes {
    free_top = max_nodes;  // Cap it at 2M
    // We "lose" some freed nodes (they're cleared but not reusable)
    // But we don't crash!
}
```

## Summary

**The hotel analogy:**
- **Rooms** = nodes in the tree
- **Front desk** = alloc_counter (tracks total rooms rented)
- **Recycling bin** = free_list (cleaned rooms ready to re-rent)
- **Bin counter** = free_top (how many rooms in recycling bin)
- **Pruning** = evicting guests from old wings, cleaning rooms
- **Freeze bug** = recycling bin overflowed, front desk reading garbage

The GPU does everything **massively parallel** (2048 cleaners working at once), which is why we have race conditions!
