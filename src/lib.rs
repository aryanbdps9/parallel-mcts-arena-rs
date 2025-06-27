use rand::Rng;
use std::collections::HashMap;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use parking_lot::{RwLock, Mutex};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

// Thread-local storage for move generation to avoid allocations
thread_local! {
    static MOVE_BUFFER: std::cell::RefCell<Vec<(usize, usize)>> = std::cell::RefCell::new(Vec::new());
}

/// A pool for recycling nodes to reduce memory allocations
struct NodePool<M: Clone + Eq + std::hash::Hash> {
    /// Pool of available nodes that can be reused
    available_nodes: Arc<Mutex<Vec<Arc<Node<M>>>>>,
}

impl<M: Clone + Eq + std::hash::Hash> NodePool<M> {
    fn new() -> Self {
        Self {
            available_nodes: Arc::new(Mutex::new(Vec::with_capacity(1000000))),
        }
    }

    /// Return multiple nodes to the pool in batch
    fn return_nodes(&self, nodes: Vec<Arc<Node<M>>>) {
        let mut pool = self.available_nodes.lock();
        for node in nodes {
            // Only recycle nodes that have unique ownership
            if let Ok(mut node) = Arc::try_unwrap(node) {
                node.reset();
                pool.push(Arc::new(node));
            }
        }
        // Limit pool size to prevent unbounded growth
        if pool.len() > 4000000 {
            pool.truncate(1000000);
        }
    }
}

/// The state of the game. Must be cloneable to be used in the MCTS.
/// `Send` and `Sync` are required for parallel processing.
pub trait GameState: Clone + Send + Sync {
    /// The type of a move in the game.
    type Move: Clone + Eq + std::hash::Hash + std::fmt::Debug + Send + Sync;

    /// Returns a vector of all possible moves from the current state.
    fn get_possible_moves(&self) -> Vec<Self::Move>;
    /// Applies a move to the state, modifying it.
    fn make_move(&mut self, mv: &Self::Move);
    /// Returns true if the game is over.
    fn is_terminal(&self) -> bool;
    /// Returns the winner of the game, if any.
    /// Should return `Some(player_id)` if a player has won, `None` for a draw or if the game is not over.
    fn get_winner(&self) -> Option<i32>;
    /// Returns the player whose turn it is to move.
    fn get_current_player(&self) -> i32;
}

/// A node in the Monte Carlo Search Tree.
/// It is wrapped in an `Arc` to allow for shared ownership across threads.
struct Node<M: Clone + Eq + std::hash::Hash> {
    /// A map from a move to the child node.
    children: RwLock<HashMap<M, Arc<Node<M>>>>,
    /// The number of times this node has been visited. Atomic for lock-free updates.
    visits: AtomicI32,
    /// Sum of rewards from the parent's perspective, stored as an integer.
    /// 2 for a win, 1 for a draw, 0 for a loss. Uses atomic operations for lock-free updates.
    wins: AtomicI32,
    /// Virtual losses applied to this node. Used to reduce thread contention in parallel search.
    virtual_losses: AtomicI32,
    /// Depth of this node in the tree (0 for root)
    depth: u32,
}

impl<M: Clone + Eq + std::hash::Hash> Node<M> {
    /// Creates a new, empty node.
    fn new() -> Self {
        Node {
            children: RwLock::new(HashMap::new()),
            visits: AtomicI32::new(0),
            wins: AtomicI32::new(0),
            virtual_losses: AtomicI32::new(0),
            depth: 0,
        }
    }

    /// Resets the node to its initial state for reuse
    fn reset(&mut self) {
        *self.children.write() = HashMap::new();
        self.visits.store(0, Ordering::Relaxed);
        self.wins.store(0, Ordering::Relaxed);
        self.virtual_losses.store(0, Ordering::Relaxed);
        self.depth = 0;
    }

    /// Collects all descendant nodes for batch recycling
    fn collect_subtree_nodes(&self) -> Vec<Arc<Node<M>>> {
        let mut nodes = Vec::new();
        let mut stack = Vec::new();
        
        // Start with immediate children
        {
            let children = self.children.read();
            stack.extend(children.values().cloned());
        }
        
        while let Some(current) = stack.pop() {
            // Add the current node to the result
            nodes.push(current.clone());
            
            // Add its children to the stack
            let children = current.children.read();
            stack.extend(children.values().cloned());
        }
        
        nodes
    }

    /// Prunes weak children to save memory
    /// Keeps only children with visit count >= threshold
    fn prune_weak_children(&self, min_visits: i32) -> Vec<Arc<Node<M>>> {
        let mut children = self.children.write();
        let mut pruned_nodes = Vec::new();
        
        children.retain(|_, node| {
            let visits = node.visits.load(Ordering::Relaxed);
            if visits < min_visits {
                // Collect the pruned subtree for recycling
                pruned_nodes.extend(node.collect_subtree_nodes());
                pruned_nodes.push(node.clone());
                false
            } else {
                true
            }
        });
        
        pruned_nodes
    }

    /// Applies virtual loss to this node.
    fn apply_virtual_loss(&self) {
        self.virtual_losses.fetch_add(1, Ordering::Relaxed);
    }

    /// Removes virtual loss from this node.
    fn remove_virtual_loss(&self) {
        self.virtual_losses.fetch_sub(1, Ordering::Relaxed);
    }

    /// Calculates the PUCT (Predictor + Upper Confidence bounds applied to Trees) score for this node.
    /// This is a more sophisticated version of UCB1 that includes a prior probability term.
    /// Now includes virtual losses to discourage other threads from selecting the same path.
    ///
    /// # Arguments
    /// * `parent_visits` - The no. of visits to the parent node.
    /// * `exploration_parameter` - A constant to tune the level of exploration (C_puct).
    /// * `prior_probability` - The prior probability of selecting this move (usually from a neural network).
    fn puct(&self, parent_visits: i32, exploration_parameter: f64, prior_probability: f64) -> f64 {
        let visits = self.visits.load(Ordering::Relaxed);
        let virtual_losses = self.virtual_losses.load(Ordering::Relaxed);
        let effective_visits = visits + virtual_losses;
        
        if effective_visits == 0 {
            // For unvisited nodes, return only the exploration term
            exploration_parameter * prior_probability * (parent_visits as f64).sqrt()
        } else {
            let wins = self.wins.load(Ordering::Relaxed) as f64;
            let effective_visits_f = effective_visits as f64;
            // PUCT formula with virtual losses: Q(s,a) + C_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a) + VL(s,a))
            // Virtual losses effectively reduce the Q value, making the node less attractive
            let q_value = if visits > 0 {
                (wins / visits as f64) / 2.0
            } else {
                0.0 // If only virtual losses, assume worst case
            };
            let exploration_term = exploration_parameter * prior_probability * (parent_visits as f64).sqrt() / (1.0 + effective_visits_f);
            q_value + exploration_term
        }
    }
}

/// The main MCTS engine.
pub struct MCTS<S: GameState> {
    /// The root of the search tree.
    root: Arc<Node<S::Move>>,
    /// The exploration parameter for the UCB1 formula.
    exploration_parameter: f64,
    /// The rayon thread pool for parallel search.
    pool: ThreadPool,
    /// Node pool for recycling nodes
    node_pool: NodePool<S::Move>,
    /// Maximum number of nodes allowed in the tree
    max_nodes: usize,
    /// Current number of nodes in the tree (approximate)
    node_count: Arc<AtomicI32>,
}

impl<S: GameState> MCTS<S> {
    /// Creates a new MCTS engine.
    ///
    /// # Arguments
    /// * `exploration_parameter` - A constant to tune the level of exploration.
    /// * `num_threads` - The number of threads to use for the search. If 0, rayon will use the default.
    /// * `max_nodes` - Maximum number of nodes allowed in the tree.
    pub fn new(exploration_parameter: f64, num_threads: usize, max_nodes: usize) -> Self {
        let pool_builder = ThreadPoolBuilder::new();
        let pool = if num_threads > 0 {
            pool_builder.num_threads(num_threads).build().unwrap()
        } else {
            pool_builder.build().unwrap()
        };
        MCTS {
            root: Arc::new(Node::new()),
            exploration_parameter,
            pool,
            node_pool: NodePool::new(),
            max_nodes,
            node_count: Arc::new(AtomicI32::new(1)), // Start with 1 for root node
        }
    }

    /// Advances the root of the tree to the node corresponding to the given move.
    /// This version recycles unused subtrees to reduce memory allocation/deallocation.
    pub fn advance_root(&mut self, mv: &S::Move) {
        let (new_root, nodes_to_recycle, new_tree_size) = {
            let children = self.root.children.read();
            let new_root = children.get(mv).map(Arc::clone).unwrap_or_else(|| Arc::new(Node::new()));
            
            // Calculate the size of the new subtree
            let new_tree_size = if children.contains_key(mv) {
                1 + self.count_subtree_nodes(&new_root)
            } else {
                1 // Just the new root node
            };
            
            // Collect all nodes from non-selected subtrees for recycling
            let mut nodes_to_recycle = Vec::new();
            for (other_move, other_node) in children.iter() {
                if other_move != mv {
                    // Collect the entire subtree for recycling
                    nodes_to_recycle.extend(other_node.collect_subtree_nodes());
                    nodes_to_recycle.push(other_node.clone());
                }
            }
            
            (new_root, nodes_to_recycle, new_tree_size)
        };
        
        // Batch recycle all collected nodes
        if !nodes_to_recycle.is_empty() {
            self.node_pool.return_nodes(nodes_to_recycle);
        }
        
        // Update the node count to reflect the new tree size
        self.node_count.store(new_tree_size as i32, Ordering::Relaxed);
        
        self.root = new_root;
    }

    /// Counts the total number of nodes in a subtree (including the root of the subtree)
    fn count_subtree_nodes(&self, root: &Arc<Node<S::Move>>) -> usize {
        let mut count = 0;
        let mut stack = vec![root.clone()];
        
        while let Some(node) = stack.pop() {
            count += 1;
            let children = node.children.read();
            stack.extend(children.values().cloned());
        }
        
        count
    }

    /// Prunes weak children from the tree to save memory and computation.
    /// Call this periodically during search to maintain tree efficiency.
    pub fn prune_tree(&mut self, min_visits_threshold: i32) {
        let pruned_nodes = self.root.prune_weak_children(min_visits_threshold);
        if !pruned_nodes.is_empty() {
            self.node_pool.return_nodes(pruned_nodes);
        }
        
        // Recursively prune children that survived the initial pruning
        let children = self.root.children.read();
        for child in children.values() {
            let child_pruned = child.prune_weak_children(min_visits_threshold);
            if !child_pruned.is_empty() {
                self.node_pool.return_nodes(child_pruned);
            }
        }
    }

    /// Automatically prunes the tree based on visit statistics
    /// Removes children with less than 1% of the root's visits
    pub fn auto_prune(&mut self) {
        let root_visits = self.root.visits.load(Ordering::Relaxed);
        let min_visits = std::cmp::max(1, root_visits / 100); // At least 1% of root visits
        self.prune_tree(min_visits);
    }

    /// Returns statistics for the children of the root node.
    /// The stats are a map from a move to a tuple of (wins, visits).
    pub fn get_root_children_stats(&self) -> std::collections::HashMap<S::Move, (f64, i32)> {
        let children = self.root.children.read();
        children
            .iter()
            .map(|(mv, node)| {
                let wins = node.wins.load(Ordering::Relaxed) as f64;
                let visits = node.visits.load(Ordering::Relaxed);
                (mv.clone(), (wins, visits))
            })
            .collect()
    }

    /// Returns the stats for the root node.
    /// The stats are a tuple of (wins, visits).
    pub fn get_root_stats(&self) -> (f64, i32) {
        let wins = self.root.wins.load(Ordering::Relaxed) as f64;
        let visits = self.root.visits.load(Ordering::Relaxed);
        (wins, visits)
    }

    /// Returns debug information about the current MCTS state
    pub fn get_debug_info(&self) -> String {
        let root_visits = self.root.visits.load(Ordering::Relaxed);
        let root_wins = self.root.wins.load(Ordering::Relaxed);
        let node_count = self.node_count.load(Ordering::Relaxed);
        let children_count = self.root.children.read().len();
        
        format!(
            "MCTS Debug - Root: {} visits, {} wins, {} children, {} total nodes in tree",
            root_visits, root_wins, children_count, node_count
        )
    }

    /// Ensures the root node is fully expanded with all possible moves.
    /// This prevents the issue where only one move gets explored due to early exploitation.
    fn ensure_root_expanded(&mut self, state: &S) {
        let mut children_guard = self.root.children.write();
        if children_guard.is_empty() && !state.is_terminal() {
            let possible_moves = state.get_possible_moves();
            let mut new_nodes_count = 0;

            for mv in possible_moves.iter() {
                let new_node = Arc::new(Node {
                    children: RwLock::new(HashMap::new()),
                    visits: AtomicI32::new(0),
                    wins: AtomicI32::new(0),
                    virtual_losses: AtomicI32::new(0),
                    depth: 1, // Children of root are at depth 1
                });
                children_guard.insert(mv.clone(), new_node);
                new_nodes_count += 1;
            }
            
            // Update node count
            self.node_count.fetch_add(new_nodes_count, Ordering::Relaxed);
        }
    }

    /// Performs a parallel MCTS search with optional pruning.
    /// This method launches multiple simulations in parallel using `rayon`.
    ///
    /// # Arguments
    /// * `state` - The current state of the game.
    /// * `iterations` - The total number of simulations to run.
    pub fn search(&mut self, state: &S, iterations: i32) -> S::Move {
        // Ensure root node is fully expanded before starting parallel search
        self.ensure_root_expanded(state);
        
        // Run simulations in parallel with periodic pruning
        let prune_interval = std::cmp::max(1000, iterations / 10); // Prune every 10% of iterations or at least every 1000
        
        if iterations > prune_interval {
            // Split search into chunks with periodic pruning
            let chunks = iterations / prune_interval;
            let remainder = iterations % prune_interval;
            
            for _ in 0..chunks {
                self.pool.install(|| {
                    (0..prune_interval).into_par_iter().for_each(|_| {
                        self.run_simulation(state);
                    });
                });
                // Prune after each chunk - remove children with < 3% of max visits
                self.prune_children_by_percentage(0.03);
            }
            
            // Run remaining iterations
            if remainder > 0 {
                self.pool.install(|| {
                    (0..remainder).into_par_iter().for_each(|_| {
                        self.run_simulation(state);
                    });
                });
            }
        } else {
            // Run all iterations at once if too few iterations
            self.pool.install(|| {
                (0..iterations).into_par_iter().for_each(|_| {
                    self.run_simulation(state);
                });
            });
        }

        // Don't auto-prune immediately after search to preserve statistics for display
        // The pruning will be done later via explicit call or on next search

        // After all simulations, the best move is the one most visited.
        let children = self.root.children.read();
        if children.is_empty() {
            // Fallback: if no children exist, return a random valid move
            // This should rarely happen with the improved expansion logic
            let possible_moves = state.get_possible_moves();
            if possible_moves.is_empty() {
                panic!("No possible moves available - game should be terminal");
            }
            possible_moves[rand::thread_rng().gen_range(0..possible_moves.len())].clone()
        } else {
            children
                .iter()
                .max_by_key(|(_, node)| node.visits.load(Ordering::Relaxed))
                .map(|(mv, _)| mv.clone())
                .expect("Root node has children but max_by_key failed")
        }
    }

    /// Performs a parallel MCTS search with custom pruning interval.
    /// Prunes the tree every `prune_interval` iterations to maintain memory efficiency.
    ///
    /// # Arguments
    /// * `state` - The current state of the game.
    /// * `iterations` - The total number of simulations to run.
    /// * `prune_interval` - How often to prune the tree (0 = no pruning during search).
    pub fn search_with_pruning(&mut self, state: &S, iterations: i32, prune_interval: i32) -> S::Move {
        // Ensure root node is fully expanded before starting parallel search
        self.ensure_root_expanded(state);
        
        if prune_interval > 0 && iterations > prune_interval {
            // Split search into chunks with pruning
            let chunks = iterations / prune_interval;
            let remainder = iterations % prune_interval;
            
            for _ in 0..chunks {
                self.pool.install(|| {
                    (0..prune_interval).into_par_iter().for_each(|_| {
                        self.run_simulation(state);
                    });
                });
                // Prune after each chunk - remove children with < 3% of max visits
                self.prune_children_by_percentage(0.03);
            }
            
            // Run remaining iterations
            if remainder > 0 {
                self.pool.install(|| {
                    (0..remainder).into_par_iter().for_each(|_| {
                        self.run_simulation(state);
                    });
                });
            }
        } else {
            // Run all iterations at once
            self.pool.install(|| {
                (0..iterations).into_par_iter().for_each(|_| {
                    self.run_simulation(state);
                });
            });
        }

        // Don't do final pruning here - let it be done explicitly after statistics are displayed

        // Return the best move
        let children = self.root.children.read();
        if children.is_empty() {
            // Fallback: if no children exist, return a random valid move
            // This should rarely happen with the improved expansion logic
            let possible_moves = state.get_possible_moves();
            if possible_moves.is_empty() {
                panic!("No possible moves available - game should be terminal");
            }
            possible_moves[rand::thread_rng().gen_range(0..possible_moves.len())].clone()
        } else {
            children
                .iter()
                .max_by_key(|(_, node)| node.visits.load(Ordering::Relaxed))
                .map(|(mv, _)| mv.clone())
                .expect("Root node has children but max_by_key failed")
        }
    }

    /// Runs a single MCTS simulation with virtual loss support.
    fn run_simulation(&self, state: &S) {
        let mut current_state = state.clone();
        let mut path: Vec<Arc<Node<S::Move>>> = Vec::with_capacity(64); // Pre-allocate reasonable capacity
        path.push(self.root.clone());
        let mut current_node = self.root.clone();
        
        // Calculate board capacity based on initial move count for better memory allocation
        let board_capacity = current_state.get_possible_moves().len();
        let mut moves_cache = Vec::with_capacity(board_capacity);
        let mut candidates = Vec::with_capacity(board_capacity);

        // --- Selection Phase with Virtual Loss ---
        // Traverse the tree until a leaf node is reached.
        loop {
            let children_guard = current_node.children.read();
            if children_guard.is_empty() || current_state.is_terminal() {
                drop(children_guard);
                break;
            }

            moves_cache.clear();
            moves_cache.extend(current_state.get_possible_moves());
            
            let parent_visits = current_node.visits.load(Ordering::Relaxed);
            // Use uniform prior probability for all moves since we don't have a neural network
            let prior_probability = 1.0 / moves_cache.len() as f64;
            let (best_move, next_node) = {
                candidates.clear();
                candidates.extend(
                    moves_cache
                        .iter()
                        .filter_map(|m| children_guard.get(m).map(|n| (m, n)))
                        .map(|(m, n)| {
                            let puct = n.puct(parent_visits, self.exploration_parameter, prior_probability);
                            (m.clone(), n.clone(), puct)
                        })
                );
                
                // Find the maximum PUCT score and collect best indices in one pass
                let mut max_puct = f64::NEG_INFINITY;
                let mut best_indices = Vec::with_capacity(4); // Most common case is 1-4 best moves
                
                for (i, (_, _, puct)) in candidates.iter().enumerate() {
                    if *puct > max_puct {
                        max_puct = *puct;
                        best_indices.clear();
                        best_indices.push(i);
                    } else if (*puct - max_puct).abs() < 1e-10 {
                        best_indices.push(i);
                    }
                }
                
                let selected_idx = best_indices[rand::thread_rng().gen_range(0..best_indices.len())];
                let selected = &candidates[selected_idx];
                (selected.0.clone(), selected.1.clone())
            };

            drop(children_guard); // Release read lock

            // Apply virtual loss to the selected node
            next_node.apply_virtual_loss();
            
            current_state.make_move(&best_move);
            current_node = next_node;
            path.push(current_node.clone());
        }

        // --- Expansion Phase ---
        // If the node is a leaf and the game is not over, decide whether to expand based on:
        // 1. Current tree size vs max_nodes limit
        // 2. Depth-based probability (deeper nodes are less likely to expand)
        // 3. Visit count (more visited nodes are more likely to expand)
        // Special case: Always expand the root node to ensure the search can find moves
        if !current_state.is_terminal() {
            let should_expand = {
                let current_nodes = self.node_count.load(Ordering::Relaxed) as usize;
                let tree_capacity_available = current_nodes < self.max_nodes;
                
                if !tree_capacity_available {
                    false // Hard limit: no expansion if tree is full
                } else {
                    let children_guard = current_node.children.read();
                    let is_leaf = children_guard.is_empty();
                    drop(children_guard);
                    
                    if !is_leaf {
                        false // Already expanded by another thread
                    } else {
                        // Always expand the root node (depth 0) to ensure we have moves to choose from
                        let depth = current_node.depth;
                        if depth == 0 {
                            true
                        } else {
                            // Probabilistic expansion based on depth and visits for non-root nodes
                            let visits = current_node.visits.load(Ordering::Relaxed);
                            
                            // Base expansion probability decreases with depth
                            // More visits increase the likelihood of expansion
                            let depth_factor = 1.0 / (1.0 + (depth as f64) * 0.5);
                            let visit_factor = (visits as f64).sqrt() / 10.0; // Encourage expansion for well-visited nodes
                            let expansion_probability = (depth_factor + visit_factor).min(1.0);
                            
                            rand::thread_rng().gen::<f64>() < expansion_probability
                        }
                    }
                }
            };
            
            if should_expand {
                let mut children_guard = current_node.children.write();
                // Double-check it's still empty after acquiring write lock
                if children_guard.is_empty() {
                    moves_cache.clear();
                    moves_cache.extend(current_state.get_possible_moves());
                    let new_depth = current_node.depth + 1;
                    let mut new_nodes_count = 0;
                    
                    for mv in moves_cache.iter() {
                        // Create a new node with the correct depth
                        let new_node = Arc::new(Node {
                            children: RwLock::new(HashMap::new()),
                            visits: AtomicI32::new(0),
                            wins: AtomicI32::new(0),
                            virtual_losses: AtomicI32::new(0),
                            depth: new_depth,
                        });
                        children_guard.insert(mv.clone(), new_node);
                        new_nodes_count += 1;
                    }
                    
                    // Update node count
                    self.node_count.fetch_add(new_nodes_count, Ordering::Relaxed);
                }
            }
        }

        // --- Simulation Phase ---
        // Run a random playout from the new node to the end of the game.
        let mut sim_state = current_state.clone();
        while !sim_state.is_terminal() {
            moves_cache.clear();
            moves_cache.extend(sim_state.get_possible_moves());
            if moves_cache.is_empty() {
                break;
            }
            let mv = &moves_cache[rand::thread_rng().gen_range(0..moves_cache.len())];
            sim_state.make_move(mv);
        }
        let winner = sim_state.get_winner();

        // --- Backpropagation Phase with Virtual Loss Removal ---
        // Update the visit counts and win statistics for all nodes in the path.
        // Also remove virtual losses that were applied during selection.
        let mut player_for_reward = current_state.get_current_player();
        for (i, node) in path.iter().rev().enumerate() {
            // Remove virtual loss from all nodes except the last one (the leaf/terminal node)
            // which didn't have virtual loss applied during selection
            if i < path.len() - 1 {
                node.remove_virtual_loss();
            }
            
            node.visits.fetch_add(1, Ordering::Relaxed);
            let reward = match winner {
                Some(w) if w == player_for_reward => 0, // Loss for parent
                Some(w) if w == -player_for_reward => 2, // Win for parent
                _ => 1,                                 // Draw
            };
            node.wins.fetch_add(reward, Ordering::Relaxed);
            player_for_reward = -player_for_reward;
        }
    }

    /// Prunes children based on visit percentage relative to the best child
    /// Removes children with less than the specified percentage of the most visited child's visits
    pub fn prune_children_by_percentage(&mut self, min_percentage: f64) {
        let mut children = self.root.children.write();
        
        if children.len() <= 1 {
            return; // No need to prune if we have 1 or fewer children
        }
        
        // Find the maximum visit count among children
        let max_visits = children
            .values()
            .map(|node| node.visits.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0);
        
        if max_visits == 0 {
            return; // No visits yet, nothing to prune
        }
        
        let min_visits_threshold = ((max_visits as f64) * min_percentage).ceil() as i32;
        let mut pruned_nodes = Vec::new();
        
        children.retain(|_, node| {
            let visits = node.visits.load(Ordering::Relaxed);
            if visits < min_visits_threshold {
                // Collect the pruned subtree for recycling
                pruned_nodes.extend(node.collect_subtree_nodes());
                pruned_nodes.push(node.clone());
                false
            } else {
                true
            }
        });
        
        // Batch recycle all pruned nodes
        if !pruned_nodes.is_empty() {
            self.node_pool.return_nodes(pruned_nodes);
        }
    }
}
