use rand::Rng;
use std::collections::HashMap;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

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
struct Node<M: Clone + Eq + std::hash::Hash + Send + Sync> {
    /// Sum of rewards from this node's perspective. Protected by a Mutex for concurrent access.
    wins: Mutex<f64>,
    /// Number of times this node has been visited. Atomic for lock-free updates.
    visits: AtomicI32,
    /// Child nodes, representing possible next states. The HashMap is protected by a Mutex.
    children: Mutex<HashMap<M, Arc<Node<M>>>>,
}

impl<M: Clone + Eq + std::hash::Hash + Send + Sync> Node<M> {
    /// Creates a new, empty node.
    fn new() -> Self {
        Node {
            wins: Mutex::new(0.0),
            visits: AtomicI32::new(0),
            children: Mutex::new(HashMap::new()),
        }
    }

    /// Calculates the UCB1 (Upper Confidence Bound 1) score for this node.
    /// This score balances exploration and exploitation.
    ///
    /// # Arguments
    /// * `parent_visits` - The number of visits to the parent node.
    /// * `exploration_parameter` - A constant to tune the level of exploration.
    fn ucb1(&self, parent_visits: i32, exploration_parameter: f64) -> f64 {
        let visits = self.visits.load(Ordering::Relaxed);
        if visits == 0 {
            std::f64::INFINITY
        } else {
            let wins = *self.wins.lock();
            // UCB1 formula
            wins / visits as f64
                + exploration_parameter * ((parent_visits as f64).ln() / visits as f64).sqrt()
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
}

impl<S: GameState> MCTS<S> {
    /// Creates a new MCTS engine.
    ///
    /// # Arguments
    /// * `exploration_parameter` - A constant to tune the level of exploration.
    /// * `num_threads` - The number of threads to use for the search. If 0, rayon will use the default.
    pub fn new(exploration_parameter: f64, num_threads: usize) -> Self {
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
        }
    }

    /// Advances the root of the tree to the node corresponding to the given move.
    /// This is useful to preserve the search tree between moves.
    pub fn advance_root(&mut self, mv: &S::Move) {
        let new_root = {
            let children = self.root.children.lock();
            children.get(mv).cloned().unwrap_or_else(|| Arc::new(Node::new()))
        };
        self.root = new_root;
    }

    /// Returns statistics for the children of the root node.
    /// The stats are a map from a move to a tuple of (wins, visits).
    pub fn get_root_children_stats(&self) -> std::collections::HashMap<S::Move, (f64, i32)> {
        let children = self.root.children.lock();
        children
            .iter()
            .map(|(mv, node)| {
                let wins = *node.wins.lock();
                let visits = node.visits.load(Ordering::Relaxed);
                (mv.clone(), (wins, visits))
            })
            .collect()
    }

    /// Performs a parallel MCTS search.
    /// This method launches multiple simulations in parallel using `rayon`.
    ///
    /// # Arguments
    /// * `state` - The current state of the game.
    /// * `iterations` - The total number of simulations to run.
    pub fn search(&self, state: &S, iterations: i32) -> S::Move {
        // Run simulations in parallel within the custom thread pool.
        self.pool.install(|| {
            (0..iterations).into_par_iter().for_each(|_| {
                self.run_simulation(state);
            });
        });

        // After all simulations, the best move is the one most visited.
        let children = self.root.children.lock();
        children.iter()
            .max_by_key(|(_, node)| node.visits.load(Ordering::Relaxed))
            .map(|(mv, _)| mv.clone())
            .expect("Search returned no moves. The root node has no children.")
    }

    /// Runs a single MCTS simulation.
    fn run_simulation(&self, state: &S) {
        let mut current_state = state.clone();
        let mut path: Vec<Arc<Node<S::Move>>> = vec![self.root.clone()];
        let mut current_node = self.root.clone();

        // --- Selection Phase ---
        // Traverse the tree until a leaf node is reached.
        loop {
            let parent_visits = current_node.visits.load(Ordering::Relaxed);
            let mut children_guard = current_node.children.lock();

            if children_guard.is_empty() || current_state.is_terminal() {
                // --- Expansion Phase ---
                // If the node is a leaf and the game is not over, expand it.
                if !current_state.is_terminal() {
                    let moves = current_state.get_possible_moves();
                    for mv in moves {
                        children_guard.insert(mv, Arc::new(Node::new()));
                    }
                }
                break;
            }

            let moves = current_state.get_possible_moves();
            let (best_move, next_node) = moves
                .iter()
                .filter_map(|m| children_guard.get(m).map(|n| (m, n)))
                .max_by(|(_, a), (_, b)| {
                    let a_ucb = a.ucb1(parent_visits, self.exploration_parameter);
                    let b_ucb = b.ucb1(parent_visits, self.exploration_parameter);
                    a_ucb.partial_cmp(&b_ucb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(m, n)| (m.clone(), n.clone()))
                .unwrap_or_else(|| {
                    // If some moves don't have corresponding nodes, pick one to explore.
                    // This can happen with a partial expansion.
                    let mv = moves[0].clone();
                    let node = children_guard.entry(mv.clone()).or_insert_with(|| Arc::new(Node::new()));
                    (mv, node.clone())
                });

            drop(children_guard); // Release lock before state update

            current_state.make_move(&best_move);
            current_node = next_node;
            path.push(current_node.clone());
        }

        // --- Simulation Phase ---
        // Run a random playout from the new node to the end of the game.
        let mut sim_state = current_state.clone();
        while !sim_state.is_terminal() {
            let moves = sim_state.get_possible_moves();
            if moves.is_empty() { break; }
            let mv = &moves[rand::thread_rng().gen_range(0..moves.len())];
            sim_state.make_move(mv);
        }
        let winner = sim_state.get_winner();

        // --- Backpropagation Phase ---
        // Update the visit counts and win statistics for all nodes in the path.
        let mut player_for_reward = current_state.get_current_player();
        for node in path.iter().rev() {
            node.visits.fetch_add(1, Ordering::Relaxed);
            let reward = match winner {
                Some(w) if w == player_for_reward => 0.0, // Opponent won
                Some(w) if w == -player_for_reward => 1.0, // We won
                _ => 0.5, // Draw
            };
            *node.wins.lock() += reward;
            player_for_reward = -player_for_reward;
        }
    }
}
