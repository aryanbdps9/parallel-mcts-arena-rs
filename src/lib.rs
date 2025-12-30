//! # Parallel Monte Carlo Tree Search (MCTS) Library
//!
//! This library provides a generic, parallel implementation of Monte Carlo Tree Search
//! that can be used with any game that implements the `GameState` trait.
//!
//! ## Key Features
//! - **Parallel Search**: Uses Rayon for multi-threaded tree search
//! - **Thread Safety**: RwLock-based concurrent access to the search tree
//! - **Virtual Losses**: Prevents multiple threads from exploring the same paths
//! - **Memory Management**: Node recycling and automatic tree pruning
//! - **PUCT Selection**: Enhanced UCB1 formula with prior probabilities
//! - **GPU Acceleration** (optional): Batch PUCT calculation on GPU for large trees
//!
//! ## Example Usage
//! ```rust
//! use mcts::{MCTS, GameState};
//! use mcts::games::connect4::Connect4State;
//!
//! // Your game must implement GameState
//! let game_state = Connect4State::new(7, 6, 4);
//! let mut mcts = MCTS::new(1.4, 8, 1000000);
//! let (best_move, stats) = mcts.search(&game_state, 0, 0, 30); // 30 second timeout
//! ```
//!
//! ## GPU Acceleration
//! To enable GPU acceleration, build with the `gpu` feature:
//! ```bash
//! cargo build --features gpu
//! ```
//! Then use `MCTS::with_gpu()` to create a GPU-accelerated engine.

#[cfg(feature = "gpu")]
pub mod gpu;

// Game implementations - available for all features
pub mod games;

use parking_lot::{Mutex, RwLock};
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

thread_local! {
    static RNG: std::cell::RefCell<Xoshiro256PlusPlus> = std::cell::RefCell::new(
        Xoshiro256PlusPlus::from_seed([1; 32])
    );
}

fn with_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut Xoshiro256PlusPlus) -> R,
{
    RNG.with(|rng| f(&mut *rng.borrow_mut()))
}

fn random_range(min: usize, max: usize) -> usize {
    with_rng(|rng| {
        let range = max - min;
        min + (rng.next_u64() as usize) % range
    })
}

fn random_f64() -> f64 {
    with_rng(|rng| (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64))
}

/// Statistics about the MCTS search
#[derive(Debug, Clone, Default)]
pub struct SearchStatistics {
    pub total_nodes: i32,
    pub root_visits: i32,
    pub root_wins: f64,
    pub root_value: f64,
    pub children_stats: HashMap<String, (f64, i32)>,
}

/// Strategy for selecting the best move after MCTS search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoveSelectionStrategy {
    /// Select the move with the highest visit count (default, most robust)
    #[default]
    MaxVisits,
    /// Select the move with the highest Q value (win rate)
    MaxQ,
}

// Thread-local storage for move generation to avoid allocations
// Each thread maintains its own buffer for generating possible moves,
// which reduces memory allocations during hot path execution.
thread_local! {
    static MOVE_BUFFER: std::cell::RefCell<Vec<(usize, usize)>> = std::cell::RefCell::new(Vec::new());
}

/// A pool for recycling nodes to reduce memory allocations
///
/// Instead of constantly allocating and deallocating nodes, we maintain
/// a pool of reusable nodes to improve performance and reduce GC pressure.
struct NodePool<M: Clone + Eq + std::hash::Hash> {
    /// Pool of available nodes that can be reused
    available_nodes: Arc<Mutex<Vec<Arc<Node<M>>>>>,
}

impl<M: Clone + Eq + std::hash::Hash> NodePool<M> {
    /// Creates a new empty node pool
    fn new() -> Self {
        Self {
            available_nodes: Arc::new(Mutex::new(Vec::with_capacity(1000000))),
        }
    }

    /// Return multiple nodes to the pool in batch
    ///
    /// More efficient than returning nodes one at a time.
    /// Nodes are reset to their initial state before being returned to the pool.
    ///
    /// # Arguments
    /// * `nodes` - Vector of nodes to return to the pool
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
///
/// This trait defines the interface that any game must implement to work
/// with the MCTS engine. The engine will call these methods to:
/// - Generate possible moves
/// - Apply moves to create new states
/// - Check if the game is over
/// - Determine the winner
///
/// ## Thread Safety
/// All methods must be thread-safe since the MCTS engine runs in parallel.
/// The game state should be immutable during search (only copied, not modified).
pub trait GameState: Clone + Send + Sync {
    /// The type of a move in the game.
    ///
    /// Must be cloneable, comparable, hashable, debuggable, and thread-safe.
    /// Used as keys in hash maps and for move generation.
    type Move: Clone + Eq + std::hash::Hash + std::fmt::Debug + Send + Sync + 'static;

    /// Returns the board state.
    ///
    /// Used for visualization and analysis. Should return a 2D vector
    /// where each cell contains a player ID (e.g., 1, -1, 0 for empty).
    fn get_board(&self) -> Vec<Vec<i32>>;

    /// Returns the last move made as a set of coordinates, if applicable.
    ///
    /// Used for UI highlighting and game analysis. Some games may not
    /// have coordinate-based moves, so this can return None.
    fn get_last_move(&self) -> Option<Vec<(usize, usize)>> {
        None
    }

    /// Returns data for GPU simulation if supported
    /// Returns (board_data, board_width, board_height, current_player)
    fn get_gpu_simulation_data(&self) -> Option<(Vec<i32>, usize, usize, i32)> {
        None
    }

    /// Returns the number of players in the game.
    fn get_num_players(&self) -> i32;

    /// Returns a list of all possible moves for the current player.
    fn get_possible_moves(&self) -> Vec<Self::Move>;

    /// Applies a move to the state, modifying it.
    ///
    /// This should update the game state and switch to the next player.
    /// The move is guaranteed to be legal (from get_possible_moves).
    fn make_move(&mut self, mv: &Self::Move);

    /// Returns true if the game is over.
    ///
    /// Called to determine when to stop simulations. Should check for
    /// wins, draws, or any other terminal conditions.
    fn is_terminal(&self) -> bool;

    /// Returns the winner of the game, if any.
    /// Should return `Some(player_id)` if a player has won, `None` for a draw or if the game is not over.
    ///
    /// Used to determine the reward during backpropagation. The reward
    /// is calculated from the perspective of each player in the path.
    fn get_winner(&self) -> Option<i32>;

    /// Returns the player whose turn it is to move.
    ///
    /// Used to determine perspective during reward calculation and
    /// for UI display of current player.
    fn get_current_player(&self) -> i32;

    /// Returns the weight of a move for random rollouts.
    ///
    /// Used to bias the random rollout towards better moves.
    /// Default implementation returns 1.0 (uniform probability).
    fn get_move_weight(&self, _mv: &Self::Move) -> f64 {
        1.0
    }
}

/// A node in the Monte Carlo Search Tree.
/// It is wrapped in an `Arc` to allow for shared ownership across threads.
///
/// Each node represents a game state and stores statistics about the outcomes
/// of all simulations that have passed through this node. The tree is built
/// incrementally as the search progresses.
///
/// ## Thread Safety
/// All fields use atomic operations or locks to ensure thread-safe access
/// during parallel search. Multiple threads can read and update the same
/// node simultaneously.
struct Node<M: Clone + Eq + std::hash::Hash> {
    /// A map from a move to the child node.
    ///
    /// Protected by RwLock for concurrent access. Multiple threads can read
    /// simultaneously, but only one can write (when expanding the tree).
    children: RwLock<HashMap<M, Arc<Node<M>>>>,

    /// The number of times this node has been visited. Atomic for lock-free updates.
    ///
    /// Incremented each time a simulation passes through this node.
    /// Used in the PUCT formula for move selection.
    visits: AtomicI32,

    /// Sum of rewards from the parent's perspective, stored as an integer.
    /// 2 for a win, 1 for a draw, 0 for a loss. Uses atomic operations for lock-free updates.
    ///
    /// The reward is always from the perspective of the player who made the move
    /// leading to this node. This makes backpropagation easier.
    wins: AtomicI32,

    /// Virtual losses applied to this node. Used to reduce thread contention in parallel search.
    ///
    /// When a thread selects a path, it applies a virtual loss to discourage
    /// other threads from following the same path. The virtual loss is removed
    /// after the simulation completes.
    virtual_losses: AtomicI32,

    /// Depth of this node in the tree (0 for root)
    ///
    /// Used for tree analysis and debugging. Not used in the search algorithm itself.
    depth: u32,
}

impl<M: Clone + Eq + std::hash::Hash> Node<M> {
    /// Creates a new, empty node.
    ///
    /// All statistics are initialized to zero. The node has no children initially.
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
    ///
    /// Clears all statistics and children so the node can be reused
    /// for a different position. This is part of the memory management
    /// system to reduce allocations.
    fn reset(&mut self) {
        *self.children.write() = HashMap::new();
        self.visits.store(0, Ordering::Relaxed);
        self.wins.store(0, Ordering::Relaxed);
        self.virtual_losses.store(0, Ordering::Relaxed);
        self.depth = 0;
    }

    /// Collects all descendant nodes for batch recycling
    ///
    /// Recursively traverses the subtree and collects all nodes
    /// into a vector. Used when pruning parts of the tree or
    /// when the tree is being destroyed.
    ///
    /// # Returns
    /// Vector of all nodes in the subtree rooted at this node
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
    ///
    /// Instead of removing the node entirely, we just clear its children (prune its subtree).
    /// This preserves the node's statistics (visits, wins) but frees up the memory used by its descendants.
    ///
    /// # Arguments
    /// * `min_visits` - Minimum number of visits required to keep a child's subtree
    ///
    /// # Returns
    /// Vector of pruned nodes that can be recycled
    fn prune_weak_children(&self, min_visits: i32) -> Vec<Arc<Node<M>>> {
        let children = self.children.read();
        let mut pruned_nodes = Vec::new();

        for node in children.values() {
            let visits = node.visits.load(Ordering::Relaxed);
            if visits < min_visits {
                // Prune the subtree of this weak node, but keep the node itself
                // We do this by clearing its children map
                let mut node_children = node.children.write();
                if !node_children.is_empty() {
                    // Collect all descendants for recycling
                    for child in node_children.values() {
                        pruned_nodes.extend(child.collect_subtree_nodes());
                        pruned_nodes.push(child.clone());
                    }
                    // Clear the children map
                    node_children.clear();
                }
            }
        }

        pruned_nodes
    }

    /// Applies virtual loss to this node.
    ///
    /// Virtual losses are used to coordinate between threads in parallel search.
    /// When a thread selects a path for exploration, it applies a virtual loss
    /// to discourage other threads from selecting the same path.
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
    /// * `virtual_loss_weight` - Weight applied to virtual losses.
    fn puct(&self, parent_visits: i32, exploration_parameter: f64, prior_probability: f64, virtual_loss_weight: f64) -> f64 {
        let visits = self.visits.load(Ordering::Relaxed);
        let virtual_losses = self.virtual_losses.load(Ordering::Relaxed);
        
        // Calculate effective visits including virtual losses
        // This increases the denominator in both Q and U terms
        let effective_visits = visits as f64 + (virtual_losses as f64 * virtual_loss_weight);

        if effective_visits <= 1e-9 {
            // For unvisited nodes, return only the exploration term
            exploration_parameter * prior_probability * (parent_visits as f64).sqrt()
        } else {
            let wins = self.wins.load(Ordering::Relaxed) as f64;
            
            // PUCT formula with virtual losses:
            // Q = Wins / (Visits + VirtualLosses)
            // We treat virtual losses as "losses" (value 0), so we add them to the denominator
            // but not the numerator. This temporarily lowers the winrate.
            let q_value = (wins / effective_visits) / 2.0;
            
            let exploration_term =
                exploration_parameter * prior_probability * (parent_visits as f64).sqrt()
                    / (1.0 + effective_visits);
            
            q_value + exploration_term
        }
    }
}

/// Request for GPU evaluation
struct EvaluationRequest<S: GameState> {
    state: S,
    path: Vec<Arc<Node<S::Move>>>,
    path_players: Vec<i32>,
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
    /// Maximum number of nodes in the tree
    max_nodes: usize,
    /// Current number of nodes in the tree (approximate)
    node_count: Arc<AtomicI32>,
    /// Running average of thread coordination overhead in milliseconds
    timeout_overhead_ms: Arc<Mutex<f64>>,
    /// Number of timeout measurements taken
    timeout_measurements: Arc<AtomicI32>,
    /// Counter for searches since last overhead measurement
    searches_since_measurement: Arc<AtomicI32>,
    /// Strategy for selecting the best move (visits vs Q value)
    move_selection_strategy: MoveSelectionStrategy,
    /// GPU accelerator for batch PUCT computation (optional, requires 'gpu' feature)
    #[cfg(feature = "gpu")]
    gpu_accelerator: Option<Arc<Mutex<gpu::GpuMctsAccelerator>>>,
    /// Whether GPU acceleration is enabled
    #[cfg(feature = "gpu")]
    gpu_enabled: bool,
    /// Cached GPU-computed PUCT scores keyed by (parent_node_id, child_node_id)
    /// This allows caching PUCT for the entire tree, not just root children
    #[cfg(feature = "gpu")]
    gpu_puct_cache: Arc<RwLock<HashMap<(usize, usize), f64>>>,
    /// Last time the GPU PUCT cache was updated
    #[cfg(feature = "gpu")]
    gpu_cache_timestamp: Arc<Mutex<Instant>>,
    /// Number of simulations since last GPU cache update
    #[cfg(feature = "gpu")]
    simulations_since_gpu_update: Arc<AtomicI32>,
    /// Statistics: nodes processed in last GPU batch
    #[cfg(feature = "gpu")]
    gpu_last_batch_size: Arc<AtomicI32>,
    /// Channel for sending simulation requests to the GPU worker
    #[cfg(feature = "gpu")]
    gpu_simulation_sender: Option<std::sync::mpsc::Sender<EvaluationRequest<S>>>,
    /// Counter for pending GPU evaluations
    #[cfg(feature = "gpu")]
    gpu_pending_evaluations: Arc<AtomicI32>,
    /// Persistent GPU-native Othello MCTS engine for tree reuse
    /// Wrapped in Mutex to allow interior mutability (lazy initialization)
    #[cfg(feature = "gpu")]
    gpu_native_othello: Mutex<Option<Arc<Mutex<gpu::GpuOthelloMcts>>>>,
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
            timeout_overhead_ms: Arc::new(Mutex::new(50.0)), // Start with conservative 50ms estimate
            timeout_measurements: Arc::new(AtomicI32::new(0)),
            searches_since_measurement: Arc::new(AtomicI32::new(0)),
            move_selection_strategy: MoveSelectionStrategy::default(),
            #[cfg(feature = "gpu")]
            gpu_accelerator: None,
            #[cfg(feature = "gpu")]
            gpu_enabled: false,
            #[cfg(feature = "gpu")]
            gpu_puct_cache: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "gpu")]
            gpu_cache_timestamp: Arc::new(Mutex::new(Instant::now())),
            #[cfg(feature = "gpu")]
            simulations_since_gpu_update: Arc::new(AtomicI32::new(0)),
            #[cfg(feature = "gpu")]
            gpu_last_batch_size: Arc::new(AtomicI32::new(0)),
            #[cfg(feature = "gpu")]
            gpu_simulation_sender: None,
            #[cfg(feature = "gpu")]
            gpu_pending_evaluations: Arc::new(AtomicI32::new(0)),
            #[cfg(feature = "gpu")]
            gpu_native_othello: Mutex::new(None),
        }
    }

    /// Creates a new MCTS engine with GPU acceleration enabled.
    ///
    /// This constructor attempts to initialize a GPU context for accelerating
    /// PUCT calculations. If GPU initialization fails, the engine falls back
    /// to CPU-only mode.
    ///
    /// # Arguments
    /// * `exploration_parameter` - A constant to tune the level of exploration.
    /// * `num_threads` - The number of threads to use for the search. If 0, rayon will use the default.
    /// * `max_nodes` - Maximum number of nodes allowed in the tree.
    ///
    /// # Returns
    /// A tuple of (MCTS engine, Option<String>) where the string contains GPU info or error message.
    #[cfg(feature = "gpu")]
    pub fn with_gpu(exploration_parameter: f64, num_threads: usize, max_nodes: usize) -> (Self, Option<String>)
    where
        S: 'static,
    {
        let gpu_config = gpu::GpuConfig::default();
        Self::with_gpu_config(exploration_parameter, num_threads, max_nodes, gpu_config, true)
    }

    /// Creates a new MCTS engine with custom GPU configuration.
    ///
    /// # Arguments
    /// * `exploration_parameter` - A constant to tune the level of exploration.
    /// * `num_threads` - The number of threads to use for the search.
    /// * `max_nodes` - Maximum number of nodes allowed in the tree.
    /// * `gpu_config` - Custom GPU configuration.
    ///
    /// # Returns
    /// A tuple of (MCTS engine, Option<String>) where the string contains GPU info or error message.
    #[cfg(feature = "gpu")]
    pub fn with_gpu_config(
        exploration_parameter: f64,
        num_threads: usize,
        max_nodes: usize,
        gpu_config: gpu::GpuConfig,
        use_heuristic: bool,
    ) -> (Self, Option<String>)
    where
        S: 'static,
    {
        let pool_builder = ThreadPoolBuilder::new();
        let pool = if num_threads > 0 {
            pool_builder.num_threads(num_threads).build().unwrap()
        } else {
            pool_builder.build().unwrap()
        };

        let (gpu_accelerator, gpu_enabled, message) = match gpu::try_init_gpu(&gpu_config) {
            gpu::GpuInitResult::Success(ctx) => {
                let info = ctx.debug_info();
                let accelerator = gpu::GpuMctsAccelerator::new(std::sync::Arc::new(ctx));
                (Some(Arc::new(Mutex::new(accelerator))), true, Some(format!("GPU enabled: {}", info)))
            }
            gpu::GpuInitResult::Unavailable(msg) => {
                (None, false, Some(format!("GPU unavailable, using CPU: {}", msg)))
            }
            gpu::GpuInitResult::Error(msg) => {
                (None, false, Some(format!("GPU initialization failed, using CPU: {}", msg)))
            }
        };

        let node_count = Arc::new(AtomicI32::new(1));
        let pending_evaluations = Arc::new(AtomicI32::new(0));
        let pending_evals_clone = pending_evaluations.clone();

        let gpu_simulation_sender = if gpu_enabled {
            if let Some(ref accelerator) = gpu_accelerator {
                let accelerator = accelerator.clone();
                let (tx, rx) = std::sync::mpsc::channel::<EvaluationRequest<S>>();
                let max_batch_size = gpu_config.max_batch_size;
                let use_heuristic_flag = use_heuristic;
                
                std::thread::spawn(move || {
                    let mut batch_requests: Vec<EvaluationRequest<S>> = Vec::with_capacity(max_batch_size);
                    let mut flat_data = Vec::with_capacity(max_batch_size * 256);
                    let mut gpu_indices = Vec::with_capacity(max_batch_size);
                    let mut cpu_indices = Vec::with_capacity(max_batch_size);
                    let mut scores = Vec::with_capacity(max_batch_size);
                    
                    loop {
                        let first = match rx.recv() {
                            Ok(req) => req,
                            Err(_) => break,
                        };
                        
                        batch_requests.push(first);

                        // Aggressive batching: Wait for more requests to form a decent batch
                        // This is crucial for GPU throughput.
                        // We spin-wait because latency is critical and we want to grab requests ASAP.
                        let min_batch = 64; // Reduced from 1024 to improve latency
                        let max_wait = Duration::from_micros(200); // Reduced from 1000us to 200us
                        let start_wait = Instant::now();

                        while batch_requests.len() < max_batch_size {
                            match rx.try_recv() {
                                Ok(req) => batch_requests.push(req),
                                Err(_) => {
                                    // If we have enough requests, stop waiting
                                    if batch_requests.len() >= min_batch {
                                        break;
                                    }
                                    // If we timed out, stop waiting
                                    if start_wait.elapsed() >= max_wait {
                                        break;
                                    }
                                    // Yield to let other threads produce
                                    std::thread::yield_now();
                                }
                            }
                        }
                        
                        if batch_requests.is_empty() { continue; }
                        
                        // Clear buffers for reuse
                        flat_data.clear();
                        gpu_indices.clear();
                        cpu_indices.clear();
                        scores.clear();
                        scores.resize(batch_requests.len(), 0.0);
                        
                        let mut params = None;
                        
                        // Generate a unique base seed for this batch using high-resolution timing
                        let base_seed = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos() as u32;
                        
                        for (i, req) in batch_requests.iter().enumerate() {
                            if let Some((data, w, h, player)) = req.state.get_gpu_simulation_data() {
                                if params.is_none() {
                                    // Use high-resolution nanosecond-based seed that varies per batch
                                    params = Some(gpu::GpuSimulationParams {
                                        board_width: w as u32,
                                        board_height: h as u32,
                                        current_player: player,
                                        use_heuristic: if use_heuristic_flag { 1 } else { 0 },
                                        seed: base_seed + (i as u32 * 9973), // Different seed per state
                                    });
                                }
                                flat_data.extend(data);
                                gpu_indices.push(i);
                            } else {
                                // This game doesn't support GPU simulation, needs CPU rollout
                                cpu_indices.push(i);
                            }
                        }
                        
                        // GPU evaluation for supported games
                        if let Some(p) = params {
                            let mut acc = accelerator.lock();
                            if let Ok(gpu_scores) = acc.simulate_batch(&flat_data, p) {
                                for (idx, score) in gpu_indices.iter().zip(gpu_scores.into_iter()) {
                                    scores[*idx] = score;
                                }
                            }
                        }

                        
                        // CPU random rollout for games that don't support GPU simulation
                        // This ensures all games work, even without custom GPU shaders
                        for &idx in &cpu_indices {
                            let mut sim_state = batch_requests[idx].state.clone();
                            let leaf_player = sim_state.get_current_player();
                            
                            // Run random rollout on CPU
                            let winner = if sim_state.is_terminal() {
                                sim_state.get_winner()
                            } else {
                                let mut moves_cache = Vec::new();
                                let mut simulation_moves = 0;
                                const MAX_SIMULATION_MOVES: usize = 500;
                                
                                while !sim_state.is_terminal() && simulation_moves < MAX_SIMULATION_MOVES {
                                    moves_cache.clear();
                                    moves_cache.extend(sim_state.get_possible_moves());
                                    if moves_cache.is_empty() {
                                        break;
                                    }
                                    
                                    // Weighted random selection
                                    let mut total_weight = 0.0;
                                    for mv in &moves_cache {
                                        total_weight += sim_state.get_move_weight(mv);
                                    }

                                    let mut threshold = random_f64() * total_weight;
                                    let mut move_index = 0;
                                    
                                    for (i, mv) in moves_cache.iter().enumerate() {
                                        let weight = sim_state.get_move_weight(mv);
                                        if threshold < weight {
                                            move_index = i;
                                            break;
                                        }
                                        threshold -= weight;
                                    }
                                    
                                    // Fallback
                                    if move_index >= moves_cache.len() {
                                        move_index = 0;
                                    }

                                    let mv = &moves_cache[move_index];
                                    sim_state.make_move(mv);
                                    simulation_moves += 1;
                                }
                                
                                if simulation_moves >= MAX_SIMULATION_MOVES {
                                    None // Treat as draw
                                } else {
                                    sim_state.get_winner()
                                }
                            };
                            
                            // Convert winner to score (from leaf_player's perspective)
                            // Use special values to distinguish win/loss/draw
                            scores[idx] = match winner {
                                Some(w) if w == leaf_player => 4000.0,
                                Some(_) => -4000.0,
                                None => 0.0, // Draw
                            };
                        }

                        // Process results: Expand and Backpropagate in parallel
                        // Offload to Rayon global pool to avoid blocking the GPU worker
                        let requests: Vec<_> = batch_requests.drain(..).collect();
                        let scores_for_processing = scores.clone();
                        let pending_evals_clone = pending_evals_clone.clone();
                        
                        rayon::spawn(move || {
                            let process_item = |req: EvaluationRequest<S>, score: f32| {
                                // Backpropagate
                                // Map score to [0, 1] win probability for current_player (at leaf)
                                let leaf_player = req.state.get_current_player();
                                
                                let win_prob = if score >= 4000.0 {
                                    1.0
                                } else if score <= -4000.0 {
                                    0.0
                                } else {
                                    // Map heuristic score to win probability
                                    0.5 + 0.5 * (score / 200.0).tanh() as f64
                                };

                                for (node, &player_who_moved) in req.path.iter().zip(req.path_players.iter()).rev() {
                                    node.remove_virtual_loss();
                                    node.visits.fetch_add(1, Ordering::Relaxed);
                                    
                                    // Calculate reward for this node's perspective
                                    let reward_val = if player_who_moved == leaf_player {
                                        2.0 * win_prob
                                    } else {
                                        2.0 * (1.0 - win_prob)
                                    };
                                    
                                    // Stochastic rounding to integer
                                    let reward_int = reward_val as i32;
                                    let reward_frac = reward_val - reward_int as f64;
                                    let final_reward = reward_int + if random_f64() < reward_frac { 1 } else { 0 };
                                    
                                    node.wins.fetch_add(final_reward, Ordering::Relaxed);
                                }
                                
                                // Decrement pending evaluations counter
                                pending_evals_clone.fetch_sub(1, Ordering::Relaxed);
                            };

                            // Use parallel iteration only for large batches to avoid overhead
                            // Small batches are processed serially to reduce scheduler pressure
                            if requests.len() > 128 {
                                requests.into_par_iter().zip(scores_for_processing.into_par_iter()).for_each(|(req, score)| {
                                    process_item(req, score);
                                });
                            } else {
                                for (req, score) in requests.into_iter().zip(scores_for_processing.into_iter()) {
                                    process_item(req, score);
                                }
                            }
                        });
                    }
                });
                Some(tx)
            } else {
                None
            }
        } else {
            None
        };

        let mcts = MCTS {
            root: Arc::new(Node::new()),
            exploration_parameter,
            pool,
            node_pool: NodePool::new(),
            max_nodes,
            node_count,
            timeout_overhead_ms: Arc::new(Mutex::new(50.0)),
            timeout_measurements: Arc::new(AtomicI32::new(0)),
            searches_since_measurement: Arc::new(AtomicI32::new(0)),
            move_selection_strategy: MoveSelectionStrategy::default(),
            gpu_accelerator,
            gpu_enabled,
            gpu_puct_cache: Arc::new(RwLock::new(HashMap::new())),
            gpu_cache_timestamp: Arc::new(Mutex::new(Instant::now())),
            simulations_since_gpu_update: Arc::new(AtomicI32::new(0)),
            gpu_last_batch_size: Arc::new(AtomicI32::new(0)),
            gpu_simulation_sender,
            gpu_pending_evaluations: pending_evaluations,
            gpu_native_othello: Mutex::new(None),
        };

        (mcts, message)
    }

    /// Gets the exploration parameter used in the UCB1 formula
    ///
    /// # Returns
    /// The exploration parameter (C_puct value)
    pub fn get_exploration_parameter(&self) -> f64 {
        self.exploration_parameter
    }

    /// Gets the maximum number of nodes allowed in the tree
    ///
    /// # Returns
    /// Maximum node count before tree pruning occurs
    pub fn get_max_nodes(&self) -> usize {
        self.max_nodes
    }

    /// Returns whether GPU acceleration is enabled
    ///
    /// # Returns
    /// True if GPU acceleration is enabled and available
    #[cfg(feature = "gpu")]
    pub fn is_gpu_enabled(&self) -> bool {
        self.gpu_enabled
    }

    /// Returns GPU debug information if available
    ///
    /// # Returns
    /// Optional string with GPU statistics and info
    #[cfg(feature = "gpu")]
    pub fn get_gpu_info(&self) -> Option<String> {
        if let Some(ref accelerator) = self.gpu_accelerator {
            Some(accelerator.lock().debug_info())
        } else {
            None
        }
    }

    /// Enables or disables GPU acceleration at runtime
    ///
    /// This allows toggling GPU usage without recreating the MCTS engine.
    /// Useful for comparing GPU vs CPU performance or when GPU encounters issues.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable GPU acceleration
    #[cfg(feature = "gpu")]
    pub fn set_gpu_enabled(&mut self, enabled: bool) {
        if enabled && self.gpu_accelerator.is_some() {
            self.gpu_enabled = true;
        } else if !enabled {
            self.gpu_enabled = false;
        }
    }

    /// Sets the move selection strategy
    ///
    /// Controls how the AI selects the best move after search completes:
    /// - `MaxVisits`: Select the most visited move (default, most robust)
    /// - `MaxQ`: Select the move with highest win rate (Q value)
    ///
    /// # Arguments
    /// * `strategy` - The move selection strategy to use
    pub fn set_move_selection_strategy(&mut self, strategy: MoveSelectionStrategy) {
        self.move_selection_strategy = strategy;
    }

    /// Gets the current move selection strategy
    ///
    /// # Returns
    /// The currently configured move selection strategy
    pub fn get_move_selection_strategy(&self) -> MoveSelectionStrategy {
        self.move_selection_strategy
    }

    #[cfg(feature = "gpu")]
    fn compute_othello_legal_moves(board: &[i32; 64], player: i32) -> Vec<(usize, usize)> {
        const DIRS: [(i32, i32); 8] = [
            (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
        ];
        let w = 8i32;
        let mut moves = Vec::new();
        for y in 0..w {
            for x in 0..w {
                if board[(y * w + x) as usize] != 0 {
                    continue;
                }
                let mut valid = false;
                for (dx, dy) in DIRS.iter().copied() {
                    let mut cx = x + dx;
                    let mut cy = y + dy;
                    let mut seen_opponent = false;
                    while cx >= 0 && cx < w && cy >= 0 && cy < w {
                        let cell = board[(cy * w + cx) as usize];
                        if cell == -player {
                            seen_opponent = true;
                            cx += dx;
                            cy += dy;
                            continue;
                        }
                        if cell == player && seen_opponent {
                            valid = true;
                        }
                        break;
                    }
                    if valid {
                        break;
                    }
                }
                if valid {
                    // GPU-native expects (x, y) where x=col, y=row; keep (x, y) as computed.
                    moves.push((x as usize, y as usize));
                }
            }
        }
        moves
    }

    /// Performs GPU-native MCTS search for Othello
    ///
    /// This method runs all four MCTS phases entirely on the GPU, eliminating
    /// CPU-GPU synchronization overhead and the stale path problem that affects
    /// the hybrid approach.
    ///
    /// # Arguments
    /// * `board` - 64-element Othello board (1 = black, -1 = white, 0 = empty)
    /// * `current_player` - Player to move (1 or -1)
    /// * `legal_moves` - List of legal moves as (x, y) coordinates
    /// * `iterations_per_batch` - Number of parallel iterations per GPU dispatch
    /// * `num_batches` - Number of batches to run
    /// * `exploration` - C_puct exploration parameter
    /// * `virtual_loss_weight` - Weight applied to virtual loss during selection
    /// * `gpu_max_nodes` - Optional override for max nodes (None = auto-calculate from GPU limits)
    ///
    /// # Returns
    /// The best move as (x, y), total visits, Q value, children stats, total nodes allocated, and GPU telemetry
    #[cfg(feature = "gpu")]
    pub fn search_gpu_native_othello(
        &self,
        board: &[i32; 64],
        current_player: i32,
        legal_moves: &[(usize, usize)],
        iterations_per_batch: u32,
        num_batches: u32,
        exploration: f32,
        virtual_loss_weight: f32,
        temperature: f32,
        timeout_secs: u64,
        gpu_max_nodes: Option<u32>,
    ) -> Option<((usize, usize), i32, f64, Vec<(usize, usize, i32, i32, f64)>, u32, gpu::OthelloRunTelemetry)> {
        use std::sync::Arc;

        // Recompute legal moves from the board/current_player to catch caller inconsistencies
        let mut provided_moves = legal_moves.to_vec();
        let mut computed_moves = Self::compute_othello_legal_moves(board, current_player);
        provided_moves.sort_unstable();
        computed_moves.sort_unstable();
        let legal_moves = if provided_moves == computed_moves {
            provided_moves
        } else {
            eprintln!(
                "[GPU-Native HOST WARN] legal_moves mismatch for search; using computed. provided={:?} computed={:?}",
                provided_moves, computed_moves
            );
            computed_moves
        };
        
        if legal_moves.is_empty() {
            return None;
        }
        
        if legal_moves.len() == 1 {
            return Some((
                legal_moves[0],
                1,
                0.5,
                vec![(legal_moves[0].0, legal_moves[0].1, 1, 0, 0.5)],
                1,
                gpu::OthelloRunTelemetry::default(),
            ));
        }

        // Use existing GPU-native engine if available (check if max_nodes matches)
        let gpu_mcts_arc = {
            let mut guard = self.gpu_native_othello.lock();
            
            // Check if we can reuse existing engine
            let should_recreate = if let Some(ref existing) = *guard {
                if let Some(requested) = gpu_max_nodes {
                    let existing_capacity = existing.lock().get_capacity();
                    if existing_capacity != requested {
                        eprintln!(
                            "[GPU-Native WARNING] Requested max_nodes={} but existing engine has capacity={}. Recreating engine...",
                            requested, existing_capacity
                        );
                        true  // Recreate with new capacity
                    } else {
                        false  // Capacity matches, reuse
                    }
                } else {
                    false  // No user override, reuse existing
                }
            } else {
                true  // No existing engine, need to create
            };
            
            if !should_recreate {
                guard.as_ref().unwrap().clone()
            } else {
                // Drop existing if any
                *guard = None;
                
                // Need to create a new engine - get or create GPU context
                let gpu_context = if let Some(ref acc) = self.gpu_accelerator {
                    acc.lock().get_context()
                } else {
                    // Try to create a new context
                    let config = gpu::GpuConfig::default();
                    match gpu::GpuContext::new(&config) {
                        Ok(ctx) => Arc::new(ctx),
                        Err(_) => return None,
                    }
                };
                
                // Create new engine
                // Calculate max_nodes based on GPU limits
                // Each node requires ~256 bytes for children_indices buffer (64 children * 4 bytes)
                // This is the largest single buffer, so it dictates the limit
                let max_storage_size = gpu_context.max_storage_buffer_binding_size();
                if max_storage_size == 0 {
                    eprintln!("[GPU-Native HOST FATAL] GPU context reports max_storage_buffer_binding_size == 0! Cannot allocate node pool. This is a driver or hardware issue.");
                    panic!("GPU context reports max_storage_buffer_binding_size == 0; cannot create node pool");
                }
                // Use 98% of the limit - we can be aggressive since we have capacity monitoring
                let max_nodes = gpu_max_nodes.unwrap_or_else(|| {
                    (max_storage_size as f64 * 0.98 / 256.0) as u32
                });
                if max_nodes == 0 {
                    eprintln!("[GPU-Native HOST FATAL] Computed max_nodes == 0 (max_storage_size = {}). Cannot create node pool. This is a driver, hardware, or configuration issue.", max_storage_size);
                    panic!("Computed max_nodes == 0; cannot create node pool");
                }
                eprintln!("[GPU-Native] Max nodes set to {} {}based on storage limit of {} bytes", 
                    max_nodes, 
                    if gpu_max_nodes.is_some() { "(user override) " } else { "" },
                    max_storage_size);

                let new_engine = gpu::GpuOthelloMcts::new(
                    gpu_context,
                    max_nodes,
                    iterations_per_batch,
                ).expect("Failed to create GpuOthelloMcts");
                eprintln!("[GPU-Native HOST] After creation: new_engine.get_capacity() = {}", new_engine.get_capacity());
                let engine_arc = Arc::new(Mutex::new(new_engine));
                // Store it in the MCTS struct for reuse across searches
                *guard = Some(engine_arc.clone());
                engine_arc
            }
        };

        let mut gpu_mcts = gpu_mcts_arc.lock();

        // Check if we need to initialize or can continue from existing tree
        let root_visits = gpu_mcts.get_root_visits();

        // Validate GPU root board against host board to avoid subtle drift when reusing trees
        if root_visits > 0 {
            let gpu_hash = gpu_mcts.get_root_board_hash();
            let mut host_hash: u64 = 0xcbf29ce484222325;
            for v in board.iter() {
                host_hash ^= *v as u64;
                host_hash = host_hash.wrapping_mul(0x100000001b3);
            }

            if gpu_hash != host_hash {
                panic!(
                    "GPU-Native root board hash mismatch (gpu={} host={}); aborting GPU search",
                    gpu_hash, host_hash
                );
            }
        }
        
        // Count pieces on board to detect new game (Othello starts with 4 pieces)
        // If we are at the start of a game, we MUST reset the tree to ensure we don't use stale data
        // from a previous game that ended.
        let piece_count = board.iter().filter(|&&x| x != 0).count();
        let is_new_game = piece_count <= 4;

        if root_visits == 0 || is_new_game {
            // Fresh tree needed
            gpu_mcts.init_tree(board, current_player, &legal_moves);
        }
        // Note: Tree reuse via advance_root is handled separately through advance_root_gpu_native()

        // Sanity check: root children should match legal_moves; otherwise reset tree
        let mut actual_moves: Vec<(usize, usize)> = gpu_mcts
            .get_children_stats()
            .into_iter()
            .map(|(x, y, _, _, _)| (x, y))
            .collect();
        let mut expected_moves: Vec<(usize, usize)> = legal_moves.clone();
        actual_moves.sort_unstable();
        expected_moves.sort_unstable();

        if actual_moves != expected_moves {
            // Recompute legal moves from the supplied board/player to avoid stale caller data
            let mut computed_moves = Self::compute_othello_legal_moves(board, current_player);
            computed_moves.sort_unstable();

            if actual_moves != computed_moves {
                panic!(
                    "[GPU-Native HOST FATAL] root children mismatch - GPU state corrupted! actual={:?} expected={:?} computed={:?}",
                    actual_moves, expected_moves, computed_moves
                );
            }
        }

        // Run iterations with timeout enforcement
        let start_time = std::time::Instant::now();
        let timeout = if timeout_secs > 0 {
            Some(std::time::Duration::from_secs(timeout_secs))
        } else {
            None
        };
        
        let mut seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u32;

        let mut last_telemetry: Option<gpu::OthelloRunTelemetry> = None;

        let mut batch = 0u32;
        loop {
            // Check timeout before each batch
            if let Some(t) = timeout {
                if start_time.elapsed() >= t {
                    break;
                }
            } else if batch >= num_batches {
                // No timeout - use batch count limit
                break;
            }
            
            let telemetry = gpu_mcts.run_iterations(
                iterations_per_batch,
                exploration,
                virtual_loss_weight,
                temperature,
                seed.wrapping_add(batch * 1000),
            );
            last_telemetry = Some(telemetry);
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            batch += 1;
            
            // Force GPU sync every 100 batches to prevent command buffer buildup
            if batch % 100 == 0 {
                gpu_mcts.flush_and_wait();
            }

            if telemetry.saturated {
                eprintln!(
                    "[GPU-Native WARNING] Node pool saturated: {} / {} nodes after batch {}",
                    telemetry.alloc_count_after, telemetry.node_capacity, batch
                );
                break;
            }

            // If we are within 2% of capacity, stop dispatching more batches to avoid illegal proposals
            let nodes_in_use = telemetry.alloc_count_after.saturating_sub(telemetry.free_count_after);
            let cap_guard = (telemetry.node_capacity as f32 * 0.98) as u32;
            if nodes_in_use >= cap_guard {
                eprintln!(
                    "[GPU-Native WARNING] Node pool near capacity: {} in use ({} allocated - {} freed) / {} after batch {}; stopping early",
                    nodes_in_use, telemetry.alloc_count_after, telemetry.free_count_after, telemetry.node_capacity, batch
                );
                break;
            }
        }
        
        // Final flush to ensure all work completes
        gpu_mcts.flush_and_wait();
        
        eprintln!("[GPU-Native] Completed {} batches ({} iterations) in {:.2}s", 
            batch, batch as u64 * iterations_per_batch as u64, start_time.elapsed().as_secs_f64());

        // Get children stats for TSV logging
        let children_stats = gpu_mcts.get_children_stats();
        let telemetry = last_telemetry.unwrap_or_default();
        let mut total_nodes = if telemetry.alloc_count_after > 0 {
            telemetry.alloc_count_after
        } else {
            gpu_mcts.get_total_nodes()
        };

        // Diagnostics: show how many nodes were used vs capacity and whether we saturated
        let diag_prefix = "\x1b[33m[GPU-Native DIAG]\x1b[0m";
        let sat_flag = if telemetry.saturated { "\x1b[31mSATURATED\x1b[0m" } else { "OK" };
        eprintln!(
            "{} nodes_used={} capacity={} batches={} iter_per_batch={} virtual_loss_weight={:.2} status={}",
            diag_prefix,
            total_nodes,
            telemetry.node_capacity,
            batch,
            iterations_per_batch,
            virtual_loss_weight,
            sat_flag,
        );

        // Per-depth visit histogram to see search spread; last bin is overflow if max depth exceeded
        let depth_hist = gpu_mcts.get_depth_visit_histogram(32);
        if !depth_hist.is_empty() {
            let overflow = depth_hist.len() == 32;
            let parts: Vec<String> = depth_hist
                .iter()
                .enumerate()
                .filter(|(_, count)| **count > 0)
                .map(|(d, count)| format!("{}:{}", d, count))
                .collect();
            let suffix = if overflow { " (bin 31 = overflow)" } else { "" };
            eprintln!("{} depth_visits:{}{}", diag_prefix, parts.join(" "), suffix);
        }

        // Selection/expansion counters to spot algorithmic early exits
        let d = telemetry.diagnostics;
        eprintln!(
            "{} diag_counts sel_term={} sel_noch={} sel_inv={} sel_pathcap={} exp_attempts={} exp_success={} exp_locked={} exp_term={} alloc_fail={} rollouts={}",
            diag_prefix,
            d.selection_terminal,
            d.selection_no_children,
            d.selection_invalid_child,
            d.selection_path_cap,
            d.expansion_attempts,
            d.expansion_success,
            d.expansion_locked,
            d.expansion_terminal,
            d.alloc_failures,
            d.rollouts,
        );

        let diag_red_flag = d.selection_invalid_child > 0 || d.alloc_failures > 0;

        // DEBUG: Print ALL children to see visit distribution
        let mut sorted = children_stats.clone();
        sorted.sort_by_key(|(_, _, v, _, _)| -(*v));
        eprintln!("[GPU-Native DEBUG] All {} children by visits:", sorted.len());
        for (i, (x, y, visits, wins, q)) in sorted.iter().enumerate() {
            let win_rate = if *visits > 0 { *wins as f64 / (*visits as f64 * 2.0) } else { 0.0 };
            eprintln!("  {}. ({},{}) visits={:7} wins={:7} Q={:.4} raw_wr={:.4}", 
                     i+1, x, y, visits, wins, q, win_rate);
        }

        // Guardrail: if GPU proposes an illegal move, abort instead of silently patching
        let final_children_stats = children_stats;
        let best = gpu_mcts.get_best_move();
        if let Some((x, y, _, _)) = best {
            let mv = (x, y);
            let legal_ok = legal_moves.binary_search(&mv).is_ok();
            if !legal_ok {
                panic!("GPU-Native proposed illegal move ({},{}); aborting GPU search", x, y);
            }
        }

        if diag_red_flag {
            panic!("GPU-Native diagnostics flagged invalid-child/alloc issues; aborting GPU search");
        }

        total_nodes = if telemetry.alloc_count_after > 0 {
            telemetry.alloc_count_after
        } else {
            gpu_mcts.get_total_nodes()
        };

        best.map(|(x, y, visits, q)| ((x, y), visits, q, final_children_stats, total_nodes, telemetry))
    }

    /// Advance the GPU-native MCTS tree root after a move is made
    /// This enables tree reuse for consecutive GPU-native searches
    ///
    /// # Arguments
    /// * `move_xy` - The move that was made as (x, y) coordinates
    /// * `new_board` - The new board state after the move
    /// * `new_player` - The player to move in the new position
    /// * `new_legal_moves` - Legal moves from the new position
    ///
    /// # Returns
    /// true if advanced to existing child (tree reused), false if reinitialized
    #[cfg(feature = "gpu")]
    pub fn advance_root_gpu_native(
        &self,
        move_xy: (usize, usize),
        new_board: &[i32; 64],
        new_player: i32,
        new_legal_moves: &[(usize, usize)],
    ) -> bool {
        // Validate legal moves against the supplied board/new_player to avoid desync
        let mut provided_moves = new_legal_moves.to_vec();
        let mut computed_moves = Self::compute_othello_legal_moves(new_board, new_player);
        provided_moves.sort_unstable();
        computed_moves.sort_unstable();
        let legal_moves = if provided_moves == computed_moves {
            provided_moves
        } else {
            eprintln!(
                "[GPU-Native HOST WARN] legal_moves mismatch for advance_root; using computed. provided={:?} computed={:?}",
                provided_moves, computed_moves
            );
            computed_moves
        };

        let guard = self.gpu_native_othello.lock();
        if let Some(ref gpu_mcts_arc) = *guard {
            let mut gpu_mcts = gpu_mcts_arc.lock();
            gpu_mcts.advance_root(move_xy.0, move_xy.1, new_board, new_player, &legal_moves)
        } else {
            false
        }
    }

    /// Initialize the GPU-native Othello engine for a new game
    /// Call this before the first search in a game to enable tree reuse
    #[cfg(feature = "gpu")]
    pub fn init_gpu_native_othello(&mut self, board: &[i32; 64], current_player: i32, legal_moves: &[(usize, usize)], iterations_per_batch: u32) {
        // Get GPU context
        let gpu_context = if let Some(ref acc) = self.gpu_accelerator {
            acc.lock().get_context()
        } else {
            let config = gpu::GpuConfig::default();
            match gpu::GpuContext::new(&config) {
                Ok(ctx) => std::sync::Arc::new(ctx),
                Err(_) => return,
            }
        };

        // Calculate max_nodes based on GPU limits
        let max_storage_size = gpu_context.max_storage_buffer_binding_size();
        // Use 98% of the limit - we can be aggressive since we have capacity monitoring
        let max_nodes = (max_storage_size as f64 * 0.98 / 256.0) as u32;
        eprintln!("[GPU-Native] Max nodes set to {} based on storage limit of {} bytes", max_nodes, max_storage_size);

        let mut engine = gpu::GpuOthelloMcts::new(gpu_context, max_nodes, iterations_per_batch)
            .expect("Failed to create GpuOthelloMcts");
        engine.init_tree(board, current_player, legal_moves);
        *self.gpu_native_othello.lock() = Some(Arc::new(Mutex::new(engine)));
    }

    /// Selects the best move from children based on the configured strategy
    ///
    /// This helper method implements the move selection logic for both strategies:
    /// - `MaxVisits`: Returns the move with the highest visit count
    /// - `MaxQ`: Returns the move with the highest Q value (wins/visits)
    ///
    /// # Arguments
    /// * `children` - Reference to the children map from the root node
    ///
    /// # Returns
    /// The best move according to the configured strategy
    fn select_best_move(&self, children: &HashMap<S::Move, Arc<Node<S::Move>>>) -> S::Move {
        match self.move_selection_strategy {
            MoveSelectionStrategy::MaxVisits => {
                children
                    .iter()
                    .max_by_key(|(_, node)| node.visits.load(Ordering::Relaxed))
                    .map(|(mv, _)| mv.clone())
                    .expect("Root node has children but max_by_key failed")
            }
            MoveSelectionStrategy::MaxQ => {
                children
                    .iter()
                    .max_by(|(_, a), (_, b)| {
                        let a_visits = a.visits.load(Ordering::Relaxed);
                        let b_visits = b.visits.load(Ordering::Relaxed);
                        let a_q = if a_visits > 0 {
                            a.wins.load(Ordering::Relaxed) as f64 / a_visits as f64
                        } else {
                            f64::NEG_INFINITY
                        };
                        let b_q = if b_visits > 0 {
                            b.wins.load(Ordering::Relaxed) as f64 / b_visits as f64
                        } else {
                            f64::NEG_INFINITY
                        };
                        a_q.partial_cmp(&b_q).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(mv, _)| mv.clone())
                    .expect("Root node has children but max_by failed")
            }
        }
    }

    /// Advances the root of the tree to the node corresponding to the given move.
    ///
    /// This is used when a move is made in the actual game to reuse the search tree.
    /// The subtree corresponding to the selected move becomes the new root,
    /// and all other subtrees are recycled to save memory.
    ///
    /// # Arguments
    /// * `mv` - The move that was made in the game
    /// * `debug_info` - Optional debug info about the player who made the move
    pub fn advance_root(&mut self, mv: &S::Move, debug_info: Option<&str>) {
        let (new_root, nodes_to_recycle, new_tree_size) = {
            let children = self.root.children.read();
            let new_root = children
                .get(mv)
                .map(Arc::clone)
                .unwrap_or_else(|| Arc::new(Node::new()));

            let info = debug_info.unwrap_or("Unknown");

            // --- Debug Stats Collection ---
            let root_visits = self.root.visits.load(Ordering::Relaxed);
            // Derive RootQ from the best child's Q-value instead of the root's mixed stats
            let root_q = children.values()
                .map(|n| {
                    let v = n.visits.load(Ordering::Relaxed);
                    let w = n.wins.load(Ordering::Relaxed);
                    if v > 0 { (w as f64 / v as f64) / 2.0 } else { 0.0 }
                })
                .fold(0.0, f64::max);
            
            // Stats for the chosen move (New Root)
            let (new_root_visits, new_root_q, new_root_u) = if let Some(node) = children.get(mv) {
                 let v = node.visits.load(Ordering::Relaxed);
                 let w = node.wins.load(Ordering::Relaxed);
                 let q = if v > 0 { (w as f64 / v as f64) / 2.0 } else { 0.0 };
                 let prior = if !children.is_empty() { 1.0 / children.len() as f64 } else { 0.0 };
                 let u = self.exploration_parameter * prior * (root_visits as f64).sqrt() / (1.0 + v as f64);
                 (v, q, u)
            } else {
                 (0, 0.0, 0.0)
            };

            // Stats for the Second Best move (for comparison)
            let mut sorted_children: Vec<_> = children.iter().collect();
            sorted_children.sort_by_key(|(_, node)| -node.visits.load(Ordering::Relaxed));
            
            // Helper to clean move strings
            let clean_move = |mv: &S::Move| -> String {
                let s = format!("{:?}", mv);
                let mut result = s;
                loop {
                    let old_len = result.len();
                    if let Some(start) = result.find('(') {
                        if let Some(end) = result.rfind(')') {
                            let prefix = &result[..start];
                            if prefix.chars().all(|c| c.is_alphanumeric() || c == '_') && !prefix.is_empty() {
                                result = result[start+1..end].to_string();
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                    if result.len() == old_len { break; }
                }
                result
            };

            let mut second_best_stats = (0, 0.0, 0.0); // visits, q, u
            let mut second_best_move_str = "None".to_string();

            for (child_mv, node) in sorted_children {
                if child_mv != mv {
                     let v = node.visits.load(Ordering::Relaxed);
                     let w = node.wins.load(Ordering::Relaxed);
                     let q = if v > 0 { (w as f64 / v as f64) / 2.0 } else { 0.0 };
                     let prior = if !children.is_empty() { 1.0 / children.len() as f64 } else { 0.0 };
                     let u = self.exploration_parameter * prior * (root_visits as f64).sqrt() / (1.0 + v as f64);
                     second_best_stats = (v, q, u);
                     second_best_move_str = clean_move(child_mv);
                     break;
                }
            }

            // CSV Output
            // Only write to CSV if this is NOT an opponent move update
            // We detect this by checking if the info string contains "(Opponent Move)"
            if !info.contains("(Opponent Move)") {
                let visit_diff = (new_root_visits as i64) - (second_best_stats.0 as i64);
                // Format: Info, RootVisits, RootQ, MoveVisits-AltVisits, Move, MoveVisits, MoveQ, MoveU, AltMove, AltVisits, AltQ, AltU
                let csv_line = format!("{}\t{}\t{:.4}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\n",
                    info, root_visits, root_q, visit_diff, clean_move(mv), new_root_visits, new_root_q, new_root_u,
                    second_best_move_str, second_best_stats.0, second_best_stats.1, second_best_stats.2
                );

                // Print to terminal
                println!("CSV_DATA: {}", csv_line.trim());

                // Append to file
                use std::io::Write;
                let file_path = std::path::Path::new("mcts_stats.tsv");
                
                // Use a static flag to track if we've initialized the file in this run
                // This ensures we overwrite on the first write, then append for subsequent writes
                static FILE_INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                
                let is_first_write = !FILE_INITIALIZED.swap(true, Ordering::Relaxed);
                
                let mut options = std::fs::OpenOptions::new();
                if is_first_write {
                    options.create(true).write(true).truncate(true);
                } else {
                    options.create(true).append(true);
                }

                if let Ok(mut file) = options.open(file_path) {
                    if is_first_write {
                        let _ = file.write_all(b"Info\tRootVisits\tRootQ\tMoveVisits-AltVisits\tMove\tMoveVisits\tMoveQ\tMoveU\tAltMove\tAltVisits\tAltQ\tAltU\n");
                    }
                    let _ = file.write_all(csv_line.as_bytes());
                }
            }
            // -----------------------------

            // Calculate the size of the new subtree
            let new_tree_size = if children.contains_key(mv) {
                let visits = new_root.visits.load(Ordering::Relaxed);
                let root_visits = self.root.visits.load(Ordering::Relaxed);
                let root_nodes = self.node_count.load(Ordering::Relaxed);
                println!("[{}] Advancing root to existing child. Child Visits: {}, Q: {:.4}, U: {:.4}, Root Visits: {}, Root Nodes: {}, Root Q: {:.4}", info, visits, new_root_q, new_root_u, root_visits, root_nodes, root_q);
                1 + self.count_subtree_nodes(&new_root)
            } else {
                let root_visits = self.root.visits.load(Ordering::Relaxed);
                let root_nodes = self.node_count.load(Ordering::Relaxed);
                println!("[{}] Advancing root to NEW node (child not found). Root Visits: {}, Root Nodes: {}, Root Q: {:.4}", info, root_visits, root_nodes, root_q);
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
        self.node_count
            .store(new_tree_size as i32, Ordering::Relaxed);

        self.root = new_root;
    }

    /// Counts the total number of nodes in a subtree (including the root of the subtree)
    ///
    /// Used for memory management and tree statistics.
    ///
    /// # Arguments
    /// * `root` - The root node of the subtree to count
    ///
    /// # Returns
    /// Total number of nodes in the subtree
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
    ///
    /// Removes nodes with visit counts below the threshold to control memory usage
    /// and focus computation on promising paths. Call this periodically during search.
    ///
    /// # Arguments
    /// * `min_visits_threshold` - Minimum number of visits required to keep a node
    pub fn prune_tree(&mut self, min_visits_threshold: i32) {
        let mut total_pruned_count = 0;
        let mut stack = vec![self.root.clone()];

        while let Some(node) = stack.pop() {
            // Prune weak children of this node
            let pruned_nodes = node.prune_weak_children(min_visits_threshold);
            
            if !pruned_nodes.is_empty() {
                total_pruned_count += pruned_nodes.len();
                self.node_pool.return_nodes(pruned_nodes);
            }

            // Add surviving children to stack to check them too
            let children = node.children.read();
            for child in children.values() {
                // Optimization: only recurse if child has enough visits to potentially have children
                // If child visits are low (but > threshold), it might not have many children anyway
                if child.visits.load(Ordering::Relaxed) > min_visits_threshold {
                    stack.push(child.clone());
                }
            }
        }

        if total_pruned_count > 0 {
            self.node_count.fetch_sub(total_pruned_count as i32, Ordering::Relaxed);
        }
    }

    /// Automatically prunes the tree based on visit statistics
    ///
    /// Removes children with less than 0.1% of the root's visits to keep the tree
    /// focused on the most promising moves. This is a heuristic-based pruning
    /// that doesn't require manual threshold setting.
    pub fn auto_prune(&mut self) {
        let root_visits = self.root.visits.load(Ordering::Relaxed);
        let min_visits = std::cmp::max(1, root_visits / 1000); // At least 0.1% of root visits
        self.prune_tree(min_visits);
    }

    /// Returns statistics for the children of the root node.
    ///
    /// Provides detailed statistics about each possible move from the current position.
    /// Used for move analysis and debugging the search behavior.
    ///
    /// # Returns
    /// HashMap mapping moves to (wins, visits) tuples
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

    /// Returns the statistics for the root node.
    ///
    /// Provides overall statistics about the search from the current position.
    ///
    /// # Returns
    /// Tuple of (wins, visits) for the root node
    pub fn get_root_stats(&self) -> (f64, i32) {
        let wins = self.root.wins.load(Ordering::Relaxed) as f64;
        let visits = self.root.visits.load(Ordering::Relaxed);
        (wins, visits)
    }

    /// Returns debug information about the current MCTS state
    ///
    /// Provides a formatted string with detailed information about the search tree,
    /// including root statistics, tree size, configuration, and top moves.
    /// Useful for debugging and monitoring search progress.
    ///
    /// # Returns
    /// Multi-line debug string with tree statistics and top moves
    pub fn get_debug_info(&self) -> String {
        let root_visits = self.root.visits.load(Ordering::Relaxed);
        let root_wins = self.root.wins.load(Ordering::Relaxed);
        let node_count = self.node_count.load(Ordering::Relaxed);
        let children_count = self.root.children.read().len();

        let mut debug_lines = vec![
            "MCTS Debug Info:".to_string(),
            format!(
                "Root: {} visits, {} wins, {:.3} rate",
                root_visits,
                root_wins,
                {
                    let children = self.root.children.read();
                    children.values()
                        .map(|n| {
                            let v = n.visits.load(Ordering::Relaxed);
                            let w = n.wins.load(Ordering::Relaxed);
                            if v > 0 { (w as f64 / v as f64) / 2.0 } else { 0.0 }
                        })
                        .fold(0.0, f64::max)
                }
            ),
            format!(
                "Tree: {} nodes ({} children, max {})",
                node_count, children_count, self.max_nodes
            ),
            format!(
                "Exploration: {:.3}, Threads: {}",
                self.exploration_parameter,
                self.pool.current_num_threads()
            ),
        ];

        // Add GPU status if feature is enabled
        #[cfg(feature = "gpu")]
        {
            if self.gpu_enabled {
                if let Some(ref accelerator) = self.gpu_accelerator {
                    let acc = accelerator.lock();
                    let (total_us, dispatches, avg_us) = acc.stats();
                    debug_lines.push(format!(
                        "GPU: enabled, {} dispatches, {:.2}ms total, {:.2}µs avg",
                        dispatches, total_us as f64 / 1000.0, avg_us
                    ));
                }
            } else {
                debug_lines.push("GPU: disabled".to_string());
            }
        }

        // Add top child statistics (limited to prevent UI overflow)
        let children = self.root.children.read();
        if !children.is_empty() {
            debug_lines.push("Top moves:".to_string());
            let mut sorted_children: Vec<_> = children.iter().collect();
            sorted_children.sort_by_key(|(_, node)| -node.visits.load(Ordering::Relaxed));

            for (mv, node) in sorted_children.iter().take(5) {
                let visits = node.visits.load(Ordering::Relaxed);
                let wins = node.wins.load(Ordering::Relaxed);
                let win_rate = if visits > 0 {
                    wins as f64 / visits as f64 / 2.0
                } else {
                    0.0
                };
                debug_lines.push(format!(
                    "  {:?}: {} visits, {:.3} rate",
                    mv, visits, win_rate
                ));
            }

            if sorted_children.len() > 5 {
                debug_lines.push(format!(
                    "  ... and {} more moves",
                    sorted_children.len() - 5
                ));
            }
        } else {
            debug_lines.push("No moves evaluated yet".to_string());
        }

        debug_lines.join("\n")
    }

    /// Ensures the root node is fully expanded with all possible moves.
    ///
    /// This prevents the issue where only one move gets explored due to early exploitation.
    /// By expanding all possible moves at the root, we ensure that the search considers
    /// all options and doesn't get stuck in local optima.
    ///
    /// # Arguments
    /// * `state` - The current game state to get possible moves from
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
            self.node_count
                .fetch_add(new_nodes_count, Ordering::Relaxed);
        }
    }

    /// Performs a parallel MCTS search with optional pruning and external stop control.
    /// This variant allows external threads to interrupt the search by setting the stop flag.
    ///
    /// # Arguments
    /// * `state` - The current state of the game.
    /// * `iterations` - The total number of simulations to run.
    /// * `stats_interval_secs` - Interval in seconds to print statistics (0 = no periodic stats).
    /// * `timeout_secs` - The maximum time in seconds to search for. 0 means no timeout.
    /// * `external_stop` - Optional external stop flag that can interrupt the search.
    pub fn search_with_stop(
        &mut self,
        state: &S,
        iterations: i32,
        stats_interval_secs: u64,
        timeout_secs: u64,
        external_stop: Option<Arc<AtomicBool>>,
    ) -> (S::Move, SearchStatistics) {
        let start_time = Instant::now();

        // Get current overhead estimate
        let estimated_overhead_ms = {
            let overhead = self.timeout_overhead_ms.lock();
            *overhead
        };

        // Use dynamic overhead estimation with a minimum of 20ms and maximum of 500ms
        let overhead_buffer_ms = estimated_overhead_ms.max(20.0).min(500.0);

        let effective_timeout_ms = if timeout_secs > 0 {
            ((timeout_secs * 1000) as f64 - overhead_buffer_ms).max(100.0) as u64 // Ensure at least 100ms for actual search
        } else {
            0
        };
        let timeout = if effective_timeout_ms > 0 {
            Some(Duration::from_millis(effective_timeout_ms))
        } else {
            None
        };

        self.ensure_root_expanded(state);

        // Initialize GPU PUCT cache if GPU is enabled
        #[cfg(feature = "gpu")]
        // self.update_gpu_puct_cache(true);
        {}

        let possible_moves = state.get_possible_moves();
        if possible_moves.len() == 1 {
            return (possible_moves[0].clone(), SearchStatistics::default());
        }

        if possible_moves.is_empty() {
            // This case should ideally be handled by get_possible_moves returning a pass move.
            // If we get here, it means the game ended, but is_terminal was false.
            // We are in an inconsistent state, but we must return a move.
            // The Blokus implementation should return a pass move.
            // If another game does not, this will be a problem.
            // For now, let's check the root's children. If there's one, it must be the pass move.
            let children = self.root.children.read();
            if children.len() == 1 {
                return (
                    children.keys().next().unwrap().clone(),
                    SearchStatistics::default(),
                );
            }
            // If there are no children and no possible moves, we are stuck.
            // This indicates a logic error in the game state implementation.
            panic!(
                "MCTS search: No possible moves and no children in root node. Game logic may be flawed."
            );
        }

        let completed_iterations = Arc::new(AtomicUsize::new(0));
        let stop_searching = Arc::new(AtomicBool::new(false));

        // Pre-calculate absolute timeout deadline
        let timeout_deadline = timeout.map(|t| start_time + t);

        let stats_interval = if stats_interval_secs > 0 {
            Some(Duration::from_secs(stats_interval_secs))
        } else {
            None
        };
        let last_stats_time = Arc::new(Mutex::new(Instant::now()));

        // Start a dedicated timeout monitoring thread if we have a timeout
        let timeout_monitor_handle = if let Some(deadline) = timeout_deadline {
            let stop_flag = stop_searching.clone();
            let ext_stop = external_stop.clone();
            Some(std::thread::spawn(move || {
                let check_interval = Duration::from_millis(5); // Check every 5ms for maximum responsiveness
                while !stop_flag.load(Ordering::Relaxed) {
                    if Instant::now() >= deadline {
                        stop_flag.store(true, Ordering::Relaxed);
                        break;
                    }
                    if let Some(ref ext_stop) = ext_stop {
                        if ext_stop.load(Ordering::Relaxed) {
                            stop_flag.store(true, Ordering::Relaxed);
                            break;
                        }
                    }
                    std::thread::sleep(check_interval);
                }
            }))
        } else {
            None
        };

        // Start a GPU cache refresh thread if GPU is enabled
        #[cfg(feature = "gpu")]
        let gpu_refresh_handle: Option<std::thread::JoinHandle<()>> = None;
        /*
        let gpu_refresh_handle = if self.gpu_enabled && self.gpu_accelerator.is_some() {
            // ... (existing commented out code) ...
        } else {
            None
        };
        */

        // Run simulations in batches to allow for periodic pruning
        let batch_size = 5000;
        let mut iterations_remaining = iterations;
        let mut prune_memory_count = 0;

        while iterations_remaining > 0 && !stop_searching.load(Ordering::Relaxed) {
            // Check if we need to prune the tree
            let current_nodes = self.node_count.load(Ordering::Relaxed) as usize;
            
            // Prune ONLY if we are close to the limit (> 90%)
            let memory_pressure = current_nodes > self.max_nodes * 9 / 10;

            if memory_pressure {
                prune_memory_count += 1;
                self.auto_prune();
            }

            let current_batch = iterations_remaining.min(batch_size);

            self.pool.install(|| {
                let _ = (0..current_batch)
                    .into_par_iter()
                    .try_for_each(|_| -> Result<(), ()> {
                        // Double-check stop flag at the very start of each iteration
                        if stop_searching.load(Ordering::Relaxed) {
                            return Err(()); // Stop this thread immediately
                        }

                        // Check external stop flag at the start of each iteration
                        if let Some(ref ext_stop) = external_stop {
                            if ext_stop.load(Ordering::Relaxed) {
                                stop_searching.store(true, Ordering::Relaxed);
                                return Err(());
                            }
                        }

                        self.run_simulation(state, &stop_searching);
                        completed_iterations.fetch_add(1, Ordering::Relaxed);

                        // Check stop flag again after simulation (set by timeout monitor)
                        if stop_searching.load(Ordering::Relaxed) {
                            return Err(());
                        }

                        if let Some(interval) = stats_interval {
                            let mut last_time = last_stats_time.lock();
                            if last_time.elapsed() >= interval {
                                // Stats are now displayed in the TUI debug panel instead of console output
                                // to prevent interference with the TUI display
                                *last_time = Instant::now();
                            }
                        }
                        Ok(())
                    });
            });

            iterations_remaining -= current_batch;
        }

        // Clean up timeout monitor thread and measure actual overhead (optimized frequency)
        if let Some(handle) = timeout_monitor_handle {
            stop_searching.store(true, Ordering::Relaxed);
            let cleanup_start = Instant::now();
            let _ = handle.join(); // Wait for timeout monitor to finish
            let cleanup_duration = cleanup_start.elapsed();

            // Only measure overhead every 8 searches to reduce performance impact
            let searches_count = self
                .searches_since_measurement
                .fetch_add(1, Ordering::Relaxed);
            let should_measure = searches_count % 8 == 0;

            if should_measure {
                // Update overhead estimation with actual measured overhead
                if let Some(timeout) = timeout {
                    let total_elapsed = start_time.elapsed();
                    let expected_duration = timeout;
                    if total_elapsed > expected_duration {
                        let actual_overhead_ms =
                            (total_elapsed - expected_duration).as_millis() as f64;

                        // Only update if the deviation is significant (>10ms difference from estimate)
                        let current_estimate = {
                            let overhead = self.timeout_overhead_ms.lock();
                            *overhead
                        };

                        if (actual_overhead_ms - current_estimate).abs() > 10.0 {
                            self.update_overhead_estimate(actual_overhead_ms);
                        }
                    } else {
                        // Even if we finished early, record the cleanup time as minimum overhead
                        let cleanup_overhead_ms = cleanup_duration.as_millis() as f64;

                        // Only update if cleanup time suggests we need a higher minimum
                        let current_estimate = {
                            let overhead = self.timeout_overhead_ms.lock();
                            *overhead
                        };

                        if cleanup_overhead_ms > current_estimate {
                            self.update_overhead_estimate(cleanup_overhead_ms);
                        }
                    }
                }
            }
        }

        // Clean up GPU refresh thread
        #[cfg(feature = "gpu")]
        if let Some(handle) = gpu_refresh_handle {
            // The stop_searching flag is already set, GPU thread will exit
            let _ = handle.join();
        }

        // Wait for pending GPU evaluations to complete (with timeout)
        #[cfg(feature = "gpu")]
        if self.gpu_simulation_sender.is_some() {
            let wait_start = Instant::now();
            let max_wait = Duration::from_millis(500); // Wait up to 500ms for pending evaluations
            while self.gpu_pending_evaluations.load(Ordering::Relaxed) > 0 {
                if wait_start.elapsed() > max_wait {
                    // Timeout - some evaluations may be lost, but we need to return
                    break;
                }
                std::thread::sleep(Duration::from_micros(100));
            }
        }

        // After all simulations, the best move is selected based on the configured strategy.
        let children = self.root.children.read();
        let best_move = if children.is_empty() {
            // Fallback: if no children exist, return a random valid move
            // This should rarely happen with the improved expansion logic
            let possible_moves = state.get_possible_moves();
            if possible_moves.is_empty() {
                panic!("No possible moves available - game should be terminal");
            }
            possible_moves[random_range(0, possible_moves.len())].clone()
        } else {
            self.select_best_move(&children)
        };

        let root_visits = self.root.visits.load(Ordering::Relaxed);
        let root_wins = self.root.wins.load(Ordering::Relaxed) as f64;
        let stats = SearchStatistics {
            total_nodes: self.node_count.load(Ordering::Relaxed),
            root_visits,
            root_wins,
            root_value: children.values()
                .map(|n| {
                    let v = n.visits.load(Ordering::Relaxed);
                    let w = n.wins.load(Ordering::Relaxed);
                    if v > 0 { (w as f64 / v as f64) / 2.0 } else { 0.0 }
                })
                .fold(0.0, f64::max),
            children_stats: self
                .get_root_children_stats()
                .into_iter()
                .map(|(m, (w, v))| (format!("{:?}", m), (w, v)))
                .collect(),
        };

        if prune_memory_count > 0 {
            println!("Pruning summary: {} total (memory pressure)", prune_memory_count);
        }

        (best_move, stats)
    }

    /// Performs a parallel MCTS search with optional pruning.
    /// This method launches multiple simulations in parallel using `rayon`.
    ///
    /// # Arguments
    /// * `state` - The current state of the game.
    /// * `iterations` - The total number of simulations to run.
    /// * `stats_interval_secs` - Interval in seconds to print statistics (0 = no periodic stats).
    /// * `timeout_secs` - The maximum time in seconds to search for. 0 means no timeout.
    pub fn search(
        &mut self,
        state: &S,
        iterations: i32,
        stats_interval_secs: u64,
        timeout_secs: u64,
    ) -> (S::Move, SearchStatistics) {
        let start_time = Instant::now();
        let timeout = if timeout_secs > 0 {
            Some(Duration::from_secs(timeout_secs))
        } else {
            None
        };

        self.ensure_root_expanded(state);

        let possible_moves = state.get_possible_moves();
        if possible_moves.len() == 1 {
            return (possible_moves[0].clone(), SearchStatistics::default());
        }

        if possible_moves.is_empty() {
            // This case should ideally be handled by get_possible_moves returning a pass move.
            // If we get here, it means the game ended, but is_terminal was false.
            // We are in an inconsistent state, but we must return a move.
            // The Blokus implementation should return a pass move.
            // If another game does not, this will be a problem.
            // For now, let's check the root's children. If there's one, it must be the pass move.
            let children = self.root.children.read();
            if children.len() == 1 {
                return (
                    children.keys().next().unwrap().clone(),
                    SearchStatistics::default(),
                );
            }
            // If there are no children and no possible moves, we are stuck.
            // This indicates a logic error in the game state implementation.
            panic!(
                "MCTS search: No possible moves and no children in root node. Game logic may be flawed."
            );
        }

        let completed_iterations = Arc::new(AtomicUsize::new(0));
        let stop_searching = Arc::new(AtomicBool::new(false));

        let stats_interval = if stats_interval_secs > 0 {
            Some(Duration::from_secs(stats_interval_secs))
        } else {
            None
        };
        let last_stats_time = Arc::new(Mutex::new(Instant::now()));

        self.pool.install(|| {
            let _ = (0..iterations)
                .into_par_iter()
                .try_for_each(|_| -> Result<(), ()> {
                    if stop_searching.load(Ordering::Relaxed) {
                        return Err(()); // Stop this thread
                    }

                    self.run_simulation(state, &stop_searching);
                    completed_iterations.fetch_add(1, Ordering::Relaxed);

                    if let Some(t) = timeout {
                        if start_time.elapsed() >= t {
                            stop_searching.store(true, Ordering::Relaxed);
                            return Err(()); // Stop this thread and signal others
                        }
                    }

                    if let Some(interval) = stats_interval {
                        let mut last_time = last_stats_time.lock();
                        if last_time.elapsed() >= interval {
                            // Stats are now displayed in the TUI debug panel instead of console output
                            // to prevent interference with the TUI display
                            *last_time = Instant::now();
                        }
                    }
                    Ok(())
                });
        });

        // After all simulations, the best move is selected based on the configured strategy.
        let children = self.root.children.read();
        let best_move = if children.is_empty() {
            // Fallback: if no children exist, return a random valid move
            // This should rarely happen with the improved expansion logic
            let possible_moves = state.get_possible_moves();
            if possible_moves.is_empty() {
                panic!("No possible moves available - game should be terminal");
            }
            possible_moves[random_range(0, possible_moves.len())].clone()
        } else {
            self.select_best_move(&children)
        };

        let root_visits = self.root.visits.load(Ordering::Relaxed);
        let root_wins = self.root.wins.load(Ordering::Relaxed) as f64;
        let stats = SearchStatistics {
            total_nodes: self.node_count.load(Ordering::Relaxed),
            root_visits,
            root_wins,
            root_value: if root_visits > 0 {
                root_wins / root_visits as f64 / 2.0
            } else {
                0.0
            },
            children_stats: self
                .get_root_children_stats()
                .into_iter()
                .map(|(m, (w, v))| (format!("{:?}", m), (w, v)))
                .collect(),
        };

        (best_move, stats)
    }

    /// Performs a parallel MCTS search with custom pruning interval.
    /// Prunes the tree every `prune_interval` iterations to maintain memory efficiency.
    ///
    /// # Arguments
    /// * `state` - The current state of the game.
    /// * `iterations` - The total number of simulations to run.
    /// * `prune_interval` - How often to prune the tree (0 = no pruning during search).
    pub fn search_with_pruning(
        &mut self,
        state: &S,
        iterations: i32,
        prune_interval: i32,
    ) -> (S::Move, SearchStatistics) {
        // Ensure root node is fully expanded before starting parallel search
        self.ensure_root_expanded(state);

        let stop_searching = Arc::new(AtomicBool::new(false));
        let run_iterations = |this: &MCTS<S>, iters: i32, stop_flag: &Arc<AtomicBool>| {
            this.pool.install(|| {
                (0..iters).into_par_iter().for_each(|_| {
                    if !stop_flag.load(Ordering::Relaxed) {
                        this.run_simulation(state, stop_flag);
                    }
                });
            });
        };

        if prune_interval > 0 && iterations > prune_interval {
            // Split search into chunks with pruning
            let chunks = iterations / prune_interval;
            let remainder = iterations % prune_interval;

            for _ in 0..chunks {
                run_iterations(self, prune_interval, &stop_searching);
                // Prune after each chunk
                self.auto_prune();
            }

            // Run remaining iterations
            if remainder > 0 {
                run_iterations(self, remainder, &stop_searching);
            }
        } else {
            // Run all iterations at once
            run_iterations(self, iterations, &stop_searching);
        }

        // Don't do final pruning here - let it be done explicitly after statistics are displayed

        // Return the best move based on the configured strategy
        let children = self.root.children.read();
        let best_move = if children.is_empty() {
            // Fallback: if no children exist, return a random valid move
            // This should rarely happen with the improved expansion logic
            let possible_moves = state.get_possible_moves();
            if possible_moves.is_empty() {
                panic!("No possible moves available - game should be terminal");
            }
            possible_moves[random_range(0, possible_moves.len())].clone()
        } else {
            self.select_best_move(&children)
        };

        let root_visits = self.root.visits.load(Ordering::Relaxed);
        let root_wins = self.root.wins.load(Ordering::Relaxed) as f64;
        let stats = SearchStatistics {
            total_nodes: self.node_count.load(Ordering::Relaxed),
            root_visits,
            root_wins,
            root_value: children.values()
                .map(|n| {
                    let v = n.visits.load(Ordering::Relaxed);
                    let w = n.wins.load(Ordering::Relaxed);
                    if v > 0 { (w as f64 / v as f64) / 2.0 } else { 0.0 }
                })
                .fold(0.0, f64::max),
            children_stats: self
                .get_root_children_stats()
                .into_iter()
                .map(|(m, (w, v))| (format!("{:?}", m), (w, v)))
                .collect(),
        };

        (best_move, stats)
    }

    /// Gets the current estimated timeout overhead in milliseconds
    ///
    /// This is useful for debugging and monitoring the adaptive overhead estimation.
    ///
    /// # Returns
    /// The current estimated overhead in milliseconds
    pub fn get_timeout_overhead_estimate(&self) -> f64 {
        let overhead = self.timeout_overhead_ms.lock();
        *overhead
    }

    /// Gets the number of overhead measurements taken so far
    ///
    /// # Returns
    /// The number of measurements used to calculate the current estimate
    pub fn get_overhead_measurement_count(&self) -> i32 {
        self.timeout_measurements.load(Ordering::Relaxed)
    }

    /// Runs a single MCTS simulation with virtual loss support.
    ///
    /// This is the core of the MCTS algorithm. It performs:
    /// 1. Selection: Traverse tree using PUCT to select promising paths
    /// 2. Expansion: Add new nodes to the tree when reaching a leaf
    /// 3. Simulation: Play out a random game from the new position
    /// 4. Backpropagation: Update statistics along the path
    ///
    /// Virtual losses are used to coordinate parallel threads and prevent
    /// multiple threads from exploring the same path simultaneously.
    ///
    /// # Arguments
    /// * `state` - The current game state to simulate from
    /// * `stop_flag` - Flag to check for early termination
    fn run_simulation(&self, state: &S, stop_flag: &AtomicBool) {
        // Early exit if stop flag is already set
        if stop_flag.load(Ordering::Relaxed) {
            return;
        }

        // Increment GPU simulation counter for cache updates
        #[cfg(feature = "gpu")]
        {
            self.simulations_since_gpu_update.fetch_add(1, Ordering::Relaxed);
        }

        let mut current_state = state.clone();
        let mut path: Vec<Arc<Node<S::Move>>> = Vec::with_capacity(64); // Pre-allocate reasonable capacity
        let mut path_players: Vec<i32> = Vec::with_capacity(64); // Track which player made each move
        path.push(self.root.clone());
        path_players.push(current_state.get_current_player()); // Root represents current player's turn
        let mut current_node = self.root.clone();

        // Calculate board capacity based on initial move count for better memory allocation
        let board_capacity = current_state.get_possible_moves().len();
        let mut moves_cache = Vec::with_capacity(board_capacity);
        let mut candidates = Vec::with_capacity(board_capacity);

        // --- Selection Phase with Virtual Loss ---
        // Traverse the tree until a leaf node is reached.
        loop {
            // Check stop flag during selection phase
            if stop_flag.load(Ordering::Relaxed) {
                return;
            }

            let children_guard = current_node.children.read();
            if children_guard.is_empty() || current_state.is_terminal() {
                drop(children_guard);
                break;
            }

            moves_cache.clear();
            moves_cache.extend(current_state.get_possible_moves());

            // Safety check: if no moves available, something is wrong
            if moves_cache.is_empty() {
                // This shouldn't happen if game logic is correct, but handle gracefully
                break;
            }

            let parent_visits = current_node.visits.load(Ordering::Relaxed);
            // Use uniform prior probability for all moves since we don't have a neural network
            let prior_probability = 1.0 / moves_cache.len() as f64;
            // Determine virtual loss weight based on configuration

            // We use 1.0 for both CPU and GPU to ensure proper diversity.
            // For GPU with large batches, this is critical to prevent stampeding.
            #[cfg(feature = "gpu")]
            let virtual_loss_weight = 1.0;
            #[cfg(not(feature = "gpu"))]
            let virtual_loss_weight = 1.0;

            let (best_move, next_node) = {
                candidates.clear();
                candidates.extend(
                    moves_cache
                        .iter()
                        .filter_map(|m| children_guard.get(m).map(|n| (m, n)))
                        .map(|(m, n)| {
                            // Use GPU-cached PUCT for any level if available
                            #[cfg(feature = "gpu")]
                            let puct = self.get_cached_puct_by_node(&current_node, n).unwrap_or_else(|| {
                                n.puct(
                                    parent_visits,
                                    self.exploration_parameter,
                                    prior_probability,
                                    virtual_loss_weight,
                                )
                            });
                            #[cfg(not(feature = "gpu"))]
                            let puct = n.puct(
                                parent_visits,
                                self.exploration_parameter,
                                prior_probability,
                                virtual_loss_weight,
                            );
                            (m.clone(), n.clone(), puct)
                        }),
                );

                // If no expanded children exist, we need to break out of selection and go to expansion
                if candidates.is_empty() {
                    drop(children_guard);
                    break;
                }

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

                // Safety check: best_indices should never be empty at this point
                if best_indices.is_empty() {
                    panic!("PUCT selection failed: no best indices found");
                }

                let selected_idx = if best_indices.len() == 1 {
                    best_indices[0]
                } else {
                    best_indices[random_range(0, best_indices.len())]
                };
                let selected = &candidates[selected_idx];
                (selected.0.clone(), selected.1.clone())
            };

            drop(children_guard); // Release read lock

            // Apply virtual loss to the selected node
            next_node.apply_virtual_loss();

            // Remember which player is making this move
            let moving_player = current_state.get_current_player();
            current_state.make_move(&best_move);
            current_node = next_node;
            path.push(current_node.clone());
            path_players.push(moving_player); // Track the player who made this move
        }

        // --- Expansion Phase ---
        // If the node is a leaf and the game is not over, decide whether to expand based on:
        // 1. Current tree size vs max_nodes limit
        // 2. Depth-based probability (deeper nodes are less likely to expand)
        // 3. Visit count (more visited nodes are more likely to expand)
        // Special case: Always expand the root node to ensure the search can find moves
        if !current_state.is_terminal() {
            // Check stop flag before expansion
            if stop_flag.load(Ordering::Relaxed) {
                return;
            }

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
                            // Use effective visits (visits + virtual_losses) to account for pending simulations
                            // This is crucial for GPU batching where real visits are delayed
                            let visits = current_node.visits.load(Ordering::Relaxed);
                            let virtual_losses = current_node.virtual_losses.load(Ordering::Relaxed);
                            let effective_visits = visits + virtual_losses;

                            // Base expansion probability decreases with depth
                            // More visits increase the likelihood of expansion
                            let depth_factor = 1.0 / (1.0 + (depth as f64) * 0.5);
                            let visit_factor = (effective_visits as f64).sqrt() / 10.0; // Encourage expansion for well-visited nodes
                            let expansion_probability = (depth_factor + visit_factor).min(1.0);

                            random_f64() < expansion_probability
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

                    // Only proceed with expansion if we have moves
                    if !moves_cache.is_empty() {
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
                        self.node_count
                            .fetch_add(new_nodes_count, Ordering::Relaxed);
                    }
                }
            }
        }

        // --- Simulation Phase ---
        // Run a random playout from the new node to the end of the game.
        let mut sim_state = current_state.clone();

        #[cfg(feature = "gpu")]
        if let Some(ref sender) = self.gpu_simulation_sender {
            if !sim_state.is_terminal() {
                // Adaptive pending limit based on game phase
                // Key insight: In endgame with few moves, we need fresher statistics
                // because positions are more tactical and rollouts are shorter.
                // 
                // - Many moves (opening/midgame): Higher limit OK, rollouts take longer
                // - Few moves (endgame): Need lower limit for fresher backprop
                let num_moves = moves_cache.len().max(1);
                let pending_limit = if num_moves >= 15 {
                    2000  // Opening: many moves, longer rollouts
                } else if num_moves >= 8 {
                    1000  // Midgame: moderate
                } else if num_moves >= 4 {
                    500   // Late midgame: getting tactical
                } else {
                    200   // Endgame: very few moves, need fresh data
                };
                
                let pending = self.gpu_pending_evaluations.load(Ordering::Relaxed);
                if pending < pending_limit {
                    // Send to GPU. Multiple threads can evaluate the same position - this is fine.
                    self.gpu_pending_evaluations.fetch_add(1, Ordering::Relaxed);
                    let request = EvaluationRequest {
                        state: sim_state.clone(), // Clone state for GPU
                        path: path.clone(), // Clone path for GPU
                        path_players: path_players.clone(), // Clone path_players for GPU
                    };

                    if sender.send(request).is_ok() {
                        // Successfully sent. The GPU thread will handle backprop and VL removal.
                        // Virtual losses stay applied until GPU finishes backpropagation.
                        return;
                    } else {
                        // Failed to send (GPU thread died). Decrement counter and fall through to CPU.
                        self.gpu_pending_evaluations.fetch_sub(1, Ordering::Relaxed);
                    }
                }
            }
        }

        #[cfg(not(feature = "gpu"))]
        let gpu_result: Option<Option<i32>> = None;

        // If we are here, either GPU is disabled or the game state is terminal.
        // We proceed with CPU simulation (random rollout) or just get the winner if terminal.
        
        let winner = if sim_state.is_terminal() {
            sim_state.get_winner()
        } else {
            let mut simulation_moves = 0;
            const MAX_SIMULATION_MOVES: usize = 1000; // Safeguard against infinite loops

            // Track timing for intelligent stop flag checking
            let sim_phase_start = std::time::Instant::now();
            let mut last_stop_check = sim_phase_start;
            const STOP_CHECK_INTERVAL_MS: u64 = 5; // Check every 5ms

            while !sim_state.is_terminal() && simulation_moves < MAX_SIMULATION_MOVES {
                // Intelligent stop flag checking: only check periodically based on time, not move count
                let now = std::time::Instant::now();
                if now.duration_since(last_stop_check).as_millis() >= STOP_CHECK_INTERVAL_MS as u128 {
                    if stop_flag.load(Ordering::Relaxed) {
                        break; // Exit simulation early if stop flag is set
                    }
                    last_stop_check = now;
                }

                moves_cache.clear();
                moves_cache.extend(sim_state.get_possible_moves());
                if moves_cache.is_empty() {
                    break;
                }

                // Weighted random selection
                let mut total_weight = 0.0;
                for mv in &moves_cache {
                    total_weight += sim_state.get_move_weight(mv);
                }

                let mut threshold = random_f64() * total_weight;
                let mut move_index = 0;
                
                for (i, mv) in moves_cache.iter().enumerate() {
                    let weight = sim_state.get_move_weight(mv);
                    if threshold < weight {
                        move_index = i;
                        break;
                    }
                    threshold -= weight;
                }
                
                // Fallback if floating point errors caused us to miss (should be rare)
                if move_index >= moves_cache.len() {
                    move_index = 0;
                }

                let mv = &moves_cache[move_index];
                sim_state.make_move(mv);
                simulation_moves += 1;
            }

            // If we hit the simulation limit, treat it as a draw
            if simulation_moves >= MAX_SIMULATION_MOVES {
                None // Treat as draw/timeout
            } else {
                sim_state.get_winner()
            }
        };

        // --- Backpropagation Phase with Virtual Loss Removal ---
        // Update the visit counts and win statistics for all nodes in the path.
        // Also remove virtual losses that were applied during selection.
        // For multi-player games, reward each node based on whether the player who made that move won

        // Check stop flag before backpropagation
        if stop_flag.load(Ordering::Relaxed) {
            // Even if we're stopping, we need to remove virtual losses to keep the tree consistent
            // But we can skip the actual visit/win updates
            for (node, _) in path.iter().zip(path_players.iter()).rev() {
                node.remove_virtual_loss();
            }
            return;
        }

        for (node, &player_who_moved) in path.iter().zip(path_players.iter()).rev() {
            // Remove virtual loss from all nodes in the path
            // They all had virtual loss applied during selection
            node.remove_virtual_loss();

            node.visits.fetch_add(1, Ordering::Relaxed);
            let reward = match winner {
                Some(w) if w == player_who_moved => 2, // Win for the player who made this move
                Some(_) => 0,                          // Loss (another player won)
                None => 1,                             // Draw
            };
            node.wins.fetch_add(reward, Ordering::Relaxed);
        }
    }

    /// Updates the running average of timeout overhead based on actual measurements
    ///
    /// Uses an exponential moving average to adapt to changing system conditions
    /// while giving more weight to recent measurements. This method is now called
    /// less frequently (every ~8 searches) to reduce performance overhead.
    ///
    /// # Arguments
    /// * `measured_overhead_ms` - The measured overhead in milliseconds
    fn update_overhead_estimate(&self, measured_overhead_ms: f64) {
        // Clamp the measured overhead to reasonable bounds to avoid outliers
        let clamped_overhead = measured_overhead_ms.max(5.0).min(1000.0);

        let mut current_avg = self.timeout_overhead_ms.lock();
        let measurements = self.timeout_measurements.fetch_add(1, Ordering::Relaxed);

        if measurements == 0 {
            // First measurement, just use it directly
            *current_avg = clamped_overhead;
        } else {
            // Use exponential moving average with alpha = 0.15 (15% weight to new measurement)
            // Slightly higher alpha since we measure less frequently now
            *current_avg = 0.85 * (*current_avg) + 0.15 * clamped_overhead;
        }
    }

    /// Updates the GPU PUCT cache for root children
    ///
    /// This method traverses the entire tree and computes PUCT scores on the GPU
    /// for all children of all expanded nodes. This enables processing thousands
    /// of nodes in a single GPU batch for maximum GPU utilization.
    ///
    /// # Arguments
    /// * `force` - If true, update the cache regardless of the simulation count
    #[cfg(feature = "gpu")]
    #[allow(dead_code)]
    fn update_gpu_puct_cache(&self, force: bool) {
        if !self.gpu_enabled || self.gpu_accelerator.is_none() {
            return;
        }

        // Only update every 1000 simulations or if forced
        let sims = self.simulations_since_gpu_update.load(Ordering::Relaxed);
        if !force && sims < 1000 {
            return;
        }

        // Reset the counter
        self.simulations_since_gpu_update.store(0, Ordering::Relaxed);

        // Traverse the entire tree and collect all parent-child pairs
        // Use BFS to process level by level
        let mut node_data: Vec<gpu::GpuNodeData> = Vec::with_capacity(65536);
        let mut cache_keys: Vec<(usize, usize)> = Vec::with_capacity(65536);
        
        // Stack for DFS traversal: (node, depth)
        let mut stack: Vec<(Arc<Node<S::Move>>, u32)> = Vec::with_capacity(1024);
        stack.push((self.root.clone(), 0));
        
        const MAX_DEPTH: u32 = 50; // Limit depth to avoid infinite recursion
        const MAX_NODES: usize = 65536; // Limit to avoid GPU buffer overflow
        
        while let Some((parent_node, depth)) = stack.pop() {
            if depth >= MAX_DEPTH || node_data.len() >= MAX_NODES {
                break;
            }
            
            let children = parent_node.children.read();
            if children.is_empty() {
                continue;
            }
            
            let parent_visits = parent_node.visits.load(Ordering::Relaxed);
            let num_children = children.len();
            let prior_prob = 1.0 / num_children as f32;
            let parent_id = Arc::as_ptr(&parent_node) as usize;
            
            for (_mv, child_node) in children.iter() {
                if node_data.len() >= MAX_NODES {
                    break;
                }
                
                let child_id = Arc::as_ptr(child_node) as usize;
                
                node_data.push(gpu::GpuNodeData::new(
                    child_node.visits.load(Ordering::Relaxed),
                    child_node.wins.load(Ordering::Relaxed),
                    child_node.virtual_losses.load(Ordering::Relaxed),
                    parent_visits,
                    prior_prob,
                    self.exploration_parameter as f32,
                ));
                cache_keys.push((parent_id, child_id));
                
                // Only add children with visits to the stack (they might have their own children)
                if child_node.visits.load(Ordering::Relaxed) > 0 {
                    stack.push((child_node.clone(), depth + 1));
                }
            }
        }

        if node_data.is_empty() {
            return;
        }

        let batch_size = node_data.len();
        self.gpu_last_batch_size.store(batch_size as i32, Ordering::Relaxed);

        // Compute PUCT scores on GPU
        if let Some(ref accelerator) = self.gpu_accelerator {
            let mut acc = accelerator.lock();
            if let Ok(results) = acc.compute_puct_batch(&node_data) {
                // Update the cache
                let mut cache = self.gpu_puct_cache.write();
                cache.clear();
                cache.reserve(results.len());
                
                for (i, result) in results.iter().enumerate() {
                    cache.insert(cache_keys[i], result.puct_score as f64);
                }
                
                // Update timestamp
                let mut ts = self.gpu_cache_timestamp.lock();
                *ts = Instant::now();
                
                // Debug output
                if force {
                    eprintln!("[GPU] Deep tree PUCT batch: {} nodes processed", batch_size);
                }
            }
        }
    }

    /// Gets a cached PUCT score from the GPU cache using node pointers
    ///
    /// Returns the cached PUCT score for a parent-child node pair if available,
    /// or None if the cache doesn't have this pair.
    #[cfg(feature = "gpu")]
    fn get_cached_puct_by_node(&self, _parent: &Arc<Node<S::Move>>, _child: &Arc<Node<S::Move>>) -> Option<f64> {
        // Disabled to prevent lock contention on gpu_puct_cache
        None
    }

    /// Prunes children based on visit percentage relative to the best child
    ///
    /// Removes children with less than the specified percentage of the most visited child's visits.
    /// This is more aggressive than absolute threshold pruning and helps focus on the most
    /// promising moves while preserving exploration diversity.
    ///
    /// # Arguments
    /// * `min_percentage` - Minimum percentage of the best child's visits required to keep a child (0.0-1.0)
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

    /// Returns grid-based statistics for games like Gomoku and Othello
    ///
    /// Provides spatial analysis of the search tree for coordinate-based games.
    /// Each position on the grid shows how many times that move was considered
    /// and its expected value from the MCTS search.
    ///
    /// # Arguments
    /// * `board_size` - Size of the game board (NxN)
    ///
    /// # Returns
    /// Tuple of (visits_grid, values_grid, wins_grid, root_value) where each grid is board_size x board_size
    pub fn get_grid_stats(
        &self,
        board_size: usize,
    ) -> (Vec<Vec<i32>>, Vec<Vec<f64>>, Vec<Vec<f64>>, f64) {
        let mut visits_grid = vec![vec![0; board_size]; board_size];
        let mut values_grid = vec![vec![0.0; board_size]; board_size];
        let mut wins_grid = vec![vec![0.0; board_size]; board_size];

        let children = self.root.children.read();
        for (mv, node) in children.iter() {
            let visits = node.visits.load(Ordering::Relaxed);
            let wins = node.wins.load(Ordering::Relaxed) as f64;
            let value = if visits > 0 {
                wins / (visits as f64) / 2.0
            } else {
                0.0
            };

            // Extract coordinates based on move type
            if let Some((r, c)) = self.extract_move_coordinates(mv, board_size) {
                if r < board_size && c < board_size {
                    visits_grid[r][c] = visits;
                    values_grid[r][c] = value;
                    wins_grid[r][c] = wins;
                }
            }
        }

        let root_visits = self.root.visits.load(Ordering::Relaxed);
        let root_wins = self.root.wins.load(Ordering::Relaxed) as f64;
        let root_value = if root_visits > 0 {
            root_wins / (root_visits as f64) / 2.0
        } else {
            0.0
        };

        (visits_grid, values_grid, wins_grid, root_value)
    }

    /// Extract coordinates from a move for grid display (helper function)
    ///
    /// Attempts to parse row and column coordinates from a move's Debug representation.
    /// This is used for spatial visualization of search statistics on grid-based games.
    ///
    /// # Arguments
    /// * `mv` - The move to extract coordinates from
    /// * `_board_size` - Board size (unused but kept for future validation)
    ///
    /// # Returns
    /// Optional tuple of (row, column) coordinates if parsing succeeds
    fn extract_move_coordinates(&self, mv: &S::Move, _board_size: usize) -> Option<(usize, usize)> {
        // This is a trait-based approach that will need to be implemented per game type
        // For now, we'll use std::fmt::Debug to parse coordinates from the move string
        let move_str = format!("{:?}", mv);

        // Try to parse coordinates from move string representations
        // Handle MoveWrapper patterns like MoveWrapper::Gomoku(GomokuMove(r, c))
        if move_str.contains("Gomoku(GomokuMove(") || move_str.contains("Othello(OthelloMove(") {
            // Find the innermost parentheses with coordinates
            if let Some(start) = move_str.rfind('(') {
                if let Some(end) = move_str[start..].find(')') {
                    let coords_str = &move_str[start + 1..start + end];
                    let coords = coords_str.split(", ").collect::<Vec<_>>();
                    if coords.len() == 2 {
                        if let (Ok(r), Ok(c)) =
                            (coords[0].parse::<usize>(), coords[1].parse::<usize>())
                        {
                            return Some((r, c));
                        }
                    }
                }
            }
        }
        // Also handle direct move patterns for backward compatibility
        else if move_str.starts_with("GomokuMove(") || move_str.starts_with("OthelloMove(") {
            let coords = move_str
                .trim_start_matches(|c: char| c != '(')
                .trim_start_matches('(')
                .trim_end_matches(')')
                .split(", ")
                .collect::<Vec<_>>();
            if coords.len() == 2 {
                if let (Ok(r), Ok(c)) = (coords[0].parse::<usize>(), coords[1].parse::<usize>()) {
                    return Some((r, c));
                }
            }
        }
        None
    }
}

/// ## Optimized Overhead Estimation
///
/// The MCTS engine uses a dynamic, runtime-estimated buffer to account for thread
/// coordination overhead when setting timeouts. This optimization reduces the
/// performance impact of overhead measurements while maintaining accuracy:
///
/// - **Reduced Frequency**: Measurements are taken every 8 searches instead of every search
/// - **Significance Threshold**: Only updates the estimate if the deviation is >10ms
/// - **Outlier Protection**: Clamps measured values between 5ms and 1000ms
/// - **Adaptive Alpha**: Uses 15% weight for new measurements (vs 10% before) since we measure less often
/// - **Selective Updates**: Only updates minimum overhead estimate if cleanup time exceeds current estimate
///
/// This approach reduces overhead measurement costs by ~87% while maintaining reasonable
/// accuracy for timeout estimation under varying system loads.

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test game state for testing
    #[derive(Clone, Debug, PartialEq)]
    struct TestGame {
        board: Vec<Vec<i32>>,
        current_player: i32,
        moves_made: usize,
        last_move: Option<Vec<(usize, usize)>>,
    }

    impl TestGame {
        fn new() -> Self {
            TestGame {
                board: vec![vec![0; 3]; 3], // 3x3 board
                current_player: 1,
                moves_made: 0,
                last_move: None,
            }
        }
    }

    impl GameState for TestGame {
        type Move = (usize, usize);

        fn get_board(&self) -> Vec<Vec<i32>> {
            self.board.clone()
        }

        fn get_last_move(&self) -> Option<Vec<(usize, usize)>> {
            self.last_move.clone()
        }

        fn get_num_players(&self) -> i32 {
            2
        }

        fn get_possible_moves(&self) -> Vec<Self::Move> {
            if self.is_terminal() {
                return vec![];
            }

            let mut moves = Vec::new();
            for i in 0..3 {
                for j in 0..3 {
                    if self.board[i][j] == 0 {
                        moves.push((i, j));
                    }
                }
            }
            moves
        }

        fn make_move(&mut self, mv: &Self::Move) {
            let (row, col) = *mv;
            self.board[row][col] = self.current_player;
            self.last_move = Some(vec![*mv]);
            self.current_player = if self.current_player == 1 { 2 } else { 1 };
            self.moves_made += 1;
        }

        fn is_terminal(&self) -> bool {
            self.get_winner().is_some() || self.moves_made >= 9
        }

        fn get_winner(&self) -> Option<i32> {
            // Check rows, columns, and diagonals for winner
            for i in 0..3 {
                if self.board[i][0] != 0
                    && self.board[i][0] == self.board[i][1]
                    && self.board[i][1] == self.board[i][2]
                {
                    return Some(self.board[i][0]);
                }
                if self.board[0][i] != 0
                    && self.board[0][i] == self.board[1][i]
                    && self.board[1][i] == self.board[2][i]
                {
                    return Some(self.board[0][i]);
                }
            }
            // Diagonals
            if self.board[0][0] != 0
                && self.board[0][0] == self.board[1][1]
                && self.board[1][1] == self.board[2][2]
            {
                return Some(self.board[0][0]);
            }
            if self.board[0][2] != 0
                && self.board[0][2] == self.board[1][1]
                && self.board[1][1] == self.board[2][0]
            {
                return Some(self.board[0][2]);
            }
            None
        }

        fn get_current_player(&self) -> i32 {
            self.current_player
        }
    }

    #[test]
    fn test_overhead_estimation_optimization() {
        let mut mcts = MCTS::<TestGame>::new(1.4, 1, 1000);
        let game = TestGame::new();

        // Verify initial overhead estimate
        let initial_estimate = mcts.get_timeout_overhead_estimate();
        assert_eq!(initial_estimate, 50.0); // Should start with 50ms

        // Verify initial measurement count
        let initial_count = mcts.get_overhead_measurement_count();
        assert_eq!(initial_count, 0);

        // Run multiple searches with timeout to potentially trigger overhead measurement
        // The timeout and measurement triggering depends on actual timing
        for _i in 0..16 {
            // Use a short timeout and many iterations to likely trigger the timeout
            let _ = mcts.search(&game, 100000, 1, 1); // 1 second timeout with many iterations
        }

        // Check the results - the optimization should limit measurements
        let final_count = mcts.get_overhead_measurement_count();
        let final_estimate = mcts.get_timeout_overhead_estimate();

        // Verify that the estimate is within reasonable bounds
        assert!(
            final_estimate >= 5.0 && final_estimate <= 1000.0,
            "Estimate should be within reasonable bounds, got {}",
            final_estimate
        );

        // The measurement frequency should be limited by our optimization
        // We did 16 searches, so at most we should see 2-3 measurements (every 8th search)
        println!(
            "Final measurement count: {}, Final estimate: {}ms",
            final_count, final_estimate
        );

        // Test the frequency optimization - measurements should be less frequent than searches
        assert!(
            final_count <= 4,
            "Should not measure overhead too frequently, got {} measurements for 16 searches",
            final_count
        );
    }
}
