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
//!
//! // Your game must implement GameState
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
    fn get_board(&self) -> &Vec<Vec<i32>>;

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
    /// Removes children that haven't been visited enough times.
    /// This helps control memory usage in long-running searches.
    /// The pruned nodes are returned so they can be recycled.
    ///
    /// # Arguments
    /// * `min_visits` - Minimum number of visits required to keep a child
    ///
    /// # Returns
    /// Vector of pruned nodes that can be recycled
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
            let exploration_term =
                exploration_parameter * prior_probability * (parent_visits as f64).sqrt()
                    / (1.0 + effective_visits_f);
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
        let node_count_clone = node_count.clone();
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
                    
                    // Track execution time to adapt batching strategy
                    let mut last_execution_time = Duration::from_micros(0);
                    
                    loop {
                        let first = match rx.recv() {
                            Ok(req) => req,
                            Err(_) => break,
                        };
                        
                        batch_requests.push(first);

                        // Adaptive batching: if GPU is slow (>2ms), wait up to 500us to fill larger batches
                        let heavy_load = last_execution_time.as_micros() > 2000;
                        let target_batch = if heavy_load { 
                            (max_batch_size / 2).max(64).min(2048) 
                        } else { 
                            (max_batch_size / 4).max(64).min(512) 
                        };
                        
                        let deadline = if heavy_load { 
                            Some(Instant::now() + Duration::from_micros(500)) 
                        } else { 
                            None 
                        };

                        while batch_requests.len() < target_batch {
                            match rx.try_recv() {
                                Ok(req) => batch_requests.push(req),
                                Err(_) => {
                                    match deadline {
                                        Some(d) => {
                                            if batch_requests.len() >= 64 || Instant::now() >= d { break; }
                                            std::thread::yield_now();
                                        }
                                        None => break,
                                    }
                                }
                            }
                        }
                        
                        if batch_requests.is_empty() { continue; }
                        
                        let mut flat_data = Vec::new();
                        let mut gpu_indices = Vec::new();
                        let mut cpu_indices = Vec::new();
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
                                        seed: base_seed,
                                    });
                                }
                                flat_data.extend(data);
                                gpu_indices.push(i);
                            } else {
                                // This game doesn't support GPU simulation, needs CPU rollout
                                cpu_indices.push(i);
                            }
                        }
                        
                        let mut scores = vec![0.0f32; batch_requests.len()];

                        // GPU evaluation for supported games
                        let start_time = Instant::now();
                        if let Some(p) = params {
                            let mut acc = accelerator.lock();
                            if let Ok(gpu_scores) = acc.simulate_batch(&flat_data, p) {
                                for (idx, score) in gpu_indices.into_iter().zip(gpu_scores.into_iter()) {
                                    scores[idx] = score;
                                }
                            }
                        }
                        // Update execution time metric (only if GPU was actually used)
                        if params.is_some() {
                            last_execution_time = start_time.elapsed();
                        } else {
                            last_execution_time = Duration::from_micros(0);
                        }
                        
                        // CPU random rollout for games that don't support GPU simulation
                        // This ensures all games work, even without custom GPU shaders
                        for idx in cpu_indices {
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
                                    let move_index = random_range(0, moves_cache.len());
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
                        let requests: Vec<_> = batch_requests.drain(..).collect();
                        requests.into_par_iter().zip(scores.into_par_iter()).for_each(|(req, score)| {
                            let leaf_node = req.path.last().unwrap();
                            
                            // 1. Expand
                            if !req.state.is_terminal() {
                                // Check max_nodes to respect tree size limit
                                let current_nodes = node_count_clone.load(Ordering::Relaxed) as usize;
                                if current_nodes < max_nodes {
                                    let mut children_guard = leaf_node.children.write();
                                    if children_guard.is_empty() {
                                        let possible_moves = req.state.get_possible_moves();
                                        let new_depth = leaf_node.depth + 1;
                                        let mut new_nodes_count = 0;
                                        
                                        for mv in possible_moves {
                                            let new_node = Arc::new(Node {
                                                children: RwLock::new(HashMap::new()),
                                                visits: AtomicI32::new(0),
                                                wins: AtomicI32::new(0),
                                                virtual_losses: AtomicI32::new(0),
                                                depth: new_depth,
                                            });
                                            children_guard.insert(mv, new_node);
                                            new_nodes_count += 1;
                                        }
                                        node_count_clone.fetch_add(new_nodes_count, Ordering::Relaxed);
                                    }
                                }
                            }

                            // 2. Backpropagate
                            // Map score to [0, 1] win probability for current_player (at leaf)
                            let leaf_player = req.state.get_current_player();
                            
                            let win_prob = if score >= 4000.0 {
                                1.0
                            } else if score <= -4000.0 {
                                0.0
                            } else {
                                // Map heuristic score to win probability
                                // Use tanh to squash score into [-1, 1], then map to [0, 1]
                                // Adjusted scale: score of 500 (5 four-patterns) gives ~99% win probability
                                // score of 100 (1 four-pattern) gives ~76% win probability  
                                // score of 10 (1 three-pattern) gives ~54% win probability
                                0.5 + 0.5 * (score / 200.0).tanh() as f64
                            };

                            for (node, &player_who_moved) in req.path.iter().zip(req.path_players.iter()).rev() {
                                node.remove_virtual_loss();
                                node.visits.fetch_add(1, Ordering::Relaxed);
                                
                                // Calculate reward for this node's perspective
                                // If the player who made the move is the same as the one favored by the score,
                                // they get a higher reward.
                                let reward_val = if player_who_moved == leaf_player {
                                    2.0 * win_prob
                                } else {
                                    2.0 * (1.0 - win_prob)
                                };
                                
                                // Stochastic rounding to integer to preserve fractional value in expectation
                                let reward_int = reward_val as i32;
                                let reward_frac = reward_val - reward_int as f64;
                                let final_reward = reward_int + if random_f64() < reward_frac { 1 } else { 0 };
                                
                                node.wins.fetch_add(final_reward, Ordering::Relaxed);
                            }
                            
                            // Decrement pending evaluations counter
                            pending_evals_clone.fetch_sub(1, Ordering::Relaxed);
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
            gpu_accelerator,
            gpu_enabled,
            gpu_puct_cache: Arc::new(RwLock::new(HashMap::new())),
            gpu_cache_timestamp: Arc::new(Mutex::new(Instant::now())),
            simulations_since_gpu_update: Arc::new(AtomicI32::new(0)),
            gpu_last_batch_size: Arc::new(AtomicI32::new(0)),
            gpu_simulation_sender,
            gpu_pending_evaluations: pending_evaluations,
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

    /// Advances the root of the tree to the node corresponding to the given move.
    ///
    /// This is used when a move is made in the actual game to reuse the search tree.
    /// The subtree corresponding to the selected move becomes the new root,
    /// and all other subtrees are recycled to save memory.
    ///
    /// # Arguments
    /// * `mv` - The move that was made in the game
    pub fn advance_root(&mut self, mv: &S::Move) {
        let (new_root, nodes_to_recycle, new_tree_size) = {
            let children = self.root.children.read();
            let new_root = children
                .get(mv)
                .map(Arc::clone)
                .unwrap_or_else(|| Arc::new(Node::new()));

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
    ///
    /// Removes children with less than 1% of the root's visits to keep the tree
    /// focused on the most promising moves. This is a heuristic-based pruning
    /// that doesn't require manual threshold setting.
    pub fn auto_prune(&mut self) {
        let root_visits = self.root.visits.load(Ordering::Relaxed);
        let min_visits = std::cmp::max(1, root_visits / 100); // At least 1% of root visits
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
                if root_visits > 0 {
                    root_wins as f64 / root_visits as f64 / 2.0
                } else {
                    0.0
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
        self.update_gpu_puct_cache(true);

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
        let gpu_refresh_handle = if self.gpu_enabled && self.gpu_accelerator.is_some() {
            let stop_flag = stop_searching.clone();
            let gpu_accelerator = self.gpu_accelerator.clone();
            let root = self.root.clone();
            let exploration_parameter = self.exploration_parameter;
            let gpu_puct_cache = self.gpu_puct_cache.clone();
            let simulations_counter = self.simulations_since_gpu_update.clone();
            
            Some(std::thread::spawn(move || {
                let refresh_interval = Duration::from_millis(50); // Refresh every 50ms for better GPU utilization
                const MAX_DEPTH: u32 = 50;
                const MAX_NODES: usize = 65536;
                
                while !stop_flag.load(Ordering::Relaxed) {
                    std::thread::sleep(refresh_interval);
                    
                    // Check if enough simulations have passed
                    let sims = simulations_counter.load(Ordering::Relaxed);
                    if sims < 500 {
                        continue;
                    }
                    simulations_counter.store(0, Ordering::Relaxed);
                    
                    // Deep tree traversal to collect all parent-child pairs
                    let mut node_data: Vec<gpu::GpuNodeData> = Vec::with_capacity(MAX_NODES);
                    let mut cache_keys: Vec<(usize, usize)> = Vec::with_capacity(MAX_NODES);
                    let mut stack: Vec<(Arc<Node<S::Move>>, u32)> = Vec::with_capacity(1024);
                    stack.push((root.clone(), 0));
                    
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
                                exploration_parameter as f32,
                            ));
                            cache_keys.push((parent_id, child_id));
                            
                            // Add visited children to stack for deeper traversal
                            if child_node.visits.load(Ordering::Relaxed) > 0 {
                                stack.push((child_node.clone(), depth + 1));
                            }
                        }
                    }
                    
                    if node_data.is_empty() {
                        continue;
                    }
                    
                    // Compute PUCT on GPU
                    if let Some(ref accelerator) = gpu_accelerator {
                        let mut acc = accelerator.lock();
                        if let Ok(results) = acc.compute_puct_batch(&node_data) {
                            let mut cache = gpu_puct_cache.write();
                            cache.clear();
                            cache.reserve(results.len());
                            for (i, result) in results.iter().enumerate() {
                                cache.insert(cache_keys[i], result.puct_score as f64);
                            }
                        }
                    }
                }
            }))
        } else {
            None
        };

        self.pool.install(|| {
            let _ = (0..iterations)
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

        // After all simulations, the best move is the one most visited.
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
            children
                .iter()
                .max_by_key(|(_, node)| node.visits.load(Ordering::Relaxed))
                .map(|(mv, _)| mv.clone())
                .expect("Root node has children but max_by_key failed")
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

        // After all simulations, the best move is the one most visited.
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
            children
                .iter()
                .max_by_key(|(_, node)| node.visits.load(Ordering::Relaxed))
                .map(|(mv, _)| mv.clone())
                .expect("Root node has children but max_by_key failed")
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

        // Return the best move
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
            children
                .iter()
                .max_by_key(|(_, node)| node.visits.load(Ordering::Relaxed))
                .map(|(mv, _)| mv.clone())
                .expect("Root node has children but max_by_key failed")
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
                                )
                            });
                            #[cfg(not(feature = "gpu"))]
                            let puct = n.puct(
                                parent_visits,
                                self.exploration_parameter,
                                prior_probability,
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
                            let visits = current_node.visits.load(Ordering::Relaxed);

                            // Base expansion probability decreases with depth
                            // More visits increase the likelihood of expansion
                            let depth_factor = 1.0 / (1.0 + (depth as f64) * 0.5);
                            let visit_factor = (visits as f64).sqrt() / 10.0; // Encourage expansion for well-visited nodes
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
                // Check pending evaluations to prevent huge backlog
                // If GPU is saturated, fall back to CPU simulation
                let pending = self.gpu_pending_evaluations.load(Ordering::Relaxed);
                if pending < 10000 {
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

                let move_index = random_range(0, moves_cache.len());
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
            for (i, (node, _)) in path.iter().zip(path_players.iter()).rev().enumerate() {
                if i < path.len() - 1 {
                    node.remove_virtual_loss();
                }
            }
            return;
        }

        for (i, (node, &player_who_moved)) in path.iter().zip(path_players.iter()).rev().enumerate()
        {
            // Remove virtual loss from all nodes except the last one (the leaf/terminal node)
            // which didn't have virtual loss applied during selection
            if i < path.len() - 1 {
                node.remove_virtual_loss();
            }

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
    fn get_cached_puct_by_node(&self, parent: &Arc<Node<S::Move>>, child: &Arc<Node<S::Move>>) -> Option<f64> {
        if !self.gpu_enabled {
            return None;
        }
        
        let parent_id = Arc::as_ptr(parent) as usize;
        let child_id = Arc::as_ptr(child) as usize;
        
        let cache = self.gpu_puct_cache.read();
        cache.get(&(parent_id, child_id)).copied()
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

        fn get_board(&self) -> &Vec<Vec<i32>> {
            &self.board
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
