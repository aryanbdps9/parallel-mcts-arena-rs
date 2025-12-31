//! # GUI Application State
//!
//! Manages the application state for the Windows GUI, including game state,
//! AI coordination, and UI state.
//!
//! ## Architecture
//! The GUI application uses a central GameController to own the authoritative
//! game state. This ensures:
//! - Clear separation between game logic and UI
//! - Proper move validation before any state changes
//! - Thread-safe communication with the AI worker

use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, atomic::AtomicBool};
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime};

use crate::game_controller::{GameController, MoveResult};
use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::games::blokus::BlokusState;
use crate::games::connect4::Connect4State;
use crate::games::gomoku::GomokuState;
use crate::games::hive::HiveState;
use crate::games::othello::OthelloState;
use mcts::{GameState, MCTS, SearchStatistics};

use super::game_renderers::{GameRenderer, create_renderer_for_game};

/// Player type (human or AI)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerType {
    Human,
    AiCpu,
    AiGpu,
    /// GPU-Native MCTS (currently only for Othello) - all 4 MCTS phases run on GPU
    AiGpuNative,
}

/// Current screen/mode of the GUI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiMode {
    /// Game selection menu
    GameSelection,
    /// Settings menu
    Settings,
    /// Player configuration (Human vs AI)
    PlayerConfig,
    /// Active game
    InGame,
    /// Game over screen
    GameOver,
    /// How to play screen
    HowToPlay,
}

/// Game status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameStatus {
    InProgress,
    Win(i32),
    Draw,
}

/// Active tab in the info panel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTab {
    DebugStats,
    MoveHistory,
}

impl ActiveTab {
    pub fn next(&self) -> Self {
        match self {
            ActiveTab::DebugStats => ActiveTab::MoveHistory,
            ActiveTab::MoveHistory => ActiveTab::DebugStats,
        }
    }
}

/// Available games to choose from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameType {
    Gomoku,
    Connect4,
    Othello,
    Blokus,
    Hive,
}

impl GameType {
    pub fn name(&self) -> &'static str {
        match self {
            GameType::Gomoku => "Gomoku",
            GameType::Connect4 => "Connect 4",
            GameType::Othello => "Othello",
            GameType::Blokus => "Blokus",
            GameType::Hive => "Hive",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            GameType::Gomoku => "Get five stones in a row to win",
            GameType::Connect4 => "Drop pieces to connect four",
            GameType::Othello => "Flip opponent's pieces by surrounding",
            GameType::Blokus => "Place polyomino pieces strategically",
            GameType::Hive => "Surround the opponent's Queen Bee",
        }
    }

    pub fn all() -> &'static [GameType] {
        &[GameType::Gomoku, GameType::Connect4, GameType::Othello, GameType::Blokus, GameType::Hive]
    }
}

/// Move history entry
#[derive(Debug, Clone)]
pub struct MoveEntry {
    pub timestamp: SystemTime,
    pub player: i32,
    pub move_made: MoveWrapper,
}

/// AI worker messages
#[derive(Debug)]
pub enum AIRequest {
    Search(GameWrapper, u64, PlayerType, i32),
    /// Advance root with: (move_made, debug_info, new_game_state)
    AdvanceRoot(MoveWrapper, String, GameWrapper),
    Stop,
}

#[derive(Debug)]
pub enum AIResponse {
    BestMove(MoveWrapper, SearchStatistics),
}

/// AI worker that runs searches in a background thread
pub struct AIWorker {
    handle: Option<JoinHandle<()>>,
    tx: Sender<AIRequest>,
    rx: Receiver<AIResponse>,
    stop_flag: Arc<AtomicBool>,
}

impl AIWorker {
    pub fn new(
        cpu_exploration_constant: f64,
        gpu_exploration_constant: f64,
        num_threads: usize,
        max_nodes: usize,
        search_iterations: u32,
        shared_tree: bool,
        gpu_threads: usize,
        gpu_use_heuristic: bool,
        cpu_select_by_q: bool,
        gpu_select_by_q: bool,
        gpu_native_batch_size: u32,
        gpu_virtual_loss_weight: f32,
        gpu_temperature: f32,
        gpu_max_nodes: Option<u32>,
    ) -> Self {
        use std::sync::mpsc::channel;
        use std::collections::HashMap;
        use mcts::MoveSelectionStrategy;

        let (tx_req, rx_req) = channel();
        let (tx_resp, rx_resp) = channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_clone = stop_flag.clone();

        let cpu_move_selection_strategy = if cpu_select_by_q {
            MoveSelectionStrategy::MaxQ
        } else {
            MoveSelectionStrategy::MaxVisits
        };

        let gpu_move_selection_strategy = if gpu_select_by_q {
            MoveSelectionStrategy::MaxQ
        } else {
            MoveSelectionStrategy::MaxVisits
        };

        let handle = std::thread::spawn(move || {
            let mut mcts_cpu_map: HashMap<i32, MCTS<GameWrapper>> = HashMap::new();
            let mut mcts_gpu_map: HashMap<i32, MCTS<GameWrapper>> = HashMap::new();
            // Persistent MCTS for GPU-native Othello with tree reuse
            #[cfg(feature = "gpu")]
            let mut mcts_gpu_native: Option<MCTS<GameWrapper>> = None;

            for request in rx_req {
                match request {
                    AIRequest::Search(state, timeout, player_type, player_id) => {
                        if stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }

                        let key = if shared_tree { 0 } else { player_id };

                        // Handle GPU-native search for Othello separately
                        #[cfg(feature = "gpu")]
                        if let PlayerType::AiGpuNative = player_type {
                            if let GameWrapper::Othello(ref othello_state) = state {
                                use crate::games::othello::OthelloMove;
                                
                                // Extract board state for GPU
                                let board_2d = othello_state.get_board();
                                let mut board = [0i32; 64];
                                for (r, row) in board_2d.iter().enumerate() {
                                    for (c, &cell) in row.iter().enumerate() {
                                        board[r * 8 + c] = cell;
                                    }
                                }
                                
                                // Get legal moves
                                let legal_moves = othello_state.get_possible_moves();
                                let legal_moves_xy: Vec<(usize, usize)> = legal_moves
                                    .iter()
                                    .map(|m| (m.1, m.0)) // OthelloMove is (row, col), GPU expects (x, y)
                                    .collect();
                                
                                let current_player = othello_state.get_current_player();
                                
                                // Calculate number of batches based on timeout
                                let num_batches = (search_iterations / gpu_native_batch_size).max(1);
                                
                                // Get or create persistent GPU-native MCTS
                                // Use minimal CPU MCTS - GPU-native handles everything on GPU
                                // Don't use with_gpu_config as that creates heavy CPU thread pool and GPU simulation thread
                                let mcts = mcts_gpu_native.get_or_insert_with(|| {
                                    // Create lightweight MCTS - only need it for GPU-native engine hosting
                                    // Use small max_nodes since CPU tree won't be used
                                    let mut new_mcts = MCTS::new(
                                        gpu_exploration_constant, 
                                        1,      // Minimal threads - won't be used
                                        1000,   // Minimal nodes - won't be used for GPU-native
                                    );
                                    new_mcts.set_move_selection_strategy(gpu_move_selection_strategy);
                                    eprintln!("[AI GPU-Native] Using pure GPU-native MCTS (no CPU tree)");
                                    // Initialize the GPU-native Othello engine
                                    new_mcts.init_gpu_native_othello(&board, current_player, &legal_moves_xy, gpu_native_batch_size);
                                    new_mcts
                                });
                                
                                if let Some(((x, y), visits, q, children_stats, total_nodes, telemetry)) = mcts.search_gpu_native_othello(
                                    &board,
                                    current_player,
                                    &legal_moves_xy,
                                    gpu_native_batch_size,
                                    num_batches,
                                    gpu_exploration_constant as f32,
                                    gpu_virtual_loss_weight,
                                    gpu_temperature,
                                    timeout,
                                    gpu_max_nodes,
                                ) {
                                    if telemetry.saturated {
                                        eprintln!(
                                            "[GPU-Native WARNING] Node pool saturated: {} / {} nodes",
                                            telemetry.alloc_count_after, telemetry.node_capacity
                                        );
                                    }
                                    // Convert back to OthelloMove (row, col)
                                    let best_move = MoveWrapper::Othello(OthelloMove(y, x));
                                    
                                    // Build children_stats HashMap for UI display
                                    let mut stats_map = std::collections::HashMap::new();
                                    for (cx, cy, cv, _cw, cq) in &children_stats {
                                        // Format as (row, col) to match OthelloMove and CPU debug output
                                        let move_str = format!("({},{})", cy, cx);
                                        stats_map.insert(move_str, (*cq, *cv));
                                    }
                                    
                                    // === TSV Logging for GPU-Native ===
                                    // Sort children by visits to find best and second-best
                                    let mut sorted_children = children_stats.clone();
                                    sorted_children.sort_by_key(|(_, _, v, _, _)| -(*v));
                                    
                                    if sorted_children.len() >= 2 {
                                        let best = &sorted_children[0];
                                        let second = &sorted_children[1];
                                        
                                        let visit_diff = best.2 - second.2;
                                        let best_move_str = format!("{},{}", best.0, best.1);
                                        let second_move_str = format!("{},{}", second.0, second.1);
                                        
                                        // Calculate U values (PUCT exploration term)
                                        let prior = 1.0 / sorted_children.len() as f64;
                                        let best_u = gpu_exploration_constant * prior * (visits as f64).sqrt() / (1.0 + best.2 as f64);
                                        let second_u = gpu_exploration_constant * prior * (visits as f64).sqrt() / (1.0 + second.2 as f64);
                                        
                                        let csv_line = format!("GPU-Native (Player {})\t{}\t{:.4}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\n",
                                            current_player, visits, q, visit_diff,
                                            best_move_str, best.2, best.4, best_u,
                                            second_move_str, second.2, second.4, second_u
                                        );
                                        
                                        // Print to terminal
                                        println!("CSV_DATA: {}", csv_line.trim());
                                        
                                        // Append to file
                                        use std::io::Write;
                                        let file_path = std::path::Path::new("mcts_stats.tsv");
                                        
                                        if let Ok(mut file) = std::fs::OpenOptions::new()
                                            .create(true)
                                            .append(true)
                                            .open(file_path) 
                                        {
                                            let _ = file.write_all(csv_line.as_bytes());
                                        }
                                    }
                                    // === End TSV Logging ===
                                    
                                    // Create SearchStatistics for UI display
                                    let stats = SearchStatistics {
                                        total_nodes: total_nodes as i32,
                                        root_visits: visits,
                                        root_wins: q * visits as f64,
                                        root_value: q,
                                        children_stats: stats_map,
                                    };
                                    
                                    if !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                                        let _ = tx_resp.send(AIResponse::BestMove(best_move, stats));
                                    }
                                }
                                continue;
                            } else {
                                eprintln!("[AI] GPU-Native only supported for Othello, falling back to hybrid GPU");
                            }
                        }

                        let mcts_opt = match player_type {
                            PlayerType::AiCpu => {
                                Some(mcts_cpu_map.entry(key).or_insert_with(|| {
                                    let mut mcts = MCTS::new(cpu_exploration_constant, num_threads, max_nodes);
                                    mcts.set_move_selection_strategy(cpu_move_selection_strategy);
                                    mcts
                                }))
                            },
                            PlayerType::AiGpu | PlayerType::AiGpuNative => {
                                #[cfg(feature = "gpu")]
                                {
                                    if !mcts_gpu_map.contains_key(&key) {
                                        let gpu_config = mcts::gpu::GpuConfig {
                                            max_batch_size: gpu_threads,
                                            ..Default::default()
                                        };
                                        let (mut new_mcts, gpu_msg) = MCTS::with_gpu_config(gpu_exploration_constant, num_threads, max_nodes, gpu_config, gpu_use_heuristic);
                                        new_mcts.set_move_selection_strategy(gpu_move_selection_strategy);
                                        if let Some(msg) = gpu_msg {
                                            eprintln!("[AI] {}", msg);
                                        }
                                        mcts_gpu_map.insert(key, new_mcts);
                                    }
                                    mcts_gpu_map.get_mut(&key)
                                }
                                #[cfg(not(feature = "gpu"))]
                                {
                                    eprintln!("[AI] GPU not available, falling back to CPU");
                                    Some(mcts_cpu_map.entry(key).or_insert_with(|| {
                                        let mut mcts = MCTS::new(cpu_exploration_constant, num_threads, max_nodes);
                                        mcts.set_move_selection_strategy(cpu_move_selection_strategy);
                                        mcts
                                    }))
                                }
                            },
                            _ => None,
                        };

                        if let Some(mcts) = mcts_opt {
                            let (best_move, stats) = mcts.search_with_stop(
                                &state,
                                search_iterations as i32,
                                1,
                                timeout,
                                Some(stop_clone.clone()),
                            );

                            if !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                                let _ = tx_resp.send(AIResponse::BestMove(best_move, stats));
                            }
                        }
                    }
                    AIRequest::AdvanceRoot(move_made, debug_info, new_state) => {
                        // Only advance the root for the AI that actually made the move
                        // The debug_info string contains "AiCpu" or "AiGpu" which we can use to filter
                        
                        // Handle GPU-native Othello tree advancement
                        #[cfg(feature = "gpu")]
                        if let Some(ref mut mcts) = mcts_gpu_native {
                            if let GameWrapper::Othello(ref othello_state) = new_state {
                                if let MoveWrapper::Othello(ref mv) = move_made {
                                    use std::sync::atomic::{AtomicU32, Ordering};
                                    static ADVANCE_LOG_COUNT: AtomicU32 = AtomicU32::new(0);

                                    // Extract new board state
                                    let board_2d = othello_state.get_board();
                                    let mut new_board = [0i32; 64];
                                    for (r, row) in board_2d.iter().enumerate() {
                                        for (c, &cell) in row.iter().enumerate() {
                                            new_board[r * 8 + c] = cell;
                                        }
                                    }
                                    
                                    // Get legal moves for new position
                                    let legal_moves = othello_state.get_possible_moves();
                                    let legal_moves_xy: Vec<(usize, usize)> = legal_moves
                                        .iter()
                                        .map(|m| (m.1, m.0))
                                        .collect();
                                    
                                    let new_player = othello_state.get_current_player();

                                    // Log a few samples to debug GPU-native advance_root mismatches
                                    if ADVANCE_LOG_COUNT.fetch_add(1, Ordering::Relaxed) < 8 {
                                        // Build ASCII board for readability
                                        let mut ascii_rows: Vec<String> = Vec::with_capacity(8);
                                        for r in 0..8 {
                                            let mut line = String::with_capacity(8);
                                            for c in 0..8 {
                                                let cell = new_board[r * 8 + c];
                                                let ch = match cell {
                                                    1 => 'X',
                                                    -1 => 'O',
                                                    _ => '.',
                                                };
                                                line.push(ch);
                                            }
                                            ascii_rows.push(line);
                                        }
                                        println!(
                                            "[GPU-Native HOST] advance_root input current_player={} legal_moves={:?} board={:?}",
                                            new_player,
                                            legal_moves_xy,
                                            ascii_rows
                                        );
                                    }
                                    
                                    // Advance GPU-native tree
                                    // mv.0 is row, mv.1 is col, but advance_root expects (x, y) = (col, row)
                                    let reused = mcts.advance_root_gpu_native(
                                        (mv.1, mv.0), // Convert (row, col) to (x, y)
                                        &new_board,
                                        new_player,
                                        &legal_moves_xy,
                                    );
                                    if reused {
                                        println!("[GPU-Native] Tree reuse successful");
                                    }
                                }
                            }
                        }
                        
                        if debug_info.contains("AiCpu") {
                            for mcts in mcts_cpu_map.values_mut() {
                                mcts.advance_root(&move_made, Some(&debug_info));
                            }
                            // For the OTHER AI (GPU), we also need to advance the root so it stays in sync
                            // with the game state, but we might want to log it differently or not log stats
                            for mcts in mcts_gpu_map.values_mut() {
                                mcts.advance_root(&move_made, Some(&format!("{} (Opponent Move)", debug_info)));
                            }
                        } else if debug_info.contains("AiGpuNative") {
                            // GPU-native tree already advanced above
                            // Sync CPU and hybrid GPU AIs as opponent moves
                            for mcts in mcts_cpu_map.values_mut() {
                                mcts.advance_root(&move_made, Some(&format!("{} (Opponent Move)", debug_info)));
                            }
                            for mcts in mcts_gpu_map.values_mut() {
                                mcts.advance_root(&move_made, Some(&format!("{} (Opponent Move)", debug_info)));
                            }
                        } else if debug_info.contains("AiGpu") {
                            for mcts in mcts_gpu_map.values_mut() {
                                mcts.advance_root(&move_made, Some(&debug_info));
                            }
                            // Sync CPU AI
                            for mcts in mcts_cpu_map.values_mut() {
                                mcts.advance_root(&move_made, Some(&format!("{} (Opponent Move)", debug_info)));
                            }
                        } else {
                            // Fallback for human moves or unknown sources
                            for mcts in mcts_cpu_map.values_mut() {
                                mcts.advance_root(&move_made, Some(&debug_info));
                            }
                            for mcts in mcts_gpu_map.values_mut() {
                                mcts.advance_root(&move_made, Some(&debug_info));
                            }
                        }
                    }
                    AIRequest::Stop => break,
                }
            }
        });

        Self {
            handle: Some(handle),
            tx: tx_req,
            rx: rx_resp,
            stop_flag,
        }
    }

    pub fn start_search(&self, state: GameWrapper, timeout: u64, player_type: PlayerType, player_id: i32) {
        let _ = self.tx.send(AIRequest::Search(state, timeout, player_type, player_id));
    }

    pub fn try_recv(&self) -> Option<AIResponse> {
        self.rx.try_recv().ok()
    }

    pub fn stop(&self) {
        self.stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = self.tx.send(AIRequest::Stop);
    }

    /// Advance the MCTS tree root after a move is made
    /// 
    /// This allows the AI to reuse previous search results by promoting
    /// the child node corresponding to the move as the new root.
    pub fn advance_root(&self, move_made: &MoveWrapper, debug_info: String, new_state: GameWrapper) {
        let _ = self.tx.send(AIRequest::AdvanceRoot(move_made.clone(), debug_info, new_state));
    }
}

impl Drop for AIWorker {
    fn drop(&mut self) {
        self.stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Main GUI application state
pub struct GuiApp {
    // Current mode
    pub mode: GuiMode,
    pub should_quit: bool,

    // Game selection
    pub selected_game_index: usize,
    pub selected_game_type: GameType,

    // Player configuration
    pub player_types: Vec<(i32, PlayerType)>,
    pub selected_player_index: usize,

    // Game state - using GameController as the authoritative source
    pub game_controller: GameController,
    /// UI-facing game wrapper for rendering (synced with controller)
    pub game: GameWrapper,
    pub game_status: GameStatus,
    pub move_history: Vec<MoveEntry>,

    // Game renderer
    pub game_renderer: Box<dyn GameRenderer>,

    // AI
    pub ai_worker: AIWorker,
    pub ai_thinking: bool,
    pub ai_thinking_start: Option<Instant>,
    pub last_search_stats: Option<SearchStatistics>,

    // Settings (editable in settings menu)
    pub board_size: usize,
    pub line_size: usize,
    pub timeout_secs: u64,
    pub ai_threads: usize,
    pub gpu_threads: usize,
    pub max_nodes: usize,
    pub search_iterations: u32,
    pub cpu_exploration_constant: f64,
    pub gpu_exploration_constant: f64,
    pub stats_interval_secs: u64,
    pub ai_only: bool,
    pub shared_tree: bool,
    pub gpu_use_heuristic: bool,
    pub cpu_select_by_q: bool,
    pub gpu_select_by_q: bool,
    /// Batch size for GPU-native MCTS (iterations per GPU dispatch)
    pub gpu_native_batch_size: u32,
    /// Virtual loss weight for GPU-native PUCT
    pub gpu_virtual_loss_weight: f32,
    /// Temperature for GPU-native softmax selection
    pub gpu_temperature: f32,
    /// Optional override for max nodes in GPU-native MCTS
    pub gpu_max_nodes: Option<u32>,
    pub selected_settings_index: usize,

    // For urgent GPU log polling
    last_gpu_log_poll: std::time::Instant,

    // UI state
    pub needs_redraw: bool,
    pub hover_button: Option<usize>,
    pub active_tab: ActiveTab,
    pub debug_scroll: i32,
    pub history_scroll: i32,
    pub how_to_play_scroll: i32,
    pub selected_how_to_play_game: usize,
    pub game_selection_scroll: i32,
    pub settings_scroll: i32,
        // Removed stray line
        // virtual_los

    // Resizable panel state
    /// Width of the right info panel (0.0 to 1.0 as percentage of available width)
    pub info_panel_ratio: f32,
    /// Whether we're currently dragging the splitter
    pub is_dragging_splitter: bool,
    /// Whether we're currently right-click dragging (for tilt adjustment)
    pub is_right_dragging: bool,
    /// Last mouse position for drag delta calculation
    pub last_drag_pos: Option<(f32, f32)>,
    /// Minimum panel width in pixels
    pub min_panel_width: f32,
    /// Maximum panel ratio
    pub max_panel_ratio: f32,
}

impl GuiApp {
    pub fn new(
        cpu_exploration_constant: f64,
        gpu_exploration_constant: f64,
        num_threads: usize,
        max_nodes: usize,
        search_iterations: u32,
        shared_tree: bool,
        gpu_threads: usize,
        gpu_use_heuristic: bool,
        board_size: usize,
        line_size: usize,
        timeout_secs: u64,
        stats_interval_secs: u64,
        ai_only: bool,
        cpu_select_by_q: bool,
        gpu_select_by_q: bool,
        gpu_native_batch_size: u32,
        gpu_virtual_loss_weight: f32,
        gpu_temperature: f32,
        gpu_max_nodes: Option<u32>,
    ) -> Self {
        let default_game = GameWrapper::Gomoku(GomokuState::new(board_size, line_size));
        let game_controller = GameController::new(default_game.clone());
        let renderer = create_renderer_for_game(&default_game);

        Self {
            mode: GuiMode::GameSelection,
            should_quit: false,
            selected_game_index: 0,
            selected_game_type: GameType::Gomoku,
            player_types: vec![(1, PlayerType::Human), (-1, PlayerType::AiCpu)],
            selected_player_index: 0,
            game_controller,
            game: default_game,
            game_status: GameStatus::InProgress,
            move_history: Vec::new(),
            game_renderer: renderer,
            ai_worker: AIWorker::new(cpu_exploration_constant, gpu_exploration_constant, num_threads, max_nodes, search_iterations, shared_tree, gpu_threads, gpu_use_heuristic, cpu_select_by_q, gpu_select_by_q, gpu_native_batch_size, gpu_virtual_loss_weight, gpu_temperature, gpu_max_nodes),
            ai_thinking: false,
            ai_thinking_start: None,
            last_search_stats: None,
            board_size,
            line_size,
            timeout_secs,
            ai_threads: num_threads,
            gpu_threads,
            max_nodes,
            search_iterations,
            cpu_exploration_constant,
            gpu_exploration_constant,
            stats_interval_secs,
            ai_only,
            shared_tree,
            gpu_use_heuristic,
            cpu_select_by_q,
            gpu_select_by_q,
            gpu_native_batch_size,
            gpu_virtual_loss_weight,
            gpu_temperature,
            gpu_max_nodes,
            selected_settings_index: 0,
            needs_redraw: true,
            hover_button: None,
            active_tab: ActiveTab::DebugStats,
            debug_scroll: 0,
            history_scroll: 0,
            how_to_play_scroll: 0,
            selected_how_to_play_game: 0,
            game_selection_scroll: 0,
            settings_scroll: 0,
            info_panel_ratio: 0.25, // Default 25% of game area width
            is_dragging_splitter: false,
            is_right_dragging: false,
            last_drag_pos: None,
            min_panel_width: 200.0,
            max_panel_ratio: 0.5,
            last_gpu_log_poll: std::time::Instant::now(),
        }
    }

    /// Start a new game with current settings
    pub fn start_game(&mut self) {
        // If AI-only mode, set all players to AI
        if self.ai_only {
            for (_, pt) in &mut self.player_types {
                if *pt == PlayerType::Human {
                    *pt = PlayerType::AiCpu;
                }
            }
        }

        let new_game = match self.selected_game_type {
            GameType::Gomoku => GameWrapper::Gomoku(GomokuState::new(self.board_size, self.line_size)),
            GameType::Connect4 => GameWrapper::Connect4(Connect4State::new(7, 6, self.line_size)),
            GameType::Othello => GameWrapper::Othello(OthelloState::new(8)),
            GameType::Blokus => GameWrapper::Blokus(BlokusState::new()),
            GameType::Hive => GameWrapper::Hive(HiveState::new()),
        };

        // Reset the game controller with new state
        self.game_controller.reset(new_game.clone());
        self.game = new_game;

        self.game_renderer = create_renderer_for_game(&self.game);
        self.game_renderer.reset();
        self.game_status = GameStatus::InProgress;
        self.move_history.clear();
        self.ai_thinking = false;
        self.ai_thinking_start = None;
        self.last_search_stats = None;
        self.mode = GuiMode::InGame;
        self.needs_redraw = true;

        // Create a new AI worker (the old one may have been stopped when going back to menu)
        self.ai_worker = AIWorker::new(
            self.cpu_exploration_constant,
            self.gpu_exploration_constant,
            self.ai_threads,
            self.max_nodes,
            self.search_iterations,
            self.shared_tree,
            self.gpu_threads,
            self.gpu_use_heuristic,
            self.cpu_select_by_q,
            self.gpu_select_by_q,
            self.gpu_native_batch_size,
            self.gpu_virtual_loss_weight,
            self.gpu_temperature,
            self.gpu_max_nodes,
        );

        // Check if AI should move first
        self.check_ai_turn();
    }

    /// Check if it's AI's turn and start search if needed
    pub fn check_ai_turn(&mut self) {
        if self.game_status != GameStatus::InProgress {
            return;
        }

        let current_player = self.game.get_current_player();
        let player_type = self.player_types
            .iter()
            .find(|(id, _)| *id == current_player)
            .map(|(_, pt)| *pt)
            .unwrap_or(PlayerType::Human);

        let is_ai = matches!(player_type, PlayerType::AiCpu | PlayerType::AiGpu | PlayerType::AiGpuNative);

        if is_ai && !self.ai_thinking {
            self.ai_thinking = true;
            self.ai_thinking_start = Some(Instant::now());
            self.ai_worker.start_search(self.game.clone(), self.timeout_secs, player_type, current_player);
            self.needs_redraw = true;
        }
    }

    /// Process a move (from human or AI)
    ///
    /// Uses the GameController to validate and apply the move.
    /// Rejects invalid moves and updates game status accordingly.
    pub fn make_move(&mut self, mv: MoveWrapper) {
        // Use GameController for move validation and application
        match self.game_controller.try_make_move(mv.clone()) {
            MoveResult::Success { player, game_over, winner, .. } => {
                // Sync UI game state with controller
                self.game = self.game_controller.get_state_for_search();
                
                // Advance MCTS tree root to maintain tree reuse
                // This must happen for EVERY move (human or AI) so the tree stays in sync
                let player_type = self.player_types
                    .iter()
                    .find(|(id, _)| *id == player)
                    .map(|(_, pt)| *pt)
                    .unwrap_or(PlayerType::Human);
                
                let debug_info = format!("Player {} ({:?})", player, player_type);
                self.ai_worker.advance_root(&mv, debug_info, self.game.clone());
                
                // Add to move history for UI display
                self.move_history.push(MoveEntry {
                    timestamp: SystemTime::now(),
                    player,
                    move_made: mv,
                });

                // Auto-scroll history to bottom
                self.history_scroll = i32::MAX;

                // Check game status
                if game_over {
                    self.game_status = match winner {
                        Some(w) => GameStatus::Win(w),
                        None => GameStatus::Draw,
                    };
                    self.mode = GuiMode::GameOver;
                }

                self.ai_thinking = false;
                self.needs_redraw = true;

                // Check if AI should move next
                self.check_ai_turn();
            }
            MoveResult::Invalid { reason: _reason } => {
                // Move was rejected - log for debugging
                #[cfg(debug_assertions)]
                eprintln!("Move rejected: {}", _reason);
                self.needs_redraw = true;
            }
            MoveResult::GameOver => {
                // Game is already over, shouldn't happen
                self.needs_redraw = true;
            }
        }
    }

    /// Update application state (called periodically)
    pub fn update(&mut self) {
        // Check for AI response
        if self.ai_thinking {
            if let Some(AIResponse::BestMove(mv, stats)) = self.ai_worker.try_recv() {
                self.last_search_stats = Some(stats);
                
                // Validate AI move before applying
                if let Err(reason) = self.game_controller.validate_move(&mv) {
                    // AI proposed an invalid move - this is a critical error
                    // Dump state and exit
                    self.dump_invalid_ai_move(&mv, &reason);
                    self.should_quit = true;
                    return;
                }
                
                self.make_move(mv);
            }
        }

        // === Urgent GPU log polling for GPU-native Othello ===
        // Only poll if Othello is active and at least one player is AiGpuNative
        let is_gpu_native_othello =
            matches!(self.selected_game_type, GameType::Othello)
            && self.player_types.iter().any(|&(_, pt)| pt == PlayerType::AiGpuNative);
        if is_gpu_native_othello && self.last_gpu_log_poll.elapsed().as_millis() >= 100 {
            // Try to access the GPU-native engine and poll logs
            #[cfg(feature = "gpu")]
            {
                // use mcts::gpu::GpuMctsEngine; // removed unused import
                // Try to access the GPU-native engine via the AI worker's MCTS instance
                // This is a bit hacky, but we can reach it via the Option<MCTS<GameWrapper>> in the AI worker thread
                // For now, we use a static method to poll all known engines (if any)
                // (If you want to be more precise, you could expose a method on AIWorker to do this)
                // mcts::gpu::poll_all_gpu_native_debug_events(); // Removed: function does not exist
            }
            self.last_gpu_log_poll = std::time::Instant::now();
        }
    }

    /// Dump game state and history when AI proposes an invalid move
    fn dump_invalid_ai_move(&self, invalid_move: &MoveWrapper, reason: &crate::game_controller::MoveValidationError) {
        use std::io::Write;
        
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let filename = format!("ai_invalid_move_{}.txt", timestamp);
        
        let mut content = String::new();
        content.push_str("=== AI INVALID MOVE DUMP ===\n\n");
        content.push_str(&format!("Timestamp: {}\n", timestamp));
        content.push_str(&format!("Game: {:?}\n", self.selected_game_type));
        content.push_str(&format!("Invalid Move: {:?}\n", invalid_move));
        content.push_str(&format!("Rejection Reason: {}\n\n", reason));
        
        content.push_str("=== Move History ===\n");
        content.push_str(&self.game_controller.format_history_for_clipboard());
        
        content.push_str("\n=== Current Game State ===\n");
        content.push_str(&format!("Current Player: {}\n", self.game.get_current_player()));
        content.push_str(&format!("Is Terminal: {}\n", self.game.is_terminal()));
        
        // Add board state
        content.push_str("\n=== Board State ===\n");
        let board = self.game.get_board();
        for row in board {
            for cell in row {
                let c = match cell {
                    0 => '.',
                    1 => 'X',
                    -1 => 'O',
                    n => std::char::from_digit(n.unsigned_abs() as u32, 10).unwrap_or('?'),
                };
                content.push_str(&format!("{} ", c));
            }
            content.push('\n');
        }
        
        content.push_str("\n=== Legal Moves ===\n");
        let legal_moves = self.game_controller.get_legal_moves();
        for mv in legal_moves.iter().take(50) {
            content.push_str(&format!("{:?}\n", mv));
        }
        if legal_moves.len() > 50 {
            content.push_str(&format!("... and {} more moves\n", legal_moves.len() - 50));
        }
        
        // Write to file
        if let Ok(mut file) = std::fs::File::create(&filename) {
            let _ = file.write_all(content.as_bytes());
            eprintln!("CRITICAL: AI proposed invalid move! Dumped state to: {}", filename);
        } else {
            eprintln!("CRITICAL: AI proposed invalid move! Failed to write dump file.");
            eprintln!("{}", content);
        }
    }

    /// Handle game selection
    pub fn select_game(&mut self, index: usize) {
        let games = GameType::all();
        if index < games.len() {
            self.selected_game_index = index;
            self.selected_game_type = games[index];
            
            // Update player types based on game
            self.player_types = match self.selected_game_type {
                GameType::Blokus => vec![
                    (1, PlayerType::Human),
                    (2, PlayerType::AiCpu),
                    (3, PlayerType::AiCpu),
                    (4, PlayerType::AiCpu),
                ],
                _ => vec![
                    (1, PlayerType::Human),
                    (-1, PlayerType::AiCpu),
                ],
            };
            
            // In AI-only mode, skip player config and start game directly
            if self.ai_only {
                self.start_game();
            } else {
                self.mode = GuiMode::PlayerConfig;
                self.needs_redraw = true;
            }
        }
    }

    /// Toggle player type
    pub fn toggle_player(&mut self, index: usize) {
        if index < self.player_types.len() {
            let (id, pt) = &self.player_types[index];
            // Cycle through: Human -> CPU -> GPU (Hybrid) -> GPU-Native -> Human
            // GPU-Native is only available for Othello
            let next_type = match pt {
                PlayerType::Human => PlayerType::AiCpu,
                PlayerType::AiCpu => PlayerType::AiGpu,
                PlayerType::AiGpu => {
                    // GPU-Native only available for Othello
                    if self.selected_game_type == GameType::Othello {
                        PlayerType::AiGpuNative
                    } else {
                        PlayerType::Human
                    }
                },
                PlayerType::AiGpuNative => PlayerType::Human,
            };
            self.player_types[index] = (*id, next_type);
            self.needs_redraw = true;
        }
    }

    /// Go back to previous screen
    pub fn go_back(&mut self) {
        match self.mode {
            GuiMode::Settings | GuiMode::HowToPlay => self.mode = GuiMode::GameSelection,
            GuiMode::PlayerConfig => self.mode = GuiMode::GameSelection,
            GuiMode::InGame | GuiMode::GameOver => {
                self.ai_worker.stop();
                self.mode = GuiMode::GameSelection;
            }
            GuiMode::GameSelection => self.should_quit = true,
        }
        self.needs_redraw = true;
    }

    /// Toggle the active tab between Debug Stats and Move History
    pub fn toggle_tab(&mut self) {
        self.active_tab = self.active_tab.next();
        self.needs_redraw = true;
    }

    /// Scroll debug stats up
    pub fn scroll_debug_up(&mut self) {
        self.debug_scroll = (self.debug_scroll - 1).max(0);
        self.needs_redraw = true;
    }

    /// Scroll debug stats down
    pub fn scroll_debug_down(&mut self) {
        self.debug_scroll += 1;
        self.needs_redraw = true;
    }

    /// Scroll move history up
    pub fn scroll_history_up(&mut self) {
        self.history_scroll = (self.history_scroll - 1).max(0);
        self.needs_redraw = true;
    }

    /// Scroll move history down
    pub fn scroll_history_down(&mut self) {
        self.history_scroll += 1;
        self.needs_redraw = true;
    }

    /// Scroll how to play up
    pub fn scroll_how_to_play_up(&mut self) {
        self.how_to_play_scroll = (self.how_to_play_scroll - 1).max(0);
        self.needs_redraw = true;
    }

    /// Scroll how to play down
    pub fn scroll_how_to_play_down(&mut self) {
        self.how_to_play_scroll += 1;
        self.needs_redraw = true;
    }

    /// Adjust a settings value
    pub fn adjust_setting(&mut self, index: usize, delta: i32) {
        match index {
            0 => { // Board Size
                self.board_size = ((self.board_size as i32 + delta).max(5).min(25)) as usize;
            }
            1 => { // Line Size
                self.line_size = ((self.line_size as i32 + delta).max(3).min(10)) as usize;
            }
            2 => { // AI Threads
                self.ai_threads = ((self.ai_threads as i32 + delta).max(1).min(64)) as usize;
            }
            3 => { // Max Nodes
                let step = if delta > 0 { 100000 } else { -100000 };
                self.max_nodes = ((self.max_nodes as i64 + step).max(100000).min(50000000)) as usize;
            }
            4 => { // Search Iterations
                let step = if delta > 0 { 100000 } else { -100000 };
                self.search_iterations = ((self.search_iterations as i64 + step).max(10000).min(100000000)) as u32;
            }
            5 => { // CPU Exploration Constant
                let step = if delta > 0 { 0.1 } else { -0.1 };
                self.cpu_exploration_constant = (self.cpu_exploration_constant + step).max(0.1).min(10.0);
            }
            6 => { // GPU Exploration Constant
                let step = if delta > 0 { 0.1 } else { -0.1 };
                self.gpu_exploration_constant = (self.gpu_exploration_constant + step).max(0.1).min(10.0);
            }
            7 => { // GPU Virtual Loss Weight
                let step = if delta > 0 { 0.5 } else { -0.5 };
                self.gpu_virtual_loss_weight = (self.gpu_virtual_loss_weight + step).max(0.1).min(20.0);
            }
            8 => { // Timeout
                self.timeout_secs = ((self.timeout_secs as i64 + delta as i64).max(1).min(600)) as u64;
            }
            9 => { // Stats Interval
                self.stats_interval_secs = ((self.stats_interval_secs as i64 + delta as i64).max(1).min(120)) as u64;
            }
            10 => { // AI Only
                self.ai_only = !self.ai_only;
            }
            11 => { // Shared Tree
                self.shared_tree = !self.shared_tree;
            }
            12 => { // GPU Threads
                let step = if delta > 0 { 256 } else { -256 };
                self.gpu_threads = ((self.gpu_threads as i32 + step).max(256).min(16384)) as usize;
            }
            13 => { // GPU-Native Batch Size
                let step = if delta > 0 { 1024 } else { -1024 };
                self.gpu_native_batch_size = ((self.gpu_native_batch_size as i32 + step).max(1024).min(32768)) as u32;
            }
            _ => {}
        }
        self.needs_redraw = true;
    }

    /// Get settings display strings
    pub fn get_settings_items(&self) -> Vec<(String, String)> {
        vec![
            ("Board Size".to_string(), self.board_size.to_string()),
            ("Line Size".to_string(), self.line_size.to_string()),
            ("AI Threads".to_string(), self.ai_threads.to_string()),
            ("Max Nodes".to_string(), format!("{}K", self.max_nodes / 1000)),
            ("Search Iterations".to_string(), format!("{}K", self.search_iterations / 1000)),
            ("CPU Exploration".to_string(), format!("{:.2}", self.cpu_exploration_constant)),
            ("GPU Exploration".to_string(), format!("{:.2}", self.gpu_exploration_constant)),
            ("GPU Virtual Loss".to_string(), format!("{:.2}", self.gpu_virtual_loss_weight)),
            ("Timeout (secs)".to_string(), self.timeout_secs.to_string()),
            ("Stats Interval (secs)".to_string(), self.stats_interval_secs.to_string()),
            ("AI Only Mode".to_string(), if self.ai_only { "Yes" } else { "No" }.to_string()),
            ("Shared Tree".to_string(), if self.shared_tree { "Yes" } else { "No" }.to_string()),
            ("GPU Threads".to_string(), self.gpu_threads.to_string()),
            ("GPU-Native Batch".to_string(), self.gpu_native_batch_size.to_string()),
        ]
    }

    /// Check if current player is AI
    pub fn is_current_player_ai(&self) -> bool {
        let current_player = self.game.get_current_player();
        self.player_types
            .iter()
            .find(|(id, _)| *id == current_player)
            .map(|(_, pt)| matches!(pt, PlayerType::AiCpu | PlayerType::AiGpu | PlayerType::AiGpuNative))
            .unwrap_or(false)
    }

    /// Get formatted move history for display
    pub fn get_formatted_history(&self) -> Vec<String> {
        self.move_history
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let player_name = self.game_renderer.player_name(entry.player);
                format!("{}. {} - {}", i + 1, player_name, entry.move_made)
            })
            .collect()
    }

    /// Get formatted debug stats for display
    /// 
    /// Returns formatted lines that fit within typical panel widths.
    /// Long move strings are shortened for better display.
    pub fn get_debug_stats_lines(&self) -> Vec<String> {
        let mut lines = vec!["Debug Statistics".to_string(), String::new()];
        
        if let Some(stats) = &self.last_search_stats {
            lines.push("AI Status: Active".to_string());
            lines.push(format!("Total Nodes: {}", stats.total_nodes));
            lines.push(format!("Root Visits: {}", stats.root_visits));
            lines.push(format!("Root Value: {:.3}", stats.root_value));
            lines.push(String::new());
            
            // Sort children by visits
            let mut sorted_children: Vec<_> = stats.children_stats.iter().collect();
            sorted_children.sort_by_key(|(_, (_, visits))| *visits);
            sorted_children.reverse();
            
            lines.push("Top AI Moves (Row, Col):".to_string());
            for (i, (move_str, (value, visits))) in sorted_children.iter().take(10).enumerate() {
                // Shorten the move string for display - extract just the move part
                let short_move = shorten_move_string(move_str);
                lines.push(format!("{}. {}", i + 1, short_move));
                lines.push(format!("   Q={:.3} ({} visits)", value, visits));
            }
        } else {
            lines.push("AI Status: Idle".to_string());
            lines.push("Waiting for MCTS...".to_string());
        }
        
        lines
    }

    /// Copy move history to clipboard
    ///
    /// Formats the move history as a readable string and copies it to the system clipboard.
    /// Returns true if successful, false otherwise.
    pub fn copy_history_to_clipboard(&self) -> bool {
        let history_text = self.game_controller.format_history_for_clipboard();
        crate::clipboard::copy_history_to_clipboard(&history_text)
    }

    /// Get clipboard-ready formatted history
    ///
    /// Returns the move history formatted for clipboard, using the GameController's
    /// authoritative history.
    pub fn get_clipboard_history(&self) -> String {
        self.game_controller.format_history_for_clipboard()
    }
}

/// Shorten a move string for display in the debug panel
/// 
/// Extracts the essential move information from verbose move strings like
/// "Gomoku(GomokuMove(3, 3))" -> "G(3,3)"
fn shorten_move_string(move_str: &str) -> String {
    // Try to extract coordinates from patterns like "Move(x, y)" or "(x, y)"
    if let Some(start) = move_str.find('(') {
        if let Some(end) = move_str.rfind(')') {
            // Get game prefix (first letter)
            let prefix = move_str.chars().next().unwrap_or('?');
            // Get the inner content
            let inner = &move_str[start..=end];
            // Try to simplify nested parens like "GomokuMove(3, 3)" -> "(3,3)"
            if let Some(inner_start) = inner.find('(') {
                if inner_start > 0 {
                    // There's nested content, extract innermost
                    let coords = &inner[inner_start..];
                    // Remove spaces for compactness
                    let compact = coords.replace(" ", "");
                    return format!("{}{}", prefix, compact);
                }
            }
            // Just use the parenthetical part
            let compact = inner.replace(" ", "");
            return format!("{}{}", prefix, compact);
        }
    }
    // Fallback: truncate long strings
    if move_str.len() > 25 {
        format!("{}...", &move_str[..22])
    } else {
        move_str.to_string()
    }
}
