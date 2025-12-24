//! # GUI Application State
//!
//! Manages the application state for the Windows GUI, including game state,
//! AI coordination, and UI state.

use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, atomic::AtomicBool};
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime};

use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::games::blokus::BlokusState;
use crate::games::connect4::Connect4State;
use crate::games::gomoku::GomokuState;
use crate::games::othello::OthelloState;
use mcts::{GameState, MCTS, SearchStatistics};

use super::game_renderers::{GameRenderer, create_renderer_for_game};

/// Player type (human or AI)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerType {
    Human,
    AI,
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
}

impl GameType {
    pub fn name(&self) -> &'static str {
        match self {
            GameType::Gomoku => "Gomoku",
            GameType::Connect4 => "Connect 4",
            GameType::Othello => "Othello",
            GameType::Blokus => "Blokus",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            GameType::Gomoku => "Get five stones in a row to win",
            GameType::Connect4 => "Drop pieces to connect four",
            GameType::Othello => "Flip opponent's pieces by surrounding",
            GameType::Blokus => "Place polyomino pieces strategically",
        }
    }

    pub fn all() -> &'static [GameType] {
        &[GameType::Gomoku, GameType::Connect4, GameType::Othello, GameType::Blokus]
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
    Search(GameWrapper, u64),
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
        exploration_constant: f64,
        num_threads: usize,
        max_nodes: usize,
        search_iterations: u32,
    ) -> Self {
        use std::sync::mpsc::channel;

        let (tx_req, rx_req) = channel();
        let (tx_resp, rx_resp) = channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_clone = stop_flag.clone();

        let handle = std::thread::spawn(move || {
            let mut mcts: Option<MCTS<GameWrapper>> = None;

            for request in rx_req {
                match request {
                    AIRequest::Search(state, timeout) => {
                        if stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }

                        if mcts.is_none() {
                            mcts = Some(MCTS::new(exploration_constant, num_threads, max_nodes));
                        }

                        let (best_move, stats) = mcts.as_mut().unwrap().search_with_stop(
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

    pub fn start_search(&self, state: GameWrapper, timeout: u64) {
        let _ = self.tx.send(AIRequest::Search(state, timeout));
    }

    pub fn try_recv(&self) -> Option<AIResponse> {
        self.rx.try_recv().ok()
    }

    pub fn stop(&self) {
        self.stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = self.tx.send(AIRequest::Stop);
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

    // Game state
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
    pub max_nodes: usize,
    pub search_iterations: u32,
    pub exploration_constant: f64,
    pub stats_interval_secs: u64,
    pub ai_only: bool,
    pub shared_tree: bool,
    pub selected_settings_index: usize,

    // UI state
    pub needs_redraw: bool,
    pub hover_button: Option<usize>,
    pub active_tab: ActiveTab,
    pub debug_scroll: i32,
    pub history_scroll: i32,
    pub how_to_play_scroll: i32,
    pub selected_how_to_play_game: usize,
}

impl GuiApp {
    pub fn new(
        exploration_constant: f64,
        num_threads: usize,
        search_iterations: u32,
        max_nodes: usize,
        board_size: usize,
        line_size: usize,
        timeout_secs: u64,
        stats_interval_secs: u64,
        ai_only: bool,
        shared_tree: bool,
    ) -> Self {
        let default_game = GameWrapper::Gomoku(GomokuState::new(board_size, line_size));
        let renderer = create_renderer_for_game(&default_game);

        Self {
            mode: GuiMode::GameSelection,
            should_quit: false,
            selected_game_index: 0,
            selected_game_type: GameType::Gomoku,
            player_types: vec![(1, PlayerType::Human), (-1, PlayerType::AI)],
            selected_player_index: 0,
            game: default_game,
            game_status: GameStatus::InProgress,
            move_history: Vec::new(),
            game_renderer: renderer,
            ai_worker: AIWorker::new(exploration_constant, num_threads, max_nodes, search_iterations),
            ai_thinking: false,
            ai_thinking_start: None,
            last_search_stats: None,
            board_size,
            line_size,
            timeout_secs,
            ai_threads: num_threads,
            max_nodes,
            search_iterations,
            exploration_constant,
            stats_interval_secs,
            ai_only,
            shared_tree,
            selected_settings_index: 0,
            needs_redraw: true,
            hover_button: None,
            active_tab: ActiveTab::DebugStats,
            debug_scroll: 0,
            history_scroll: 0,
            how_to_play_scroll: 0,
            selected_how_to_play_game: 0,
        }
    }

    /// Start a new game with current settings
    pub fn start_game(&mut self) {
        // If AI-only mode, set all players to AI
        if self.ai_only {
            for (_, pt) in &mut self.player_types {
                *pt = PlayerType::AI;
            }
        }

        self.game = match self.selected_game_type {
            GameType::Gomoku => GameWrapper::Gomoku(GomokuState::new(self.board_size, self.line_size)),
            GameType::Connect4 => GameWrapper::Connect4(Connect4State::new(7, 6, self.line_size)),
            GameType::Othello => GameWrapper::Othello(OthelloState::new(8)),
            GameType::Blokus => GameWrapper::Blokus(BlokusState::new()),
        };

        self.game_renderer = create_renderer_for_game(&self.game);
        self.game_renderer.reset();
        self.game_status = GameStatus::InProgress;
        self.move_history.clear();
        self.ai_thinking = false;
        self.ai_thinking_start = None;
        self.last_search_stats = None;
        self.mode = GuiMode::InGame;
        self.needs_redraw = true;

        // Check if AI should move first
        self.check_ai_turn();
    }

    /// Check if it's AI's turn and start search if needed
    pub fn check_ai_turn(&mut self) {
        if self.game_status != GameStatus::InProgress {
            return;
        }

        let current_player = self.game.get_current_player();
        let is_ai = self.player_types
            .iter()
            .find(|(id, _)| *id == current_player)
            .map(|(_, pt)| *pt == PlayerType::AI)
            .unwrap_or(false);

        if is_ai && !self.ai_thinking {
            self.ai_thinking = true;
            self.ai_thinking_start = Some(Instant::now());
            self.ai_worker.start_search(self.game.clone(), self.timeout_secs);
            self.needs_redraw = true;
        }
    }

    /// Process a move (from human or AI)
    pub fn make_move(&mut self, mv: MoveWrapper) {
        let player = self.game.get_current_player();
        
        self.game.make_move(&mv);
        self.move_history.push(MoveEntry {
            timestamp: SystemTime::now(),
            player,
            move_made: mv,
        });

        // Check game status
        if self.game.is_terminal() {
            if let Some(winner) = self.game.get_winner() {
                self.game_status = GameStatus::Win(winner);
            } else {
                self.game_status = GameStatus::Draw;
            }
            self.mode = GuiMode::GameOver;
        }

        self.ai_thinking = false;
        self.needs_redraw = true;

        // Check if AI should move next
        self.check_ai_turn();
    }

    /// Update application state (called periodically)
    pub fn update(&mut self) {
        // Check for AI response
        if self.ai_thinking {
            if let Some(AIResponse::BestMove(mv, stats)) = self.ai_worker.try_recv() {
                self.last_search_stats = Some(stats);
                self.make_move(mv);
            }
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
                    (2, PlayerType::AI),
                    (3, PlayerType::AI),
                    (4, PlayerType::AI),
                ],
                _ => vec![
                    (1, PlayerType::Human),
                    (-1, PlayerType::AI),
                ],
            };
            
            self.mode = GuiMode::PlayerConfig;
            self.needs_redraw = true;
        }
    }

    /// Toggle player type
    pub fn toggle_player(&mut self, index: usize) {
        if index < self.player_types.len() {
            let (id, pt) = &self.player_types[index];
            self.player_types[index] = (*id, match pt {
                PlayerType::Human => PlayerType::AI,
                PlayerType::AI => PlayerType::Human,
            });
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
            5 => { // Exploration Constant
                let step = if delta > 0 { 0.1 } else { -0.1 };
                self.exploration_constant = (self.exploration_constant + step).max(0.1).min(10.0);
            }
            6 => { // Timeout
                self.timeout_secs = ((self.timeout_secs as i64 + delta as i64).max(1).min(600)) as u64;
            }
            7 => { // Stats Interval
                self.stats_interval_secs = ((self.stats_interval_secs as i64 + delta as i64).max(1).min(120)) as u64;
            }
            8 => { // AI Only
                self.ai_only = !self.ai_only;
            }
            9 => { // Shared Tree
                self.shared_tree = !self.shared_tree;
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
            ("Exploration Constant".to_string(), format!("{:.2}", self.exploration_constant)),
            ("Timeout (secs)".to_string(), self.timeout_secs.to_string()),
            ("Stats Interval (secs)".to_string(), self.stats_interval_secs.to_string()),
            ("AI Only Mode".to_string(), if self.ai_only { "Yes" } else { "No" }.to_string()),
            ("Shared Tree".to_string(), if self.shared_tree { "Yes" } else { "No" }.to_string()),
        ]
    }

    /// Check if current player is AI
    pub fn is_current_player_ai(&self) -> bool {
        let current_player = self.game.get_current_player();
        self.player_types
            .iter()
            .find(|(id, _)| *id == current_player)
            .map(|(_, pt)| *pt == PlayerType::AI)
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
    pub fn get_debug_stats_lines(&self) -> Vec<String> {
        let mut lines = vec!["Debug Statistics".to_string(), String::new()];
        
        if let Some(stats) = &self.last_search_stats {
            lines.push(format!("AI Status: Active"));
            lines.push(format!("Total Nodes: {}", stats.total_nodes));
            lines.push(format!("Root Visits: {}", stats.root_visits));
            lines.push(format!("Root Value: {:.3}", stats.root_value));
            lines.push(String::new());
            
            // Sort children by visits
            let mut sorted_children: Vec<_> = stats.children_stats.iter().collect();
            sorted_children.sort_by_key(|(_, (_, visits))| *visits);
            sorted_children.reverse();
            
            lines.push("Top AI Moves:".to_string());
            for (i, (move_str, (value, visits))) in sorted_children.iter().take(10).enumerate() {
                lines.push(format!("{}. {}: {:.3} ({})", i + 1, move_str, value, visits));
            }
        } else {
            lines.push("AI Status: Idle".to_string());
            lines.push("Waiting for MCTS statistics...".to_string());
        }
        
        lines
    }
}
