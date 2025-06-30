//! # Application State and Core Components
//!
//! This module defines the core data structures and components that manage the
//! application's state, including UI state, player types, AI workers, and communication
//! channels between the UI, game logic, and AI threads.

use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::tui::layout::LayoutConfig;
use crate::tui::mouse::DragState;
use crate::tui::blokus_ui::BlokusUIConfig;
use mcts::{GameState, MCTS};
use ratatui::widgets::ListState;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, atomic::AtomicBool};
use std::thread::{self, JoinHandle};
use std::time::SystemTime;

/// Represents a single move in the game's move history
///
/// Tracks when a move was made, which player made it, and what the move was.
/// Used for game replay and analysis.
#[derive(Debug, Clone)]
pub struct MoveHistoryEntry {
    pub timestamp: SystemTime,
    pub player: i32,
    pub a_move: MoveWrapper,
}

impl MoveHistoryEntry {
    pub fn new(player: i32, a_move: MoveWrapper) -> Self {
        Self {
            timestamp: SystemTime::now(),
            player,
            a_move,
        }
    }
}

/// Messages sent to AI worker threads
///
/// Controls AI behavior and requests information from the AI engine.
#[derive(Debug)]
pub enum AIRequest {
    Search(GameWrapper, u64), // GameWrapper and timeout in seconds
    AdvanceRoot(MoveWrapper), // Advance the MCTS tree root to reflect a move that was made
    Stop,
}

/// Messages received from AI worker threads
///
/// Provides AI moves, status updates, and analysis information.
#[derive(Debug)]
pub enum AIResponse {
    Move(MoveWrapper, mcts::SearchStatistics),
}

/// The AI worker that runs in a separate thread
///
/// Handles MCTS search requests and manages the search tree.
pub struct AIWorker {
    handle: Option<JoinHandle<()>>,
    tx_req: Sender<AIRequest>,
    rx_resp: Receiver<AIResponse>,
    stop_flag: Arc<AtomicBool>,
}

impl AIWorker {
    pub fn new(exploration_constant: f64, num_threads: usize, max_nodes: usize) -> Self {
        let (tx_req, rx_req) = mpsc::channel();
        let (tx_resp, rx_resp) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = stop_flag.clone();

        let handle = thread::spawn(move || {
            let mut mcts: Option<MCTS<GameWrapper>> = None;

            for request in rx_req {
                match request {
                    AIRequest::Search(state, timeout_secs) => {
                        if stop_flag_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        
                        if mcts.is_none() {
                            mcts = Some(MCTS::new(
                                exploration_constant,
                                num_threads,
                                max_nodes,
                            ));
                        }
                        let mcts_ref = mcts.as_mut().unwrap();
                        
                        // Use timeout_secs directly and pass the stop flag for external interruption
                        let (best_move, stats) =
                            mcts_ref.search_with_stop(&state, 1000000, 1, timeout_secs, Some(stop_flag_clone.clone()));
                        
                        // Only send response if we haven't been stopped
                        if !stop_flag_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            tx_resp
                                .send(AIResponse::Move(best_move, stats))
                                .ok(); // Ignore send errors if receiver is dropped
                        }
                    }
                    AIRequest::AdvanceRoot(move_made) => {
                        if let Some(mcts_ref) = mcts.as_mut() {
                            mcts_ref.advance_root(&move_made);
                        }
                    }
                    AIRequest::Stop => break,
                }
            }
        });

        Self {
            handle: Some(handle),
            tx_req,
            rx_resp,
            stop_flag,
        }
    }

    pub fn start_search(&self, state: GameWrapper, timeout_secs: u64) {
        self.tx_req.send(AIRequest::Search(state, timeout_secs)).unwrap();
    }

    pub fn try_recv(&self) -> Option<AIResponse> {
        self.rx_resp.try_recv().ok()
    }

    /// Explicitly stop the AI worker
    pub fn stop(&self) {
        // Set the stop flag first to interrupt any ongoing search
        self.stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        // Then send the stop message to break the worker loop
        self.tx_req.send(AIRequest::Stop).ok();
    }

    pub fn advance_root(&self, move_made: &MoveWrapper) {
        self.tx_req.send(AIRequest::AdvanceRoot(move_made.clone())).ok();
    }
}

impl Drop for AIWorker {
    fn drop(&mut self) {
        // Stop the worker gracefully
        self.stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        self.tx_req.send(AIRequest::Stop).ok();
        
        // Wait for the thread to finish, but with a timeout to avoid hanging
        if let Some(handle) = self.handle.take() {
            // Give the thread up to 1 second to finish gracefully
            // If it doesn't finish in time, it will be forcefully terminated
            // when the process exits
            let _ = handle.join();
        }
    }
}

/// Type of player (human or AI)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    Human,
    AI,
}

/// Current state of the application
///
/// Controls which screen/menu is currently displayed to the user.
/// The application transitions between these states based on user input.
#[derive(PartialEq)]
pub enum AppMode {
    GameSelection,
    Settings,
    PlayerConfig,
    InGame,
    GameOver,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameStatus {
    InProgress,
    Win(i32),
    Draw,
}

/// The main application state
///
/// This struct holds all the state required to run the application,
/// including the game state, UI state, AI workers, and communication channels.
pub struct App {
    pub should_quit: bool,
    pub mode: AppMode,
    pub games: Vec<(&'static str, Box<dyn Fn() -> GameWrapper>)>, // (name, factory)
    pub game_selection_state: ListState,
    pub game_wrapper: GameWrapper,
    pub game_status: GameStatus,
    pub player_options: Vec<(i32, Player)>, // (player_id, type)
    pub selected_player_config_index: usize,
    pub ai_worker: AIWorker,
    pub last_search_stats: Option<mcts::SearchStatistics>,
    pub move_history: Vec<MoveHistoryEntry>,
    pub show_debug: bool,
    pub board_cursor: (u16, u16),
    pub selected_blokus_piece: Option<(usize, usize)>,
    pub history_scroll: u16,
    pub debug_scroll: u16,
    // Auto-scroll for move history
    pub history_auto_scroll: bool,
    pub history_user_scroll_time: Option<std::time::Instant>,
    pub history_auto_scroll_reset_duration: std::time::Duration,
    // AI timing and status
    pub ai_thinking_start: Option<std::time::Instant>,
    pub ai_minimum_display_duration: std::time::Duration,
    pub pending_ai_response: Option<(MoveWrapper, mcts::SearchStatistics)>,
    // Settings
    pub settings_board_size: usize,
    pub settings_line_size: usize,
    pub settings_ai_threads: usize,
    pub settings_max_nodes: usize,
    pub settings_exploration_constant: f64,
    pub selected_settings_index: usize,
    // AI behavior settings
    pub timeout_secs: u64,
    pub stats_interval_secs: u64,
    pub ai_only: bool,
    pub shared_tree: bool,
    // Enhanced UI components
    pub layout_config: LayoutConfig,
    pub drag_state: DragState,
    pub blokus_ui_config: BlokusUIConfig,
}

impl App {
    pub fn new(
        exploration_constant: f64,
        num_threads: usize,
        max_nodes: usize,
        game_name: Option<String>,
        board_size: usize,
        line_size: usize,
        timeout_secs: u64,
        stats_interval_secs: u64,
        ai_only: bool,
        shared_tree: bool,
    ) -> Self {
        // Set default values if not provided
        let gomoku_board_size = if board_size == 15 { 15 } else { board_size };
        let gomoku_line_size = if line_size == 5 { 5 } else { line_size };
        
        let connect4_width = if board_size == 15 { 7 } else { board_size }; // Default Connect4 width is 7
        let connect4_height = if board_size == 15 { 6 } else { board_size.saturating_sub(1).max(4) }; // Default Connect4 height is 6
        let connect4_line_size = if line_size == 5 { 4 } else { line_size }; // Default Connect4 line is 4
        
        let othello_board_size = if board_size == 15 { 8 } else { 
            // Ensure even number for Othello
            if board_size % 2 == 0 { board_size } else { board_size + 1 }
        };

        let games: Vec<(&'static str, Box<dyn Fn() -> GameWrapper>)> = vec![
            (
                "Gomoku",
                Box::new(move || {
                    GameWrapper::Gomoku(crate::games::gomoku::GomokuState::new(
                        gomoku_board_size, gomoku_line_size,
                    ))
                }),
            ),
            (
                "Connect4",
                Box::new(move || {
                    GameWrapper::Connect4(crate::games::connect4::Connect4State::new(
                        connect4_width,
                        connect4_height,
                        connect4_line_size,
                    ))
                }),
            ),
            (
                "Othello",
                Box::new(move || {
                    GameWrapper::Othello(crate::games::othello::OthelloState::new(othello_board_size))
                }),
            ),
            (
                "Blokus",
                Box::new(|| GameWrapper::Blokus(crate::games::blokus::BlokusState::new())),
            ),
        ];

        let has_specific_game = game_name.is_some();
        let (initial_mode, initial_game_index) = if let Some(name) = game_name {
            let game_index = games
                .iter()
                .position(|(game_name, _)| *game_name == name)
                .unwrap_or(0);
            // If AI-only mode is enabled, skip player config and go straight to game
            if ai_only {
                (AppMode::InGame, game_index)
            } else {
                (AppMode::PlayerConfig, game_index)
            }
        } else {
            (AppMode::GameSelection, 0)
        };

        let game_wrapper = games[initial_game_index].1();
        let mut player_options: Vec<(i32, Player)> = (1..=game_wrapper.get_num_players())
            .map(|i| (i, Player::Human))
            .collect();

        // If AI-only mode and a specific game was selected, configure all players as AI
        let is_ai_only_with_game = ai_only && has_specific_game;
        if is_ai_only_with_game {
            for (_, player_type) in &mut player_options {
                *player_type = Player::AI;
            }
        }

        // Set initial cursor position for AI-only mode
        let initial_cursor = if is_ai_only_with_game {
            match &game_wrapper {
                GameWrapper::Gomoku(_) => {
                    let board = game_wrapper.get_board();
                    let size = board.len();
                    (size / 2, size / 2)
                }
                GameWrapper::Connect4(_) => {
                    let board = game_wrapper.get_board();
                    let width = if !board.is_empty() { board[0].len() } else { 7 };
                    (0, width / 2)
                }
                GameWrapper::Othello(_) => {
                    let board = game_wrapper.get_board();
                    let size = board.len();
                    (size / 2 - 1, size / 2 - 1)
                }
                GameWrapper::Blokus(_) => (0, 0), // Start at corner for first move
            }
        } else {
            (0, 0)
        };

        let mut game_selection_state = ListState::default();
        game_selection_state.select(Some(initial_game_index));

        Self {
            should_quit: false,
            mode: initial_mode,
            games,
            game_selection_state,
            game_wrapper,
            game_status: GameStatus::InProgress,
            player_options,
            selected_player_config_index: 0,
            ai_worker: AIWorker::new(exploration_constant, num_threads, max_nodes),
            last_search_stats: None,
            move_history: Vec::new(),
            show_debug: false,
            board_cursor: (initial_cursor.0 as u16, initial_cursor.1 as u16),
            selected_blokus_piece: None,
            history_scroll: 0,
            debug_scroll: 0,
            // Auto-scroll for move history
            history_auto_scroll: true,
            history_user_scroll_time: None,
            history_auto_scroll_reset_duration: std::time::Duration::from_secs(20), // 20 seconds as requested
            // AI timing and status
            ai_thinking_start: None,
            ai_minimum_display_duration: std::time::Duration::from_millis(500), // Minimum 0.5 seconds display
            pending_ai_response: None,
            // Initialize settings with current values
            settings_board_size: if board_size == 15 { 15 } else { board_size }, // Keep 15 as standard Gomoku default
            settings_line_size: if line_size == 5 { 5 } else { line_size }, // Keep 5 as standard Gomoku default
            settings_ai_threads: num_threads,
            settings_max_nodes: max_nodes,
            settings_exploration_constant: exploration_constant,
            selected_settings_index: 0,
            // AI behavior settings
            timeout_secs,
            stats_interval_secs,
            ai_only,
            shared_tree,
            // Enhanced UI components
            layout_config: LayoutConfig::default(),
            drag_state: DragState::default(),
            blokus_ui_config: BlokusUIConfig::default(),
        }
    }

    pub fn update(&mut self) {
        if self.mode == AppMode::InGame {
            if self.game_status == GameStatus::InProgress {
                if self.is_current_player_ai() {
                    if self.ai_thinking_start.is_none() {
                        self.ai_thinking_start = Some(std::time::Instant::now());
                        self.ai_worker.start_search(self.game_wrapper.clone(), self.timeout_secs);
                    }
                }

                // Check if we have a pending AI response that we're ready to process
                if let Some((best_move, stats)) = self.pending_ai_response.take() {
                    // Ensure the AI timer has been displayed for at least the minimum duration
                    let should_process_move = if let Some(start_time) = self.ai_thinking_start {
                        start_time.elapsed() >= self.ai_minimum_display_duration
                    } else {
                        true // No timer was set, process immediately
                    };

                    if should_process_move {
                        self.ai_thinking_start = None; // Reset thinking timer
                        self.move_history.push(MoveHistoryEntry::new(
                            self.game_wrapper.get_current_player(),
                            best_move.clone(),
                        ));
                        self.on_move_added(); // Auto-scroll to bottom
                        self.game_wrapper.make_move(&best_move);
                        self.last_search_stats = Some(stats);
                        
                        // Advance the AI worker's MCTS tree root to reflect the move that was just made
                        self.ai_worker.advance_root(&best_move);
                        
                        // Clear selected piece if it becomes unavailable after move
                        if matches!(self.game_wrapper, GameWrapper::Blokus(_)) {
                            self.clear_selected_piece_if_unavailable();
                        }
                        
                        self.check_game_over();
                    } else {
                        // Put the response back until the minimum time has elapsed
                        self.pending_ai_response = Some((best_move, stats));
                    }
                }

                // Check for new AI responses
                if let Some(response) = self.ai_worker.try_recv() {
                    match response {
                        AIResponse::Move(best_move, stats) => {
                            // Store the response for delayed processing
                            self.pending_ai_response = Some((best_move, stats));
                        }
                    }
                }
            }
        }
        
        // Handle auto-scroll reset timer for move history
        self.update_history_auto_scroll();
    }

    pub fn get_selected_game_name(&self) -> &'static str {
        self.games[self.game_selection_state.selected().unwrap_or(0)].0
    }

    pub fn select_next_game(&mut self) {
        let i = match self.game_selection_state.selected() {
            Some(i) => (i + 1) % (self.games.len() + 2), // +2 for Settings and Quit
            None => 0,
        };
        self.game_selection_state.select(Some(i));
    }

    pub fn select_prev_game(&mut self) {
        let i = match self.game_selection_state.selected() {
            Some(i) => (i + self.games.len() + 1) % (self.games.len() + 2),
            None => 0,
        };
        self.game_selection_state.select(Some(i));
    }

    pub fn start_game(&mut self) {
        if let Some(selected) = self.game_selection_state.selected() {
            if selected < self.games.len() {
                let factory = &self.games[selected].1;
                self.game_wrapper = factory();
                self.game_status = GameStatus::InProgress;
                self.last_search_stats = None;
                self.move_history.clear();

                let num_players = self.game_wrapper.get_num_players();

                // Only reset player options if we don't have the right number of players
                // or if we don't have any player options configured yet
                if self.player_options.is_empty() || self.player_options.len() != num_players as usize {
                    self.player_options = (1..=num_players).map(|i| (i, Player::Human)).collect();
                    self.selected_player_config_index = 0;
                }

                // If AI-only mode is enabled, skip player config and go straight to game
                if self.ai_only {
                    // Set all players to AI
                    for (_, player_type) in &mut self.player_options {
                        *player_type = Player::AI;
                    }
                    
                    // Set initial cursor position and go straight to game
                    let (initial_row, initial_col) = match &self.game_wrapper {
                        GameWrapper::Gomoku(_) => {
                            let board = self.game_wrapper.get_board();
                            let size = board.len();
                            (size / 2, size / 2)
                        }
                        GameWrapper::Connect4(_) => {
                            let board = self.game_wrapper.get_board();
                            let width = if !board.is_empty() { board[0].len() } else { 7 };
                            (0, width / 2)
                        }
                        GameWrapper::Othello(_) => {
                            let board = self.game_wrapper.get_board();
                            let size = board.len();
                            (size / 2 - 1, size / 2 - 1)
                        }
                        GameWrapper::Blokus(_) => (10, 10), // Center of Blokus board
                    };
                    
                    self.board_cursor = (initial_row as u16, initial_col as u16);
                    self.mode = AppMode::InGame;
                } else {
                    // Normal mode: go to player configuration
                    self.mode = AppMode::PlayerConfig;
                }
            } else if selected == self.games.len() + 1 {
                // This is the "Quit" button
                self.should_quit = true;
            }
        }
    }

    pub fn select_next_player_config(&mut self) {
        // Include the "Start Game" option in navigation
        let max_index = self.player_options.len(); // Start Game is at index len()
        self.selected_player_config_index =
            (self.selected_player_config_index + 1) % (max_index + 1);
    }

    pub fn select_prev_player_config(&mut self) {
        // Include the "Start Game" option in navigation
        let max_index = self.player_options.len(); // Start Game is at index len()
        self.selected_player_config_index =
            (self.selected_player_config_index + max_index) % (max_index + 1);
    }

    pub fn cycle_player_type(&mut self) {
        if self.selected_player_config_index < self.player_options.len() {
            let (_, player_type) = &mut self.player_options[self.selected_player_config_index];
            *player_type = match *player_type {
                Player::Human => Player::AI,
                Player::AI => Player::Human,
            };
        }
    }

    pub fn confirm_player_config(&mut self) {
        self.mode = AppMode::InGame;
        
        // Set initial cursor position based on game type
        let (initial_row, initial_col) = match &self.game_wrapper {
            GameWrapper::Gomoku(_) => {
                let board = self.game_wrapper.get_board();
                let size = board.len();
                (size / 2, size / 2)
            }
            GameWrapper::Connect4(_) => {
                let board = self.game_wrapper.get_board();
                let width = if !board.is_empty() { board[0].len() } else { 7 };
                (0, width / 2)
            }
            GameWrapper::Othello(_) => {
                let board = self.game_wrapper.get_board();
                let size = board.len();
                (size / 2 - 1, size / 2 - 1)
            }
            GameWrapper::Blokus(_) => (10, 10), // Center of Blokus board
        };
        
        self.board_cursor = (initial_row as u16, initial_col as u16);
    }

    pub fn reset_game(&mut self) {
        // Get the currently selected game and reset its state without changing player config
        if let Some(selected) = self.game_selection_state.selected() {
            if selected < self.games.len() {
                let factory = &self.games[selected].1;
                self.game_wrapper = factory();
                self.game_status = GameStatus::InProgress;
                self.last_search_stats = None;
                self.move_history.clear();

                // Reset cursor position based on game type
                let (initial_row, initial_col) = match &self.game_wrapper {
                    GameWrapper::Gomoku(_) => {
                        let board = self.game_wrapper.get_board();
                        let size = board.len();
                        (size / 2, size / 2)
                    }
                    GameWrapper::Connect4(_) => {
                        let board = self.game_wrapper.get_board();
                        let width = if !board.is_empty() { board[0].len() } else { 7 };
                        (0, width / 2)
                    }
                    GameWrapper::Othello(_) => {
                        let board = self.game_wrapper.get_board();
                        let size = board.len();
                        (size / 2 - 1, size / 2 - 1)
                    }
                    GameWrapper::Blokus(_) => (10, 10), // Center of Blokus board
                };
                
                self.board_cursor = (initial_row as u16, initial_col as u16);
                self.mode = AppMode::InGame;
                
                // Clear any Blokus-specific selections
                if matches!(self.game_wrapper, GameWrapper::Blokus(_)) {
                    self.blokus_ui_config.selected_piece_idx = None;
                    self.blokus_ui_config.selected_transformation_idx = 0;
                }
            }
        }
    }

    // Settings navigation methods
    pub fn select_next_setting(&mut self) {
        self.selected_settings_index = (self.selected_settings_index + 1) % 11; // 9 settings + separator + back
    }

    pub fn select_prev_setting(&mut self) {
        self.selected_settings_index = (self.selected_settings_index + 10) % 11;
    }

    pub fn increase_setting(&mut self) {
        match self.selected_settings_index {
            0 => self.settings_board_size = (self.settings_board_size + 1).min(25),
            1 => self.settings_line_size = (self.settings_line_size + 1).min(10),
            2 => self.settings_ai_threads = (self.settings_ai_threads + 1).min(16),
            3 => self.settings_max_nodes = (self.settings_max_nodes + 100000).min(10000000),
            4 => self.settings_exploration_constant = (self.settings_exploration_constant + 0.1).min(10.0),
            5 => self.timeout_secs = (self.timeout_secs + 10).min(600), // Max 10 minutes
            6 => self.stats_interval_secs = (self.stats_interval_secs + 5).min(120), // Max 2 minutes
            7 => self.ai_only = !self.ai_only, // Toggle
            8 => self.shared_tree = !self.shared_tree, // Toggle
            _ => {} // separator or back
        }
    }

    pub fn decrease_setting(&mut self) {
        match self.selected_settings_index {
            0 => self.settings_board_size = self.settings_board_size.saturating_sub(1).max(3),
            1 => self.settings_line_size = self.settings_line_size.saturating_sub(1).max(3),
            2 => self.settings_ai_threads = self.settings_ai_threads.saturating_sub(1).max(1),
            3 => self.settings_max_nodes = self.settings_max_nodes.saturating_sub(100000).max(10000),
            4 => self.settings_exploration_constant = (self.settings_exploration_constant - 0.1).max(0.1),
            5 => self.timeout_secs = self.timeout_secs.saturating_sub(10).max(5), // Min 5 seconds
            6 => self.stats_interval_secs = self.stats_interval_secs.saturating_sub(5).max(5), // Min 5 seconds
            7 => self.ai_only = !self.ai_only, // Toggle
            8 => self.shared_tree = !self.shared_tree, // Toggle
            _ => {} // separator or back
        }
    }

    /// Gracefully shut down the application
    /// This ensures all threads are properly stopped before exiting
    pub fn shutdown(&mut self) {
        // Explicitly stop the AI worker
        self.ai_worker.stop();
        
        // Give threads more time to shut down gracefully
        // This is especially important when AI is in the middle of a search
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    // Debug and history scrolling methods
    pub fn scroll_debug_up(&mut self) {
        self.debug_scroll = self.debug_scroll.saturating_sub(1);
    }

    pub fn scroll_debug_down(&mut self) {
        self.debug_scroll = self.debug_scroll.saturating_add(1);
    }

    pub fn scroll_move_history_up(&mut self) {
        self.history_scroll = self.history_scroll.saturating_sub(1);
        self.on_user_history_scroll();
    }

    pub fn scroll_move_history_down(&mut self) {
        self.history_scroll = self.history_scroll.saturating_add(1);
        self.on_user_history_scroll();
    }

    // Enhanced Blokus functionality
    pub fn blokus_select_piece(&mut self, piece_idx: usize) {
        // Only allow selection of available pieces
        if let GameWrapper::Blokus(state) = &self.game_wrapper {
            let available_pieces = state.get_available_pieces(state.get_current_player());
            if available_pieces.contains(&piece_idx) {
                self.blokus_ui_config.select_piece(piece_idx);
                // Try to find a valid cursor position for this piece
                self.find_valid_cursor_position_for_piece(piece_idx, 0);
            }
            // If piece is not available, do nothing (no selection change)
        }
    }

    /// Check if a Blokus piece would fit within board bounds at the given position
    fn would_blokus_piece_fit_at_cursor(&self, piece_idx: usize, transformation_idx: usize) -> bool {
        if let GameWrapper::Blokus(state) = &self.game_wrapper {
            let board = state.get_board();
            let board_height = board.len();
            let board_width = if board_height > 0 { board[0].len() } else { 0 };
            
            // Check if this piece is available for the current player
            let current_player = state.get_current_player();
            let available_pieces = state.get_available_pieces(current_player);
            if !available_pieces.contains(&piece_idx) {
                return false; // Piece not available
            }
            
            // Get the piece and its transformation
            let pieces = crate::games::blokus::get_blokus_pieces();
            if let Some(piece) = pieces.iter().find(|p| p.id == piece_idx) {
                if transformation_idx < piece.transformations.len() {
                    let shape = &piece.transformations[transformation_idx];
                    let cursor_row = self.board_cursor.0 as i32;
                    let cursor_col = self.board_cursor.1 as i32;
                    
                    // Check if all blocks of the piece would be within bounds
                    for &(dr, dc) in shape {
                        let board_r = cursor_row + dr;
                        let board_c = cursor_col + dc;
                        
                        // If any block would be out of bounds, piece doesn't fit
                        if board_r < 0 || board_r >= board_height as i32 || board_c < 0 || board_c >= board_width as i32 {
                            return false;
                        }
                    }
                    return true;
                }
            }
        }
        false
    }

    pub fn blokus_rotate_piece(&mut self) {
        if let Some(piece_idx) = self.blokus_ui_config.selected_piece_idx {
            // Get the selected piece to find out how many transformations it has
            if let GameWrapper::Blokus(state) = &self.game_wrapper {
                let available_pieces = state.get_available_pieces(state.get_current_player());
                if available_pieces.contains(&piece_idx) {
                    let pieces = crate::games::blokus::get_blokus_pieces();
                    if let Some(piece) = pieces.iter().find(|p| p.id == piece_idx) {
                        let current_transformation = self.blokus_ui_config.selected_transformation_idx;
                        let total_transformations = piece.transformations.len();
                        
                        if total_transformations > 0 {
                            // Calculate the next transformation index
                            let next_transformation = (current_transformation + 1) % total_transformations;
                            
                            // Check if the piece would fit at the current cursor position with the new transformation
                            if self.would_blokus_piece_fit_at_cursor(piece_idx, next_transformation) {
                                // Fits at current position, just rotate
                                self.blokus_ui_config.selected_transformation_idx = next_transformation;
                            } else {
                                // Doesn't fit at current position, find a new position and then rotate
                                // Temporarily set the transformation to see if we can find a valid position
                                let old_transformation = self.blokus_ui_config.selected_transformation_idx;
                                self.blokus_ui_config.selected_transformation_idx = next_transformation;
                                
                                if self.find_valid_cursor_position_for_piece(piece_idx, next_transformation) {
                                    // Found a valid position, keep the new transformation and position
                                    // (cursor was already moved by find_valid_cursor_position_for_piece)
                                } else {
                                    // Couldn't find a valid position, revert transformation
                                    self.blokus_ui_config.selected_transformation_idx = old_transformation;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn blokus_flip_piece(&mut self) {
        // TODO: Implement piece flipping logic
        // This would apply a reflection transformation
    }

    pub fn blokus_place_piece(&mut self) {
        if let Some((piece_idx, transformation_idx)) = self.blokus_ui_config.get_selected_piece_info() {
            if let GameWrapper::Blokus(state) = &mut self.game_wrapper {
                let blokus_move = crate::games::blokus::BlokusMove(
                    piece_idx,
                    transformation_idx,
                    self.board_cursor.0 as usize,
                    self.board_cursor.1 as usize,
                );
                
                // Check if the move is legal
                if state.is_legal(&blokus_move) {
                    let move_wrapper = crate::game_wrapper::MoveWrapper::Blokus(blokus_move);
                    
                    // Record the move in history
                    self.move_history.push(crate::app::MoveHistoryEntry::new(
                        self.game_wrapper.get_current_player(),
                        move_wrapper.clone(),
                    ));
                    self.on_move_added(); // Auto-scroll to bottom
                    
                    // Make the move
                    self.game_wrapper.make_move(&move_wrapper);
                    
                    // Advance the AI worker's MCTS tree root
                    self.ai_worker.advance_root(&move_wrapper);
                    
                    // Clear selection after placing
                    self.blokus_ui_config.selected_piece_idx = None;
                    self.blokus_ui_config.selected_transformation_idx = 0;
                    
                    // Check if game is over
                    self.check_game_over();
                }
            }
        }
    }

    pub fn blokus_pass_move(&mut self) {
        if let GameWrapper::Blokus(_) = &mut self.game_wrapper {
            // Create a pass move (usize::MAX is the PASS_MOVE constant)
            let pass_move = crate::games::blokus::BlokusMove(usize::MAX, 0, 0, 0);
            let move_wrapper = crate::game_wrapper::MoveWrapper::Blokus(pass_move);
            
            // Record the move in history
            self.move_history.push(crate::app::MoveHistoryEntry::new(
                self.game_wrapper.get_current_player(),
                move_wrapper.clone(),
            ));
            self.on_move_added(); // Auto-scroll to bottom
            
            // Make the pass move
            self.game_wrapper.make_move(&move_wrapper);
            
            // Advance AI tree
            self.ai_worker.advance_root(&move_wrapper);
            
            // Check for game over
            self.check_game_over();
        }
    }

    pub fn blokus_expand_all(&mut self) {
        self.blokus_ui_config.expand_all();
    }

    pub fn blokus_collapse_all(&mut self) {
        self.blokus_ui_config.collapse_all();
    }

    pub fn blokus_toggle_player_expand(&mut self, player_idx: usize) {
        self.blokus_ui_config.toggle_player_expand(player_idx);
    }

    pub fn reset_debug_scroll(&mut self) {
        self.debug_scroll = 0;
    }

    pub fn reset_history_scroll(&mut self) {
        self.history_scroll = 0;
        self.history_auto_scroll = true;
        self.history_user_scroll_time = None;
    }

    pub fn blokus_scroll_panel_up(&mut self) {
        self.blokus_ui_config.scroll_panel_up();
    }

    pub fn blokus_scroll_panel_down(&mut self) {
        self.blokus_ui_config.scroll_panel_down();
    }

    /// Update auto-scroll behavior for move history
    fn update_history_auto_scroll(&mut self) {
        // Check if we should reset auto-scroll after user interaction
        if let Some(user_scroll_time) = self.history_user_scroll_time {
            if user_scroll_time.elapsed() >= self.history_auto_scroll_reset_duration {
                self.history_auto_scroll = true;
                self.history_user_scroll_time = None;
            }
        }
    }

    /// Called when user manually scrolls the history - disables auto-scroll temporarily
    pub fn on_user_history_scroll(&mut self) {
        self.history_auto_scroll = false;
        self.history_user_scroll_time = Some(std::time::Instant::now());
    }

    /// Called when a new move is added - ensures we scroll to bottom if auto-scroll is enabled
    pub fn on_move_added(&mut self) {
        if self.history_auto_scroll {
            self.history_scroll = 0; // Reset to bottom (0 means show latest moves)
        }
    }

    /// Manually enable auto-scroll for move history
    pub fn enable_history_auto_scroll(&mut self) {
        self.history_auto_scroll = true;
        self.history_user_scroll_time = None;
        self.history_scroll = 0; // Go to bottom immediately
    }

    /// Map game player ID to UI player ID
    /// 
    /// Games like Connect4, Gomoku, and Othello use 1 and -1 for players,
    /// but our UI uses 1 and 2. This method provides the mapping.
    fn map_game_player_to_ui_player(&self, game_player_id: i32) -> i32 {
        match &self.game_wrapper {
            GameWrapper::Blokus(_) => game_player_id, // Blokus already uses 1,2,3,4
            _ => {
                // For 2-player games, map 1->1 and -1->2
                if game_player_id == 1 {
                    1
                } else if game_player_id == -1 {
                    2
                } else {
                    game_player_id // fallback
                }
            }
        }
    }

    /// Check if the current player is controlled by AI
    pub fn is_current_player_ai(&self) -> bool {
        let game_player_id = self.game_wrapper.get_current_player();
        let ui_player_id = self.map_game_player_to_ui_player(game_player_id);
        self.player_options
            .iter()
            .find(|(id, _)| *id == ui_player_id)
            .map(|(_, p_type)| *p_type == Player::AI)
            .unwrap_or(false)
    }

    /// Check if the game is over and update game status
    fn check_game_over(&mut self) {
        if self.game_wrapper.is_terminal() {
            self.game_status = match self.game_wrapper.get_winner() {
                Some(winner) => GameStatus::Win(winner),
                None => GameStatus::Draw,
            };
            self.mode = AppMode::GameOver;
        }
    }

    /// Clear selected Blokus piece if it becomes unavailable
    pub fn clear_selected_piece_if_unavailable(&mut self) {
        if let (Some(piece_idx), GameWrapper::Blokus(state)) = (self.blokus_ui_config.selected_piece_idx, &self.game_wrapper) {
            let available_pieces = state.get_available_pieces(state.get_current_player());
            if !available_pieces.contains(&piece_idx) {
                self.blokus_ui_config.selected_piece_idx = None;
                self.blokus_ui_config.selected_transformation_idx = 0;
            }
        }
    }

    /// Find a valid cursor position for the given Blokus piece and transformation
    fn find_valid_cursor_position_for_piece(&mut self, piece_idx: usize, transformation_idx: usize) -> bool {
        if let GameWrapper::Blokus(state) = &self.game_wrapper {
            let board = state.get_board();
            let board_height = board.len();
            let board_width = if board_height > 0 { board[0].len() } else { 0 };
            
            // Try positions starting from current cursor position, then spiral outward
            let start_row = self.board_cursor.0 as usize;
            let start_col = self.board_cursor.1 as usize;
            
            // First try the current position
            if self.would_blokus_piece_fit_at_cursor(piece_idx, transformation_idx) {
                return true;
            }
            
            // Try positions in expanding squares around the current position
            for radius in 1..=10 {
                for row in start_row.saturating_sub(radius)..=(start_row + radius).min(board_height - 1) {
                    for col in start_col.saturating_sub(radius)..=(start_col + radius).min(board_width - 1) {
                        // Only check the border of the current radius
                        if (row == start_row.saturating_sub(radius) || row == (start_row + radius).min(board_height - 1)) ||
                           (col == start_col.saturating_sub(radius) || col == (start_col + radius).min(board_width - 1)) {
                            
                            self.board_cursor = (row as u16, col as u16);
                            if self.would_blokus_piece_fit_at_cursor(piece_idx, transformation_idx) {
                                return true;
                            }
                        }
                    }
                }
            }
            
            // If no valid position found, reset cursor to original position
            self.board_cursor = (start_row as u16, start_col as u16);
        }
        false
    }

    /// Get the effective scroll offset for move history, considering auto-scroll
    pub fn get_history_scroll_offset(&self, content_height: usize, visible_height: usize) -> usize {
        if self.history_auto_scroll {
            // Auto-scroll: always show the bottom
            content_height.saturating_sub(visible_height)
        } else {
            // Manual scroll: use user-set offset
            let max_scroll = content_height.saturating_sub(visible_height);
            (self.history_scroll as usize).min(max_scroll)
        }
    }

    /// Get the color for a player that matches the board display
    pub fn get_player_color(&self, player_id: i32) -> ratatui::prelude::Color {
        use ratatui::prelude::Color;
        
        match &self.game_wrapper {
            GameWrapper::Connect4(_) => {
                // Connect4 uses game player IDs 1 and -1, map to UI colors
                if player_id == 1 { Color::Red } else { Color::Yellow }
            }
            GameWrapper::Othello(_) => {
                // Othello uses game player IDs 1 and -1, both display as white for contrast
                Color::White
            }
            GameWrapper::Blokus(_) => {
                // Blokus uses player IDs 1,2,3,4 directly
                match player_id {
                    1 => Color::Red,
                    2 => Color::Blue,
                    3 => Color::Green,
                    4 => Color::Yellow,
                    _ => Color::White,
                }
            }
            _ => { // Gomoku and others
                // Game uses 1 and -1, map to UI colors
                if player_id == 1 { Color::Red } else { Color::Blue }
            }
        }
    }

    /// Get the player symbol/marker for display
    pub fn get_player_symbol(&self, player_id: i32) -> &'static str {
        match &self.game_wrapper {
            GameWrapper::Connect4(_) => {
                if player_id == 1 { "🔴" } else { "🟡" }
            }
            GameWrapper::Othello(_) => {
                if player_id == 1 { "⚫" } else { "⚪" }
            }
            GameWrapper::Blokus(_) => {
                match player_id {
                    1 => "🟥", // Red square
                    2 => "🟦", // Blue square
                    3 => "🟩", // Green square
                    4 => "🟨", // Yellow square
                    _ => "⬜",
                }
            }
            _ => { // Gomoku and others
                if player_id == 1 { "❌" } else { "⭕" }
            }
        }
    }
}
