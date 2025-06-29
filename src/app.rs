//! # Application State and Core Components
//!
//! This module defines the core data structures and components that manage the
//! application's state, including UI state, player types, AI workers, and communication
//! channels between the UI, game logic, and AI threads.

use crate::game_wrapper::{GameWrapper, MoveWrapper};
use mcts::{GameState, MCTS};
use ratatui::widgets::ListState;
use std::sync::mpsc::{self, Receiver, Sender};
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
    Search(GameWrapper),
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
}

impl AIWorker {
    pub fn new(exploration_constant: f64, num_threads: usize, max_nodes: usize) -> Self {
        let (tx_req, rx_req) = mpsc::channel();
        let (tx_resp, rx_resp) = mpsc::channel();

        let handle = thread::spawn(move || {
            let mut mcts: Option<MCTS<GameWrapper>> = None;

            for request in rx_req {
                match request {
                    AIRequest::Search(state) => {
                        if mcts.is_none() {
                            mcts = Some(MCTS::new(
                                exploration_constant,
                                num_threads,
                                max_nodes,
                            ));
                        }
                        let mcts_ref = mcts.as_mut().unwrap();
                        let (best_move, stats) =
                            mcts_ref.search(&state, 10000, 1, u64::MAX);
                        tx_resp
                            .send(AIResponse::Move(best_move, stats))
                            .unwrap();
                    }
                    AIRequest::Stop => break,
                }
            }
        });

        Self {
            handle: Some(handle),
            tx_req,
            rx_resp,
        }
    }

    pub fn start_search(&self, state: GameWrapper) {
        self.tx_req.send(AIRequest::Search(state)).unwrap();
    }

    pub fn try_recv(&self) -> Option<AIResponse> {
        self.rx_resp.try_recv().ok()
    }

    /// Explicitly stop the AI worker
    pub fn stop(&self) {
        self.tx_req.send(AIRequest::Stop).ok();
    }
}

impl Drop for AIWorker {
    fn drop(&mut self) {
        self.tx_req.send(AIRequest::Stop).ok();
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap();
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
    // Settings
    pub settings_board_size: usize,
    pub settings_line_size: usize,
    pub settings_ai_threads: usize,
    pub settings_max_nodes: usize,
    pub settings_exploration_constant: f64,
    pub selected_settings_index: usize,
}

impl App {
    pub fn new(
        exploration_constant: f64,
        num_threads: usize,
        max_nodes: usize,
        game_name: Option<String>,
        board_size: usize,
        line_size: usize,
    ) -> Self {
        // Set default values if not provided
        let gomoku_board_size = if board_size == 0 { 15 } else { board_size };
        let gomoku_line_size = if line_size == 0 { 5 } else { line_size };
        
        let connect4_width = if board_size == 0 { 7 } else { board_size };
        let connect4_height = if board_size == 0 { 6 } else { board_size.saturating_sub(1).max(4) };
        let connect4_line_size = if line_size == 0 { 4 } else { line_size };
        
        let othello_board_size = if board_size == 0 { 8 } else { 
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

        let (initial_mode, initial_game_index) = if let Some(name) = game_name {
            let game_index = games
                .iter()
                .position(|(game_name, _)| *game_name == name)
                .unwrap_or(0);
            (AppMode::PlayerConfig, game_index)
        } else {
            (AppMode::GameSelection, 0)
        };

        let game_wrapper = games[initial_game_index].1();
        let player_options = (1..=game_wrapper.get_num_players())
            .map(|i| (i, Player::Human))
            .collect();

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
            board_cursor: (0, 0),
            selected_blokus_piece: None,
            history_scroll: 0,
            debug_scroll: 0,
            // Initialize settings with current values
            settings_board_size: if board_size == 0 { 15 } else { board_size },
            settings_line_size: if line_size == 0 { 5 } else { line_size },
            settings_ai_threads: num_threads,
            settings_max_nodes: max_nodes,
            settings_exploration_constant: exploration_constant,
            selected_settings_index: 0,
        }
    }

    pub fn update(&mut self) {
        if self.mode == AppMode::InGame {
            if self.game_status == GameStatus::InProgress {
                if self.is_current_player_ai() {
                    self.ai_worker.start_search(self.game_wrapper.clone());
                }

                if let Some(response) = self.ai_worker.try_recv() {
                    match response {
                        AIResponse::Move(best_move, stats) => {
                            self.move_history.push(MoveHistoryEntry::new(
                                self.game_wrapper.get_current_player(),
                                best_move.clone(),
                            ));
                            self.game_wrapper.make_move(&best_move);
                            self.last_search_stats = Some(stats);
                            self.check_game_over();
                        }
                    }
                }
            }
        }
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
                self.player_options = (1..=num_players).map(|i| (i, Player::Human)).collect();
                self.selected_player_config_index = 0;

                self.mode = AppMode::PlayerConfig;
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
        let (_, player_type) = &mut self.player_options[self.selected_player_config_index];
        *player_type = match *player_type {
            Player::Human => Player::AI,
            Player::AI => Player::Human,
        };
    }

    pub fn confirm_player_config(&mut self) {
        self.mode = AppMode::InGame;
    }

    pub fn reset_game(&mut self) {
        self.start_game();
    }

    // Settings navigation methods
    pub fn select_next_setting(&mut self) {
        self.selected_settings_index = (self.selected_settings_index + 1) % 7; // 5 settings + separator + back
    }

    pub fn select_prev_setting(&mut self) {
        self.selected_settings_index = (self.selected_settings_index + 6) % 7;
    }

    pub fn increase_setting(&mut self) {
        match self.selected_settings_index {
            0 => self.settings_board_size = (self.settings_board_size + 1).min(25),
            1 => self.settings_line_size = (self.settings_line_size + 1).min(10),
            2 => self.settings_ai_threads = (self.settings_ai_threads + 1).min(16),
            3 => self.settings_max_nodes = (self.settings_max_nodes + 100000).min(10000000),
            4 => self.settings_exploration_constant = (self.settings_exploration_constant + 0.1).min(3.0),
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
            _ => {} // separator or back
        }
    }

    /// Gracefully shut down the application
    /// This ensures all threads are properly stopped before exiting
    pub fn shutdown(&mut self) {
        // Explicitly stop the AI worker
        self.ai_worker.stop();
        
        // Give threads a moment to shut down gracefully
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    fn is_current_player_ai(&self) -> bool {
        let current_player_id = self.game_wrapper.get_current_player();
        self.player_options
            .iter()
            .any(|(id, p_type)| *id == current_player_id && *p_type == Player::AI)
    }

    fn check_game_over(&mut self) {
        if self.game_wrapper.is_terminal() {
            self.game_status = match self.game_wrapper.get_winner() {
                Some(winner) => GameStatus::Win(winner),
                None => GameStatus::Draw,
            };
            self.mode = AppMode::GameOver;
        }
    }
}
