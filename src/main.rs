pub mod tui;

pub mod games;
pub mod game_wrapper;

use clap::Parser;
use std::io;
use std::fs::OpenOptions;
use std::io::Write;
use mcts::{GameState, MCTS};
use crate::games::gomoku::{GomokuMove, GomokuState};
use crate::games::connect4::{Connect4Move, Connect4State};
use crate::games::blokus::{BlokusMove, BlokusState};
use crate::games::othello::{OthelloMove, OthelloState};
use crate::game_wrapper::{GameWrapper, MoveWrapper};
use ratatui::layout::Constraint;
use std::sync::mpsc::{Receiver, Sender};
use std::time::SystemTime;

// Move history tracking
#[derive(Debug, Clone)]
pub struct MoveHistoryEntry {
    pub timestamp: SystemTime,
    pub move_number: u32,
    pub player: i32,
    pub move_data: MoveWrapper,
}

impl MoveHistoryEntry {
    pub fn new(move_number: u32, player: i32, move_data: MoveWrapper) -> Self {
        Self {
            timestamp: SystemTime::now(),
            move_number,
            player,
            move_data,
        }
    }
}

// Centralized move processing
pub enum GameRequest {
    MakeMove(MoveWrapper),
}

// AI Worker Communication
#[derive(Debug)]
pub enum AIRequest {
    Search {
        game_state: GameWrapper,
        timeout_secs: u64,
        request_id: u64,
    },
    UpdateSettings {
        exploration_parameter: f64,
        num_threads: usize,
        max_nodes: usize,
        iterations: i32,
        stats_interval_secs: u64,
    },
    AdvanceRoot { last_move: MoveWrapper },
    GetGridStats { board_size: usize },
    GetDebugInfo,
    Stop,
}

#[derive(Debug)]
pub enum AIResponse {
    MoveReady(MoveWrapper, u64), // move, request_id
    Thinking(u64), // request_id
    GridStats {
        visits_grid: Vec<Vec<i32>>,
        values_grid: Vec<Vec<f64>>,
        wins_grid: Vec<Vec<f64>>,
        root_value: f64,
    },
    DebugInfo(String),
    Error(String),
}

pub struct AIWorker {
    ai: MCTS<GameWrapper>,
    iterations: i32,
    stats_interval_secs: u64,
    current_request_id: u64,
}

impl AIWorker {
    pub fn new(exploration_parameter: f64, num_threads: usize, max_nodes: usize) -> Self {
        Self {
            ai: MCTS::new(exploration_parameter, num_threads, max_nodes),
            iterations: 10000,
            stats_interval_secs: 0,
            current_request_id: 0,
        }
    }

    pub fn run(mut self, rx: std::sync::mpsc::Receiver<AIRequest>, tx: std::sync::mpsc::Sender<AIResponse>) {
        while let Ok(request) = rx.recv() {
            match request {
                AIRequest::Search { game_state, timeout_secs, request_id } => {
                    // Update current request ID and ignore old requests
                    if request_id < self.current_request_id {
                        continue;
                    }
                    self.current_request_id = request_id;
                    
                    let _ = tx.send(AIResponse::Thinking(request_id));
                    let best_move = self.ai.search(&game_state, self.iterations, self.stats_interval_secs, timeout_secs);
                    
                    // Only send response if this is still the current request
                    if request_id == self.current_request_id {
                        let _ = tx.send(AIResponse::MoveReady(best_move, request_id));
                    }
                }
                AIRequest::UpdateSettings { exploration_parameter, num_threads, max_nodes, iterations, stats_interval_secs } => {
                    self.ai = MCTS::new(exploration_parameter, num_threads, max_nodes);
                    self.iterations = iterations;
                    self.stats_interval_secs = stats_interval_secs;
                }
                AIRequest::AdvanceRoot { last_move } => {
                    // Advance the MCTS tree to the node corresponding to the last move
                    // This preserves the search tree instead of starting from scratch
                    self.ai.advance_root(&last_move);
                }
                AIRequest::GetGridStats { board_size } => {
                    let (visits_grid, values_grid, wins_grid, root_value) = self.ai.get_grid_stats(board_size);
                    let _ = tx.send(AIResponse::GridStats {
                        visits_grid,
                        values_grid,
                        wins_grid,
                        root_value,
                    });
                }
                AIRequest::GetDebugInfo => {
                    let debug_info = self.ai.get_debug_info();
                    let _ = tx.send(AIResponse::DebugInfo(debug_info));
                }
                AIRequest::Stop => break,
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum AIState {
    Idle,
    Thinking,
    Ready,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerType {
    Human,
    AI,
}

#[derive(PartialEq)]
pub enum AppState {
    Menu,
    PlayerConfig, // New: second-level menu for player type selection
    Settings,
    Playing,
    GameOver,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DragBoundary {
    BoardInstructions,  // Boundary between board and instructions panes
    InstructionsStats,  // Boundary between instructions and stats panes
    StatsHistory,       // Boundary between debug stats and move history panes
    BlokusPieceSelectionLeft,   // Left wall of Blokus piece selection panel
    BlokusPieceSelectionRight,  // Right wall of Blokus piece selection panel
}

pub struct App<'a> {
    pub titles: Vec<&'a str>,
    pub index: usize,
    pub state: AppState,
    pub game_type: String,
    pub game: GameWrapper,
    pub cursor: (usize, usize),
    pub winner: Option<i32>,
    pub ai_state: AIState,
    pub ai_tx: Sender<AIRequest>,
    pub ai_rx: Receiver<AIResponse>,
    pub game_tx: Sender<GameRequest>,
    pub game_rx: Receiver<GameRequest>,
    pub pending_ai_move: Option<MoveWrapper>,
    pub ai_only: bool,
    pub shared_tree: bool,
    pub iterations: i32,
    pub num_threads: usize,
    pub stats_interval_secs: u64,
    pub timeout_secs: u64,
    pub debug_scroll_offset: usize,
    // Settings for game configuration
    pub settings_index: usize,
    pub settings_titles: Vec<String>,
    pub gomoku_board_size: usize,
    pub gomoku_line_size: usize,
    pub connect4_width: usize,
    pub connect4_height: usize,
    pub connect4_line_size: usize,
    pub othello_board_size: usize,
    pub exploration_parameter: f64,
    pub max_nodes: usize,
    // Responsive layout fields
    pub board_height_percent: u16,
    pub instructions_height_percent: u16,
    pub stats_height_percent: u16,
    pub stats_width_percent: u16,      // Width of debug stats vs move history
    pub is_dragging: bool,
    pub drag_boundary: Option<DragBoundary>,
    pub last_terminal_size: (u16, u16),
    // MCTS statistics for display
    pub mcts_visits_grid: Option<Vec<Vec<i32>>>,
    pub mcts_values_grid: Option<Vec<Vec<f64>>>,
    pub mcts_wins_grid: Option<Vec<Vec<f64>>>,
    pub mcts_root_value: Option<f64>,
    pub mcts_debug_info: Option<String>,
    pub ai_thinking_start_time: Option<std::time::Instant>,
    pub stats_request_counter: u32,
    pub last_stats_request_time: Option<std::time::Instant>,
    pub next_request_id: u64,
    pub current_request_id: u64,
    // Move history tracking
    pub move_history: Vec<MoveHistoryEntry>,
    pub move_counter: u32,
    pub move_history_scroll_offset: usize,
    pub moves_in_current_round: usize, // Track how many moves made in current round
    
    // Blokus-specific UI state
    pub blokus_selected_piece_idx: Option<usize>,    // Currently selected piece index
    pub blokus_selected_transformation: usize,       // Current transformation/orientation
    pub blokus_piece_preview_pos: (usize, usize),   // Preview position on board
    pub blokus_show_piece_preview: bool,             // Whether to show piece preview on board
    pub blokus_piece_selection_scroll: usize,        // Scroll offset for piece selection panel (per-player)
    pub blokus_panel_scroll_offset: usize,           // Scroll offset for entire Available Pieces panel
    pub blokus_piece_selection_width: u16,           // Width of the piece selection panel (for drag resizing)
    pub blokus_last_rotation_time: Option<std::time::Instant>, // Track last rotation to prevent rapid changes
    pub blokus_players_expanded: Vec<bool>,          // Track which players' piece sections are expanded

    // Player type configuration for current game
    pub player_types: Vec<PlayerType>, // length = number of players for current game
    pub player_config_index: usize,    // which player is selected in config menu
}

impl<'a> App<'a> {
    fn new(args: Args) -> App<'a> {
        let (game, game_type, should_start_playing) = if let Some(game_name) = args.game {
            // Game was explicitly specified
            let game = match game_name.as_str() {
                "gomoku" => GameWrapper::Gomoku(GomokuState::new(args.board_size, args.line_size)),
                "connect4" => GameWrapper::Connect4(Connect4State::new(7, 6, 4)),
                "blokus" => GameWrapper::Blokus(BlokusState::new()),
                "othello" => GameWrapper::Othello(OthelloState::new(8)),
                _ => panic!("Unknown game type: {}", game_name),
            };
            (game, game_name, true) // Always skip menu when game is explicitly specified
        } else {
            // No game specified, use default but always show menu
            let default_game = GameWrapper::Gomoku(GomokuState::new(args.board_size, args.line_size));
            (default_game, "gomoku".to_string(), false)
        };
        
        // Create AI worker thread communication channels
        let (ai_tx, worker_rx) = std::sync::mpsc::channel::<AIRequest>();
        let (worker_tx, ai_rx) = std::sync::mpsc::channel::<AIResponse>();

        // Create game move channel
        let (game_tx, game_rx) = std::sync::mpsc::channel::<GameRequest>();
        
        // Spawn AI worker thread
        let ai_worker = AIWorker::new(args.exploration_parameter, args.num_threads, args.max_nodes);
        std::thread::spawn(move || {
            ai_worker.run(worker_rx, worker_tx);
        });
        
        let mut app = App {
            titles: vec!["Gomoku", "Connect4", "Blokus", "Othello", "Settings", "Quit"],
            index: 0,
            state: AppState::Menu,
            game_type,
            game,
            cursor: (0, 0),
            winner: None,
            ai_state: AIState::Idle,
            ai_tx,
            ai_rx,
            game_tx,
            game_rx,
            pending_ai_move: None,
            ai_only: args.ai_only,
            shared_tree: args.shared_tree,
            iterations: args.iterations,
            num_threads: args.num_threads,
            stats_interval_secs: args.stats_interval_secs,
            timeout_secs: args.timeout_secs,
            debug_scroll_offset: 0,
            // Initialize settings
            settings_index: 0,
            settings_titles: vec![], // Will be populated by update_settings_display
            gomoku_board_size: args.board_size,
            gomoku_line_size: args.line_size,
            connect4_width: 7,
            connect4_height: 6,
            connect4_line_size: 4,
            othello_board_size: 8,
            exploration_parameter: args.exploration_parameter,
            max_nodes: args.max_nodes,
            // Responsive layout fields - will be set by initialize_layout
            board_height_percent: 50, // Temporary default, will be overridden
            instructions_height_percent: 20, // Temporary default, will be overridden
            stats_height_percent: 30, // Temporary default, will be overridden
            stats_width_percent: 60, // Default: 60% for debug stats, 40% for move history
            is_dragging: false,
            drag_boundary: None,
            last_terminal_size: (0, 0),
            // MCTS statistics for display
            mcts_visits_grid: None,
            mcts_values_grid: None,
            mcts_wins_grid: None,
            mcts_root_value: None,
            mcts_debug_info: None,
            ai_thinking_start_time: None,
            stats_request_counter: 0,
            last_stats_request_time: None,
            next_request_id: 1,
            current_request_id: 0,
            move_history: vec![],
            move_counter: 0,
            move_history_scroll_offset: 0,
            moves_in_current_round: 0,
            blokus_selected_piece_idx: None,
            blokus_selected_transformation: 0,
            blokus_piece_preview_pos: (0, 0),
            blokus_show_piece_preview: false,
            blokus_piece_selection_scroll: 0,
            blokus_panel_scroll_offset: 0,
            blokus_piece_selection_width: 40, // Default width
            blokus_last_rotation_time: None,
            blokus_players_expanded: vec![true; 4], // Start with all players expanded
            // Player type configuration for current game
            player_types: vec![PlayerType::Human; 2], // Will be set properly below
            player_config_index: 0,
        };
        
        // Set initial player types based on command line args
        if args.ai_only {
            // Set all players to AI if --ai-only flag is used
            let player_count = app.get_player_count();
            app.player_types = vec![PlayerType::AI; player_count];
        }
        
        app.update_settings_display();
        
        // Initialize AI worker with current game state and settings
        let _ = app.ai_tx.send(AIRequest::UpdateSettings {
            exploration_parameter: args.exploration_parameter,
            num_threads: args.num_threads,
            max_nodes: args.max_nodes,
            iterations: args.iterations,
            stats_interval_secs: args.stats_interval_secs,
        });
        
        // If game was explicitly specified, automatically start playing
        if should_start_playing {
            app.state = AppState::Playing;
        }
        
        app
    }

    /// Initialize layout percentages based on minimum content requirements
    pub fn initialize_layout(&mut self, terminal_height: u16) {
        let min_board_percent = self.get_minimum_board_height(terminal_height);
        let min_instructions_percent = self.get_minimum_instructions_height(terminal_height);
        let min_stats_percent = 5u16;
        
        // Set initial layout with minimum heights as default (no extra space)
        self.board_height_percent = min_board_percent;
        self.instructions_height_percent = min_instructions_percent;
        self.stats_height_percent = 100 - self.board_height_percent - self.instructions_height_percent;
        
        // Ensure stats has at least its minimum
        if self.stats_height_percent < min_stats_percent {
            self.stats_height_percent = min_stats_percent;
            // Adjust board height if needed, but keep instructions at minimum
            self.board_height_percent = 100 - self.instructions_height_percent - self.stats_height_percent;
        }
    }

    pub fn set_game(&mut self, index: usize) {
        log_debug(&format!("set_game() called with index={} ({})", index, self.titles[index]));
        self.game_type = self.titles[index].to_lowercase();
        self.game = match self.game_type.as_str() {
            "gomoku" => GameWrapper::Gomoku(GomokuState::new(self.gomoku_board_size, self.gomoku_line_size)),
            "connect4" => GameWrapper::Connect4(Connect4State::new(self.connect4_width, self.connect4_height, self.connect4_line_size)),
            "blokus" => GameWrapper::Blokus(BlokusState::new()),
            "othello" => GameWrapper::Othello(OthelloState::new(self.othello_board_size)),
            _ => panic!("Unknown game type"),
        };
        // Set cursor position based on game type and board size
        self.cursor = match self.game_type.as_str() {
            "gomoku" => (self.gomoku_board_size / 2, self.gomoku_board_size / 2), // Center of board
            "connect4" => (0, self.connect4_width / 2), // Top row, center column - will be updated below
            "blokus" => (10, 10), // Center of 20x20 board
            "othello" => (self.othello_board_size / 2 - 1, self.othello_board_size / 2 - 1), // Starting position for Othello
            _ => (0, 0),
        };
        
        // For Connect4, update cursor to the correct row position
        if self.game_type == "connect4" {
            self.update_connect4_cursor_row();
        }
        self.debug_scroll_offset = 0;
        self.move_history.clear();
        self.move_counter = 0;
        self.move_history_scroll_offset = 0;
        self.moves_in_current_round = 0;
        self.ai_state = AIState::Idle;
        self.pending_ai_move = None;
        
        // Reset Blokus UI state when changing games
        self.blokus_selected_piece_idx = None;
        self.set_blokus_transformation(0, "set_game");
        self.blokus_piece_preview_pos = (0, 0);
        self.blokus_show_piece_preview = false;
        self.blokus_piece_selection_scroll = 0;
        self.blokus_panel_scroll_offset = 0;
        self.blokus_last_rotation_time = None;
        
        // Initialize layout based on the new game's requirements
        if self.last_terminal_size.1 > 0 {
            self.initialize_layout(self.last_terminal_size.1);
        }
        
        // Update AI with current settings
        let _ = self.ai_tx.send(AIRequest::UpdateSettings {
            exploration_parameter: self.exploration_parameter,
            num_threads: self.num_threads,
            max_nodes: self.max_nodes,
            iterations: self.iterations,
            stats_interval_secs: self.stats_interval_secs,
        });
        
        // Reset player config for the new game
        self.reset_player_config();
    }

    pub fn tick(&mut self) -> bool {
        let mut ui_changed = false;
        
        // Check for any messages from the AI thread
        while let Ok(response) = self.ai_rx.try_recv() {
            log_debug("tick(): AI response received");
            ui_changed = true; // AI response received, UI needs update
            match response {
                AIResponse::MoveReady(mv, request_id) => {
                    if request_id == self.current_request_id {
                        self.pending_ai_move = Some(mv);
                        self.ai_state = AIState::Ready;
                    }
                }
                AIResponse::GridStats { visits_grid, values_grid, wins_grid, root_value } => {
                    self.mcts_visits_grid = Some(visits_grid);
                    self.mcts_values_grid = Some(values_grid);
                    self.mcts_wins_grid = Some(wins_grid);
                    self.mcts_root_value = Some(root_value);
                }
                AIResponse::DebugInfo(info) => {
                    self.mcts_debug_info = Some(info);
                }
                AIResponse::Thinking(request_id) => {
                    if request_id == self.current_request_id {
                        self.ai_state = AIState::Thinking;
                        self.ai_thinking_start_time = Some(std::time::Instant::now());
                    }
                }
                AIResponse::Error(_) => {
                    self.ai_state = AIState::Idle;
                }
            }
        }

        // Process ready AI move by submitting it to the central game channel
        if self.ai_state == AIState::Ready {
            if let Some(mv) = self.pending_ai_move.take() {
                log_debug("tick(): AI move ready");
                // The AI has a move, send it to the game loop for processing
                let _ = self.game_tx.send(GameRequest::MakeMove(mv));
                self.ai_state = AIState::Idle; // Reset state, new move will be requested if needed after processing
                ui_changed = true;
            }
        }

        // Process any pending game requests (e.g., moves from player or AI)
        if let Ok(game_request) = self.game_rx.try_recv() {
            log_debug("tick(): Game request received");
            ui_changed = true; // Game state changed, UI needs update
            match game_request {
                GameRequest::MakeMove(mv) => {
                    if self.game.is_legal(&mv) {
                        // Track move history before applying the move
                        let current_player = self.game.get_current_player();
                        
                        // Increment moves in current round
                        self.moves_in_current_round += 1;
                        
                        // If this is the first move of a new round, increment move counter
                        if self.moves_in_current_round == 1 {
                            self.move_counter += 1;
                        }
                        
                        let history_entry = MoveHistoryEntry::new(self.move_counter, current_player, mv.clone());
                        self.move_history.push(history_entry);
                        
                        // Apply the move to our game state
                        self.game.make_move(&mv);

                        // Check if we completed a full round of players
                        let player_count = self.get_player_count();
                        if self.moves_in_current_round >= player_count {
                            self.moves_in_current_round = 0; // Reset for next round
                        }

                        // If not in AI-only mode, or if in AI-only mode with a shared tree, advance the root.
                        if !self.is_ai_only_mode() || self.shared_tree {
                            let _ = self.ai_tx.send(AIRequest::AdvanceRoot { last_move: mv });
                        }

                        self.ai_thinking_start_time = None;

                        // After making a move, check if game is over
                        if self.game.is_terminal() {
                            self.winner = self.game.get_winner();
                            self.state = AppState::GameOver;
                        } else {
                            // If it's the AI's turn next, request a move
                            if self.is_current_player_ai() && self.ai_state == AIState::Idle {
                                self.send_search_request(self.timeout_secs);
                            }
                        }
                    } else {
                        // The move was illegal. If it came from an AI, reset state.
                        if self.ai_state == AIState::Ready {
                             self.ai_state = AIState::Idle;
                        }
                    }
                }
            }
        }

        // Request MCTS statistics periodically for games that support grid display
        // Use the stats_interval_secs parameter to control frequency
        if self.ai_state == AIState::Thinking && self.stats_interval_secs > 0 {
            let should_request_stats = if let Some(last_request) = self.last_stats_request_time {
                last_request.elapsed().as_secs() >= self.stats_interval_secs
            } else {
                true // First request
            };

            if should_request_stats {
                if matches!(self.game, GameWrapper::Gomoku(_) | GameWrapper::Othello(_)) {
                    // Request grid stats
                    let board_size = self.game.get_board_size();
                    let _ = self.ai_tx.send(AIRequest::GetGridStats { board_size });
                    
                    // Request debug info
                    let _ = self.ai_tx.send(AIRequest::GetDebugInfo);
                    
                    // Update last request time
                    self.last_stats_request_time = Some(std::time::Instant::now());
                }
            }
        }

        // Request AI move if needed
        if self.state == AppState::Playing && self.is_current_player_ai() && self.ai_state == AIState::Idle {
            if !self.game.is_terminal() {
                self.send_search_request(self.timeout_secs);
            }
        }
        
        ui_changed // Return whether UI needs to be redrawn
    }

    pub fn next(&mut self) {
        self.index = (self.index + 1) % self.titles.len();
    }

    pub fn previous(&mut self) {
        if self.index > 0 {
            self.index -= 1;
        } else {
            self.index = self.titles.len() - 1;
        }
    }

    pub fn move_cursor_down(&mut self) {
        let board_size = self.game.get_board().len();
        if self.cursor.0 < board_size - 1 {
            self.cursor.0 += 1;
        }
    }

    pub fn move_cursor_up(&mut self) {
        if self.cursor.0 > 0 {
            self.cursor.0 -= 1;
        }
    }

    pub fn move_cursor_left(&mut self) {
        if self.game_type == "connect4" {
            // For Connect4, only move horizontally between columns
            if self.cursor.1 > 0 {
                self.cursor.1 -= 1;
                // Update cursor row to the lowest empty row in the new column
                self.update_connect4_cursor_row();
            }
        } else {
            if self.cursor.1 > 0 {
                self.cursor.1 -= 1;
            }
        }
    }

    pub fn move_cursor_right(&mut self) {
        if self.game_type == "connect4" {
            // For Connect4, only move horizontally between columns
            let board = self.game.get_board();
            let board_width = if board.len() > 0 { board[0].len() } else { 0 };
            if self.cursor.1 < board_width - 1 {
                self.cursor.1 += 1;
                // Update cursor row to the lowest empty row in the new column
                self.update_connect4_cursor_row();
            }
        } else {
            let board_size = self.game.get_board().len();
            if self.cursor.1 < board_size - 1 {
                self.cursor.1 += 1;
            }
        }
    }

    pub fn submit_move(&mut self) {
        let (r, c) = self.cursor;
        let player_move = match self.game {
            GameWrapper::Gomoku(_) => MoveWrapper::Gomoku(GomokuMove(r, c)),
            GameWrapper::Connect4(_) => MoveWrapper::Connect4(Connect4Move(c)),
            GameWrapper::Blokus(_) => {
                // For Blokus, we'll use the first available piece and transformation as a placeholder
                // This is a simplified move selection for UI purposes
                MoveWrapper::Blokus(BlokusMove(0, 0, r, c))
            },
            GameWrapper::Othello(_) => MoveWrapper::Othello(OthelloMove(r, c)),
        };

        if self.game.is_legal(&player_move) {
            let _ = self.game_tx.send(GameRequest::MakeMove(player_move));
        }
    }

    pub fn reset(&mut self) {
        log_debug("reset() called - resetting all Blokus state");
        self.state = AppState::Menu;
        self.game = match self.game_type.as_str() {
            "gomoku" => GameWrapper::Gomoku(GomokuState::new(self.gomoku_board_size, self.gomoku_line_size)),
            "connect4" => GameWrapper::Connect4(Connect4State::new(self.connect4_width, self.connect4_height, self.connect4_line_size)),
            "blokus" => GameWrapper::Blokus(BlokusState::new()),
            "othello" => GameWrapper::Othello(OthelloState::new(self.othello_board_size)),
            _ => panic!("Unknown game type"),
        };
        
        // Reset move history
        self.move_history.clear();
        self.move_counter = 0;
        self.move_history_scroll_offset = 0;
        self.moves_in_current_round = 0;
        
        // Reset Blokus state completely
        self.blokus_selected_piece_idx = None;
        self.set_blokus_transformation(0, "reset");
        self.blokus_piece_preview_pos = (0, 0);
        self.blokus_show_piece_preview = false;
        self.blokus_piece_selection_scroll = 0;
        self.blokus_panel_scroll_offset = 0;
        self.blokus_last_rotation_time = None;
        
        // Reset player config
        self.reset_player_config();
        
        let _ = self.ai_tx.send(AIRequest::UpdateSettings {
            exploration_parameter: self.exploration_parameter,
            num_threads: self.num_threads,
            max_nodes: self.max_nodes,
            iterations: self.iterations,
            stats_interval_secs: self.stats_interval_secs,
        });
        self.ai_state = AIState::Idle;
        self.pending_ai_move = None;
        self.winner = None;
        self.debug_scroll_offset = 0;
        self.cursor = match self.game_type.as_str() {
            "gomoku" => (self.gomoku_board_size / 2, self.gomoku_board_size / 2), // Center of board
            "connect4" => (0, self.connect4_width / 2), // Top row, center column
            "blokus" => (10, 10), // Center of 20x20 board
            "othello" => (self.othello_board_size / 2 - 1, self.othello_board_size / 2 - 1), // Starting position for Othello
            _ => (0, 0),
        };
    }

    pub fn scroll_debug_up(&mut self) {
        self.debug_scroll_offset = self.debug_scroll_offset.saturating_sub(1);
    }

    pub fn scroll_debug_down(&mut self) {
        if self.debug_scroll_offset < 100 { // Safety limit
            self.debug_scroll_offset += 1;
        }
    }

    pub fn reset_debug_scroll(&mut self) {
        self.debug_scroll_offset = 0;
    }

    pub fn scroll_move_history_up(&mut self) {
        if self.move_history_scroll_offset > 0 {
            self.move_history_scroll_offset -= 1;
        }
    }

    pub fn scroll_move_history_down(&mut self) {
        self.move_history_scroll_offset += 1;
        // The bounds will be clamped when update_move_history_scroll_bounds is called
        // or when the UI renders
    }

    pub fn reset_move_history_scroll(&mut self) {
        self.move_history_scroll_offset = 0;
    }

    pub fn settings_next(&mut self) {
        self.settings_index = (self.settings_index + 1) % self.settings_titles.len();
    }

    pub fn settings_previous(&mut self) {
        if self.settings_index > 0 {
            self.settings_index -= 1;
        } else {
            self.settings_index = self.settings_titles.len() - 1;
        }
    }

    pub fn increase_setting(&mut self) {
        match self.settings_index {
            0 => { // Gomoku Board Size
                if self.gomoku_board_size < 25 {
                    self.gomoku_board_size += 2; // Keep odd for center positioning
                    self.update_settings_display();
                    if self.game_type == "gomoku" {
                        self.refresh_current_game();
                    }
                }
            }
            1 => { // Gomoku Line Size
                if self.gomoku_line_size < 10 {
                    self.gomoku_line_size += 1;
                    self.update_settings_display();
                    if self.game_type == "gomoku" {
                        self.refresh_current_game();
                    }
                }
            }
            2 => { // Connect4 Width
                if self.connect4_width < 12 {
                    self.connect4_width += 1;
                    self.update_settings_display();
                    if self.game_type == "connect4" {
                        self.refresh_current_game();
                    }
                }
            }
            3 => { // Connect4 Height
                if self.connect4_height < 10 {
                    self.connect4_height += 1;
                    self.update_settings_display();
                    if self.game_type == "connect4" {
                        self.refresh_current_game();
                    }
                }
            }
            4 => { // Connect4 Line Size
                if self.connect4_line_size < 8 {
                    self.connect4_line_size += 1;
                    self.update_settings_display();
                    if self.game_type == "connect4" {
                        self.refresh_current_game();
                    }
                }
            }
            5 => { // Othello Board Size
                if self.othello_board_size < 12 {
                    self.othello_board_size += 2; // Keep even for othello
                    self.update_settings_display();
                    if self.game_type == "othello" {
                        self.refresh_current_game();
                    }
                }
            }
            _ => {}
        }
    }

    pub fn decrease_setting(&mut self) {
        match self.settings_index {
            0 => { // Gomoku Board Size
                if self.gomoku_board_size > 9 {
                    self.gomoku_board_size -= 2; // Keep odd for center positioning
                    self.update_settings_display();
                    if self.game_type == "gomoku" {
                        self.refresh_current_game();
                    }
                }
            }
            1 => { // Gomoku Line Size
                if self.gomoku_line_size > 3 {
                    self.gomoku_line_size -= 1;
                    self.update_settings_display();
                    if self.game_type == "gomoku" {
                        self.refresh_current_game();
                    }
                }
            }
            2 => { // Connect4 Width
                if self.connect4_width > 4 {
                    self.connect4_width -= 1;
                    self.update_settings_display();
                    if self.game_type == "connect4" {
                        self.refresh_current_game();
                    }
                }
            }
            3 => { // Connect4 Height
                if self.connect4_height > 4 {
                    self.connect4_height -= 1;
                    self.update_settings_display();
                    if self.game_type == "connect4" {
                        self.refresh_current_game();
                    }
                }
            }
            4 => { // Connect4 Line Size
                if self.connect4_line_size > 3 {
                    self.connect4_line_size -= 1;
                    self.update_settings_display();
                    if self.game_type == "connect4" {
                        self.refresh_current_game();
                    }
                }
            }
            5 => { // Othello Board Size
                if self.othello_board_size > 6 {
                    self.othello_board_size -= 2; // Keep even for othello
                    self.update_settings_display();
                    if self.game_type == "othello" {
                        self.refresh_current_game();
                    }
                }
            }
            _ => {}
        }
    }

    fn update_settings_display(&mut self) {
        self.settings_titles = vec![
            format!("Gomoku Board Size: {}", self.gomoku_board_size),
            format!("Gomoku Line Size: {}", self.gomoku_line_size),
            format!("Connect4 Width: {}", self.connect4_width),
            format!("Connect4 Height: {}", self.connect4_height),
            format!("Connect4 Line Size: {}", self.connect4_line_size),
            format!("Othello Board Size: {}", self.othello_board_size),
            "Back to Menu".to_string()
        ];
    }

    fn update_ai_settings(&mut self) {
        let _ = self.ai_tx.send(AIRequest::UpdateSettings {
            exploration_parameter: self.exploration_parameter,
            num_threads: self.num_threads,
            max_nodes: self.max_nodes,
            iterations: self.iterations,
            stats_interval_secs: self.stats_interval_secs,
        });
    }

    pub fn is_ai_thinking(&self) -> bool {
        self.ai_state == AIState::Thinking
    }

    /// Get the time remaining for AI to move
    pub fn get_ai_time_remaining(&self) -> Option<f64> {
        if let Some(start_time) = self.ai_thinking_start_time {
            if self.timeout_secs > 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let remaining = self.timeout_secs as f64 - elapsed;
                Some(remaining.max(0.0))
            } else {
                None // No timeout set
            }
        } else {
            None // AI not thinking
        }
    }

    /// Calculate the minimum height needed for the board section
    fn get_minimum_board_height(&self, terminal_height: u16) -> u16 {
        let min_height = match self.game {
            GameWrapper::Gomoku(_) => (self.game.get_board_size() as u16) + 4, // +4 for borders and title
            GameWrapper::Connect4(_) => 10, // Connect4 is typically 6 high + borders
            GameWrapper::Blokus(_) => 24,   // 20x20 board + borders
            GameWrapper::Othello(_) => 12,  // 8x8 board + borders
        };
        
        // Return percentage of terminal height, but ensure minimum absolute height
        let min_percent = (min_height * 100) / terminal_height.max(min_height);
        min_percent.min(80) // Cap at 80% to leave room for other UI elements
    }

    /// Calculate the minimum height needed for the instructions section
    pub fn get_minimum_instructions_height(&self, terminal_height: u16) -> u16 {
        // Instructions need: border (2) + content (1) = 3 lines minimum
        let absolute_min = 3u16;
        // Convert to percentage
        let min_percent = ((absolute_min as f32 / terminal_height as f32) * 100.0).ceil() as u16;
        min_percent.clamp(5, 15) // Reasonable bounds
    }

    /// Update move history scroll bounds based on current terminal size and content
    pub fn update_move_history_scroll_bounds(&mut self, terminal_height: u16) {
        // Calculate actual move history area height
        let stats_height = (terminal_height as f32 * self.stats_height_percent as f32 / 100.0) as u16;
        let visible_height = (stats_height.saturating_sub(2) as usize).min(20);
        
        // Count the number of move groups
        let mut unique_moves = std::collections::HashSet::new();
        for entry in &self.move_history {
            unique_moves.insert(entry.move_number);
        }
        let content_height = unique_moves.len();
        
        // Calculate and clamp scroll offset
        let max_scroll = content_height.saturating_sub(visible_height);
        self.move_history_scroll_offset = self.move_history_scroll_offset.min(max_scroll);
    }

    // Responsive layout methods
    pub fn handle_window_resize(&mut self, width: u16, height: u16) {
        self.last_terminal_size = (width, height);
        // Reset scroll if content might have changed
        self.debug_scroll_offset = 0;
        
        // Recalculate layout to ensure minimum heights are respected
        // Get current game's minimum requirements
        let min_board_percent = self.get_minimum_board_height(height);
        let min_instructions_percent = self.get_minimum_instructions_height(height);
        let min_stats_percent = 5u16;
        
        // Only adjust if current percentages are below minimums
        let needs_board_adjustment = self.board_height_percent < min_board_percent;
        let needs_instructions_adjustment = self.instructions_height_percent < min_instructions_percent;
        
        if needs_board_adjustment || needs_instructions_adjustment {
            // Ensure total is 100% and all sections have minimum space
            let total_min_required = min_board_percent + min_instructions_percent + min_stats_percent;
            
            if total_min_required <= 100 {
                // We can fit all minimums - set sections to minimum and distribute remaining to stats
                self.board_height_percent = min_board_percent;
                self.instructions_height_percent = min_instructions_percent;
                self.stats_height_percent = 100 - min_board_percent - min_instructions_percent;
            } else {
                // Very constrained space - use absolute minimums even if they exceed 100%
                self.board_height_percent = min_board_percent;
                self.instructions_height_percent = min_instructions_percent;
                self.stats_height_percent = min_stats_percent;
            }
        } else {
            // Current layout is valid, just ensure total is 100%
            let total_used = self.board_height_percent + self.instructions_height_percent;
            self.stats_height_percent = (100u16).saturating_sub(total_used).max(min_stats_percent);
            
            // If stats was forced to minimum, adjust the others proportionally
            if self.stats_height_percent == min_stats_percent && total_used > 100 - min_stats_percent {
                let available = 100 - min_stats_percent;
                let current_total = self.board_height_percent + self.instructions_height_percent;
                if current_total > 0 {
                    self.board_height_percent = (self.board_height_percent * available / current_total).max(min_board_percent);
                    self.instructions_height_percent = (available - self.board_height_percent).max(min_instructions_percent);
                }
            }
        }
    }

    pub fn start_drag(&mut self, boundary: DragBoundary) {
        self.is_dragging = true;
        self.drag_boundary = Some(boundary);
    }

    pub fn stop_drag(&mut self) {
        self.is_dragging = false;
        self.drag_boundary = None;
        // Reset scroll positions after layout change to prevent display issues
        self.debug_scroll_offset = 0;
        self.move_history_scroll_offset = 0;
    }

    pub fn handle_drag(&mut self, mouse_row: u16, terminal_height: u16) {
        if !self.is_dragging || self.drag_boundary.is_none() {
            return;
        }

        let boundary = self.drag_boundary.unwrap();
        let row_percent = ((mouse_row as f32 / terminal_height as f32) * 100.0) as u16;

        // Calculate minimum heights based on content requirements
        let min_board_percent = self.get_minimum_board_height(terminal_height);
        let min_instructions_percent = self.get_minimum_instructions_height(terminal_height);
        let min_stats_percent = 5u16; // Stats section can be very small (scrollable)

        match boundary {
            DragBoundary::BoardInstructions => {
                // Ensure board doesn't go below its minimum height or above reasonable maximum
                let max_board_percent = 100 - min_instructions_percent - min_stats_percent;
                let new_board_percent = row_percent.clamp(min_board_percent, max_board_percent);
                let remaining = 100 - new_board_percent;
                
                // Maintain the relative ratio between instructions and stats, respecting minimums
                let instructions_ratio = self.instructions_height_percent as f32 / (self.instructions_height_percent + self.stats_height_percent) as f32;
                let desired_instructions = (remaining as f32 * instructions_ratio) as u16;
                let desired_stats = remaining - desired_instructions;
                
                // Ensure both sections meet their minimum requirements
                if desired_instructions >= min_instructions_percent && desired_stats >= min_stats_percent {
                    self.board_height_percent = new_board_percent;
                    self.instructions_height_percent = desired_instructions;
                    self.stats_height_percent = desired_stats;
                } else {
                    // Adjust to meet minimums
                    self.board_height_percent = new_board_percent;
                    self.instructions_height_percent = min_instructions_percent.max(desired_instructions);
                    self.stats_height_percent = remaining - self.instructions_height_percent;
                }
            }
            DragBoundary::InstructionsStats => {
                // Calculate which part of the non-board area we're in
                let non_board_start = self.board_height_percent;
                if row_percent > non_board_start {
                    let non_board_percent = 100 - self.board_height_percent;
                    let relative_pos = row_percent - non_board_start;

                    // Ensure instructions doesn't go below its minimum
                    let max_instructions = non_board_percent - min_stats_percent;
                    let instructions_percent = relative_pos.clamp(min_instructions_percent, max_instructions);
                    let stats_percent = non_board_percent - instructions_percent;
                    
                    // Only update if both sections meet their minimums
                    if instructions_percent >= min_instructions_percent && stats_percent >= min_stats_percent {
                        self.instructions_height_percent = instructions_percent;
                        self.stats_height_percent = stats_percent;
                    }
                }
            }
            DragBoundary::StatsHistory => {
                // For horizontal dragging within the stats area, we don't use row_percent
                // This case should be handled by a separate method for horizontal dragging
                // For now, do nothing as this will be handled by handle_horizontal_drag
            }
            DragBoundary::BlokusPieceSelectionLeft | DragBoundary::BlokusPieceSelectionRight => {
                // Horizontal dragging for Blokus piece selection panel resizing
                // This will be handled by handle_horizontal_drag method
            }
        }
    }

    pub fn handle_horizontal_drag(&mut self, col: u16, terminal_width: u16) {
        if let Some(boundary) = self.drag_boundary {
            match boundary {
                DragBoundary::StatsHistory => {
                    let col_percent = (col as f32 / terminal_width as f32 * 100.0) as u16;
                    
                    // Allow stats to be 20% to 80% of the width
                    let min_stats_width = 20u16;
                    let max_stats_width = 80u16;
                    
                    let new_stats_width = col_percent.clamp(min_stats_width, max_stats_width);
                    self.stats_width_percent = new_stats_width;
                }
                DragBoundary::BlokusPieceSelectionLeft => {
                    // Dragging left wall - decrease width as we drag right
                    let min_width = 25u16;
                    let max_width = 60u16;
                    let new_width = (terminal_width - col).clamp(min_width, max_width);
                    self.blokus_piece_selection_width = new_width;
                }
                DragBoundary::BlokusPieceSelectionRight => {
                    // Dragging right wall - increase width as we drag right
                    let min_width = 25u16;
                    let max_width = 60u16;
                    let new_width = col.clamp(min_width, max_width);
                    self.blokus_piece_selection_width = new_width;
                }
                _ => {}
            }
        }
    }

    pub fn get_layout_constraints(&self) -> [Constraint; 3] {
        [
            Constraint::Percentage(self.board_height_percent),
            Constraint::Percentage(self.instructions_height_percent),
            Constraint::Percentage(self.stats_height_percent),
        ]
    }

    pub fn get_drag_area(&self, terminal_height: u16) -> (u16, u16) {
        // Return the row ranges where dragging is allowed
        let board_end = (terminal_height as f32 * self.board_height_percent as f32 / 100.0) as u16;
        let instructions_end = board_end + (terminal_height as f32 * self.instructions_height_percent as f32 / 100.0) as u16;
        (board_end.saturating_sub(1), instructions_end.saturating_sub(1))
    }

    /// Generate a new request ID and send a search request
    fn send_search_request(&mut self, timeout_secs: u64) {
        self.next_request_id += 1;
        self.current_request_id = self.next_request_id;
        self.ai_state = AIState::Thinking;  // Immediately set to thinking to prevent duplicate requests
        
        // In AI vs AI mode with non-shared tree, reset the MCTS tree for the new player's turn.
        if self.is_ai_only_mode() && !self.shared_tree {
            self.update_ai_settings();
        }

        let _ = self.ai_tx.send(AIRequest::Search {
            game_state: self.game.clone(),
            timeout_secs,
            request_id: self.current_request_id,
        });
    }

    fn refresh_current_game(&mut self) {
        // If we're currently playing this game type, refresh the game state with new settings
        if matches!(self.state, AppState::Playing | AppState::GameOver) {
            self.game = match self.game_type.as_str() {
                "gomoku" => GameWrapper::Gomoku(GomokuState::new(self.gomoku_board_size, self.gomoku_line_size)),
                "connect4" => GameWrapper::Connect4(Connect4State::new(self.connect4_width, self.connect4_height, self.connect4_line_size)),
                "blokus" => GameWrapper::Blokus(BlokusState::new()),
                "othello" => GameWrapper::Othello(OthelloState::new(self.othello_board_size)),
                _ => return,
            };
            
            // Reset cursor position based on new dimensions
            self.cursor = match self.game_type.as_str() {
                "gomoku" => (self.gomoku_board_size / 2, self.gomoku_board_size / 2),
                "connect4" => (0, self.connect4_width / 2),
                "blokus" => (10, 10),
                "othello" => (self.othello_board_size / 2 - 1, self.othello_board_size / 2 - 1),
                _ => (0, 0),
            };
            
            // Reset game state
            self.state = AppState::Playing;
            self.winner = None;
            self.move_history.clear();
            self.move_counter = 0;
            self.move_history_scroll_offset = 0;
            self.moves_in_current_round = 0;
            self.ai_state = AIState::Idle;
            self.pending_ai_move = None;
        }
    }

    // Helper method to update cursor row for Connect4 to the lowest empty row in the current column
    pub fn update_connect4_cursor_row(&mut self) {
        if self.game_type == "connect4" {
            let board = self.game.get_board();
            let board_height = board.len();
            let col = self.cursor.1;
            
            // Find the lowest empty row in this column
            for r in (0..board_height).rev() {
                if board[r][col] == 0 {
                    self.cursor.0 = r;
                    return;
                }
            }
            // If column is full, keep cursor at the top
            self.cursor.0 = 0;
        }
    }

    // Blokus-specific methods
    
    pub fn blokus_select_piece(&mut self, piece_idx: usize) {
        // DEBUG: Print when piece selection is called
        log_debug(&format!("blokus_select_piece called with piece_idx={}", piece_idx));
        
        if let GameWrapper::Blokus(blokus_state) = &self.game {
            // Only allow piece selection if it's human's turn
            if self.is_current_player_ai() || self.ai_state != AIState::Idle {
                log_debug(&format!("Rejecting piece selection - is_ai={}, ai_state={:?}", self.is_current_player_ai(), self.ai_state));
                return;
            }
            
            // Additional check: ensure it's a human player's turn
            let current_player = blokus_state.get_current_player();
            if current_player < 1 || current_player > 4 {
                log_debug(&format!("Rejecting piece selection - invalid current_player={}", current_player));
                return; // Invalid player
            }
            
            let player_idx = (current_player - 1) as usize;
            // Get available pieces for current player and check if piece is available
            let available_pieces = blokus_state.get_available_pieces(current_player);
            if piece_idx < 21 && available_pieces.contains(&piece_idx) {
                // Check if this is actually a different piece or if no piece is selected
                let is_different_piece = self.blokus_selected_piece_idx != Some(piece_idx);
                let no_piece_selected = self.blokus_selected_piece_idx.is_none();
                
                log_debug(&format!("is_different_piece={}, no_piece_selected={}, current_transformation={}", 
                    is_different_piece, no_piece_selected, self.get_blokus_transformation("blokus_select_piece_check")));
                
                if is_different_piece || no_piece_selected {
                    // Reset transformation only when selecting a different piece or first selection
                    log_debug(&format!("Setting transformation to 0, piece to {}", piece_idx));
                    self.set_blokus_transformation(0, "blokus_select_piece");
                    self.blokus_selected_piece_idx = Some(piece_idx);
                    self.blokus_show_piece_preview = true;
                    self.blokus_last_rotation_time = None; // Reset rotation timer
                } else {
                    log_debug("Same piece selected again, doing nothing");
                }
                // If same piece is selected again, do absolutely nothing to prevent any changes
            }
        }
    }

    pub fn blokus_rotate_piece(&mut self) {
        log_debug(&format!("blokus_rotate_piece called, current_transformation={}", self.get_blokus_transformation("blokus_rotate_piece_start")));
        
        if let (Some(_), GameWrapper::Blokus(_)) = (self.blokus_selected_piece_idx, &self.game) {
            // Only allow rotation if it's human's turn
            if self.is_current_player_ai() || self.ai_state != AIState::Idle {
                log_debug(&format!("Rejecting rotation - is_ai={}, ai_state={:?}", self.is_current_player_ai(), self.ai_state));
                return;
            }
            
            // Prevent rapid rotations - require at least 200ms between rotations
            let now = std::time::Instant::now();
            if let Some(last_rotation) = self.blokus_last_rotation_time {
                if now.duration_since(last_rotation).as_millis() < 200 {
                    log_debug(&format!("Rejecting rotation - too soon, {}ms since last", now.duration_since(last_rotation).as_millis()));
                    return; // Too soon, ignore this rotation
                }
            }
            
            // Get the piece and check how many transformations it has
            let pieces = crate::games::blokus::get_piece_info();
            if let Some(piece_idx) = self.blokus_selected_piece_idx {
                if let Some((_, max_transformations)) = pieces.get(piece_idx) {
                    let old_transformation = self.get_blokus_transformation("blokus_rotate_piece_old");
                    let new_transformation = (old_transformation + 1) % max_transformations;
                    self.set_blokus_transformation(new_transformation, "blokus_rotate_piece");
                    self.blokus_last_rotation_time = Some(now);
                    log_debug(&format!("Rotated piece {} from {} to {} (max={})", 
                        piece_idx, old_transformation, new_transformation, max_transformations));
                }
            }
        }
    }

    pub fn blokus_flip_piece(&mut self) {
        if let (Some(_), GameWrapper::Blokus(_)) = (self.blokus_selected_piece_idx, &self.game) {
            // For simplicity, we'll treat flip as another rotation
            // In a more sophisticated implementation, you'd have separate flip logic
            self.blokus_rotate_piece();
        }
    }

    pub fn blokus_move_preview(&mut self, dr: i32, dc: i32) {
        if self.blokus_show_piece_preview {
            // Calculate target position
            let old_pos = self.blokus_piece_preview_pos;
            let new_r = (self.blokus_piece_preview_pos.0 as i32 + dr).max(0).min(19) as usize;
            let new_c = (self.blokus_piece_preview_pos.1 as i32 + dc).max(0).min(19) as usize;
            let target_pos = (new_r, new_c);
            
            // Only update position if it actually changed (prevents redundant updates)
            if target_pos != old_pos {
                self.blokus_piece_preview_pos = target_pos;
            }
        }
    }

    pub fn blokus_place_piece(&mut self) {
        if let (Some(piece_idx), GameWrapper::Blokus(blokus_state)) = (self.blokus_selected_piece_idx, &self.game) {
            let blokus_move = crate::games::blokus::BlokusMove(
                piece_idx,
                self.get_blokus_transformation("blokus_place_piece"),
                self.cursor.0,  // Use cursor position instead of preview pos
                self.cursor.1,  // Use cursor position instead of preview pos
            );
            
            if blokus_state.is_legal(&blokus_move) {
                let move_wrapper = crate::game_wrapper::MoveWrapper::Blokus(blokus_move);
                let _ = self.game_tx.send(GameRequest::MakeMove(move_wrapper));
                
                // Reset selection state
                self.blokus_selected_piece_idx = None;
                self.blokus_show_piece_preview = false;
            }
        }
    }

    pub fn blokus_cycle_pieces(&mut self, forward: bool) {
        log_debug(&format!("blokus_cycle_pieces called with forward={}", forward));
        if let GameWrapper::Blokus(_) = &self.game {
            // Only allow cycling if it's human's turn
            if self.is_current_player_ai() || self.ai_state != AIState::Idle {
                return;
            }
            
            let pieces = crate::games::blokus::get_piece_info();
            let max_pieces = pieces.len();
            
            if let Some(current_idx) = self.blokus_selected_piece_idx {
                let new_idx = if forward {
                    (current_idx + 1) % max_pieces
                } else {
                    (current_idx + max_pieces - 1) % max_pieces
                };
                
                // Only cycle if we're actually selecting a different piece
                if new_idx != current_idx {
                    // Store current transformation before selecting new piece
                    let current_transformation = self.get_blokus_transformation("blokus_cycle_pieces_store");
                    log_debug(&format!("Cycling from piece {} to piece {}, storing transformation={}", current_idx, new_idx, current_transformation));
                    
                    // Reset to the new piece
                    self.blokus_selected_piece_idx = Some(new_idx);
                    self.set_blokus_transformation(0, "blokus_cycle_pieces_reset");
                    log_debug(&format!("Reset transformation to 0 for new piece {}", new_idx));
                    self.blokus_show_piece_preview = true;
                    
                    // Try to restore transformation if the new piece supports it
                    if let Some((_, max_transformations)) = pieces.get(new_idx) {
                        if current_transformation < *max_transformations {
                            self.set_blokus_transformation(current_transformation, "blokus_cycle_pieces_restore");
                            log_debug(&format!("Restored transformation to {} for piece {} (max={})", current_transformation, new_idx, max_transformations));
                        } else {
                            log_debug(&format!("Could not restore transformation {} for piece {} (max={})", current_transformation, new_idx, max_transformations));
                        }
                    }
                }
            } else if max_pieces > 0 {
                // No piece selected, select the first one
                self.blokus_select_piece(0);
            }
        }
    }

    /// Expand all player sections in Blokus piece selection
    pub fn blokus_expand_all_players(&mut self) {
        for expanded in &mut self.blokus_players_expanded {
            *expanded = true;
        }
    }

    /// Collapse all player sections in Blokus piece selection
    pub fn blokus_collapse_all_players(&mut self) {
        for expanded in &mut self.blokus_players_expanded {
            *expanded = false;
        }
    }

    /// Toggle expand/collapse for the current player's section
    pub fn blokus_toggle_current_player_expand(&mut self) {
        let current_player = self.game.get_current_player();
        let player_idx = ((current_player - 1).max(0) as usize).min(3);
        self.blokus_toggle_player_expand(player_idx);
    }

    /// Toggle expand/collapse for a specific player's section (for mouse clicks)
    pub fn blokus_toggle_player_expand(&mut self, player_idx: usize) {
        if let Some(expanded) = self.blokus_players_expanded.get_mut(player_idx) {
            *expanded = !*expanded;
        }
    }

    // Blokus piece selection scroll methods
    pub fn blokus_scroll_pieces_up(&mut self) {
        self.blokus_piece_selection_scroll = self.blokus_piece_selection_scroll.saturating_sub(1);
    }

    pub fn blokus_scroll_pieces_down(&mut self) {
        if self.blokus_piece_selection_scroll < 50 { // Safety limit
            self.blokus_piece_selection_scroll += 1;
        }
    }

    pub fn reset_blokus_piece_scroll(&mut self) {
        self.blokus_piece_selection_scroll = 0;
        self.blokus_panel_scroll_offset = 0;
    }

    // Blokus full panel scrolling methods
    pub fn blokus_scroll_panel_up(&mut self) {
        self.blokus_panel_scroll_offset = self.blokus_panel_scroll_offset.saturating_sub(1);
    }

    pub fn blokus_scroll_panel_down(&mut self) {
        if self.blokus_panel_scroll_offset < 200 { // Safety limit
            self.blokus_panel_scroll_offset += 1;
        }
    }

    pub fn reset_blokus_panel_scroll(&mut self) {
        self.blokus_panel_scroll_offset = 0;
    }

    // Missing methods that are referenced in the code
    pub fn get_player_count(&self) -> usize {
        match &self.game {
            GameWrapper::Blokus(_) => 4,
            GameWrapper::Gomoku(_) => 2,
            GameWrapper::Connect4(_) => 2,
            GameWrapper::Othello(_) => 2,
        }
    }

    pub fn is_current_player_ai(&self) -> bool {
        let current_player = self.game.get_current_player();
        let player_idx = self.get_player_index_from_id(current_player);
        self.player_types.get(player_idx).map_or(false, |pt| *pt == PlayerType::AI)
    }

    pub fn get_player_index_from_id(&self, player_id: i32) -> usize {
        match &self.game {
            GameWrapper::Blokus(_) => {
                // Blokus players are 1-4, convert to 0-3
                ((player_id - 1).max(0) as usize).min(3)
            }
            _ => {
                // Other games use -1/1, convert to 0/1
                if player_id == 1 { 0 } else { 1 }
            }
        }
    }

    pub fn is_ai_only_mode(&self) -> bool {
        self.player_types.iter().all(|pt| *pt == PlayerType::AI)
    }

    pub fn toggle_player_type(&mut self, idx: usize) {
        if let Some(pt) = self.player_types.get_mut(idx) {
            *pt = match *pt {
                PlayerType::Human => PlayerType::AI,
                PlayerType::AI => PlayerType::Human,
            };
        }
    }

    pub fn reset_player_config(&mut self) {
        let n = self.get_player_count();
        self.set_player_count(n);
    }

    pub fn set_player_count(&mut self, count: usize) {
        self.player_types = vec![PlayerType::Human; count];
        self.player_config_index = 0;
    }

    pub fn set_blokus_transformation(&mut self, new_value: usize, source: &str) {
        let old_value = self.blokus_selected_transformation;
        if old_value != new_value {
            log_debug(&format!("TRANSFORMATION CHANGE: {} -> {} (source: {})", old_value, new_value, source));
            self.blokus_selected_transformation = new_value;
        } else {
            log_debug(&format!("TRANSFORMATION NO-CHANGE: {} (source: {})", old_value, source));
        }
    }

    pub fn get_blokus_transformation(&self, _source: &str) -> usize {
        self.blokus_selected_transformation
    }
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long)]
    game: Option<String>,

    #[clap(short, long, default_value_t = 9)]
    board_size: usize,

    #[clap(short, long, default_value_t = 4)]
    line_size: usize,

    #[clap(short, long, default_value_t = 8)]
    num_threads: usize,

    #[clap(short = 'e', long, default_value_t = 4.0)]
    exploration_parameter: f64,

    #[clap(short = 'i', long, default_value_t = 1000000)]
    iterations: i32,

    #[clap(short = 'm', long, default_value_t = 1000000)]
    max_nodes: usize,

    #[clap(long, default_value_t = 20)]
    stats_interval_secs: u64,

    #[clap(long, default_value_t = 60)]
    timeout_secs: u64,

    #[clap(long, action = clap::ArgAction::SetTrue)]
    ai_only: bool,

    #[clap(long, action = clap::ArgAction::SetTrue, default_value_t = true)]
    shared_tree: bool,
}

// Helper function to log debug messages to file
fn log_debug(message: &str) {
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("blokus_debug.log")
    {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        if let Err(_) = writeln!(file, "[{}] {}", timestamp, message) {
            // If we can't write to file, fall back to stderr
            eprintln!("DEBUG: {}", message);
        }
    } else {
        // If we can't open file, fall back to stderr
        eprintln!("DEBUG: {}", message);
    }
}

fn main() -> io::Result<()> {
    log_debug("=== NEW SESSION STARTED ===");
    let args = Args::parse();
    let mut app = App::new(args);
    tui::run_tui(&mut app)?;

    // Example: add a way to enter PlayerConfig from menu (pseudo, actual key handling is in tui.rs)
    // In tui.rs, in AppState::Menu key handler, add:
    // if key.code == KeyCode::Char('p') { app.state = AppState::PlayerConfig; }

    Ok(())
}
