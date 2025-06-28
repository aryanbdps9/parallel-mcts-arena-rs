pub mod tui;

pub mod games;
pub mod game_wrapper;

use clap::Parser;
use std::io;
use mcts::{GameState, MCTS};
use crate::games::gomoku::{GomokuMove, GomokuState};
use crate::games::connect4::{Connect4Move, Connect4State};
use crate::games::blokus::{BlokusMove, BlokusState};
use crate::games::othello::{OthelloMove, OthelloState};
use crate::game_wrapper::{GameWrapper, MoveWrapper};
use ratatui::layout::Constraint;
use std::sync::mpsc::{Receiver, Sender};

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

#[derive(PartialEq)]
pub enum AppState {
    Menu,
    Settings,
    Playing,
    GameOver,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DragBoundary {
    BoardInstructions,  // Boundary between board and instructions panes
    InstructionsStats,  // Boundary between instructions and stats panes
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
}

impl<'a> App<'a> {
    fn new(args: Args) -> App<'a> {
        let game = match args.game.as_str() {
            "gomoku" => GameWrapper::Gomoku(GomokuState::new(args.board_size, args.line_size)),
            "connect4" => GameWrapper::Connect4(Connect4State::new(7, 6, 4)),
            "blokus" => GameWrapper::Blokus(BlokusState::new()),
            "othello" => GameWrapper::Othello(OthelloState::new(8)),
            _ => panic!("Unknown game type"),
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
            game_type: args.game,
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
        };
        app.update_settings_display();
        
        // Initialize AI worker with current game state and settings
        let _ = app.ai_tx.send(AIRequest::UpdateSettings {
            exploration_parameter: args.exploration_parameter,
            num_threads: args.num_threads,
            max_nodes: args.max_nodes,
            iterations: args.iterations,
            stats_interval_secs: args.stats_interval_secs,
        });
        
        // If ai_only mode, automatically start playing
        if app.ai_only {
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
            "connect4" => (0, self.connect4_width / 2), // Top row, center column
            "blokus" => (10, 10), // Center of 20x20 board
            "othello" => (self.othello_board_size / 2 - 1, self.othello_board_size / 2 - 1), // Starting position for Othello
            _ => (0, 0),
        };
        self.debug_scroll_offset = 0;
        self.ai_state = AIState::Idle;
        self.pending_ai_move = None;
        
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
    }

    pub fn tick(&mut self) {
        // Check for any messages from the AI thread
        while let Ok(response) = self.ai_rx.try_recv() {
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
                // The AI has a move, send it to the game loop for processing
                let _ = self.game_tx.send(GameRequest::MakeMove(mv));
                self.ai_state = AIState::Idle; // Reset state, new move will be requested if needed after processing
            }
        }

        // Process any pending game requests (e.g., moves from player or AI)
        if let Ok(game_request) = self.game_rx.try_recv() {
            match game_request {
                GameRequest::MakeMove(mv) => {
                    if self.game.is_legal(&mv) {
                        // Apply the move to our game state
                        self.game.make_move(&mv);

                        // If not in AI-only mode, or if in AI-only mode with a shared tree, advance the root.
                        if !self.ai_only || self.shared_tree {
                            let _ = self.ai_tx.send(AIRequest::AdvanceRoot { last_move: mv });
                        }

                        self.ai_thinking_start_time = None;

                        // After making a move, check if game is over
                        if self.game.is_terminal() {
                            self.winner = self.game.get_winner();
                            self.state = AppState::GameOver;
                        } else {
                            // If it's the AI's turn next, request a move
                            let is_ai_turn = self.ai_only || self.game.get_current_player() == -1;
                            if is_ai_turn && self.ai_state == AIState::Idle {
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
        if self.state == AppState::Playing && self.ai_only && self.ai_state == AIState::Idle {
            if !self.game.is_terminal() {
                self.send_search_request(self.timeout_secs);
            }
        }
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
        if self.cursor.1 > 0 {
            self.cursor.1 -= 1;
        }
    }

    pub fn move_cursor_right(&mut self) {
        let board_size = self.game.get_board().len();
        if self.cursor.1 < board_size - 1 {
            self.cursor.1 += 1;
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
        self.state = AppState::Menu;
        self.game = match self.game_type.as_str() {
            "gomoku" => GameWrapper::Gomoku(GomokuState::new(self.gomoku_board_size, self.gomoku_line_size)),
            "connect4" => GameWrapper::Connect4(Connect4State::new(self.connect4_width, self.connect4_height, self.connect4_line_size)),
            "blokus" => GameWrapper::Blokus(BlokusState::new()),
            "othello" => GameWrapper::Othello(OthelloState::new(self.othello_board_size)),
            _ => panic!("Unknown game type"),
        };
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
        // Add a reasonable upper bound to prevent excessive scrolling
        if self.debug_scroll_offset < 1000 {
            self.debug_scroll_offset = self.debug_scroll_offset.saturating_add(1);
        }
    }

    pub fn reset_debug_scroll(&mut self) {
        self.debug_scroll_offset = 0;
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
            0 => { // Game Mode toggle
                self.ai_only = !self.ai_only;
                self.update_settings_display();
            }
            1 => { // Gomoku Board Size
                if self.gomoku_board_size < 25 {
                    self.gomoku_board_size += 2; // Keep odd for center positioning
                    self.update_settings_display();
                }
            }
            2 => { // Gomoku Line Size
                if self.gomoku_line_size < 10 {
                    self.gomoku_line_size += 1;
                    self.update_settings_display();
                }
            }
            3 => { // Connect4 Width
                if self.connect4_width < 12 {
                    self.connect4_width += 1;
                    self.update_settings_display();
                }
            }
            4 => { // Connect4 Height
                if self.connect4_height < 10 {
                    self.connect4_height += 1;
                    self.update_settings_display();
                }
            }
            5 => { // Connect4 Line Size
                if self.connect4_line_size < 8 {
                    self.connect4_line_size += 1;
                    self.update_settings_display();
                }
            }
            6 => { // Othello Board Size
                if self.othello_board_size < 12 {
                    self.othello_board_size += 2; // Keep even for othello
                    self.update_settings_display();
                }
            }
            7 => { // AI Iterations
                if self.iterations < 5000000 {
                    self.iterations = (self.iterations as f64 * 1.5) as i32;
                    self.update_settings_display();
                }
            }
            8 => { // AI Exploration Parameter
                if self.exploration_parameter < 10.0 {
                    self.exploration_parameter += 0.5;
                    self.update_settings_display();
                    self.update_ai_settings();
                }
            }
            9 => { // AI Max Nodes
                if self.max_nodes < 1000000 {
                    self.max_nodes = (self.max_nodes as f64 * 1.5) as usize;
                    self.update_settings_display();
                    self.update_ai_settings();
                }
            }
            _ => {}
        }
    }

    pub fn decrease_setting(&mut self) {
        match self.settings_index {
            0 => { // Game Mode toggle
                self.ai_only = !self.ai_only;
                self.update_settings_display();
            }
            1 => { // Gomoku Board Size
                if self.gomoku_board_size > 9 {
                    self.gomoku_board_size -= 2; // Keep odd for center positioning
                    self.update_settings_display();
                }
            }
            2 => { // Gomoku Line Size
                if self.gomoku_line_size > 3 {
                    self.gomoku_line_size -= 1;
                    self.update_settings_display();
                }
            }
            3 => { // Connect4 Width
                if self.connect4_width > 4 {
                    self.connect4_width -= 1;
                    self.update_settings_display();
                }
            }
            4 => { // Connect4 Height
                if self.connect4_height > 4 {
                    self.connect4_height -= 1;
                    self.update_settings_display();
                }
            }
            5 => { // Connect4 Line Size
                if self.connect4_line_size > 3 {
                    self.connect4_line_size -= 1;
                    self.update_settings_display();
                }
            }
            6 => { // Othello Board Size
                if self.othello_board_size > 6 {
                    self.othello_board_size -= 2; // Keep even for othello
                    self.update_settings_display();
                }
            }
            7 => { // AI Iterations
                if self.iterations > 10000 {
                    self.iterations = (self.iterations as f64 / 1.5) as i32;
                    self.update_settings_display();
                }
            }
            8 => { // AI Exploration Parameter
                if self.exploration_parameter > 0.5 {
                    self.exploration_parameter -= 0.5;
                    self.update_settings_display();
                    self.update_ai_settings();
                }
            }
            9 => { // AI Max Nodes
                if self.max_nodes > 10000 {
                    self.max_nodes = (self.max_nodes as f64 / 1.5) as usize;
                    self.update_settings_display();
                    self.update_ai_settings();
                }
            }
            _ => {}
        }
    }

    fn update_settings_display(&mut self) {
        self.settings_titles = vec![
            if self.ai_only { "Game Mode: AI vs AI".to_string() } else { "Game Mode: Human vs AI".to_string() },
            format!("Gomoku Board Size: {}", self.gomoku_board_size),
            format!("Gomoku Line Size: {}", self.gomoku_line_size),
            format!("Connect4 Width: {}", self.connect4_width),
            format!("Connect4 Height: {}", self.connect4_height),
            format!("Connect4 Line Size: {}", self.connect4_line_size),
            format!("Othello Board Size: {}", self.othello_board_size),
            format!("AI Iterations: {}", self.iterations),
            format!("AI Exploration: {:.1}", self.exploration_parameter),
            format!("AI Max Nodes: {}", self.max_nodes),
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
    pub fn get_minimum_board_height(&self, terminal_height: u16) -> u16 {
        let board_size = self.game.get_board().len();
        // Board needs: border (2) + margin (2) + (board_size * 2 rows per cell)
        let absolute_min = (board_size * 2 + 4) as u16;
        // Convert to percentage, ensuring we don't exceed reasonable bounds
        let min_percent = ((absolute_min as f32 / terminal_height as f32) * 100.0).ceil() as u16;
        min_percent.clamp(25, 70) // Reasonable bounds even for very small/large terminals
    }

    /// Calculate the minimum height needed for the instructions section
    pub fn get_minimum_instructions_height(&self, terminal_height: u16) -> u16 {
        // Instructions need: border (2) + content (1) = 3 lines minimum
        let absolute_min = 3u16;
        // Convert to percentage
        let min_percent = ((absolute_min as f32 / terminal_height as f32) * 100.0).ceil() as u16;
        min_percent.clamp(5, 15) // Reasonable bounds
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
        // Reset scroll position after layout change to prevent display issues
        self.debug_scroll_offset = 0;
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
        if self.ai_only && !self.shared_tree {
            self.update_ai_settings();
        }

        let _ = self.ai_tx.send(AIRequest::Search {
            game_state: self.game.clone(),
            timeout_secs,
            request_id: self.current_request_id,
        });
    }
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, default_value = "gomoku")]
    game: String,

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

fn main() -> io::Result<()> {
    let args = Args::parse();
    let mut app = App::new(args);
    tui::run_tui(&mut app)
}
