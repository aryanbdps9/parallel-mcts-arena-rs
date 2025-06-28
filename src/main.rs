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

// AI Worker Communication
#[derive(Debug)]
pub enum AIRequest {
    Search {
        game_state: GameWrapper,
        iterations: i32,
        stats_interval_secs: u64,
        timeout_secs: u64,
    },
    AdvanceRoot { mv: MoveWrapper },
    UpdateSettings {
        exploration_parameter: f64,
        num_threads: usize,
        max_nodes: usize,
    },
    Stop,
}

#[derive(Debug)]
pub enum AIResponse {
    MoveReady(MoveWrapper),
    Thinking,
    Error(String),
}

pub struct AIWorker {
    ai: MCTS<GameWrapper>,
}

impl AIWorker {
    pub fn new(exploration_parameter: f64, num_threads: usize, max_nodes: usize) -> Self {
        Self {
            ai: MCTS::new(exploration_parameter, num_threads, max_nodes),
        }
    }

    pub fn run(mut self, rx: std::sync::mpsc::Receiver<AIRequest>, tx: std::sync::mpsc::Sender<AIResponse>) {
        while let Ok(request) = rx.recv() {
            match request {
                AIRequest::Search { game_state, iterations, stats_interval_secs, timeout_secs } => {
                    let _ = tx.send(AIResponse::Thinking);
                    let best_move = self.ai.search(&game_state, iterations, stats_interval_secs, timeout_secs);
                    let _ = tx.send(AIResponse::MoveReady(best_move));
                }
                AIRequest::AdvanceRoot { mv } => {
                    self.ai.advance_root(&mv);
                }
                AIRequest::UpdateSettings { exploration_parameter, num_threads, max_nodes } => {
                    self.ai = MCTS::new(exploration_parameter, num_threads, max_nodes);
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
    pub ai_tx: std::sync::mpsc::Sender<AIRequest>,
    pub ai_rx: std::sync::mpsc::Receiver<AIResponse>,
    pub pending_ai_move: Option<MoveWrapper>,
    pub ai_only: bool,
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
            pending_ai_move: None,
            ai_only: args.ai_only,
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
            // Responsive layout fields
            board_height_percent: 50,
            instructions_height_percent: 20,
            stats_height_percent: 30,
            is_dragging: false,
            drag_boundary: None,
            last_terminal_size: (0, 0),
        };
        app.update_settings_display();
        app
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
        // Update AI with current settings
        let _ = self.ai_tx.send(AIRequest::UpdateSettings {
            exploration_parameter: self.exploration_parameter,
            num_threads: self.num_threads,
            max_nodes: self.max_nodes,
        });
    }

    pub fn tick(&mut self) {
        // Check for AI responses
        if let Ok(response) = self.ai_rx.try_recv() {
            match response {
                AIResponse::MoveReady(mv) => {
                    self.pending_ai_move = Some(mv);
                    self.ai_state = AIState::Ready;
                }
                AIResponse::Thinking => {
                    self.ai_state = AIState::Thinking;
                }
                AIResponse::Error(err) => {
                    eprintln!("AI Error: {}", err);
                    self.ai_state = AIState::Idle;
                }
            }
        }

        // Apply pending AI move if ready
        if self.ai_state == AIState::Ready {
            if let Some(mv) = self.pending_ai_move.take() {
                self.game.make_move(&mv);
                let _ = self.ai_tx.send(AIRequest::AdvanceRoot { mv });
                self.ai_state = AIState::Idle;
                if self.game.is_terminal() {
                    self.winner = self.game.get_winner();
                    self.state = AppState::GameOver;
                    return;
                }
            }
        }

        // Request AI move if needed
        if self.state == AppState::Playing && self.ai_only && self.ai_state == AIState::Idle {
            if !self.game.is_terminal() {
                let _ = self.ai_tx.send(AIRequest::Search {
                    game_state: self.game.clone(),
                    iterations: self.iterations,
                    stats_interval_secs: self.stats_interval_secs,
                    timeout_secs: self.timeout_secs,
                });
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

    pub fn make_move(&mut self) {
        let (r, c) = self.cursor;
        if self.game.get_board()[r][c] == 0 {
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
            self.game.make_move(&player_move);
            let _ = self.ai_tx.send(AIRequest::AdvanceRoot { mv: player_move });
            if self.game.is_terminal() {
                self.winner = self.game.get_winner();
                self.state = AppState::GameOver;
                return;
            }

            if !self.ai_only && self.ai_state == AIState::Idle {
                let _ = self.ai_tx.send(AIRequest::Search {
                    game_state: self.game.clone(),
                    iterations: self.iterations,
                    stats_interval_secs: self.stats_interval_secs,
                    timeout_secs: self.timeout_secs,
                });
            }
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
        });
    }

    pub fn is_ai_thinking(&self) -> bool {
        self.ai_state == AIState::Thinking
    }

    // Responsive layout methods
    pub fn handle_window_resize(&mut self, width: u16, height: u16) {
        self.last_terminal_size = (width, height);
        // Reset scroll if content might have changed
        self.debug_scroll_offset = 0;
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

        match boundary {
            DragBoundary::BoardInstructions => {
                // Ensure reasonable bounds (30-80% for board)
                let new_board_percent = row_percent.clamp(30, 80);
                let remaining = 100 - new_board_percent;
                // Maintain the relative ratio between instructions and stats
                let instructions_ratio = self.instructions_height_percent as f32 / (self.instructions_height_percent + self.stats_height_percent) as f32;
                self.board_height_percent = new_board_percent;
                self.instructions_height_percent = (remaining as f32 * instructions_ratio) as u16;
                self.stats_height_percent = remaining - self.instructions_height_percent;
            }
            DragBoundary::InstructionsStats => {
                // Calculate which part of the non-board area we're in
                let non_board_start = self.board_height_percent;
                if row_percent > non_board_start {
                    let non_board_percent = 100 - self.board_height_percent;
                    let relative_pos = row_percent - non_board_start;
                    let instructions_percent = relative_pos.clamp(5, non_board_percent - 5);
                    self.instructions_height_percent = instructions_percent;
                    self.stats_height_percent = non_board_percent - instructions_percent;
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

    #[clap(long, action = clap::ArgAction::SetTrue)]
    shared_tree: bool,
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let mut app = App::new(args);
    tui::run_tui(&mut app)
}
