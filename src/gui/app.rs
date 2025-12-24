//! # GUI Application State
//!
//! Manages the application state for the Windows GUI, including game state,
//! AI coordination, and UI state.

use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, atomic::AtomicBool};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime};

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
    /// Player configuration (Human vs AI)
    PlayerConfig,
    /// Active game
    InGame,
    /// Game over screen
    GameOver,
}

/// Game status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameStatus {
    InProgress,
    Win(i32),
    Draw,
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

    // Settings
    pub board_size: usize,
    pub line_size: usize,
    pub timeout_secs: u64,

    // UI state
    pub needs_redraw: bool,
    pub hover_button: Option<usize>,
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
            needs_redraw: true,
            hover_button: None,
        }
    }

    /// Start a new game with current settings
    pub fn start_game(&mut self) {
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
            GuiMode::PlayerConfig => self.mode = GuiMode::GameSelection,
            GuiMode::InGame | GuiMode::GameOver => {
                self.ai_worker.stop();
                self.mode = GuiMode::GameSelection;
            }
            GuiMode::GameSelection => self.should_quit = true,
        }
        self.needs_redraw = true;
    }
}
