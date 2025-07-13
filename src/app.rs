//! # Application State and Core Components
//!
//! This module serves as the central orchestrator for the entire multi-game MCTS engine.
//! It defines the core data structures and components that manage application state,
//! coordinate between UI and AI threads, and handle game lifecycle management.
//!
//! ## Key Responsibilities
//! - **State Management**: Centralized application state including game state, UI state, and AI status
//! - **AI Coordination**: Thread-safe communication with background AI worker processes
//! - **Event Handling**: Processing user input and system events across all game modes
//! - **Game Abstraction**: Unified interface for multiple game types through GameWrapper
//! - **UI Orchestration**: Coordinating between different UI components and screens
//!
//! ## Architecture Overview
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │   UI Components │◄──►│    App State     │◄──►│   AI Worker     │
//! │                 │    │                  │    │                 │
//! │ • Game View     │    │ • Game Wrapper   │    │ • MCTS Engine   │
//! │ • Menu System   │    │ • Player Config  │    │ • Search Tree   │
//! │ • Settings      │    │ • Move History   │    │ • Statistics    │
//! └─────────────────┘    └──────────────────┘    └─────────────────┘
//! ```
//!
//! ## Thread Safety
//! The AI worker runs in a separate thread with message-passing communication.
//! This ensures the UI remains responsive while AI performs intensive computations.
//!
//! ## Memory Management
//! The application uses various optimization strategies:
//! - Tree reuse between moves to avoid redundant computation
//! - Automatic history scrolling with user override detection
//! - Component lifecycle management for UI elements
//! - Node recycling in the MCTS search tree

// Import dependencies for game management and UI coordination
use crate::game_wrapper::{GameWrapper, MoveWrapper}; // Unified game interface
use crate::tui::blokus_ui::BlokusUIConfig; // Blokus-specific UI configuration
use crate::tui::layout::LayoutConfig; // Responsive UI layout management
use crate::tui::mouse::DragState; // Mouse interaction state tracking
use mcts::{GameState, MCTS}; // Monte Carlo Tree Search engine
use ratatui::widgets::ListState; // TUI list widget state management
use std::sync::mpsc::{self, Receiver, Sender}; // Thread communication channels
use std::sync::{Arc, atomic::AtomicBool}; // Thread-safe shared state
use std::thread::{self, JoinHandle}; // Background thread management
use std::time::SystemTime; // Timestamp tracking for moves

/// Represents a single move in the game's move history
///
/// This structure captures all relevant information about a move for later
/// analysis, replay, or debugging purposes. Each entry is immutable once
/// created to maintain historical integrity.
///
/// # Fields
/// - `timestamp`: When the move was made (for performance analysis)
/// - `player`: Which player made the move (using game-specific player IDs)
/// - `a_move`: The actual move that was made (wrapped for type safety)
///
/// # Usage
/// Move history is primarily used for:
/// - Game replay and analysis
/// - Debugging game logic issues
/// - Performance metrics (time per move)
/// - UI display of past moves
/// - Undo functionality (future enhancement)
#[derive(Debug, Clone)]
pub struct MoveHistoryEntry {
    /// System timestamp when the move was executed
    /// Used for performance analysis and replay timing
    pub timestamp: SystemTime,

    /// Player ID who made this move
    /// Uses game-specific numbering (e.g., 1/-1 for two-player games, 1-4 for Blokus)
    pub player: i32,

    /// The move that was made, type-erased for storage
    /// Contains game-specific move data wrapped in MoveWrapper enum
    pub a_move: MoveWrapper,
}

impl MoveHistoryEntry {
    /// Creates a new move history entry with the current timestamp
    ///
    /// # Arguments
    /// * `player` - The player ID who made the move
    /// * `a_move` - The move that was made
    ///
    /// # Returns
    /// A new MoveHistoryEntry with current system time
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
/// This enum defines the communication protocol between the main application
/// thread and the background AI worker. All AI operations are asynchronous
/// to keep the UI responsive during intensive computations.
///
/// # Message Types
/// - `Search`: Initiate a new MCTS search from a given position
/// - `AdvanceRoot`: Update the search tree after a move is made
/// - `Stop`: Gracefully terminate the AI worker
///
/// # Thread Safety
/// All messages are designed to be safely sent across thread boundaries.
/// The AI worker processes these messages sequentially to maintain consistency.
#[derive(Debug)]
pub enum AIRequest {
    /// Request the AI to search for the best move from a given position
    ///
    /// # Parameters
    /// - `GameWrapper`: Current game state to search from
    /// - `u64`: Maximum search time in seconds
    ///
    /// The AI will perform MCTS until either the time limit is reached
    /// or the maximum iteration count is exceeded, whichever comes first.
    Search(GameWrapper, u64),

    /// Advance the MCTS tree root to reflect a move that was made
    ///
    /// # Parameters  
    /// - `MoveWrapper`: The move that was executed in the game
    ///
    /// This allows the AI to reuse previous search results by updating
    /// the tree structure rather than starting from scratch. This is a
    /// key optimization for maintaining AI strength across moves.
    AdvanceRoot(MoveWrapper),

    /// Signal the AI worker to stop processing and exit gracefully
    ///
    /// The worker will finish its current operation and then terminate.
    /// This is used during application shutdown or when reconfiguring AI settings.
    Stop,
}

/// Messages received from AI worker threads
///
/// Responses from the AI worker containing search results and analysis data.
/// The main thread polls for these messages to update the UI and game state.
///
/// # Design Rationale
/// Currently only contains move responses, but designed to be extensible
/// for future enhancements like progressive search updates or analysis data.
#[derive(Debug)]
pub enum AIResponse {
    /// The AI's selected move along with detailed search statistics
    ///
    /// # Parameters
    /// - `MoveWrapper`: The best move found by the AI
    /// - `mcts::SearchStatistics`: Detailed analysis including node counts,
    ///   evaluation scores, and top move candidates
    ///
    /// This provides both the actionable result (the move) and rich
    /// information for debugging and user education about AI reasoning.
    Move(MoveWrapper, mcts::SearchStatistics),
}

/// The AI worker that runs in a separate thread
///
/// This struct manages a background thread dedicated to AI computation.
/// It provides a clean interface for asynchronous AI operations while
/// maintaining thread safety and resource management.
///
/// # Architecture
/// ```text
/// Main Thread                    AI Worker Thread
/// ┌─────────────┐               ┌──────────────────┐
/// │             │──AIRequest──► │                  │
/// │ Application │               │ MCTS Engine      │
/// │             │◄─AIResponse── │                  │
/// └─────────────┘               └──────────────────┘
/// ```
///
/// # Key Features
/// - **Asynchronous Operation**: AI computation doesn't block the UI
/// - **Persistent State**: Maintains MCTS tree across multiple searches
/// - **Graceful Shutdown**: Proper cleanup when the worker is no longer needed
/// - **Error Isolation**: AI failures don't crash the main application
///
/// # Lifecycle
/// 1. Created with configuration parameters
/// 2. Processes search requests asynchronously  
/// 3. Maintains search tree for efficiency
/// 4. Cleaned up automatically when dropped
pub struct AIWorker {
    /// Handle to the background thread for proper cleanup
    /// None after the worker has been stopped and joined
    handle: Option<JoinHandle<()>>,

    /// Channel for sending requests to the AI worker
    /// Requests are processed sequentially in the worker thread
    tx_req: Sender<AIRequest>,

    /// Channel for receiving responses from the AI worker
    /// Main thread polls this non-blockingly for results
    rx_resp: Receiver<AIResponse>,

    /// Atomic flag for signaling the worker to stop current operations
    /// Allows for immediate interruption of long-running searches
    stop_flag: Arc<AtomicBool>,
}

impl AIWorker {
    /// Creates a new AI worker thread
    ///
    /// Spawns a background thread that handles MCTS search requests and maintains
    /// a persistent search tree. The worker can be controlled via message passing
    /// and will automatically stop when requested.
    ///
    /// # Arguments
    /// * `exploration_constant` - C_puct value for MCTS exploration vs exploitation balance
    /// * `num_threads` - Number of parallel threads for MCTS search
    /// * `search_iterations` - Maximum number of MCTS iterations per search
    /// * `max_nodes` - Maximum number of nodes allowed in the search tree
    ///
    /// # Returns
    /// New AIWorker instance ready to process search requests
    pub fn new(
        exploration_constant: f64,
        num_threads: usize,
        search_iterations: u32,
        max_nodes: usize,
    ) -> Self {
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
                            mcts = Some(MCTS::new(exploration_constant, num_threads, max_nodes));
                        }
                        let mcts_ref = mcts.as_mut().unwrap();

                        // Use the exact timeout - MCTS now properly respects timeouts
                        let (best_move, stats) = mcts_ref.search_with_stop(
                            &state,
                            search_iterations as i32,
                            1,
                            timeout_secs,
                            Some(stop_flag_clone.clone()),
                        );

                        // Only send response if we haven't been stopped
                        if !stop_flag_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            tx_resp.send(AIResponse::Move(best_move, stats)).ok(); // Ignore send errors if receiver is dropped
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

    /// Starts an AI search for the given game state
    ///
    /// Sends a search request to the AI worker thread. The search will run
    /// asynchronously and the result can be retrieved using try_recv().
    ///
    /// # Arguments
    /// * `state` - Current game state to search from
    /// * `timeout_secs` - Maximum time to spend searching (in seconds)
    pub fn start_search(&self, state: GameWrapper, timeout_secs: u64) {
        if let Err(_) = self.tx_req.send(AIRequest::Search(state, timeout_secs)) {
            // AI worker channel is closed - this can happen if the worker thread has exited
            // This is not fatal - the AI just won't respond, which will be handled gracefully
            eprintln!("Warning: AI worker is not available for search");
        }
    }

    /// Attempts to receive a response from the AI worker
    ///
    /// Non-blocking call that returns None if no response is available yet.
    /// Should be called periodically to check for completed searches.
    ///
    /// # Returns
    /// Some(AIResponse) if a response is available, None otherwise
    pub fn try_recv(&self) -> Option<AIResponse> {
        self.rx_resp.try_recv().ok()
    }

    /// Explicitly stop the AI worker
    ///
    /// Interrupts any ongoing search and signals the worker thread to stop.
    /// The worker will finish processing the current request and then exit.
    /// This is automatically called when the AIWorker is dropped.
    pub fn stop(&self) {
        // Set the stop flag first to interrupt any ongoing search
        self.stop_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
        // Then send the stop message to break the worker loop
        self.tx_req.send(AIRequest::Stop).ok();
    }

    /// Advances the AI's search tree root to reflect a move that was made
    ///
    /// When a move is made in the game, this tells the AI to update its internal
    /// search tree so that future searches start from the new position. This
    /// allows the AI to reuse previous search results.
    ///
    /// # Arguments
    /// * `move_made` - The move that was made in the game
    pub fn advance_root(&self, move_made: &MoveWrapper) {
        self.tx_req
            .send(AIRequest::AdvanceRoot(move_made.clone()))
            .ok();
    }
}

impl Drop for AIWorker {
    /// Cleanup when AIWorker is dropped
    ///
    /// Ensures the worker thread is properly stopped and joined to prevent
    /// resource leaks. Gives the thread a reasonable time to finish gracefully.
    fn drop(&mut self) {
        // Stop the worker gracefully
        self.stop_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.tx_req.send(AIRequest::Stop).ok();

        // Wait for the thread to finish, but with a timeout to avoid hanging
        if let Some(handle) = self.handle.take() {
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

/// Represents the active tab in the combined stats/history pane
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTab {
    Debug,
    History,
}

impl ActiveTab {
    pub fn next(&self) -> Self {
        match self {
            ActiveTab::Debug => ActiveTab::History,
            ActiveTab::History => ActiveTab::Debug,
        }
    }
}

/// The main application state
///
/// This struct holds all the state required to run the application,
/// including the game state, UI state, AI workers, and communication channels.
pub struct App {
    // Application lifecycle
    pub should_quit: bool, // Flag to signal application shutdown
    pub mode: AppMode, // Current screen/menu being displayed

    // Game management
    pub games: Vec<(&'static str, Box<dyn Fn() -> GameWrapper>)>, // Available games with factory functions
    pub game_selection_state: ListState, // UI state for game selection menu
    pub settings_state: ListState, // UI state for settings menu
    pub game_wrapper: GameWrapper, // Current active game instance
    pub game_status: GameStatus, // Whether game is in progress, won, or drawn

    // Player configuration
    pub player_options: Vec<(i32, Player)>, // Player configurations (ID, Human/AI)
    pub selected_player_config_index: usize, // Currently selected player in config menu

    // AI system
    pub ai_worker: AIWorker, // Background thread for AI computations
    pub last_search_stats: Option<mcts::SearchStatistics>, // Most recent AI analysis data
    pub move_history: Vec<MoveHistoryEntry>, // Complete history of moves made

    // UI state
    pub show_debug: bool, // Whether to display debug information
    pub board_cursor: (u16, u16), // Current cursor position on game board (row, col)
    pub selected_blokus_piece: Option<(usize, usize)>, // Deprecated: use blokus_ui_config instead
    pub history_scroll: u16, // Current scroll position in move history panel
    pub debug_scroll: u16, // Current scroll position in debug panel
    pub active_tab: ActiveTab, // Currently selected tab (Debug/History)

    // Auto-scroll behavior for move history
    pub history_auto_scroll: bool, // Whether to automatically scroll to latest move
    pub history_user_scroll_time: Option<std::time::Instant>, // When user last manually scrolled
    pub history_auto_scroll_reset_duration: std::time::Duration, // How long to wait before re-enabling auto-scroll

    // Auto-scroll behavior for Blokus piece panel
    pub piece_panel_auto_scroll: bool, // Whether to automatically scroll to current player's pieces
    pub piece_panel_user_scroll_time: Option<std::time::Instant>, // When user last manually scrolled pieces
    pub piece_panel_auto_scroll_reset_duration: std::time::Duration, // How long to wait before re-enabling auto-scroll
    pub last_current_player: i32, // Previous player ID to detect player changes

    // AI timing and status display
    pub ai_thinking_start: Option<std::time::Instant>, // When AI started thinking about current move
    pub ai_minimum_display_duration: std::time::Duration, // Minimum time to show "AI thinking" message
    pub pending_ai_response: Option<(MoveWrapper, mcts::SearchStatistics)>, // AI move waiting for minimum display time

    // Game settings (configurable via settings menu)
    pub settings_board_size: usize, // Board size for games that support it
    pub settings_line_size: usize, // Number in a row needed to win
    pub settings_ai_threads: usize, // Number of threads for AI search
    pub settings_max_nodes: usize, // Maximum nodes in MCTS search tree
    pub settings_search_iterations: u32, // Maximum MCTS iterations per search
    pub settings_exploration_constant: f64, // C_puct value for exploration vs exploitation
    pub selected_settings_index: usize, // Currently selected setting in menu

    // AI behavior settings
    pub timeout_secs: u64, // Maximum time AI can spend per move
    pub stats_interval_secs: u64, // How often to update AI statistics display
    pub ai_only: bool, // Whether to run AI vs AI games without human input
    pub shared_tree: bool, // Whether AI should reuse search trees between moves

    // Enhanced UI components
    pub layout_config: LayoutConfig, // Responsive layout configuration
    pub drag_state: DragState, // Mouse drag interaction state
    pub blokus_ui_config: BlokusUIConfig, // Blokus-specific UI state and configuration

    // Component-based UI system
    pub component_manager: crate::components::manager::ComponentManager, // Manages UI component lifecycle
}

impl App {
    /// Creates a new application instance with the specified configuration
    ///
    /// Initializes all application state including game options, AI workers, UI components,
    /// and player configurations. The application can start in different modes depending
    /// on the provided parameters.
    ///
    /// # Arguments
    /// * `exploration_constant` - C_puct value for MCTS exploration vs exploitation balance
    /// * `num_threads` - Number of parallel threads for AI search
    /// * `search_iterations` - Maximum number of MCTS iterations per search
    /// * `max_nodes` - Maximum number of nodes allowed in the MCTS search tree
    /// * `game_name` - Optional specific game to start with (skips game selection)
    /// * `board_size` - Size of the game board (game-specific interpretation)
    /// * `line_size` - Number of pieces needed in a row to win (for applicable games)
    /// * `timeout_secs` - Maximum time AI can spend per move (seconds)
    /// * `stats_interval_secs` - How often to update AI statistics (seconds)
    /// * `ai_only` - Whether to run in AI-vs-AI mode (no human players)
    /// * `shared_tree` - Whether the AI should reuse search trees between moves
    ///
    /// # Returns
    /// New App instance ready to run the game engine
    pub fn new(
        exploration_constant: f64,
        num_threads: usize,
        search_iterations: u32,
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
        let connect4_height = if board_size == 15 {
            6
        } else {
            board_size.saturating_sub(1).max(4)
        }; // Default Connect4 height is 6
        // TODO: BUG: The user will never be able to pass line_size = 5. Consider making line_size an Option.
        // Same for others (e.g. board_size)
        let connect4_line_size = if line_size == 5 { 4 } else { line_size }; // Default Connect4 line is 4

        let othello_board_size = if board_size == 15 {
            8
        } else {
            // Ensure even number for Othello
            if board_size % 2 == 0 {
                board_size
            } else {
                board_size + 1
            }
        };

        let games: Vec<(&'static str, Box<dyn Fn() -> GameWrapper>)> = vec![
            (
                "gomoku",
                Box::new(move || {
                    GameWrapper::Gomoku(crate::games::gomoku::GomokuState::new(
                        gomoku_board_size,
                        gomoku_line_size,
                    ))
                }),
            ),
            (
                "connect4",
                Box::new(move || {
                    GameWrapper::Connect4(crate::games::connect4::Connect4State::new(
                        connect4_width,
                        connect4_height,
                        connect4_line_size,
                    ))
                }),
            ),
            (
                "othello",
                Box::new(move || {
                    GameWrapper::Othello(crate::games::othello::OthelloState::new(
                        othello_board_size,
                    ))
                }),
            ),
            (
                "blokus",
                Box::new(|| GameWrapper::Blokus(crate::games::blokus::BlokusState::new())),
            ),
        ];

        let has_specific_game = game_name.is_some();
        let (initial_mode, initial_game_index) = if let Some(name) = game_name {
            let game_index = games
                .iter()
                .position(|(game_name, _)| *game_name == name.to_lowercase())
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
        let initial_current_player = game_wrapper.get_current_player();
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

        let mut settings_state = ListState::default();
        settings_state.select(Some(0));

        Self {
            should_quit: false,
            mode: initial_mode,
            games,
            game_selection_state,
            settings_state,
            game_wrapper,
            game_status: GameStatus::InProgress,
            player_options,
            selected_player_config_index: 0,
            ai_worker: AIWorker::new(
                exploration_constant,
                num_threads,
                search_iterations,
                max_nodes,
            ),
            last_search_stats: None,
            move_history: Vec::new(),
            show_debug: false,
            board_cursor: (initial_cursor.0 as u16, initial_cursor.1 as u16),
            selected_blokus_piece: None,
            history_scroll: 0,
            debug_scroll: 0,
            active_tab: ActiveTab::Debug,
            // Auto-scroll for move history
            history_auto_scroll: true,
            history_user_scroll_time: None,
            history_auto_scroll_reset_duration: std::time::Duration::from_secs(20), // 20 seconds as requested
            // Auto-scroll for Blokus piece panel
            piece_panel_auto_scroll: true,
            piece_panel_user_scroll_time: None,
            piece_panel_auto_scroll_reset_duration: std::time::Duration::from_secs(15), // 15 seconds as requested
            last_current_player: initial_current_player,
            // AI timing and status
            ai_thinking_start: None,
            ai_minimum_display_duration: std::time::Duration::from_millis(100), // Minimum 0.1 seconds display
            pending_ai_response: None,
            // Initialize settings with current values
            settings_board_size: if board_size == 15 { 15 } else { board_size }, // Keep 15 as standard Gomoku default
            settings_line_size: if line_size == 5 { 5 } else { line_size }, // Keep 5 as standard Gomoku default
            settings_ai_threads: num_threads,
            settings_max_nodes: max_nodes,
            settings_search_iterations: search_iterations,
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
            // Component-based UI system
            component_manager: {
                let mut manager = crate::components::manager::ComponentManager::new();
                let root = Box::new(crate::components::ui::root::RootComponent::new());
                manager.set_root_component(root);
                manager
            },
        }
    }

    /// Creates a game instance using current settings values
    ///
    /// This ensures that when games are started from the menu, they use the
    /// updated settings rather than the original command-line parameters.
    ///
    /// # Arguments
    /// * `game_name` - Name of the game to create
    ///
    /// # Returns
    /// GameWrapper instance configured with current settings
    pub fn create_game_with_current_settings(&self, game_name: &str) -> GameWrapper {
        match game_name {
            "gomoku" => GameWrapper::Gomoku(crate::games::gomoku::GomokuState::new(
                self.settings_board_size,
                self.settings_line_size,
            )),
            "connect4" => {
                // For Connect4, board_size becomes width, and height is derived
                let width = self.settings_board_size;
                let height = (self.settings_board_size.saturating_sub(1)).max(4); // Height is usually width - 1, min 4
                GameWrapper::Connect4(crate::games::connect4::Connect4State::new(
                    width,
                    height,
                    self.settings_line_size,
                ))
            }
            "othello" => {
                // Ensure even number for Othello
                let board_size = if self.settings_board_size % 2 == 0 {
                    self.settings_board_size
                } else {
                    self.settings_board_size + 1
                };
                GameWrapper::Othello(crate::games::othello::OthelloState::new(board_size))
            }
            "blokus" => {
                // Blokus doesn't use settings for board size (it's always 20x20)
                GameWrapper::Blokus(crate::games::blokus::BlokusState::new())
            }
            _ => {
                // Default to Gomoku
                GameWrapper::Gomoku(crate::games::gomoku::GomokuState::new(
                    self.settings_board_size,
                    self.settings_line_size,
                ))
            }
        }
    }

    /// Updates the application state for one frame
    ///
    /// This is the main update loop that handles:
    /// - AI move processing and timing
    /// - Game state updates after moves
    /// - Automatic move history scrolling
    /// - Game over detection
    /// - Background AI search coordination
    ///
    /// Should be called once per frame in the main UI loop.
    pub fn update(&mut self) {
    if self.mode == AppMode::InGame && self.game_status == GameStatus::InProgress {
            if self.is_current_player_ai() {
                if self.ai_thinking_start.is_none() {
                    self.ai_thinking_start = Some(std::time::Instant::now());
                    self.ai_worker
                        .start_search(self.game_wrapper.clone(), self.timeout_secs);
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

        // Handle auto-scroll reset timer for move history
        self.update_history_auto_scroll();
        // Update auto-scroll for Blokus piece panel based on current player
        self.update_piece_panel_auto_scroll();
    }

    /// Gets the name of the currently selected game
    ///
    /// # Returns
    /// Static string reference to the selected game's name
    pub fn get_selected_game_name(&self) -> &'static str {
        self.games[self.game_selection_state.selected().unwrap_or(0)].0
    }

    /// Moves the game selection cursor to the next option
    ///
    /// Wraps around to the beginning when reaching the end of the list.
    /// Includes settings and quit options in the navigation.
    pub fn select_next_game(&mut self) {
        let i = match self.game_selection_state.selected() {
            Some(i) => (i + 1) % (self.games.len() + 2), // +2 for Settings and Quit
            None => 0,
        };
        self.game_selection_state.select(Some(i));
    }

    /// Moves the game selection cursor to the previous option
    ///
    /// Wraps around to the end when reaching the beginning of the list.
    /// Includes settings and quit options in the navigation.
    pub fn select_prev_game(&mut self) {
        let i = match self.game_selection_state.selected() {
            Some(i) => (i + self.games.len() + 1) % (self.games.len() + 2),
            None => 0,
        };
        self.game_selection_state.select(Some(i));
    }

    /// Starts the selected game and transitions to the appropriate next state
    ///
    /// Creates a new game instance, resets game state, and either goes to
    /// player configuration (normal mode) or directly to gameplay (AI-only mode).
    /// Also handles special options like Settings and Quit.
    pub fn start_game(&mut self) {
        if let Some(selected) = self.game_selection_state.selected() {
            if selected < self.games.len() {
                // Create game using current settings instead of old factory
                let game_name = self.games[selected].0;
                self.game_wrapper = self.create_game_with_current_settings(game_name);
                self.game_status = GameStatus::InProgress;
                self.last_search_stats = None;
                self.move_history.clear();

                let num_players = self.game_wrapper.get_num_players();

                // Only reset player options if we don't have the right number of players
                // or if we don't have any player options configured yet
                if self.player_options.is_empty()
                    || self.player_options.len() != num_players as usize
                {
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

    /// Resets the current game to its initial state
    ///
    /// Creates a fresh game instance while preserving player configuration.
    /// Clears move history, resets game status, and positions the cursor
    /// appropriately for the game type.
    pub fn reset_game(&mut self) {
        // Get the currently selected game and reset its state without changing player config
        if let Some(selected) = self.game_selection_state.selected() {
            if selected < self.games.len() {
                // Create game using current settings instead of old factory
                let game_name = self.games[selected].0;
                self.game_wrapper = self.create_game_with_current_settings(game_name);
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

                // Recreate AI worker to ensure fresh state for new game
                // This prevents crashes when restarting after a game where AI was the last player
                self.recreate_ai_worker();
            }
        }
    }

    // Settings navigation methods

    /// Moves to the next setting in the settings menu
    ///
    /// Wraps around to the first setting when reaching the end.
    pub fn select_next_setting(&mut self) {
        self.selected_settings_index = (self.selected_settings_index + 1) % 12; // 10 settings + separator + back
    }

    /// Moves to the previous setting in the settings menu
    ///
    /// Wraps around to the last setting when reaching the beginning.
    pub fn select_prev_setting(&mut self) {
        self.selected_settings_index = (self.selected_settings_index + 11) % 12;
    }

    /// Increases the value of the currently selected setting
    ///
    /// Each setting has its own valid range and increment amount.
    /// Boolean settings get toggled instead of incremented.
    pub fn increase_setting(&mut self) {
        let old_ai_settings = (
            self.settings_ai_threads,
            self.settings_max_nodes,
            self.settings_search_iterations,
            self.settings_exploration_constant,
        );

        match self.selected_settings_index {
            0 => self.settings_board_size = (self.settings_board_size + 1).min(25),
            1 => self.settings_line_size = (self.settings_line_size + 1).min(10),
            2 => self.settings_ai_threads = (self.settings_ai_threads + 1).min(16),
            3 => self.settings_max_nodes = (self.settings_max_nodes + 100000).min(10000000),
            4 => {
                self.settings_search_iterations =
                    (self.settings_search_iterations + 10000).min(10000000)
            }
            5 => {
                self.settings_exploration_constant =
                    (self.settings_exploration_constant + 0.1).min(10.0)
            }
            6 => self.timeout_secs = (self.timeout_secs + 10).min(600), // Max 10 minutes
            7 => self.stats_interval_secs = (self.stats_interval_secs + 5).min(120), // Max 2 minutes
            8 => self.ai_only = !self.ai_only,                                       // Toggle
            9 => self.shared_tree = !self.shared_tree,                               // Toggle
            _ => {} // separator or back
        }

        // Recreate AI worker if AI-related settings changed
        let new_ai_settings = (
            self.settings_ai_threads,
            self.settings_max_nodes,
            self.settings_search_iterations,
            self.settings_exploration_constant,
        );
        if old_ai_settings != new_ai_settings {
            self.recreate_ai_worker();
        }
    }

    /// Decreases the value of the currently selected setting
    ///
    /// Each setting has its own valid range and decrement amount.
    /// Boolean settings get toggled instead of decremented.
    /// Values are clamped to their minimum allowed values.
    pub fn decrease_setting(&mut self) {
        let old_ai_settings = (
            self.settings_ai_threads,
            self.settings_max_nodes,
            self.settings_search_iterations,
            self.settings_exploration_constant,
        );

        match self.selected_settings_index {
            0 => self.settings_board_size = self.settings_board_size.saturating_sub(1).max(3),
            1 => self.settings_line_size = self.settings_line_size.saturating_sub(1).max(3),
            2 => self.settings_ai_threads = self.settings_ai_threads.saturating_sub(1).max(1),
            3 => {
                self.settings_max_nodes = self.settings_max_nodes.saturating_sub(100000).max(10000)
            }
            4 => {
                self.settings_search_iterations = self
                    .settings_search_iterations
                    .saturating_sub(10000)
                    .max(1000)
            }
            5 => {
                self.settings_exploration_constant =
                    (self.settings_exploration_constant - 0.1).max(0.1)
            }
            6 => self.timeout_secs = self.timeout_secs.saturating_sub(10).max(5), // Min 5 seconds
            7 => self.stats_interval_secs = self.stats_interval_secs.saturating_sub(5).max(5), // Min 5 seconds
            8 => self.ai_only = !self.ai_only,         // Toggle
            9 => self.shared_tree = !self.shared_tree, // Toggle
            _ => {}                                    // separator or back
        }

        // Recreate AI worker if AI-related settings changed
        let new_ai_settings = (
            self.settings_ai_threads,
            self.settings_max_nodes,
            self.settings_search_iterations,
            self.settings_exploration_constant,
        );
        if old_ai_settings != new_ai_settings {
            self.recreate_ai_worker();
        }
    }

    /// Gracefully shut down the application
    ///
    /// Ensures all threads are properly stopped before exiting.
    /// This is especially important when AI is in the middle of a search.
    /// Gives threads time to complete their current operations cleanly.
    pub fn shutdown(&mut self) {
        // Explicitly stop the AI worker
        self.ai_worker.stop();

        // Give threads more time to shut down gracefully
        // This is especially important when AI is in the middle of a search
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    /// Recreates the AI worker with current settings
    ///
    /// This is called when AI-related settings are changed in the settings menu
    /// to ensure the AI worker uses the updated configuration.
    pub fn recreate_ai_worker(&mut self) {
        // Stop the old worker first
        self.ai_worker.stop();

        // Create a new worker with current settings
        self.ai_worker = AIWorker::new(
            self.settings_exploration_constant,
            self.settings_ai_threads,
            self.settings_search_iterations,
            self.settings_max_nodes,
        );
    }

    /// Apply current settings to the active game
    ///
    /// Recreates the current game using updated settings if we're currently in a game.
    /// This ensures that settings changes take effect immediately without requiring manual reset.
    pub fn apply_settings_to_current_game(&mut self) {
        // Only recreate game if we're currently in game mode (not in menus)
        if matches!(self.mode, AppMode::InGame | AppMode::GameOver) {
            if let Some(selected) = self.game_selection_state.selected() {
                if selected < self.games.len() {
                    let game_name = self.games[selected].0;

                    // Store the current game state
                    let was_in_progress = self.game_status == GameStatus::InProgress;

                    // Recreate the game with current settings
                    self.game_wrapper = self.create_game_with_current_settings(game_name);

                    // Reset game state
                    self.game_status = GameStatus::InProgress;
                    self.last_search_stats = None;
                    self.move_history.clear();

                    // Reset cursor position based on new game type
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

                    // Clear any game-specific selections
                    if matches!(self.game_wrapper, GameWrapper::Blokus(_)) {
                        self.blokus_ui_config.selected_piece_idx = None;
                        self.blokus_ui_config.selected_transformation_idx = 0;
                    }

                    // Recreate AI worker to ensure fresh state
                    self.recreate_ai_worker();

                    // If we were in game mode, stay in game mode; if game over, go back to game
                    if was_in_progress || self.mode == AppMode::GameOver {
                        self.mode = AppMode::InGame;
                    }
                }
            }
        }
    }

    /// Scrolls the debug panel up by one line
    pub fn scroll_debug_up(&mut self) {
        self.debug_scroll = self.debug_scroll.saturating_sub(1);
    }

    /// Scrolls the debug panel down by one line
    pub fn scroll_debug_down(&mut self) {
        self.debug_scroll = self.debug_scroll.saturating_add(1);
    }

    /// Scrolls the move history panel up by one line
    ///
    /// Disables auto-scroll when user manually scrolls.
    pub fn scroll_move_history_up(&mut self) {
        self.history_scroll = self.history_scroll.saturating_sub(1);
        self.on_user_history_scroll();
    }

    /// Scrolls the move history panel down by one line
    ///
    /// Disables auto-scroll when user manually scrolls.
    pub fn scroll_move_history_down(&mut self) {
        self.history_scroll = self.history_scroll.saturating_add(1);
        self.on_user_history_scroll();
    }

    // Enhanced Blokus functionality

    /// Selects a Blokus piece for placement
    ///
    /// Only allows selection of pieces that are available to the current player.
    /// Automatically tries to find a valid cursor position for the selected piece.
    ///
    /// # Arguments
    /// * `piece_idx` - Index of the piece to select
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
    ///
    /// Validates that the piece transformation would not extend outside the board.
    /// Used for cursor movement validation and ghost piece display.
    ///
    /// # Arguments
    /// * `piece_idx` - Index of the piece to check
    /// * `transformation_idx` - Transformation index (rotation/reflection)
    ///
    /// # Returns
    /// true if the piece fits within bounds, false otherwise
    fn would_blokus_piece_fit_at_cursor(
        &self,
        piece_idx: usize,
        transformation_idx: usize,
    ) -> bool {
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
                        if board_r < 0
                            || board_r >= board_height as i32
                            || board_c < 0
                            || board_c >= board_width as i32
                        {
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
                        let current_transformation =
                            self.blokus_ui_config.selected_transformation_idx;
                        let total_transformations = piece.transformations.len();

                        if total_transformations > 0 {
                            // Calculate the next transformation index
                            let next_transformation =
                                (current_transformation + 1) % total_transformations;

                            // Check if the piece would fit at the current cursor position with the new transformation
                            if self.would_blokus_piece_fit_at_cursor(piece_idx, next_transformation)
                            {
                                // Fits at current position, just rotate
                                self.blokus_ui_config.selected_transformation_idx =
                                    next_transformation;
                            } else {
                                // Doesn't fit at current position, find a new position and then rotate
                                // Temporarily set the transformation to see if we can find a valid position
                                let old_transformation =
                                    self.blokus_ui_config.selected_transformation_idx;
                                self.blokus_ui_config.selected_transformation_idx =
                                    next_transformation;

                                if self.find_valid_cursor_position_for_piece(
                                    piece_idx,
                                    next_transformation,
                                ) {
                                    // Found a valid position, keep the new transformation and position
                                    // (cursor was already moved by find_valid_cursor_position_for_piece)
                                } else {
                                    // Couldn't find a valid position, revert transformation
                                    self.blokus_ui_config.selected_transformation_idx =
                                        old_transformation;
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

    // TODO: Find out why we're giving special treatment to Blokus here.
    pub fn blokus_place_piece(&mut self) {
        if let Some((piece_idx, transformation_idx)) =
            self.blokus_ui_config.get_selected_piece_info()
        {
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
        // Track user scroll interaction
        self.piece_panel_auto_scroll = false;
        self.piece_panel_user_scroll_time = Some(std::time::Instant::now());
    }

    pub fn blokus_scroll_panel_down(&mut self) {
        self.blokus_ui_config.scroll_panel_down();
        // Track user scroll interaction
        self.piece_panel_auto_scroll = false;
        self.piece_panel_user_scroll_time = Some(std::time::Instant::now());
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

    /// Update auto-scroll behavior for Blokus piece panel
    fn update_piece_panel_auto_scroll(&mut self) {
        // Check if current player has changed
        let current_player = self.game_wrapper.get_current_player();
        if current_player != self.last_current_player {
            self.last_current_player = current_player;
            // Force auto-scroll to the new current player (unless user has scrolled very recently)
            if let Some(user_scroll_time) = self.piece_panel_user_scroll_time {
                // If user scrolled less than 15 seconds ago, don't override their scroll
                if user_scroll_time.elapsed() >= self.piece_panel_auto_scroll_reset_duration {
                    self.piece_panel_auto_scroll = true;
                    self.piece_panel_user_scroll_time = None;
                }
            } else {
                // No recent user scroll, enable auto-scroll
                self.piece_panel_auto_scroll = true;
            }
        }

        // Check if we should reset auto-scroll after user interaction
        if let Some(user_scroll_time) = self.piece_panel_user_scroll_time {
            if user_scroll_time.elapsed() >= self.piece_panel_auto_scroll_reset_duration {
                self.piece_panel_auto_scroll = true;
                self.piece_panel_user_scroll_time = None;
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

            // Clear AI state when game ends
            self.ai_thinking_start = None;
            self.pending_ai_response = None;
            // Stop any ongoing AI search
            self.ai_worker.stop();
        }
    }

    /// Clear selected Blokus piece if it becomes unavailable
    pub fn clear_selected_piece_if_unavailable(&mut self) {
        if let (Some(piece_idx), GameWrapper::Blokus(state)) =
            (self.blokus_ui_config.selected_piece_idx, &self.game_wrapper)
        {
            let available_pieces = state.get_available_pieces(state.get_current_player());
            if !available_pieces.contains(&piece_idx) {
                self.blokus_ui_config.selected_piece_idx = None;
                self.blokus_ui_config.selected_transformation_idx = 0;
            }
        }
    }

    /// Find a valid cursor position for the given Blokus piece and transformation
    fn find_valid_cursor_position_for_piece(
        &mut self,
        piece_idx: usize,
        transformation_idx: usize,
    ) -> bool {
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
                for row in
                    start_row.saturating_sub(radius)..=(start_row + radius).min(board_height - 1)
                {
                    for col in
                        start_col.saturating_sub(radius)..=(start_col + radius).min(board_width - 1)
                    {
                        // Only check the border of the current radius
                        if (row == start_row.saturating_sub(radius)
                            || row == (start_row + radius).min(board_height - 1))
                            || (col == start_col.saturating_sub(radius)
                                || col == (start_col + radius).min(board_width - 1))
                        {
                            self.board_cursor = (row as u16, col as u16);
                            if self.would_blokus_piece_fit_at_cursor(piece_idx, transformation_idx)
                            {
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
                if player_id == 1 {
                    Color::Red
                } else {
                    Color::Yellow
                }
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
            _ => {
                // Gomoku and others
                // Game uses 1 and -1, map to UI colors
                if player_id == 1 {
                    Color::Red
                } else {
                    Color::Blue
                }
            }
        }
    }

    /// Get the player symbol/marker for display
    pub fn get_player_symbol(&self, player_id: i32) -> &'static str {
        match &self.game_wrapper {
            GameWrapper::Connect4(_) => {
                if player_id == 1 {
                    "🔴"
                } else {
                    "🟡"
                }
            }
            GameWrapper::Othello(_) => {
                if player_id == 1 {
                    "⚫"
                } else {
                    "⚪"
                }
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
            _ => {
                // Gomoku and others
                if player_id == 1 { "❌" } else { "⭕" }
            }
        }
    }

    /// Calculate the optimal scroll position to show the current player at the top
    /// of the piece panel (for Blokus auto-scroll feature)
    pub fn calculate_piece_panel_auto_scroll_position(&self) -> Option<usize> {
        if !self.piece_panel_auto_scroll {
            return None;
        }

        if let GameWrapper::Blokus(_blokus_state) = &self.game_wrapper {
            let current_player = self.game_wrapper.get_current_player();
            if current_player < 1 || current_player > 4 {
                return None;
            }

            // Calculate actual line position by simulating the content generation
            // This mirrors the logic in draw_blokus_piece_selection
            let pieces = crate::games::blokus::get_blokus_pieces(); //  TODO: Cache this as get_blokus_pieces() is expensive
            let mut line_count = 0usize;

            for player in 1..=4 {
                // This is where the current player's header will appear
                if player == current_player {
                    return Some(line_count);
                }

                // Player header line
                line_count += 1;

                // Check if this player is expanded
                let is_expanded = self
                    .blokus_ui_config
                    .players_expanded
                    .get((player - 1) as usize)
                    .unwrap_or(&true);

                if *is_expanded {
                    let is_current = player == current_player;
                    let visible_pieces = if is_current { 21 } else { 10 };
                    let total_pieces_to_show = if is_current {
                        21
                    } else {
                        visible_pieces.min(21)
                    };
                    let pieces_per_row = 5;

                    // Add top border for current player's grid
                    if is_current && total_pieces_to_show > 0 {
                        line_count += 1;
                    }

                    // Calculate piece rows with exact heights
                    for chunk_start in (0..total_pieces_to_show).step_by(pieces_per_row) {
                        let chunk_end = (chunk_start + pieces_per_row).min(total_pieces_to_show);

                        // Get max height for this row by examining all pieces in the row
                        let mut max_height = 1;
                        for display_idx in chunk_start..chunk_end {
                            let piece_idx = display_idx;
                            if let Some(piece) = pieces.get(piece_idx) {
                                if !piece.transformations.is_empty() {
                                    let piece_shape = &piece.transformations[0];
                                    let piece_height =
                                        Self::calculate_visual_piece_height(piece_shape);
                                    max_height = max_height.max(piece_height);
                                }
                            }
                        }

                        // Key line: 1 line
                        line_count += 1;

                        // Shape lines: max_height lines
                        line_count += max_height;

                        // Row separator (except for last row): 1 line
                        if chunk_start + pieces_per_row < total_pieces_to_show {
                            line_count += 1;
                        }
                    }

                    // Add bottom border for current player's grid
                    if is_current && total_pieces_to_show > 0 {
                        line_count += 1;
                    }
                } else {
                    // Just the summary line when collapsed
                    line_count += 1;
                }

                // Add separator between players (except after the last one)
                if player < 4 {
                    line_count += 1;
                }
            }

            // If we get here, current_player was not found (shouldn't happen)
            None
        } else {
            None
        }
    }

    /// Calculate the visual height of a piece shape
    /// This mirrors the logic in create_visual_piece_shape from blokus_ui.rs
    /// Calculate visual piece height using the same logic as create_visual_piece_shape
    /// This ensures consistency between auto-scroll calculation and rendering
    fn calculate_visual_piece_height(piece_shape: &[(i32, i32)]) -> usize {
        if piece_shape.is_empty() {
            return 1; // "▢" takes 1 line
        }

        // Calculate grid dimensions (same as create_visual_piece_shape)
        let min_r = piece_shape.iter().map(|p| p.0).min().unwrap_or(0);
        let max_r = piece_shape.iter().map(|p| p.0).max().unwrap_or(0);
        let min_c = piece_shape.iter().map(|p| p.1).min().unwrap_or(0);
        let max_c = piece_shape.iter().map(|p| p.1).max().unwrap_or(0);

        let height = (max_r - min_r + 1) as usize;
        let width = (max_c - min_c + 1) as usize;

        // Create the visual grid (same algorithm as create_visual_piece_shape)
        let mut grid = vec![vec![' '; width]; height];
        for &(r, c) in piece_shape {
            let gr = (r - min_r) as usize;
            let gc = (c - min_c) as usize;
            grid[gr][gc] = '▢';
        }

        // Convert to lines (same as create_visual_piece_shape)
        let mut result: Vec<String> = grid
            .iter()
            .map(|row| row.iter().collect::<String>())
            .collect();

        // Apply special handling for single character pieces (same as create_visual_piece_shape)
        if result.len() == 1 && result[0].trim().len() == 1 {
            result[0] = format!(" {} ", result[0].trim());
        }

        result.len()
    }
}
