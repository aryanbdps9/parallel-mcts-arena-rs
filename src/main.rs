//! # Parallel Multi-Game MCTS Engine
//!
//! This is the main entry point for a comprehensive multi-game engine that supports:
//! - **Gomoku** (Five in a Row)
//! - **Connect 4** (Four in a Row)  
//! - **Othello** (Reversi)
//! - **Blokus** (Territory Control)
//!
//! The engine uses a sophisticated parallel Monte Carlo Tree Search (MCTS) algorithm
//! for AI gameplay with advanced features like virtual losses, node recycling,
//! and adaptive time management.
//!
//! ## Architecture Overview
//! - **Component-based UI**: Modular TUI components using Ratatui
//! - **Game Abstraction**: Unified interface for all game types
//! - **Parallel AI**: Multi-threaded MCTS with configurable parameters
//! - **Event System**: Centralized event handling for user interactions
//! - **State Management**: Comprehensive application state tracking
//!
//! ## Key Features
//! - Multiple game support with unified AI engine
//! - Parallel MCTS with virtual losses and tree reuse
//! - Interactive terminal UI with full mouse support
//! - Real-time AI analysis and move statistics
//! - Complete move history with replay capability
//! - Responsive layout adapting to terminal size
//! - Configurable AI difficulty and time limits
//! - Background AI computation with progressive updates
//!
//! ## Performance Considerations
//! - Use `--release` flag for optimal AI performance
//! - Default 8 threads provide good balance for most systems
//! - Tree sharing between moves reduces redundant computation
//! - Virtual losses prevent thread collision in parallel search
//!
//! ## Usage Examples
//! ```bash
//! # Basic usage with menu selection
//! cargo run --release
//!
//! # Start specific game with custom AI settings
//! cargo run --release -- --game Gomoku --exploration-factor 1.4 --timeout-secs 30
//!
//! # AI vs AI tournament mode
//! cargo run --release -- --ai-only --num-threads 16 --search-iterations 10000000
//!
//! # Debug mode with detailed statistics
//! cargo run -- --stats-interval-secs 5 --timeout-secs 10
//! ```

// Import the main application modules
// Each module handles a specific aspect of the application:
pub mod app; // Core application state and configuration
pub mod components;
pub mod game_wrapper; // Unified interface for all games
pub mod games; // Game implementations (Gomoku, Connect4, Othello, Blokus)
pub mod tui; // Terminal user interface implementation // Modular UI components and event system

use crate::app::App;
use clap::Parser;
use std::io;

/// Command-line argument parser using clap derive macros
///
/// This struct defines all configurable parameters for the MCTS engine.
/// Default values are optimized for good performance on typical systems.
///
/// # Design Philosophy
/// - Conservative defaults that work well across different games
/// - Game-specific overrides applied in main() function
/// - All timing parameters in human-readable units (seconds)
/// - Thread count defaults to 8 for good parallelism without oversaturation
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The exploration factor (C) for the MCTS PUCT formula.
    ///
    /// Higher values encourage more exploration of untested moves.
    /// Lower values favor exploitation of known good moves.
    ///
    #[arg(short, long, default_value_t = 4.0)]
    exploration_factor: f64,

    /// Maximum number of MCTS iterations per search.
    ///
    /// More iterations generally lead to stronger play but longer thinking time.
    /// This acts as a hard cap to prevent infinite search.
    ///
    /// Practical values:
    /// - Quick games: 100,000 - 500,000
    /// - Standard play: 1,000,000 - 5,000,000  
    /// - Tournament strength: 10,000,000+
    #[arg(short, long, default_value_t = 1000000)]
    search_iterations: u32,

    /// Maximum number of nodes to store in the MCTS tree.
    ///
    /// Larger trees can store more game analysis but use more memory.
    /// When limit is reached, oldest nodes are recycled.
    ///
    /// Memory usage roughly: max_nodes * 200 bytes
    /// - 1M nodes ≈ 200MB RAM
    /// - 10M nodes ≈ 2GB RAM
    #[arg(short, long, default_value_t = 1000000)]
    max_nodes: usize,

    /// Number of parallel threads for MCTS search.
    ///
    /// More threads can speed up search but with diminishing returns.
    /// Optimal value typically equals CPU core count.
    ///
    /// Recommendations:
    /// - Laptop/Desktop: 4-8 threads
    /// - Workstation: 8-16 threads
    /// - Server: 16+ threads
    #[arg(short, long, default_value_t = 8)]
    num_threads: usize,

    /// Pre-select a specific game to play on startup.
    ///
    /// Valid options: "Gomoku", "Connect4", "Othello", "Blokus"
    /// If not specified, user will see game selection menu.
    ///
    /// Case-insensitive matching is used for convenience.
    #[arg(short, long)]
    game: Option<String>,

    /// Board size for games that support variable dimensions.
    ///
    /// Used by:
    /// - Gomoku: Board is board_size × board_size (typically 15×15 or 19×19)
    /// - Othello: Fixed 8×8 (this parameter ignored)
    /// - Connect4: Width only, height is fixed at 6
    /// - Blokus: Fixed 20×20 (this parameter ignored)
    ///
    /// Common values:
    /// - Gomoku: 15 (standard), 19 (professional)
    /// - Connect4: 7 (standard width)
    #[arg(short, long, default_value_t = 15)]
    board_size: usize,

    /// Number of pieces in a row required to win.
    ///
    /// Used by:
    /// - Gomoku: Exactly 5 pieces in a row (standard)
    /// - Connect4: Exactly 4 pieces in a row (standard)
    /// - Othello: Not applicable (territory control game)
    /// - Blokus: Not applicable (area coverage game)
    ///
    /// Note: Changing this value creates game variants
    #[arg(short, long, default_value_t = 5)]
    line_size: usize,

    /// Maximum thinking time per AI move in seconds.
    ///
    /// The AI will stop searching when this time limit is reached,
    /// even if max iterations haven't been completed.
    ///
    /// Recommended values:
    /// - Blitz games: 5-10 seconds
    /// - Casual play: 30-60 seconds  
    /// - Tournament: 120+ seconds
    ///
    /// Note: Actual move time may be slightly longer due to
    /// minimum display duration for user experience.
    #[arg(long, default_value_t = 60)]
    timeout_secs: u64,

    /// Frequency of AI statistics updates in seconds.
    ///
    /// Controls how often the UI refreshes with current search progress.
    /// More frequent updates provide better feedback but may impact performance.
    ///
    /// Recommended: 5-30 seconds depending on total thinking time.
    #[arg(long, default_value_t = 20)]
    stats_interval_secs: u64,

    /// Enable AI vs AI mode with no human interaction.
    ///
    /// When enabled:
    /// - Both players are controlled by AI
    /// - Games play automatically from start to finish
    /// - Useful for testing AI strength and game balance
    /// - Can be combined with shorter timeouts for rapid testing
    #[arg(long, action = clap::ArgAction::SetTrue)]
    ai_only: bool,

    /// Enable tree sharing between consecutive moves.
    ///
    /// When enabled, the MCTS tree is preserved after each move
    /// and reused for the next search. This can significantly improve
    /// AI strength by avoiding redundant computation.
    ///
    /// Benefits:
    /// - Faster response for expected continuations
    /// - Better long-term strategic planning
    /// - More efficient use of computation time
    ///
    /// Note: Uses more memory to maintain the tree
    #[arg(long, action = clap::ArgAction::SetTrue, default_value_t = true)]
    shared_tree: bool,
}

/// Main entry point for the Parallel Multi-Game MCTS Engine
///
/// This function orchestrates the entire application lifecycle:
/// 1. **Command Line Parsing**: Uses clap to parse and validate arguments
/// 2. **Game-Specific Configuration**: Applies optimal defaults for each game type
/// 3. **Parameter Validation**: Ensures thread count and other values are sensible
/// 4. **App Initialization**: Creates the main application instance with all settings
/// 5. **TUI Launch**: Transfers control to the terminal user interface
///
/// # Game-Specific Defaults
/// The function applies intelligent defaults based on the selected game:
/// - **Gomoku**: 15×15 board, 5-in-a-row win condition
/// - **Connect4**: 7-wide board, 4-in-a-row win condition  
/// - **Othello**: 8×8 board (fixed), territorial control
/// - **Blokus**: 20×20 board (fixed), area maximization
///
/// # Error Handling
/// Returns `io::Result<()>` to propagate any terminal initialization
/// or rendering errors from the TUI layer.
///
/// # Thread Safety
/// The function ensures at least one thread is allocated for AI computation
/// to prevent deadlock scenarios.
///
/// # Examples
/// ```bash
/// # Interactive menu selection
/// cargo run --release
///
/// # Quick Gomoku game with strong AI
/// cargo run --release -- --game Gomoku --exploration-factor 1.4 --timeout-secs 30
///
/// # AI tournament mode with detailed logging
/// cargo run --release -- --ai-only --num-threads 16 --stats-interval-secs 5
///
/// # Custom Connect4 with larger board
/// cargo run --release -- --game Connect4 --board-size 9 --line-size 4
/// ```
fn main() -> io::Result<()> {
    let mut args = Args::parse();

    // Apply game-specific default configurations
    // This ensures each game uses appropriate parameters for optimal gameplay.
    //
    // Design rationale:
    // - Each game has different complexity and optimal board sizes
    // - Standard tournament rules are respected where applicable
    // - Overrides only apply when user hasn't specified custom values
    if let Some(game_name) = &args.game {
        match game_name.as_str().to_lowercase().as_str() {
            "gomoku" => {
                // Gomoku (Five in a Row)
                // Standard tournament size is 15×15, professional is 19×19
                if args.board_size == 15 {
                    args.board_size = 15; // Keep standard size
                }
                if args.line_size == 5 {
                    args.line_size = 5; // Five in a row is the classic rule
                }
            }
            "connect4" => {
                // Connect Four
                // Standard board is 7 wide × 6 tall, 4 in a row to win
                if args.board_size == 15 {
                    // User didn't specify custom size
                    args.board_size = 7; // Standard Connect4 width
                }
                if args.line_size == 5 {
                    // User didn't specify custom line size
                    args.line_size = 4; // Four in a row is the standard
                }
            }
            "othello" => {
                // Othello (Reversi)
                // Always uses 8×8 board per official rules
                if args.board_size == 15 {
                    // User didn't specify custom size
                    args.board_size = 8; // Official Othello board size
                }
                // line_size is not used for Othello (territorial game)
            }
            "blokus" => {
                // Blokus uses a fixed 20×20 board
                // These parameters don't affect Blokus gameplay
            }
            _ => {
                // Unknown game name - let the app handle the error
                // This allows for future game additions or typo handling
            }
        }
    }

    // Ensure we have at least one thread for AI computation
    // Zero threads would cause deadlock in the thread pool
    let num_threads = if args.num_threads > 0 {
        args.num_threads
    } else {
        8 // Safe default that works well on most systems
    };

    // Create the main application instance with all configuration
    // This constructor validates parameters and sets up initial state
    let mut app = App::new(
        args.exploration_factor,  // MCTS exploration parameter
        num_threads,              // Parallel search threads
        args.search_iterations,   // Maximum iterations per search
        args.max_nodes,           // Memory limit for search tree
        args.game,                // Pre-selected game (optional)
        args.board_size,          // Board dimensions
        args.line_size,           // Win condition
        args.timeout_secs,        // AI thinking time limit
        args.stats_interval_secs, // UI update frequency
        args.ai_only,             // AI vs AI mode
        args.shared_tree,         // Tree reuse between moves
    );

    // Launch the terminal user interface
    // This transfers control to the TUI event loop until the user exits
    tui::run(&mut app)
}
