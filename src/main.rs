//! # Parallel Multi-Game MCTS Engine
//!
//! This is the main entry point for a multi-game engine that supports Gomoku, Connect 4,
//! Othello, and Blokus. The engine uses a parallel Monte Carlo Tree Search (MCTS) algorithm
//! for AI gameplay.
//!
//! The application provides a terminal user interface (TUI) built with Ratatui for interactive
//! gameplay between humans and AI opponents.
//!
//! ## Features
//! - Multiple game support with unified AI engine
//! - Parallel MCTS with configurable parameters
//! - Interactive terminal UI with mouse support
//! - Real-time AI analysis and statistics
//! - Move history tracking
//!
//! ## Usage
//! Run with `cargo run --release` for best performance.

pub mod app;
pub mod games;
pub mod game_wrapper;
pub mod tui;

use crate::app::App;
use clap::Parser;
use std::io;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The exploration factor for the MCTS algorithm.
    #[arg(short, long, default_value_t = 4.0)]
    exploration_factor: f64,

    /// The number of search iterations for the MCTS algorithm.
    #[arg(short, long, default_value_t = 1000000)]
    search_iterations: u32,

    /// The maximum number of nodes in the MCTS search tree.
    #[arg(short, long, default_value_t = 1000000)]
    max_nodes: usize,

    /// The number of threads to use for the search.
    #[arg(short, long, default_value_t = 8)]
    num_threads: usize,

    /// The game to play.
    #[arg(short, long)]
    game: Option<String>,

    /// The size of the board (for Gomoku and Othello).
    #[arg(short, long, default_value_t = 15)]
    board_size: usize,

    /// The number of pieces in a row to win (for Gomoku and Connect 4).
    #[arg(short, long, default_value_t = 5)]
    line_size: usize,

    /// Maximum time AI can think per move (in seconds).
    #[arg(long, default_value_t = 60)]
    timeout_secs: u64,

    /// How often to send statistics updates (in seconds).
    #[arg(long, default_value_t = 20)]
    stats_interval_secs: u64,

    /// Whether this is an AI vs AI only game.
    #[arg(long, action = clap::ArgAction::SetTrue)]
    ai_only: bool,

    /// Whether to share the search tree between moves.
    #[arg(long, action = clap::ArgAction::SetTrue, default_value_t = true)]
    shared_tree: bool,
}

/// Main entry point for the Parallel Multi-Game MCTS Engine
/// 
/// This function:
/// 1. Parses command line arguments using clap
/// 2. Adjusts default parameters based on the selected game
/// 3. Creates and configures the main App instance
/// 4. Launches the terminal user interface
/// 
/// # Returns
/// `io::Result<()>` - Ok if the application runs successfully, or an IO error
/// 
/// # Examples
/// ```bash
/// # Run with default settings
/// cargo run --release
/// 
/// # Run Gomoku with custom settings
/// cargo run --release -- --game Gomoku --board-size 19 --exploration-factor 1.4
/// 
/// # Run AI vs AI mode
/// cargo run --release -- --ai-only --timeout-secs 10
/// ```
fn main() -> io::Result<()> {
    let mut args = Args::parse();

    // Apply game-specific default configurations
    // This ensures that each game uses appropriate default settings
    if let Some(game_name) = &args.game {
        match game_name.as_str().to_lowercase().as_str() {
            "gomoku" => {
                if args.board_size == 15 {
                    args.board_size = 15; // Standard Gomoku board
                }
                if args.line_size == 5 {
                    args.line_size = 5; // Standard Gomoku win condition
                }
            }
            "connect4" => {
                if args.board_size == 15 { // Changed from default
                    args.board_size = 7; // Standard Connect4 width
                }
                if args.line_size == 5 { // Changed from default
                    args.line_size = 4; // Standard Connect4 win condition
                }
            }
            "othello" => {
                if args.board_size == 15 { // Changed from default  
                    args.board_size = 8; // Standard Othello board
                }
            }
            _ => {} // Blokus or other games don't need this
        }
    }

    // Ensure we have at least one thread for the AI
    let num_threads = if args.num_threads > 0 {
        args.num_threads
    } else {
        8 // Default to 8 threads
    };

    let mut app = App::new(
        args.exploration_factor,
        num_threads,
        args.search_iterations,
        args.max_nodes,
        args.game,
        args.board_size,
        args.line_size,
        args.timeout_secs,
        args.stats_interval_secs,
        args.ai_only,
        args.shared_tree,
    );

    tui::run(&mut app)
}
