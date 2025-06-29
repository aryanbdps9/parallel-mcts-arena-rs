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
use num_cpus;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The exploration factor for the MCTS algorithm.
    #[arg(short, long, default_value_t = 1.4)]
    exploration_factor: f64,

    /// The number of search iterations for the MCTS algorithm.
    #[arg(short, long, default_value_t = 10000)]
    search_iterations: u32,

    /// The maximum number of nodes in the MCTS search tree.
    #[arg(short, long, default_value_t = 1000000)]
    max_nodes: usize,

    /// The number of threads to use for the search.
    #[arg(short, long, default_value_t = 0)]
    num_threads: usize,

    /// The game to play.
    #[arg(short, long)]
    game: Option<String>,

    /// The size of the board (for Gomoku and Othello).
    #[arg(short, long, default_value_t = 0)]
    board_size: usize,

    /// The number of pieces in a row to win (for Gomoku and Connect 4).
    #[arg(short, long, default_value_t = 0)]
    line_size: usize,
}

fn main() -> io::Result<()> {
    let mut args = Args::parse();

    if let Some(game_name) = &args.game {
        match game_name.as_str() {
            "Gomoku" => {
                if args.board_size == 0 {
                    args.board_size = 15;
                }
                if args.line_size == 0 {
                    args.line_size = 5;
                }
            }
            "Connect4" => {
                if args.board_size == 0 {
                    args.board_size = 7; // width
                }
                if args.line_size == 0 {
                    args.line_size = 4;
                }
            }
            "Othello" => {
                if args.board_size == 0 {
                    args.board_size = 8;
                }
            }
            _ => {} // Blokus or other games don't need this
        }
    }

    let num_threads = if args.num_threads > 0 {
        args.num_threads
    } else {
        num_cpus::get()
    };

    let mut app = App::new(
        args.exploration_factor,
        num_threads,
        args.max_nodes,
        args.game,
        args.board_size,
        args.line_size,
    );

    tui::run(&mut app)
}
