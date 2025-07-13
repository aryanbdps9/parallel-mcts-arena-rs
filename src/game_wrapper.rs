//! # Game Wrapper Module - Unified Game Interface
//!
//! This module provides the abstraction layer that allows the MCTS engine and UI
//! components to work with any supported game type through a single, unified interface.
//! It implements the adapter pattern to bridge between game-specific implementations
//! and the generic algorithms that operate on them.
//!
//! ## Design Philosophy
//! The wrapper system serves several critical purposes:
//! - **Type Safety**: Each game maintains its specific types while being usable generically
//! - **Algorithm Reuse**: MCTS and UI code works with any game without modification
//! - **Extensibility**: New games can be added with minimal changes to existing code
//! - **Performance**: Zero-cost abstractions that compile to direct calls
//!
//! ## Architecture Overview
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │   MCTS Engine   │◄──►│   GameWrapper    │◄──►│  Game-Specific  │
//! │                 │    │                  │    │ Implementations │
//! │ • Generic AI    │    │ • Unified API    │    │                 │
//! │ • Tree Search   │    │ • Type Safety    │    │ • GomokuState   │
//! │ • Statistics    │    │ • Move Handling  │    │ • Connect4State │
//! └─────────────────┘    └──────────────────┘    │ • OthelloState  │
//!                                                 │ • BlokusState   │
//!                                                 └─────────────────┘
//! ```
//!
//! ## Key Benefits
//! - **Code Reuse**: Write UI and AI code once, use with all games
//! - **Maintainability**: Changes to game rules don't affect AI or UI code
//! - **Testing**: Easy to test algorithms against different game implementations
//! - **Performance**: Compile-time polymorphism with no runtime overhead
//!
//! ## Thread Safety
//! Both GameWrapper and MoveWrapper implement Send + Sync, making them safe
//! to use across thread boundaries. This is essential for the parallel MCTS
//! implementation where game states are shared between worker threads.
//!
//! ## Memory Efficiency
//! The wrapper enums use Rust's efficient enum representation, so there's
//! minimal memory overhead compared to using the game types directly.

// Import all game-specific types that will be wrapped
use crate::games::blokus::{BlokusMove, BlokusState}; // Multi-player territory game
use crate::games::connect4::{Connect4Move, Connect4State}; // Gravity-based 4-in-a-row game
use crate::games::gomoku::{GomokuMove, GomokuState}; // Classic 5-in-a-row game
use crate::games::othello::{OthelloMove, OthelloState}; // Reversi/Othello territory game
use mcts::GameState; // Core trait for MCTS compatibility
use std::fmt; // Formatting traits for display

/// Wrapper enum for all supported game types
///
/// This enum provides a unified interface for all game implementations while
/// maintaining type safety and zero-cost abstractions. Each variant contains
/// the complete game state for its respective game type.
///
/// ## Design Rationale
/// Using an enum rather than trait objects provides several advantages:
/// - **Performance**: No dynamic dispatch or heap allocation overhead
/// - **Type Safety**: Compile-time checking ensures all methods are implemented
/// - **Pattern Matching**: Allows game-specific optimizations when needed
/// - **Memory Efficiency**: No vtable overhead, optimal memory layout
///
/// ## Usage Patterns
/// The GameWrapper is used in two main contexts:
/// 1. **AI Engine**: MCTS algorithms operate on GameWrapper instances
/// 2. **UI System**: Rendering and input handling code works with any game type
///
/// ## Thread Safety
/// All contained game states implement Clone + Send + Sync, making the
/// wrapper safe to use in multi-threaded contexts like parallel MCTS.
#[derive(Debug, Clone)]
pub enum GameWrapper {
    /// Gomoku (Five in a Row) game state
    ///
    /// Classic board game where players alternate placing stones on a grid,
    /// trying to get five in a row horizontally, vertically, or diagonally.
    /// - Variable board size (typically 15×15 or 19×19)
    /// - Simple rules but deep strategic gameplay
    /// - Good for testing basic MCTS functionality
    Gomoku(GomokuState),

    /// Connect 4 game state
    ///
    /// Gravity-based game where pieces fall to the lowest available position
    /// in each column. Players try to get four in a row.
    /// - Typically 7 wide × 6 tall board
    /// - Fast-paced tactical gameplay
    /// - Constrained move space (only 7 possible moves per turn)
    Connect4(Connect4State),

    /// Blokus game state
    ///
    /// Complex 4-player game where players place polyomino pieces on a board,
    /// trying to maximize territory while blocking opponents.
    /// - Fixed 20×20 board with 4 players
    /// - 21 unique pieces per player with multiple orientations
    /// - Very high complexity and branching factor
    Blokus(BlokusState),

    /// Othello (Reversi) game state
    ///
    /// Territory control game where players place discs and flip opponent
    /// pieces by flanking them. Winner has most pieces at end.
    /// - Fixed 8×8 board
    /// - Complex positional evaluation
    /// - Classic AI testbed with well-understood strategy
    Othello(OthelloState),
}

/// Wrapper enum for all supported move types
///
/// Provides unified handling of moves across all game types while maintaining
/// type safety and allowing game-specific move data to be preserved.
///
/// ## Move Type Complexity
/// Different games have vastly different move complexity:
/// - **Gomoku/Othello**: Simple (row, col) coordinates
/// - **Connect4**: Single column number (gravity determines row)
/// - **Blokus**: Complex (piece_id, transformation, row, col) with validation
///
/// ## Serialization Support
/// All move types implement standard traits needed for serialization,
/// making it easy to save/load games or implement network play.
///
/// ## Hash and Equality
/// Moves implement Eq and Hash for use in data structures like HashSet
/// and HashMap, which is essential for MCTS tree node identification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MoveWrapper {
    /// Gomoku move: simple coordinate placement
    ///
    /// Contains (row, col) coordinates where the player wants to place their stone.
    /// Move validation ensures the position is empty and within board bounds.
    Gomoku(GomokuMove),

    /// Connect4 move: column selection with gravity
    ///
    /// Contains only the column number where the piece should be dropped.
    /// The actual row is determined by gravity (lowest available position).
    Connect4(Connect4Move),

    /// Blokus move: complex piece placement
    ///
    /// Contains (piece_id, transformation_id, row, col) specifying:
    /// - Which of the 21 pieces to place
    /// - Which transformation (rotation/reflection) to use
    /// - Where to place the piece's anchor point
    /// This is the most complex move type due to piece shape validation.
    Blokus(BlokusMove),

    /// Othello move: coordinate placement with captures
    ///
    /// Contains (row, col) coordinates, but move execution automatically
    /// calculates and performs all necessary piece captures in all directions.
    Othello(OthelloMove),
}

impl fmt::Display for MoveWrapper {
    /// Formats moves for display in UI and logs
    ///
    /// Each game type gets a compact string representation showing the essential move info.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MoveWrapper::Gomoku(m) => write!(f, "G({},{})", m.0, m.1),
            MoveWrapper::Connect4(m) => write!(f, "C4({})", m.0),
            MoveWrapper::Blokus(m) => write!(f, "B({})", m),
            MoveWrapper::Othello(m) => write!(f, "O({},{})", m.0, m.1),
        }
    }
}

impl fmt::Display for GameWrapper {
    /// Formats the game state for display
    ///
    /// Delegates to the specific game's Display implementation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameWrapper::Gomoku(g) => write!(f, "{}", g),
            GameWrapper::Connect4(g) => write!(f, "{}", g),
            GameWrapper::Blokus(g) => write!(f, "{}", g),
            GameWrapper::Othello(g) => write!(f, "{}", g),
        }
    }
}

impl GameState for GameWrapper {
    type Move = MoveWrapper;

    fn get_current_player(&self) -> i32 {
        match self {
            GameWrapper::Gomoku(g) => g.get_current_player(),
            GameWrapper::Connect4(g) => g.get_current_player(),
            GameWrapper::Blokus(g) => g.get_current_player(),
            GameWrapper::Othello(g) => g.get_current_player(),
        }
    }

    fn get_num_players(&self) -> i32 {
        match self {
            GameWrapper::Gomoku(g) => g.get_num_players(),
            GameWrapper::Connect4(g) => g.get_num_players(),
            GameWrapper::Blokus(g) => g.get_num_players(),
            GameWrapper::Othello(g) => g.get_num_players(),
        }
    }

    fn get_possible_moves(&self) -> Vec<Self::Move> {
        match self {
            GameWrapper::Gomoku(g) => g
                .get_possible_moves()
                .into_iter()
                .map(MoveWrapper::Gomoku)
                .collect(),
            GameWrapper::Connect4(g) => g
                .get_possible_moves()
                .into_iter()
                .map(MoveWrapper::Connect4)
                .collect(),
            GameWrapper::Blokus(g) => g
                .get_possible_moves()
                .into_iter()
                .map(MoveWrapper::Blokus)
                .collect(),
            GameWrapper::Othello(g) => g
                .get_possible_moves()
                .into_iter()
                .map(MoveWrapper::Othello)
                .collect(),
        }
    }

    fn make_move(&mut self, mv: &Self::Move) {
        match (self, mv) {
            (GameWrapper::Gomoku(g), MoveWrapper::Gomoku(m)) => g.make_move(m),
            (GameWrapper::Connect4(g), MoveWrapper::Connect4(m)) => g.make_move(m),
            (GameWrapper::Blokus(g), MoveWrapper::Blokus(m)) => g.make_move(m),
            (GameWrapper::Othello(g), MoveWrapper::Othello(m)) => g.make_move(m),
            _ => panic!("Mismatched game and move types"),
        }
    }

    fn is_terminal(&self) -> bool {
        match self {
            GameWrapper::Gomoku(g) => g.is_terminal(),
            GameWrapper::Connect4(g) => g.is_terminal(),
            GameWrapper::Blokus(g) => g.is_terminal(),
            GameWrapper::Othello(g) => g.is_terminal(),
        }
    }

    fn get_winner(&self) -> Option<i32> {
        match self {
            GameWrapper::Gomoku(g) => g.get_winner(),
            GameWrapper::Connect4(g) => g.get_winner(),
            GameWrapper::Blokus(g) => g.get_winner(),
            GameWrapper::Othello(g) => g.get_winner(),
        }
    }

    fn get_board(&self) -> &Vec<Vec<i32>> {
        match self {
            GameWrapper::Gomoku(g) => g.get_board(),
            GameWrapper::Connect4(g) => g.get_board(),
            GameWrapper::Blokus(g) => g.get_board(),
            GameWrapper::Othello(g) => g.get_board(),
        }
    }
}

impl GameWrapper {
    /// Returns the size of the game board
    ///
    /// For most games this is the board height/width, but for Connect4 it's the height.
    ///
    /// # Returns
    /// Board size as number of rows
    pub fn get_board_size(&self) -> usize {
        self.get_board().len()
    }

    /// Returns the number of pieces needed in a row to win
    ///
    /// # Returns
    /// Number of pieces needed for victory (e.g., 5 for Gomoku, 4 for Connect4)
    pub fn get_line_size(&self) -> usize {
        match self {
            GameWrapper::Gomoku(g) => g.get_line_size(),
            GameWrapper::Connect4(g) => g.get_line_size(),
            GameWrapper::Blokus(g) => g.get_line_size(),
            GameWrapper::Othello(g) => g.get_line_size(),
        }
    }

    /// Returns coordinates of the last move made, if any
    ///
    /// # Returns
    /// Optional vector of (row, col) coordinates for the last move
    pub fn get_last_move(&self) -> Option<Vec<(usize, usize)>> {
        match self {
            GameWrapper::Gomoku(g) => g.get_last_move(),
            GameWrapper::Connect4(g) => g.get_last_move(),
            GameWrapper::Blokus(g) => g.get_last_move(),
            GameWrapper::Othello(g) => g.get_last_move(),
        }
    }

    /// Checks if a move is legal in the current game state
    ///
    /// # Arguments
    /// * `mv` - The move to check
    ///
    /// # Returns
    /// True if the move is legal, false otherwise
    pub fn is_legal(&self, mv: &MoveWrapper) -> bool {
        match (self, mv) {
            (GameWrapper::Gomoku(g), MoveWrapper::Gomoku(m)) => g.is_legal(m),
            (GameWrapper::Connect4(g), MoveWrapper::Connect4(m)) => g.is_legal(m),
            (GameWrapper::Blokus(g), MoveWrapper::Blokus(m)) => g.is_legal(m),
            (GameWrapper::Othello(g), MoveWrapper::Othello(m)) => g.is_legal(m),
            _ => false, // Or panic, but false is safer
        }
    }
}
