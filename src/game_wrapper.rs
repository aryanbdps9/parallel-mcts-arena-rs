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
use crate::games::hive::{HiveMove, HiveState}; // Hex-based insect strategy game
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
#[allow(dead_code)]
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

    /// Hive game state
    ///
    /// Strategic two-player game with hexagonal insect tiles.
    /// No board - pieces form the playing surface.
    /// - Each piece type has unique movement abilities
    /// - Win by surrounding opponent's Queen Bee
    /// - High branching factor with complex movement rules
    Hive(HiveState),
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

    /// Hive move: placement or movement of insect tiles
    ///
    /// Can be a placement of a new piece from hand or movement of an
    /// existing piece on the board. Movement rules vary by piece type.
    Hive(HiveMove),
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
            MoveWrapper::Hive(m) => write!(f, "H({})", m),
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
            GameWrapper::Hive(g) => write!(f, "{}", g),
        }
    }
}

macro_rules! impl_game_dispatch {
    ($($variant:ident),*) => {
        impl GameState for GameWrapper {
            type Move = MoveWrapper;

            fn get_current_player(&self) -> i32 {
                match self {
                    $(GameWrapper::$variant(g) => g.get_current_player(),)*
                }
            }

            fn get_num_players(&self) -> i32 {
                match self {
                    $(GameWrapper::$variant(g) => g.get_num_players(),)*
                }
            }

            fn get_possible_moves(&self) -> Vec<Self::Move> {
                match self {
                    $(GameWrapper::$variant(g) => g
                        .get_possible_moves()
                        .into_iter()
                        .map(MoveWrapper::$variant)
                        .collect(),)*
                }
            }

            fn make_move(&mut self, mv: &Self::Move) {
                match (self, mv) {
                    $((GameWrapper::$variant(g), MoveWrapper::$variant(m)) => g.make_move(m),)*
                    _ => panic!("Mismatched game and move types"),
                }
            }

            fn is_terminal(&self) -> bool {
                match self {
                    $(GameWrapper::$variant(g) => g.is_terminal(),)*
                }
            }

            fn get_winner(&self) -> Option<i32> {
                match self {
                    $(GameWrapper::$variant(g) => g.get_winner(),)*
                }
            }

            fn get_board(&self) -> &Vec<Vec<i32>> {
                match self {
                    $(GameWrapper::$variant(g) => g.get_board(),)*
                }
            }

            fn get_gpu_simulation_data(&self) -> Option<(Vec<i32>, usize, usize, i32)> {
                match self {
                    $(GameWrapper::$variant(g) => g.get_gpu_simulation_data(),)*
                }
            }
        }

        #[allow(dead_code)]
        impl GameWrapper {
            /// Returns the size of the game board
            pub fn get_board_size(&self) -> usize {
                self.get_board().len()
            }

            /// Returns the number of pieces needed in a row to win
            pub fn get_line_size(&self) -> usize {
                match self {
                    $(GameWrapper::$variant(g) => g.get_line_size(),)*
                }
            }

            /// Returns coordinates of the last move made, if any
            pub fn get_last_move(&self) -> Option<Vec<(usize, usize)>> {
                match self {
                    $(GameWrapper::$variant(g) => g.get_last_move(),)*
                }
            }

            /// Checks if a move is legal in the current game state
            pub fn is_legal(&self, mv: &MoveWrapper) -> bool {
                match (self, mv) {
                    $((GameWrapper::$variant(g), MoveWrapper::$variant(m)) => g.is_legal(m),)*
                    _ => false,
                }
            }
        }
    };
}

impl_game_dispatch!(Gomoku, Connect4, Blokus, Othello, Hive);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::gomoku::{GomokuMove, GomokuState};

    #[test]
    fn test_display() {
        let move_wrapper = MoveWrapper::Gomoku(GomokuMove(1, 2));
        assert_eq!(format!("{}", move_wrapper), "G(1,2)");

        let game_wrapper = GameWrapper::Gomoku(GomokuState::new(15, 5));
        // GomokuState Display might be complex, but we can check it doesn't panic
        let _ = format!("{}", game_wrapper);
    }
}
