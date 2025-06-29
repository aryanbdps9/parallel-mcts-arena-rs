//! # Game Wrapper Module
//!
//! This module provides unified interfaces for all games in the engine.
//! The GameWrapper enum allows the MCTS engine to work with any game type
//! through a single interface, while MoveWrapper handles moves for all games.
//!
//! ## Key Components
//! - **GameWrapper**: Enum that wraps all game types into a single interface
//! - **MoveWrapper**: Enum that wraps all move types for unified handling

use crate::games::connect4::{Connect4Move, Connect4State};
use crate::games::gomoku::{GomokuMove, GomokuState};
use crate::games::blokus::{BlokusMove, BlokusState};
use crate::games::othello::{OthelloMove, OthelloState};
use mcts::GameState;
use std::fmt;

/// Wrapper enum for all supported game types
/// 
/// Allows the MCTS engine and UI to work with any game through a unified interface.
/// Each variant contains the specific game state for that game type.
#[derive(Debug, Clone)]
pub enum GameWrapper {
    /// Gomoku (Five in a Row) game state
    Gomoku(GomokuState),
    /// Connect 4 game state
    Connect4(Connect4State),
    /// Blokus game state
    Blokus(BlokusState),
    /// Othello (Reversi) game state
    Othello(OthelloState),
}

/// Wrapper enum for all supported move types
/// 
/// Allows moves from any game to be stored and passed around uniformly.
/// Each variant contains the specific move type for that game.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MoveWrapper {
    /// Gomoku move (row, col)
    Gomoku(GomokuMove),
    /// Connect4 move (column)
    Connect4(Connect4Move),
    /// Blokus move (piece_id, transformation, row, col)
    Blokus(BlokusMove),
    /// Othello move (row, col)
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
