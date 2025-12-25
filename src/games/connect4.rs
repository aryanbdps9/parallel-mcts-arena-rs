//! # Connect 4 Game Implementation
//!
//! This module implements the classic Connect 4 board game.
//! Players take turns dropping pieces into columns, trying to get 4 pieces
//! in a row (horizontally, vertically, or diagonally).
//!
//! ## Rules
//! - Players alternate dropping pieces into columns
//! - Pieces fall to the lowest available spot in the column due to gravity
//! - First player to get 4 pieces in a row wins
//! - Game is a draw if the board fills up with no winner

use crate::GameState;
use std::fmt;
use std::str::FromStr;

/// Represents a move in Connect 4
///
/// Contains the column number where a player wants to drop their piece.
/// Column numbers are 0-based indices.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Connect4Move(pub usize);

/// Represents the complete state of a Connect 4 game
///
/// Contains the board state, current player, dimensions, and move history.
/// The board uses 1 for player 1 pieces, -1 for player 2 pieces, and 0 for empty spaces.
#[derive(Debug, Clone)]
pub struct Connect4State {
    /// The game board as a 2D vector (rows x columns)
    board: Vec<Vec<i32>>,
    /// Current player (1 or -1)
    current_player: i32,
    /// Board width (number of columns)
    width: usize,
    /// Board height (number of rows)
    height: usize,
    /// Number of pieces needed in a row to win
    line_size: usize,
    /// Last move made, if any (row, column)
    last_move: Option<(usize, usize)>,
}

impl fmt::Display for Connect4State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.board {
            for &cell in row {
                let symbol = match cell {
                    1 => "X",
                    -1 => "O",
                    _ => ".",
                };
                write!(f, "{} ", symbol)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl GameState for Connect4State {
    type Move = Connect4Move; // Column to drop a piece

    fn get_num_players(&self) -> i32 {
        2
    }

    fn get_board(&self) -> &Vec<Vec<i32>> {
        &self.board
    }

    fn get_last_move(&self) -> Option<Vec<(usize, usize)>> {
        self.last_move.map(|(r, c)| vec![(r, c)])
    }

    fn get_gpu_simulation_data(&self) -> Option<(Vec<i32>, usize, usize, i32)> {
        let mut data = Vec::with_capacity(self.height * self.width);
        // Normalize board so current player is always 1
        // This allows batching states with different current players
        let multiplier = if self.current_player == 1 { 1 } else { -1 };
        for row in &self.board {
            for &cell in row {
                data.push(cell * multiplier);
            }
        }
        // Encode line_size in the player field (upper bits)
        // Format: player in bits 0-7, line_size in bits 8-15
        let encoded_params = 1 | ((self.line_size as i32) << 8);
        Some((data, self.width, self.height, encoded_params))
    }

    fn get_possible_moves(&self) -> Vec<Self::Move> {
        (0..self.width)
            .filter(|&c| self.board[0][c] == 0)
            .map(Connect4Move)
            .collect()
    }

    fn make_move(&mut self, mv: &Self::Move) {
        for r in (0..self.height).rev() {
            if self.board[r][mv.0] == 0 {
                self.board[r][mv.0] = self.current_player;
                self.last_move = Some((r, mv.0));
                self.current_player = -self.current_player;
                return;
            }
        }
    }

    fn is_terminal(&self) -> bool {
        self.get_winner().is_some() || self.get_possible_moves().is_empty()
    }

    fn get_winner(&self) -> Option<i32> {
        let last_move = self.last_move?;
        let (r, c) = last_move;
        let player = self.board[r][c];

        if player == 0 {
            return None;
        }

        // Check horizontal
        let mut count = 1;
        for i in 1..self.line_size {
            if c >= i && self.board[r][c - i] == player {
                count += 1;
            } else {
                break;
            }
        }
        for i in 1..self.line_size {
            if c + i < self.width && self.board[r][c + i] == player {
                count += 1;
            } else {
                break;
            }
        }
        if count >= self.line_size {
            return Some(player);
        }

        // Check vertical
        let mut count = 1;
        for i in 1..self.line_size {
            if r + i < self.height && self.board[r + i][c] == player {
                count += 1;
            } else {
                break;
            }
        }
        if count >= self.line_size {
            return Some(player);
        }

        // Check diagonal (top-left to bottom-right)
        let mut count = 1;
        for i in 1..self.line_size {
            if r >= i && c >= i && self.board[r - i][c - i] == player {
                count += 1;
            } else {
                break;
            }
        }
        for i in 1..self.line_size {
            if r + i < self.height && c + i < self.width && self.board[r + i][c + i] == player {
                count += 1;
            } else {
                break;
            }
        }
        if count >= self.line_size {
            return Some(player);
        }

        // Check diagonal (top-right to bottom-left)
        let mut count = 1;
        for i in 1..self.line_size {
            if r >= i && c + i < self.width && self.board[r - i][c + i] == player {
                count += 1;
            } else {
                break;
            }
        }
        for i in 1..self.line_size {
            if r + i < self.height && c >= i && self.board[r + i][c - i] == player {
                count += 1;
            } else {
                break;
            }
        }
        if count >= self.line_size {
            return Some(player);
        }

        None
    }

    fn get_current_player(&self) -> i32 {
        self.current_player
    }
}

impl Connect4State {
    /// Creates a new Connect 4 game with the specified configuration
    pub fn new(width: usize, height: usize, line_size: usize) -> Self {
        Self {
            board: vec![vec![0; width]; height],
            current_player: 1,
            width,
            height,
            line_size,
            last_move: None,
        }
    }

    /// Gets the number of pieces needed in a row to win
    ///
    /// # Returns
    /// The line size (typically 4 for standard Connect 4)
    pub fn get_line_size(&self) -> usize {
        self.line_size
    }

    /// Checks if a move is legal in the current game state
    ///
    /// A move is legal if the column is within bounds and the top row
    /// of that column is empty (pieces can be dropped).
    ///
    /// # Arguments
    /// * `mv` - The move to check
    ///
    /// # Returns
    /// true if the move is legal, false otherwise
    pub fn is_legal(&self, mv: &Connect4Move) -> bool {
        mv.0 < self.width && self.board[0][mv.0] == 0
    }
}

impl FromStr for Connect4Move {
    type Err = String;

    /// Creates a Connect4Move from a string representation
    ///
    /// Expected format is just the column number as a string.
    ///
    /// # Arguments
    /// * `s` - String containing column number (e.g., "3")
    ///
    /// # Returns
    /// Ok(Connect4Move) if parsing succeeds, Err(String) if format is invalid
    ///
    /// # Examples
    /// ```
    /// use std::str::FromStr;
    /// let move = Connect4Move::from_str("3").unwrap();
    /// assert_eq!(move.0, 3);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let c = s.trim().parse::<usize>().map_err(|e| e.to_string())?;
        Ok(Connect4Move(c))
    }
}
