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
    /// The game board as a flat vector (row-major)
    board: Vec<i32>,
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
        for r in 0..self.height {
            for c in 0..self.width {
                let cell = self.board[r * self.width + c];
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

    fn get_board(&self) -> Vec<Vec<i32>> {
        let mut rows = Vec::with_capacity(self.height);
        for r in 0..self.height {
            let start = r * self.width;
            let end = start + self.width;
            rows.push(self.board[start..end].to_vec());
        }
        rows
    }

    fn get_last_move(&self) -> Option<Vec<(usize, usize)>> {
        self.last_move.map(|(r, c)| vec![(r, c)])
    }

    fn get_gpu_simulation_data(&self) -> Option<(Vec<i32>, usize, usize, i32)> {
        let mut data = self.board.clone();
        // Normalize board so current player is always 1
        let multiplier = if self.current_player == 1 { 1 } else { -1 };
        for cell in &mut data {
            *cell *= multiplier;
        }
        // Encode line_size in the player field (upper bits)
        // Format: player in bits 0-7, line_size in bits 8-15
        let encoded_params = 1 | ((self.line_size as i32) << 8);
        Some((data, self.width, self.height, encoded_params))
    }

    fn get_possible_moves(&self) -> Vec<Self::Move> {
        (0..self.width)
            .filter(|&c| self.board[c] == 0)
            .map(Connect4Move)
            .collect()
    }

    fn make_move(&mut self, mv: &Self::Move) {
        for r in (0..self.height).rev() {
            let idx = r * self.width + mv.0;
            if self.board[idx] == 0 {
                self.board[idx] = self.current_player;
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
        let idx = r * self.width + c;
        let player = self.board[idx];

        if player == 0 {
            return None;
        }

        if mcts_shared::check_line_win(&self.board, self.width, self.height, player, self.line_size) {
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
            board: vec![0; width * height],
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
        mv.0 < self.width && self.board[mv.0] == 0
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
    /// use mcts::games::connect4::Connect4Move;
    /// let mv = Connect4Move::from_str("3").unwrap();
    /// assert_eq!(mv.0, 3);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let c = s.trim().parse::<usize>().map_err(|e| e.to_string())?;
        Ok(Connect4Move(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game() {
        let game = Connect4State::new(7, 6, 4);
        assert_eq!(game.get_num_players(), 2);
        assert_eq!(game.get_current_player(), 1);
        assert_eq!(game.get_board().len(), 6);
        assert_eq!(game.get_board()[0].len(), 7);
        assert_eq!(game.get_line_size(), 4);
    }

    #[test]
    fn test_legal_moves() {
        let game = Connect4State::new(7, 6, 4);
        let moves = game.get_possible_moves();
        assert_eq!(moves.len(), 7);
        for i in 0..7 {
            assert!(moves.contains(&Connect4Move(i)));
        }
    }

    #[test]
    fn test_make_move() {
        let mut game = Connect4State::new(7, 6, 4);
        game.make_move(&Connect4Move(3));
        assert_eq!(game.get_board()[5][3], 1);
        assert_eq!(game.get_current_player(), -1);
        
        game.make_move(&Connect4Move(3));
        assert_eq!(game.get_board()[4][3], -1);
        assert_eq!(game.get_current_player(), 1);
    }

    #[test]
    fn test_win_condition_horizontal() {
        let mut game = Connect4State::new(7, 6, 4);
        // Player 1: 0, 1, 2, 3
        // Player 2: 0, 1, 2
        game.make_move(&Connect4Move(0)); // P1
        game.make_move(&Connect4Move(0)); // P2
        game.make_move(&Connect4Move(1)); // P1
        game.make_move(&Connect4Move(1)); // P2
        game.make_move(&Connect4Move(2)); // P1
        game.make_move(&Connect4Move(2)); // P2
        game.make_move(&Connect4Move(3)); // P1 wins

        assert_eq!(game.get_winner(), Some(1));
        assert!(game.is_terminal());
    }

    #[test]
    fn test_win_condition_vertical() {
        let mut game = Connect4State::new(7, 6, 4);
        // Player 1: 0, 0, 0, 0
        // Player 2: 1, 1, 1
        game.make_move(&Connect4Move(0)); // P1
        game.make_move(&Connect4Move(1)); // P2
        game.make_move(&Connect4Move(0)); // P1
        game.make_move(&Connect4Move(1)); // P2
        game.make_move(&Connect4Move(0)); // P1
        game.make_move(&Connect4Move(1)); // P2
        game.make_move(&Connect4Move(0)); // P1 wins

        assert_eq!(game.get_winner(), Some(1));
        assert!(game.is_terminal());
    }

    #[test]
    fn test_win_condition_diagonal() {
        let mut game = Connect4State::new(7, 6, 4);
        // P1 wins with diagonal /
        // . . . .
        // . . . 1
        // . . 1 2
        // . 1 2 2
        // 1 2 1 1
        
        game.make_move(&Connect4Move(0)); // P1 (0,0)
        game.make_move(&Connect4Move(1)); // P2 (0,1)
        game.make_move(&Connect4Move(1)); // P1 (1,1)
        game.make_move(&Connect4Move(2)); // P2 (0,2)
        game.make_move(&Connect4Move(2)); // P1 (1,2) - mistake in comment logic, let's just play it out
        game.make_move(&Connect4Move(3)); // P2 (0,3)
        game.make_move(&Connect4Move(2)); // P1 (2,2)
        game.make_move(&Connect4Move(3)); // P2 (1,3)
        game.make_move(&Connect4Move(3)); // P1 (2,3)
        game.make_move(&Connect4Move(0)); // P2 (1,0) - filler
        game.make_move(&Connect4Move(3)); // P1 (3,3) wins

        assert_eq!(game.get_winner(), Some(1));
        assert!(game.is_terminal());
    }
}
