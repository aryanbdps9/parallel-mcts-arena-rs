//! # Gomoku (Five in a Row) Game Implementation
//!
//! This module implements the classic Gomoku board game, also known as Five in a Row.
//! Players alternate placing pieces on a grid, trying to get five (or a configurable number) 
//! pieces in a row horizontally, vertically, or diagonally.
//!
//! ## Rules
//! - Players alternate placing pieces on empty squares
//! - First player to get N pieces in a row wins (typically 5)
//! - The line can be horizontal, vertical, or diagonal
//! - Game is a draw if the board fills up with no winner

use crate::GameState;
use std::str::FromStr;

/// Represents a move in Gomoku
/// 
/// Contains the row and column coordinates where a player wants to place their piece.
/// Both coordinates are 0-based indices.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct GomokuMove(pub usize, pub usize);

/// Represents the complete state of a Gomoku game
/// 
/// Contains the board state, current player, game configuration, and move history.
/// The board uses 1 for player 1 pieces, -1 for player 2 pieces, and 0 for empty spaces.
#[derive(Debug, Clone)]
pub struct GomokuState {
    /// The game board as a 2D vector
    pub board: Vec<Vec<i32>>,
    /// Current player (1 or -1)
    pub current_player: i32,
    /// Size of the board (NxN)
    board_size: usize,
    /// Number of pieces needed in a row to win
    line_size: usize,
    /// Last move made, if any
    last_move: Option<(usize, usize)>,
}

impl GomokuState {
    /// Creates a new Gomoku game with the specified configuration
    /// 
    /// # Arguments
    /// * `board_size` - Size of the board (NxN)
    /// * `line_size` - Number of pieces needed in a row to win
    /// 
    /// # Returns
    /// A new GomokuState ready to play
    pub fn new(board_size: usize, line_size: usize) -> Self {
        GomokuState {
            board: vec![vec![0; board_size]; board_size],
            current_player: 1,
            board_size,
            line_size,
            last_move: None,
        }
    }

    /// Returns the board size (NxN)
    pub fn get_board_size(&self) -> usize {
        self.board_size
    }

    /// Returns the number of pieces needed in a row to win
    pub fn get_line_size(&self) -> usize {
        self.line_size
    }

    /// Checks if a move is legal in the current game state
    /// 
    /// A move is legal if it's within the board bounds and the target square is empty.
    /// 
    /// # Arguments
    /// * `mv` - The move to check
    /// 
    /// # Returns
    /// True if the move is legal, false otherwise
    pub fn is_legal(&self, mv: &GomokuMove) -> bool {
        mv.0 < self.board_size && mv.1 < self.board_size && self.board[mv.0][mv.1] == 0
    }
}

impl GameState for GomokuState {
    type Move = GomokuMove;

    fn get_board(&self) -> &Vec<Vec<i32>> {
        &self.board
    }

    fn get_last_move(&self) -> Option<Vec<(usize, usize)>> {
        self.last_move.map(|(r, c)| vec![(r, c)])
    }

    fn get_possible_moves(&self) -> Vec<Self::Move> {
        (0..self.board_size)
            .flat_map(|r| (0..self.board_size).map(move |c| (r, c)))
            .filter(|&(r, c)| self.board[r][c] == 0)
            .map(|(r, c)| GomokuMove(r, c))
            .collect()
    }

    fn make_move(&mut self, mv: &Self::Move) {
        self.board[mv.0][mv.1] = self.current_player;
        self.last_move = Some((mv.0, mv.1));
        self.current_player = -self.current_player;
    }

    fn is_terminal(&self) -> bool {
        self.get_winner().is_some() || self.get_possible_moves().is_empty()
    }

    fn get_winner(&self) -> Option<i32> {
        // If no move has been made yet, there's no winner
        let last_move = self.last_move?;
        let (r, c) = last_move;
        let player = self.board[r][c];
        
        // If the position is empty, there's no winner (shouldn't happen in normal play)
        if player == 0 {
            return None;
        }
        
        // Check horizontal (left-right through the last move)
        let mut count = 1;
        // Check left
        for i in 1..self.line_size {
            if c >= i && self.board[r][c - i] == player {
                count += 1;
            } else {
                break;
            }
        }
        // Check right
        for i in 1..self.line_size {
            if c + i < self.board_size && self.board[r][c + i] == player {
                count += 1;
            } else {
                break;
            }
        }
        if count >= self.line_size {
            return Some(player);
        }
        
        // Check vertical (up-down through the last move)
        count = 1;
        // Check up
        for i in 1..self.line_size {
            if r >= i && self.board[r - i][c] == player {
                count += 1;
            } else {
                break;
            }
        }
        // Check down
        for i in 1..self.line_size {
            if r + i < self.board_size && self.board[r + i][c] == player {
                count += 1;
            } else {
                break;
            }
        }
        if count >= self.line_size {
            return Some(player);
        }
        
        // Check diagonal (top-left to bottom-right through the last move)
        count = 1;
        // Check top-left
        for i in 1..self.line_size {
            if r >= i && c >= i && self.board[r - i][c - i] == player {
                count += 1;
            } else {
                break;
            }
        }
        // Check bottom-right
        for i in 1..self.line_size {
            if r + i < self.board_size && c + i < self.board_size && self.board[r + i][c + i] == player {
                count += 1;
            } else {
                break;
            }
        }
        if count >= self.line_size {
            return Some(player);
        }
        
        // Check diagonal (top-right to bottom-left through the last move)
        count = 1;
        // Check top-right
        for i in 1..self.line_size {
            if r >= i && c + i < self.board_size && self.board[r - i][c + i] == player {
                count += 1;
            } else {
                break;
            }
        }
        // Check bottom-left
        for i in 1..self.line_size {
            if r + i < self.board_size && c >= i && self.board[r + i][c - i] == player {
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

impl FromStr for GomokuMove {
    type Err = String;

    /// Creates a GomokuMove from a string representation
    /// 
    /// Expected format is "row,col" where both are 0-based indices.
    /// 
    /// # Arguments
    /// * `s` - String in format "r,c" (e.g., "3,4")
    /// 
    /// # Returns
    /// Ok(GomokuMove) if parsing succeeds, Err(String) if format is invalid
    /// 
    /// # Examples
    /// ```
    /// use std::str::FromStr;
    /// let move_obj = GomokuMove::from_str("3,4").unwrap();
    /// assert_eq!(move_obj.0, 3);
    /// assert_eq!(move_obj.1, 4);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(',').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return Err("Expected format: r,c".to_string());
        }
        let r = parts[0].parse::<usize>().map_err(|e| e.to_string())?;
        let c = parts[1].parse::<usize>().map_err(|e| e.to_string())?;
        Ok(GomokuMove(r, c))
    }
}