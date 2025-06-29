//! # Othello (Reversi) Game Implementation
//!
//! This module implements the classic Othello (also known as Reversi) board game.
//! Players take turns placing pieces on an 8x8 board, with the goal of having
//! the most pieces of their color when the board is full or no more moves are possible.
//!
//! ## Rules
//! - Players must place pieces that "sandwich" opponent pieces between the new piece
//!   and an existing piece of the same color
//! - All sandwiched pieces are flipped to the current player's color
//! - If a player has no legal moves, their turn is skipped
//! - Game ends when neither player can make a move
//! - Winner is determined by who has more pieces on the board

use crate::GameState;
use std::str::FromStr;

/// Represents a move in Othello
/// 
/// Contains the row and column coordinates where a player wants to place their piece.
/// Both coordinates are 0-based indices.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct OthelloMove(pub usize, pub usize);

/// Represents the complete state of an Othello game
/// 
/// Contains the board state, current player, and move history.
/// The board uses 1 for black pieces, -1 for white pieces, and 0 for empty spaces.
#[derive(Debug, Clone)]
pub struct OthelloState {
    /// The game board as a 2D vector
    board: Vec<Vec<i32>>,
    /// Current player (1 for black, -1 for white)
    current_player: i32,
    /// Size of the board (NxN)
    board_size: usize,
    /// Last move made, if any
    last_move: Option<(usize, usize)>,
}

impl GameState for OthelloState {
    type Move = OthelloMove;

    fn get_board(&self) -> &Vec<Vec<i32>> {
        &self.board
    }

    fn get_possible_moves(&self) -> Vec<Self::Move> {
        let mut moves = Vec::new();
        for r in 0..self.board_size {
            for c in 0..self.board_size {
                if self.is_valid_move((r, c)) {
                    moves.push(OthelloMove(r, c));
                }
            }
        }
        moves
    }

    fn make_move(&mut self, mv: &Self::Move) {
        let (r, c) = (mv.0, mv.1);
        self.board[r][c] = self.current_player;
        self.last_move = Some((r, c));
        self.flip_pieces(r, c);
        self.current_player = -self.current_player;

        // If the new player has no moves, skip their turn
        if self.get_possible_moves().is_empty() {
            self.current_player = -self.current_player;
        }
    }

    fn is_terminal(&self) -> bool {
        // Game is terminal if no player has any possible moves
        let mut temp_state = self.clone();
        if temp_state.get_possible_moves().is_empty() {
            temp_state.current_player = -temp_state.current_player;
            if temp_state.get_possible_moves().is_empty() {
                return true;
            }
        }
        false
    }

    fn get_last_move(&self) -> Option<Vec<(usize, usize)>> {
        self.last_move.map(|(r, c)| vec![(r, c)])
    }

    fn get_winner(&self) -> Option<i32> {
        if !self.is_terminal() {
            return None;
        }

        let mut p1_score = 0;
        let mut p2_score = 0;
        for r in 0..self.board_size {
            for c in 0..self.board_size {
                if self.board[r][c] == 1 {
                    p1_score += 1;
                } else if self.board[r][c] == -1 {
                    p2_score += 1;
                }
            }
        }

        if p1_score > p2_score {
            Some(1)
        } else if p2_score > p1_score {
            Some(-1)
        } else {
            None // Draw
        }
    }

    fn get_current_player(&self) -> i32 {
        self.current_player
    }
}

impl OthelloState {
    /// Creates a new Othello game with the standard starting position
    /// 
    /// Sets up the board with 4 pieces in the center in the traditional pattern.
    /// Black (player 1) starts first.
    /// 
    /// # Arguments
    /// * `board_size` - Size of the board (NxN), typically 8
    /// 
    /// # Returns
    /// A new OthelloState ready to play
    pub fn new(board_size: usize) -> Self {
        let mut board = vec![vec![0; board_size]; board_size];
        let center = board_size / 2;
        board[center - 1][center - 1] = -1; // White
        board[center - 1][center] = 1;     // Black
        board[center][center - 1] = 1;     // Black
        board[center][center] = -1; // White
        OthelloState {
            board,
            current_player: 1, // Black starts
            board_size,
            last_move: None,
        }
    }

    /// Returns the line size for the game
    /// 
    /// Othello doesn't use a line size concept like other games,
    /// so this returns 1 as a default value.
    pub fn get_line_size(&self) -> usize {
        1 // Othello doesn't have a line size concept, return 1 as default
    }

    /// Checks if a move is legal in the current game state
    /// 
    /// A move is legal if it's on an empty square and would flip at least one opponent piece.
    /// 
    /// # Arguments
    /// * `mv` - The move to check
    /// 
    /// # Returns
    /// True if the move is legal, false otherwise
    pub fn is_legal(&self, mv: &OthelloMove) -> bool {
        self.is_valid_move((mv.0, mv.1))
    }

    /// Internal helper to check if a move at given coordinates is valid
    /// 
    /// Checks all 8 directions from the proposed move to see if any opponent
    /// pieces would be flipped (sandwiched between the new piece and an existing piece).
    /// 
    /// # Arguments
    /// * `mv` - Coordinates (row, col) to check
    /// 
    /// # Returns
    /// True if the move would flip at least one opponent piece
    fn is_valid_move(&self, mv: (usize, usize)) -> bool {
        let (r, c) = mv;
        if self.board[r][c] != 0 {
            return false;
        }

        let opponent = -self.current_player;
        let directions = [
            (-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1),
        ];

        for (dr, dc) in directions.iter() {
            let mut line = Vec::new();
            let mut nr = r as i32 + dr;
            let mut nc = c as i32 + dc;

            while nr >= 0 && nr < self.board_size as i32 && nc >= 0 && nc < self.board_size as i32 {
                if self.board[nr as usize][nc as usize] == opponent {
                    line.push((nr as usize, nc as usize));
                } else if self.board[nr as usize][nc as usize] == self.current_player {
                    if !line.is_empty() {
                        return true;
                    }
                    break;
                } else {
                    break;
                }
                nr += dr;
                nc += dc;
            }
        }
        false
    }

    /// Flips all opponent pieces that are captured by placing a piece at (r, c)
    /// 
    /// This method is called after a move is made to flip all opponent pieces
    /// that are sandwiched between the new piece and existing pieces of the same color.
    /// It searches in all 8 directions and flips pieces in each valid direction.
    /// 
    /// # Arguments
    /// * `r` - Row coordinate of the newly placed piece
    /// * `c` - Column coordinate of the newly placed piece
    fn flip_pieces(&mut self, r: usize, c: usize) {
        let opponent = -self.current_player;
        let directions = [
            (-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1),
        ];

        for (dr, dc) in directions.iter() {
            let mut line = Vec::new();
            let mut nr = r as i32 + dr;
            let mut nc = c as i32 + dc;

            while nr >= 0 && nr < self.board_size as i32 && nc >= 0 && nc < self.board_size as i32 {
                if self.board[nr as usize][nc as usize] == opponent {
                    line.push((nr as usize, nc as usize));
                } else if self.board[nr as usize][nc as usize] == self.current_player {
                    for (fr, fc) in line {
                        self.board[fr][fc] = self.current_player;
                    }
                    break;
                } else {
                    break;
                }
                nr += dr;
                nc += dc;
            }
        }
    }
}

impl FromStr for OthelloMove {
    type Err = String;

    /// Creates an OthelloMove from a string representation
    /// 
    /// Expected format is "row,col" where both are 0-based indices.
    /// 
    /// # Arguments
    /// * `s` - String in format "r,c" (e.g., "3,4")
    /// 
    /// # Returns
    /// Ok(OthelloMove) if parsing succeeds, Err(String) if format is invalid
    /// 
    /// # Examples
    /// ```
    /// use std::str::FromStr;
    /// let move = OthelloMove::from_str("3,4").unwrap();
    /// assert_eq!(move.0, 3);
    /// assert_eq!(move.1, 4);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(',').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return Err("Expected format: r,c".to_string());
        }
        let r = parts[0].parse::<usize>().map_err(|e| e.to_string())?;
        let c = parts[1].parse::<usize>().map_err(|e| e.to_string())?;
        Ok(OthelloMove(r, c))
    }
}
