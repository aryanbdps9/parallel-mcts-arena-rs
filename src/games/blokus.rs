//! # Blokus Game Implementation
//!
//! This module implements the Blokus board game, a strategic tile-laying game
//! where players place polyomino pieces on a 20x20 grid following specific placement rules.
//!
//! ## Game Overview
//! Blokus is a territorial strategy game for 2-4 players where each player tries to place
//! all 21 of their polyomino pieces on the board while blocking opponents from doing the same.
//! The player who places the most squares (fewest remaining pieces) wins.
//!
//! ## Rules and Mechanics
//! - **Initial Placement**: Each player's first piece must cover one of the four corner squares
//! - **Corner-to-Corner**: Subsequent pieces must touch corner-to-corner with existing pieces of the same color
//! - **No Edge Contact**: Pieces cannot touch edge-to-edge with the same player's pieces
//! - **Blocking**: Players can block opponents by placing pieces adjacent to their pieces
//! - **Passing**: Players must pass if they cannot place any pieces
//! - **Game End**: Game ends when all players pass consecutively
//!
//! ## Scoring
//! - Score is based on remaining piece squares (lower is better)
//! - Player with fewest remaining squares wins
//! - Bonus points for placing all pieces or ending with the single square piece
//!
//! ## Implementation Details
//! - 20x20 game board with integer player IDs (1-4)
//! - 21 unique polyomino pieces per player (monomino through pentominoes)
//! - All piece transformations (rotations + reflections) pre-computed for efficiency
//! - Move validation includes adjacency rules and corner-touching requirements

use mcts::GameState;
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

/// Special constant representing a pass move in Blokus
/// 
/// When a player cannot place any pieces according to Blokus rules,
/// they must pass their turn. This constant uses usize::MAX as the
/// piece ID to distinguish it from valid piece placements (0-20).
const PASS_MOVE: BlokusMove = BlokusMove(usize::MAX, 0, 0, 0);

/// Represents a Blokus piece with all its possible transformations
/// 
/// Each piece represents one of the 21 unique polyomino shapes used in Blokus,
/// from the single square (monomino) up to the complex pentomino shapes.
/// All possible rotations and reflections are pre-computed and normalized
/// for efficient move generation and validation.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Piece {
    /// Unique identifier for this piece type (0-20)
    pub id: usize,
    /// All possible transformations (rotations + reflections) of this piece
    /// Each transformation is a vector of (row, col) offsets from origin (0,0)
    pub transformations: Vec<Vec<(i32, i32)>>,
}

impl Piece {
    /// Creates a new piece with all possible transformations
    /// 
    /// Generates all unique rotations and reflections of the given shape through
    /// a systematic process: 4 rotations × 2 reflections = up to 8 transformations.
    /// Duplicate transformations are automatically eliminated, and all shapes are
    /// normalized to start from coordinate (0,0) for consistent positioning.
    /// 
    /// # Arguments
    /// * `id` - Unique identifier for this piece type (0-20)
    /// * `shape` - Base shape as a list of (row, col) coordinates relative to origin
    /// 
    /// # Returns
    /// A new Piece with all unique transformations calculated and sorted
    /// 
    /// # Examples
    /// Creating the simple 2-square domino piece:
    /// ```
    /// let domino = Piece::new(1, &[(0, 0), (0, 1)]);
    /// // Results in 2 transformations: horizontal and vertical orientations
    /// assert_eq!(domino.transformations.len(), 2);
    /// ```
    pub fn new(id: usize, shape: &[(i32, i32)]) -> Self {
        let mut unique_transformations = HashSet::new();
        let mut current_shape: Vec<(i32, i32)> = shape.to_vec();

        for _ in 0..2 { // Flip
            for _ in 0..4 { // Rotate
                let min_r = current_shape.iter().map(|p| p.0).min().unwrap_or(0);
                let min_c = current_shape.iter().map(|p| p.1).min().unwrap_or(0);
                let mut normalized_shape: Vec<(i32, i32)> = current_shape.iter().map(|p| (p.0 - min_r, p.1 - min_c)).collect();
                normalized_shape.sort();
                unique_transformations.insert(normalized_shape);
                current_shape = current_shape.iter().map(|(r, c)| (-c, *r)).collect(); // rotate
            }
            current_shape = current_shape.iter().map(|(r, c)| (*r, -c)).collect(); // flip
        }

        Piece {
            id,
            transformations: {
                let mut sorted_transformations: Vec<Vec<(i32, i32)>> = unique_transformations.into_iter().collect();
                sorted_transformations.sort();
                sorted_transformations
            },
        }
    }
}

/// Returns all 21 standard Blokus pieces with their transformations
/// 
/// Creates the complete set of polyomino pieces used in standard Blokus gameplay.
/// The pieces progress from simple (monomino, domino) to complex (pentomino) shapes:
/// 
/// - **Monomino (1)**: Single square
/// - **Domino (1)**: Two connected squares  
/// - **Triomino (2)**: Three connected squares in L and line shapes
/// - **Tetromino (5)**: Four connected squares in various configurations
/// - **Pentomino (12)**: Five connected squares in all possible configurations
/// 
/// Each piece is created with all its possible transformations pre-computed,
/// making move generation efficient during gameplay.
/// 
/// # Returns
/// Vector containing all 21 Blokus pieces, each with complete transformation sets
/// 
/// # Performance Note
/// This function performs significant computation generating transformations.
/// Consider caching the result rather than calling repeatedly.
pub fn get_blokus_pieces() -> Vec<Piece> {
    vec![
        Piece::new(0, &[(0, 0)]), // 1
        Piece::new(1, &[(0, 0), (0, 1)]), // 2
        Piece::new(2, &[(0, 0), (0, 1), (1, 1)]), // 3
        Piece::new(3, &[(0, 0), (0, 1), (0, 2)]), // 3 line
        Piece::new(4, &[(0, 0), (0, 1), (1, 0), (1, 1)]), // 4 square
        Piece::new(5, &[(0, 0), (0, 1), (0, 2), (0, 3)]), // 4 line
        Piece::new(6, &[(0, 0), (0, 1), (1, 1), (1, 2)]), // 4 S
        Piece::new(7, &[(0, 1), (1, 0), (1, 1), (1, 2)]), // 4 T
        Piece::new(8, &[(0, 0), (0, 1), (0, 2), (1, 2)]), // 4 L
        Piece::new(9, &[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]), // 5 line
        Piece::new(10, &[(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]), // 5 L
        Piece::new(11, &[(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)]), // 5 P
        Piece::new(12, &[(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)]), // 5 U
        Piece::new(13, &[(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)]), // 5 T
        Piece::new(14, &[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]), // 5 V
        Piece::new(15, &[(0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]), // 5 F
        Piece::new(16, &[(0, 1), (1, 1), (1, 2), (2, 0), (2, 1)]), // 5 N
        Piece::new(17, &[(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]), // 5 X
        Piece::new(18, &[(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]), // 5 S
        Piece::new(19, &[(0, 1), (1, 0), (1, 1), (2, 1), (3, 1)]), // 5 W
        Piece::new(20, &[(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)]), // 5 Y
    ]
}

/// Returns summary information about all Blokus pieces
/// 
/// Provides a lightweight overview of piece complexity by returning
/// the number of unique transformations each piece has. This is useful
/// for UI displays, piece selection interfaces, and complexity analysis.
/// 
/// # Returns
/// Vector of (piece_id, transformation_count) tuples for all 21 pieces
/// 
/// # Usage
/// Helpful for creating piece selection menus or analyzing game complexity:
/// ```
/// let piece_info = get_piece_info();
/// for (id, transform_count) in piece_info {
///     println!("Piece {}: {} orientations", id, transform_count);
/// }
/// ```
pub fn get_piece_info() -> Vec<(usize, usize)> {
    get_blokus_pieces()
        .iter()
        .map(|p| (p.id, p.transformations.len()))
        .collect()
}

/// Represents a move in Blokus
/// 
/// Contains all information needed to place a piece on the board:
/// piece selection, orientation, and board position. The special
/// constant PASS_MOVE represents a player passing their turn.
/// 
/// # Format
/// BlokusMove(piece_id, transformation_index, row, column)
/// - `piece_id`: Which piece to place (0-20)
/// - `transformation_index`: Which rotation/reflection to use
/// - `row`: Board row position (0-19)  
/// - `column`: Board column position (0-19)
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct BlokusMove(pub usize, pub usize, pub usize, pub usize);

impl fmt::Display for BlokusMove {
    /// Formats the move for display in human-readable form
    /// 
    /// Creates a compact string representation suitable for move history,
    /// debugging, and user interfaces.
    /// 
    /// # Format
    /// - Pass moves: "PASS"
    /// - Piece placements: "P{piece}T{transformation}@({row},{col})"
    /// 
    /// # Examples
    /// - "PASS" - Player passes their turn
    /// - "P5T2@(10,7)" - Place piece 5, transformation 2, at row 10, column 7
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == PASS_MOVE {
            write!(f, "PASS")
        } else {
            write!(f, "P{}T{}@({},{})", self.0, self.1, self.2, self.3)
        }
    }
}

/// Represents the complete state of a Blokus game
/// 
/// Encapsulates all information needed to represent a Blokus game at any point:
/// board state, player inventories, turn order, and game progress tracking.
/// Implements the GameState trait for compatibility with MCTS algorithms.
/// 
/// # Game State Components
/// - **Board**: 20x20 grid with player pieces (1-4) and empty spaces (0)
/// - **Player Pieces**: Each player's remaining piece inventory
/// - **Turn Management**: Current player and first-move tracking
/// - **Move History**: Coordinates of the most recent move for highlighting
/// - **Game Progress**: Pass counting for termination detection
#[derive(Debug, Clone)]
pub struct BlokusState {
    /// The game board as a 2D vector (20x20), 0=empty, 1-4=player pieces
    board: Vec<Vec<i32>>,
    /// Current player's turn (1, 2, 3, or 4)
    current_player: i32,
    /// Available pieces for each player, indexed by player_id - 1
    player_pieces: Vec<Vec<Piece>>,
    /// Whether each player is making their first move (corner requirement)
    is_first_move: [bool; 4],
    /// Coordinates of the last move made, used for board highlighting
    last_move_coords: Option<Vec<(usize, usize)>>,
    /// Number of consecutive passes by all players (game ends at 4)
    consecutive_passes: u8,
}

impl fmt::Display for BlokusState {
    /// Formats the game board for text-based display
    /// 
    /// Creates a simple ASCII representation of the 20x20 board suitable
    /// for debugging, logging, and basic console output.
    /// 
    /// # Format
    /// - Empty squares: "."
    /// - Player pieces: "1", "2", "3", "4" (corresponding to player IDs)
    /// - Each row on a separate line with spaces between squares
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.board {
            for &cell in row {
                if cell == 0 {
                    write!(f, ". ")?;
                } else {
                    write!(f, "{} ", cell)?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl GameState for BlokusState {
    type Move = BlokusMove;

    fn get_num_players(&self) -> i32 {
        4
    }

    fn get_board(&self) -> &Vec<Vec<i32>> {
        &self.board
    }

    fn get_possible_moves(&self) -> Vec<Self::Move> {
        let player_idx = (self.current_player - 1) as usize;
        let available_pieces = &self.player_pieces[player_idx];
        
        // Early check: if player has no pieces left, they must pass
        if available_pieces.is_empty() {
            return vec![PASS_MOVE];
        }

        let mut moves = Vec::new();

        for piece in available_pieces {
            for (trans_idx, shape) in piece.transformations.iter().enumerate() {
                for r in 0..self.get_board_size() {
                    for c in 0..self.get_board_size() {
                        if self.is_valid_move(player_idx, shape, r, c) {
                            moves.push(BlokusMove(piece.id, trans_idx, r, c));
                        }
                    }
                }
            }
        }

        if moves.is_empty() {
            vec![PASS_MOVE]
        } else {
            moves
        }
    }

    fn make_move(&mut self, mv: &Self::Move) {
        if *mv == PASS_MOVE {
            self.consecutive_passes += 1;
            self.last_move_coords = None;
        } else {
            let player_idx = (self.current_player - 1) as usize;
            let piece_id = mv.0;
            let trans_idx = mv.1;
            let r = mv.2;
            let c = mv.3;

            let piece_index = self.player_pieces[player_idx]
                .iter()
                .position(|p| p.id == piece_id)
                .expect("Piece not found");
            let shape = &self.player_pieces[player_idx][piece_index].transformations[trans_idx];

            let mut move_coords = Vec::new();
            for &(dr, dc) in shape {
                let board_r = (r as i32 + dr) as usize;
                let board_c = (c as i32 + dc) as usize;
                self.board[board_r][board_c] = self.current_player;
                move_coords.push((board_r, board_c));
            }

            self.player_pieces[player_idx].remove(piece_index);
            self.is_first_move[player_idx] = false;
            self.last_move_coords = Some(move_coords);
            self.consecutive_passes = 0;
        }

        // Advance to the next player
        self.current_player = (self.current_player % 4) + 1;
    }

    fn is_terminal(&self) -> bool {
        // Primary termination condition: all 4 players pass consecutively
        if self.consecutive_passes >= 4 {
            return true;
        }
        
        // Early termination: if all players have placed all pieces
        if self.player_pieces.iter().all(|pieces| pieces.is_empty()) {
            return true;
        }
        
        // Optimization: if 3 players have passed and current player has no valid moves,
        // we can terminate early rather than wait for the 4th pass
        if self.consecutive_passes >= 3 {
            let player_idx = (self.current_player - 1) as usize;
            let available_pieces = &self.player_pieces[player_idx];
            
            if available_pieces.is_empty() {
                return true; // Current player has no pieces, will pass, making it 4 consecutive
            }
            
            // Quick check if current player has any valid moves
            // This prevents needless computation when game is effectively over
            let has_valid_moves = available_pieces.iter().any(|piece| {
                piece.transformations.iter().any(|shape| {
                    (0..20).any(|r| {
                        (0..20).any(|c| {
                            self.is_valid_move(player_idx, shape, r, c)
                        })
                    })
                })
            });
            
            if !has_valid_moves {
                return true; // Current player will be forced to pass, making it 4 consecutive
            }
        }
        
        false
    }

    fn get_winner(&self) -> Option<i32> {
        if !self.is_terminal() {
            return None;
        }

        // Calculate scores: count remaining squares for each player (lower is better)
        let mut scores = [0; 4];
        for i in 0..4 {
            scores[i] = self.player_pieces[i].iter().map(|p| p.transformations[0].len()).sum::<usize>() as i32;
        }

        let min_score = *scores.iter().min().unwrap();
        let winners: Vec<_> = scores.iter().enumerate().filter(|(_, &s)| s == min_score).collect();

        if winners.len() == 1 {
            Some((winners[0].0 + 1) as i32)
        } else {
            // In case of a tie, MCTS framework doesn't support multiple winners.
            // Returning None for a draw, or the first winner.
            // For now, let's return the first winner's ID.
            Some((winners[0].0 + 1) as i32)
        }
    }

    fn get_last_move(&self) -> Option<Vec<(usize, usize)>> {
        self.last_move_coords.clone()
    }

    fn get_current_player(&self) -> i32 {
        self.current_player
    }
}

impl BlokusState {
    /// Creates a new Blokus game with default starting state
    /// 
    /// Initializes a fresh game with an empty 20x20 board and all players
    /// having their complete set of 21 pieces. Player 1 starts first,
    /// and all players are marked as needing to make their first move
    /// (which requires covering a corner square).
    /// 
    /// # Returns
    /// A new BlokusState ready for gameplay
    /// 
    /// # Corner Assignment
    /// Players must place their first piece covering these corners:
    /// - Player 1: Top-left (0,0)
    /// - Player 2: Top-right (0,19)  
    /// - Player 3: Bottom-right (19,19)
    /// - Player 4: Bottom-left (19,0)
    pub fn new() -> Self {
        let board = vec![vec![0; 20]; 20];
        let player_pieces = vec![
            get_blokus_pieces(),
            get_blokus_pieces(),
            get_blokus_pieces(),
            get_blokus_pieces(),
        ];
        BlokusState {
            board,
            current_player: 1,
            player_pieces,
            is_first_move: [true; 4],
            last_move_coords: None,
            consecutive_passes: 0,
        }
    }

    /// Returns the board size (always 20 for standard Blokus)
    /// 
    /// # Returns
    /// Board dimension (20) - Blokus is played on a fixed 20x20 grid
    pub fn get_board_size(&self) -> usize {
        20 // Blokus board is 20x20
    }

    /// Returns the line size (not applicable for Blokus)
    /// 
    /// Blokus doesn't use a "line size" concept like Connect 4 or Gomoku.
    /// This method exists for GameState trait compatibility.
    /// 
    /// # Returns
    /// Always returns 1 as a default value
    pub fn get_line_size(&self) -> usize {
        1 // Blokus doesn't have a line size concept, return 1 as default
    }

    /// Checks if a move is legal for the current player in the current game state
    /// 
    /// Validates both pass moves (always legal) and piece placement moves.
    /// For piece placements, checks that the piece exists in the player's
    /// inventory, the transformation index is valid, and the placement
    /// follows Blokus rules (corner touching, no edge adjacency, etc.).
    /// 
    /// # Arguments
    /// * `mv` - The move to validate (either piece placement or pass)
    /// 
    /// # Returns
    /// `true` if the move is legal according to Blokus rules, `false` otherwise
    /// 
    /// # Rule Validation
    /// - Pass moves are always legal
    /// - Piece must exist in current player's inventory
    /// - Transformation index must be valid for the piece
    /// - First moves must cover the player's designated corner
    /// - Subsequent moves must touch corner-to-corner with existing pieces
    /// - No edge-to-edge contact with same player's pieces allowed
    pub fn is_legal(&self, mv: &BlokusMove) -> bool {
        // Handle pass move - always legal
        if *mv == PASS_MOVE {
            return true;
        }
        
        let player_idx = (self.current_player - 1) as usize;
        if player_idx >= self.player_pieces.len() {
            return false;
        }
        let Some(piece) = self.player_pieces[player_idx].iter().find(|p| p.id == mv.0) else {
            return false;
        };
        if mv.1 >= piece.transformations.len() {
            return false;
        }
        let shape = &piece.transformations[mv.1];
        self.is_valid_move(player_idx, shape, mv.2, mv.3)
    }

    /// Checks if a piece placement is valid at the specified position
    /// 
    /// Internal helper function that performs the core move validation logic
    /// according to Blokus rules. Handles both first-move corner requirements
    /// and subsequent corner-touching/edge-avoidance rules.
    /// 
    /// # Arguments
    /// * `player_idx` - Zero-based player index (0-3)
    /// * `shape` - The piece shape to place (transformation coordinates)
    /// * `r` - Board row position for piece origin
    /// * `c` - Board column position for piece origin
    /// 
    /// # Returns
    /// `true` if the placement is valid according to Blokus rules
    /// 
    /// # Validation Steps
    /// 1. Check all piece squares fit on board and are empty
    /// 2. For first move: verify piece covers the player's corner
    /// 3. For subsequent moves: verify corner-touching and no edge contact
    fn is_valid_move(&self, player_idx: usize, shape: &[(i32, i32)], r: usize, c: usize) -> bool {
        let player_id = (player_idx + 1) as i32;
        let mut corner_touch = false;

        // Check if all pieces of the shape fit on the board and are on empty spots
        for (dr, dc) in shape {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;

            // Out of bounds or occupied cell
            if nr < 0 || nr >= 20 || nc < 0 || nc >= 20 || self.board[nr as usize][nc as usize] != 0 {
                return false;
            }
        }

        // For first move, must cover the corner
        if self.is_first_move[player_idx] {
            let target_corners = [(0, 0), (0, 19), (19, 19), (19, 0)];
            let target = target_corners[player_idx];
            for (dr, dc) in shape {
                if (r as i32 + dr, c as i32 + dc) == target {
                    return true; // First move only needs to cover corner
                }
            }
            return false;
        }

        // For subsequent moves, check adjacency rules
        for (dr, dc) in shape {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;

            // Check that no edge is adjacent to same player
            let neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)];
            for (nnr, nnc) in &neighbors {
                let ar = nr + nnr;
                let ac = nc + nnc;
                if ar >= 0 && ar < 20 && ac >= 0 && ac < 20 && self.board[ar as usize][ac as usize] == player_id {
                    return false;
                }
            }

            // Check for corner touch with same player
            let corners = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
            for (cnr, cnc) in &corners {
                let ar = nr + cnr;
                let ac = nc + cnc;
                if ar >= 0 && ar < 20 && ac >= 0 && ac < 20 && self.board[ar as usize][ac as usize] == player_id {
                    corner_touch = true;
                }
            }
        }

        corner_touch
    }

    /// Returns the piece IDs available to a specific player
    /// 
    /// Provides a list of piece IDs that the player can still place.
    /// Useful for UI displays and move generation optimization.
    /// 
    /// # Arguments
    /// * `player` - Player ID (1-4)
    /// 
    /// # Returns
    /// Vector of piece IDs (0-20) still available to the player.
    /// Returns empty vector for invalid player IDs.
    /// 
    /// # Usage
    /// ```
    /// let available = state.get_available_pieces(1);
    /// println!("Player 1 can place pieces: {:?}", available);
    /// ```
    pub fn get_available_pieces(&self, player: i32) -> Vec<usize> {
        let player_idx = (player - 1) as usize;
        if player_idx < self.player_pieces.len() {
            self.player_pieces[player_idx].iter().map(|p| p.id).collect()
        } else {
            Vec::new()
        }
    }

    /// Returns a reference to the pieces available to a specific player
    /// 
    /// Provides direct access to the complete piece objects (including
    /// transformations) for a player. Useful for detailed analysis,
    /// UI rendering, and move generation.
    /// 
    /// # Arguments
    /// * `player` - Player ID (1-4)
    /// 
    /// # Returns
    /// Reference to the vector of Piece objects available to the player.
    /// Returns reference to empty vector for invalid player IDs.
    /// 
    /// # Performance Note
    /// Returns a reference to avoid cloning the potentially large piece data.
    /// Callers should not modify the returned pieces.
    pub fn get_player_pieces(&self, player: i32) -> &Vec<Piece> {
        let player_idx = (player - 1) as usize;
        if player_idx < self.player_pieces.len() {
            &self.player_pieces[player_idx]
        } else {
            // Return reference to a static empty vector
            static EMPTY_PIECES: Vec<Piece> = Vec::new();
            &EMPTY_PIECES
        }
    }
}

impl FromStr for BlokusMove {
    type Err = String;

    /// Creates a BlokusMove from a string representation
    /// 
    /// Parses move strings in the format "(piece_id,trans_idx,row,col)" where
    /// all values are non-negative integers. This format is used for game
    /// notation, move history, and network communication.
    /// 
    /// # Arguments
    /// * `s` - String in format "(piece,transformation,row,col)" (e.g., "(5,2,3,4)")
    /// 
    /// # Returns
    /// `Ok(BlokusMove)` if parsing succeeds, `Err(String)` with error description if invalid
    /// 
    /// # Format Requirements
    /// - Must be enclosed in parentheses
    /// - Four comma-separated integers
    /// - piece_id: 0-20 (not validated here)
    /// - trans_idx: 0+ (not validated here)  
    /// - row, col: 0-19 (not validated here)
    /// 
    /// # Error Cases
    /// - Missing or extra parentheses
    /// - Wrong number of comma-separated values
    /// - Non-numeric values
    /// - Negative numbers
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.starts_with('(') && s.ends_with(')') {
            let s = &s[1..s.len() - 1];
            let parts: Vec<&str> = s.split(',').map(|s| s.trim()).collect();
            if parts.len() != 4 {
                return Err("Expected format: (piece_idx,trans_idx,r,c)".to_string());
            }
            let p = parts[0].parse::<usize>().map_err(|e| e.to_string())?;
            let t = parts[1].parse::<usize>().map_err(|e| e.to_string())?;
            let r = parts[2].parse::<usize>().map_err(|e| e.to_string())?;
            let c = parts[3].parse::<usize>().map_err(|e| e.to_string())?;
            Ok(BlokusMove(p, t, r, c))
        } else {
            Err("Invalid move format for Blokus".to_string())
        }
    }
}
