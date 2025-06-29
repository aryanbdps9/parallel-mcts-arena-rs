use crate::GameState;
use std::collections::HashSet;
use std::str::FromStr;
use std::fmt;

const PASS_MOVE: BlokusMove = BlokusMove(usize::MAX, 0, 0, 0);

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Piece {
    pub id: usize,
    pub transformations: Vec<Vec<(i32, i32)>>,
}

impl Piece {
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

pub fn get_piece_info() -> Vec<(usize, usize)> {
    get_blokus_pieces()
        .iter()
        .map(|p| (p.id, p.transformations.len()))
        .collect()
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct BlokusMove(pub usize, pub usize, pub usize, pub usize);

impl fmt::Display for BlokusMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == PASS_MOVE {
            write!(f, "PASS")
        } else {
            write!(f, "P{}T{}@({},{})", self.0, self.1, self.2, self.3)
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlokusState {
    board: Vec<Vec<i32>>,
    current_player: i32,
    player_pieces: Vec<Vec<Piece>>,
    is_first_move: [bool; 4],
    last_move_coords: Option<Vec<(usize, usize)>>,
    consecutive_passes: u8,
}

impl GameState for BlokusState {
    type Move = BlokusMove; // piece_idx, transformation_idx, row, col

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
        // Game ends when all 4 players pass consecutively
        if self.consecutive_passes >= 4 {
            return true;
        }
        
        // Additional safety check: if all players have no pieces left, game is over
        if self.player_pieces.iter().all(|pieces| pieces.is_empty()) {
            return true;
        }
        
        // Additional safety check: if no player can make any move, game is over
        // This prevents infinite loops in case of bugs
        if self.consecutive_passes >= 3 {
            // Check if the current player can make any move
            let player_idx = (self.current_player - 1) as usize;
            let available_pieces = &self.player_pieces[player_idx];
            
            if available_pieces.is_empty() {
                return true; // Current player has no pieces, will pass, making it 4 consecutive passes
            }
            
            // Quick check if current player has any valid moves
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
                return true; // Current player will be forced to pass, making it 4 consecutive passes
            }
        }
        
        false
    }

    fn get_winner(&self) -> Option<i32> {
        if !self.is_terminal() {
            return None;
        }

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

    pub fn get_board_size(&self) -> usize {
        20 // Blokus board is 20x20
    }

    pub fn get_line_size(&self) -> usize {
        1 // Blokus doesn't have a line size concept, return 1 as default
    }

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

    pub fn get_available_pieces(&self, player: i32) -> Vec<usize> {
        let player_idx = (player - 1) as usize;
        if player_idx < self.player_pieces.len() {
            self.player_pieces[player_idx].iter().map(|p| p.id).collect()
        } else {
            Vec::new()
        }
    }

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
