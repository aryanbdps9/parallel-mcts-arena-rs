use crate::GameState;
use std::collections::HashSet;
use std::str::FromStr;

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
            transformations: unique_transformations.into_iter().collect(),
        }
    }
}

fn get_blokus_pieces() -> Vec<Piece> {
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

#[derive(Clone)]
pub struct BlokusState {
    board: Vec<Vec<i32>>,
    current_player: i32,
    player_pieces: Vec<Vec<Piece>>,
    is_first_move: [bool; 4],
    passed_players: [bool; 4],
    last_move_coords: Option<Vec<(usize, usize)>>,
}

impl GameState for BlokusState {
    type Move = BlokusMove; // piece_idx, transformation_idx, row, col

    fn get_board(&self) -> &Vec<Vec<i32>> {
        &self.board
    }

    fn get_possible_moves(&self) -> Vec<Self::Move> {
        let mut moves = Vec::new();
        let player_idx = (self.current_player - 1) as usize;

        if self.passed_players[player_idx] {
            return moves;
        }

        for (piece_idx, piece) in self.player_pieces[player_idx].iter().enumerate() {
            for (trans_idx, shape) in piece.transformations.iter().enumerate() {
                for r in 0..20 {
                    for c in 0..20 {
                        if self.is_valid_move(player_idx, shape, r, c) {
                            moves.push(BlokusMove(piece_idx, trans_idx, r, c));
                        }
                    }
                }
            }
        }
        moves
    }

    fn make_move(&mut self, mv: &Self::Move) {
        let player_idx = (self.current_player - 1) as usize;
        if mv.0 == 999 { // Pass move
            self.passed_players[player_idx] = true;
            self.last_move_coords = None;
        } else {
            let piece = &self.player_pieces[player_idx][mv.0];
            let shape = &piece.transformations[mv.1];
            let mut coords = Vec::new();
            for &(dr, dc) in shape {
                let r = (mv.2 as i32 + dr) as usize;
                let c = (mv.3 as i32 + dc) as usize;
                self.board[r][c] = self.current_player;
                coords.push((r, c));
            }
            self.last_move_coords = Some(coords);
            self.player_pieces[player_idx].remove(mv.0);
            self.is_first_move[player_idx] = false;
            self.passed_players[player_idx] = false;
        }

        // Advance to the next player who hasn't passed
        let mut next_player_found = false;
        for i in 1..=4 {
            let next_player = (self.current_player % 4) + i;
            let next_player_idx = (next_player - 1) as usize;
            if !self.passed_players[next_player_idx] {
                self.current_player = next_player;
                next_player_found = true;
                break;
            }
        }
        if !next_player_found {
            // All players have passed, game ends
            self.current_player = -1; // Or some other indicator
        }
    }

    fn is_terminal(&self) -> bool {
        self.passed_players.iter().all(|&p| p)
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
            None // Draw
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
        BlokusState {
            board: vec![vec![0; 20]; 20],
            current_player: 1,
            player_pieces: vec![
                get_blokus_pieces(),
                get_blokus_pieces(),
                get_blokus_pieces(),
                get_blokus_pieces(),
            ],
            is_first_move: [true; 4],
            passed_players: [false; 4],
            last_move_coords: None,
        }
    }

    pub fn get_line_size(&self) -> usize {
        1 // Blokus doesn't have a line size concept, return 1 as default
    }

    fn is_valid_move(&self, player_idx: usize, shape: &[(i32, i32)], r: usize, c: usize) -> bool {
        let player_id = (player_idx + 1) as i32;
        let mut corner_touch = false;

        if self.is_first_move[player_idx] {
            let target_corners = [(0, 0), (0, 19), (19, 19), (19, 0)];
            let target = target_corners[player_idx];
            let mut covers_corner = false;
            for (dr, dc) in shape {
                if (r as i32 + dr, c as i32 + dc) == target {
                    covers_corner = true;
                    break;
                }
            }
            if !covers_corner {
                return false;
            }
        } 

        for (dr, dc) in shape {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;

            if nr < 0 || nr >= 20 || nc < 0 || nc >= 20 || self.board[nr as usize][nc as usize] != 0 {
                return false;
            }

            if !self.is_first_move[player_idx] {
                let neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)];
                for (nnr, nnc) in &neighbors {
                    let ar = nr + nnr;
                    let ac = nc + nnc;
                    if ar >= 0 && ar < 20 && ac >= 0 && ac < 20 && self.board[ar as usize][ac as usize] == player_id {
                        return false;
                    }
                }
            }
        }

        if !self.is_first_move[player_idx] {
            let corners = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
            for (dr, dc) in shape {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                for (cnr, cnc) in &corners {
                    let ar = nr + cnr;
                    let ac = nc + cnc;
                    if ar >= 0 && ar < 20 && ac >= 0 && ac < 20 && self.board[ar as usize][ac as usize] == player_id {
                        corner_touch = true;
                        break;
                    }
                }
                if corner_touch { break; }
            }
        }

        self.is_first_move[player_idx] || corner_touch
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
