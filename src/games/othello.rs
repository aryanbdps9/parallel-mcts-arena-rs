use crate::GameState;
use std::str::FromStr;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct OthelloMove(pub usize, pub usize);

#[derive(Clone)]
pub struct OthelloState {
    board: Vec<Vec<i32>>,
    current_player: i32,
    board_size: usize,
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
        }
    }

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
