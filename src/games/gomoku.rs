use crate::GameState;
use std::str::FromStr;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct GomokuMove(pub usize, pub usize);

#[derive(Debug, Clone)]
pub struct GomokuState {
    pub board: Vec<Vec<i32>>,
    pub current_player: i32,
    board_size: usize,
    line_size: usize,
    last_move: Option<(usize, usize)>,
}

impl GomokuState {
    pub fn new(board_size: usize, line_size: usize) -> Self {
        GomokuState {
            board: vec![vec![0; board_size]; board_size],
            current_player: 1,
            board_size,
            line_size,
            last_move: None,
        }
    }

    pub fn get_board_size(&self) -> usize {
        self.board_size
    }

    pub fn get_line_size(&self) -> usize {
        self.line_size
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