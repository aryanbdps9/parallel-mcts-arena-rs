use crate::GameState;
use std::str::FromStr;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Connect4Move(pub usize);

#[derive(Clone)]
pub struct Connect4State {
    board: Vec<Vec<i32>>,
    current_player: i32,
    width: usize,
    height: usize,
    line_size: usize,
    last_move: Option<(usize, usize)>,
}

impl GameState for Connect4State {
    type Move = Connect4Move; // Column to drop a piece

    fn get_board(&self) -> &Vec<Vec<i32>> {
        &self.board
    }

    fn get_last_move(&self) -> Option<Vec<(usize, usize)>> {
        self.last_move.map(|(r, c)| vec![(r, c)])
    }

    fn get_possible_moves(&self) -> Vec<Self::Move> {
        (0..self.width).filter(|&c| self.board[0][c] == 0).map(Connect4Move).collect()
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
    pub fn new(width: usize, height: usize, line_size: usize) -> Self {
        Connect4State {
            board: vec![vec![0; width]; height],
            current_player: 1,
            width,
            height,
            line_size,
            last_move: None,
        }
    }
}

impl FromStr for Connect4Move {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let c = s.trim().parse::<usize>().map_err(|e| e.to_string())?;
        Ok(Connect4Move(c))
    }
}
