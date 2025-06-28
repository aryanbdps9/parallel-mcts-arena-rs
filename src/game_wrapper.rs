use crate::games::connect4::{Connect4Move, Connect4State};
use crate::games::gomoku::{GomokuMove, GomokuState};
use crate::games::blokus::{BlokusMove, BlokusState};
use crate::games::othello::{OthelloMove, OthelloState};
use mcts::GameState;

#[derive(Debug, Clone)]
pub enum GameWrapper {
    Gomoku(GomokuState),
    Connect4(Connect4State),
    Blokus(BlokusState),
    Othello(OthelloState),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MoveWrapper {
    Gomoku(GomokuMove),
    Connect4(Connect4Move),
    Blokus(BlokusMove),
    Othello(OthelloMove),
}

impl GameState for GameWrapper {
    type Move = MoveWrapper;

    fn get_current_player(&self) -> i32 {
        match self {
            GameWrapper::Gomoku(g) => g.get_current_player(),
            GameWrapper::Connect4(g) => g.get_current_player(),
            GameWrapper::Blokus(g) => g.get_current_player(),
            GameWrapper::Othello(g) => g.get_current_player(),
        }
    }

    fn get_possible_moves(&self) -> Vec<Self::Move> {
        match self {
            GameWrapper::Gomoku(g) => g
                .get_possible_moves()
                .into_iter()
                .map(MoveWrapper::Gomoku)
                .collect(),
            GameWrapper::Connect4(g) => g
                .get_possible_moves()
                .into_iter()
                .map(MoveWrapper::Connect4)
                .collect(),
            GameWrapper::Blokus(g) => g
                .get_possible_moves()
                .into_iter()
                .map(MoveWrapper::Blokus)
                .collect(),
            GameWrapper::Othello(g) => g
                .get_possible_moves()
                .into_iter()
                .map(MoveWrapper::Othello)
                .collect(),
        }
    }

    fn make_move(&mut self, mv: &Self::Move) {
        match (self, mv) {
            (GameWrapper::Gomoku(g), MoveWrapper::Gomoku(m)) => g.make_move(m),
            (GameWrapper::Connect4(g), MoveWrapper::Connect4(m)) => g.make_move(m),
            (GameWrapper::Blokus(g), MoveWrapper::Blokus(m)) => g.make_move(m),
            (GameWrapper::Othello(g), MoveWrapper::Othello(m)) => g.make_move(m),
            _ => panic!("Mismatched game and move types"),
        }
    }

    fn is_terminal(&self) -> bool {
        match self {
            GameWrapper::Gomoku(g) => g.is_terminal(),
            GameWrapper::Connect4(g) => g.is_terminal(),
            GameWrapper::Blokus(g) => g.is_terminal(),
            GameWrapper::Othello(g) => g.is_terminal(),
        }
    }

    fn get_winner(&self) -> Option<i32> {
        match self {
            GameWrapper::Gomoku(g) => g.get_winner(),
            GameWrapper::Connect4(g) => g.get_winner(),
            GameWrapper::Blokus(g) => g.get_winner(),
            GameWrapper::Othello(g) => g.get_winner(),
        }
    }

    fn get_board(&self) -> &Vec<Vec<i32>> {
        match self {
            GameWrapper::Gomoku(g) => g.get_board(),
            GameWrapper::Connect4(g) => g.get_board(),
            GameWrapper::Blokus(g) => g.get_board(),
            GameWrapper::Othello(g) => g.get_board(),
        }
    }
}

impl GameWrapper {
    pub fn get_board_size(&self) -> usize {
        self.get_board().len()
    }

    pub fn get_line_size(&self) -> usize {
        match self {
            GameWrapper::Gomoku(g) => g.get_line_size(),
            GameWrapper::Connect4(g) => g.get_line_size(),
            GameWrapper::Blokus(g) => g.get_line_size(),
            GameWrapper::Othello(g) => g.get_line_size(),
        }
    }

    pub fn get_last_move(&self) -> Option<Vec<(usize, usize)>> {
        match self {
            GameWrapper::Gomoku(g) => g.get_last_move(),
            GameWrapper::Connect4(g) => g.get_last_move(),
            GameWrapper::Blokus(g) => g.get_last_move(),
            GameWrapper::Othello(g) => g.get_last_move(),
        }
    }
}
