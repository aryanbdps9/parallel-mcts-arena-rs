pub mod tui;

pub mod games;
pub mod game_wrapper;

use clap::Parser;
use std::io;
use mcts::{GameState, MCTS};
use crate::games::gomoku::{GomokuMove, GomokuState};
use crate::games::connect4::{Connect4Move, Connect4State};
use crate::games::blokus::{BlokusMove, BlokusState};
use crate::games::othello::{OthelloMove, OthelloState};
use crate::game_wrapper::{GameWrapper, MoveWrapper};

#[derive(PartialEq)]
pub enum AppState {
    Menu,
    Playing,
    GameOver,
}

pub struct App<'a> {
    pub titles: Vec<&'a str>,
    pub index: usize,
    pub state: AppState,
    pub game_type: String,
    pub game: GameWrapper,
    pub cursor: (usize, usize),
    pub winner: Option<i32>,
    pub ai: MCTS<GameWrapper>,
    pub ai_only: bool,
    pub iterations: i32,
    pub num_threads: usize,
    pub stats_interval_secs: u64,
    pub timeout_secs: u64,
}

impl<'a> App<'a> {
    fn new(args: Args) -> App<'a> {
        let game = match args.game.as_str() {
            "gomoku" => GameWrapper::Gomoku(GomokuState::new(args.board_size, args.line_size)),
            "connect4" => GameWrapper::Connect4(Connect4State::new(7, 6, 4)),
            "blokus" => GameWrapper::Blokus(BlokusState::new()),
            "othello" => GameWrapper::Othello(OthelloState::new(8)),
            _ => panic!("Unknown game type"),
        };
        let ai = MCTS::new(args.exploration_parameter, args.num_threads, args.max_nodes);
        App {
            titles: vec!["Gomoku", "Connect4", "Blokus", "Othello", "Quit"],
            index: 0,
            state: AppState::Menu,
            game_type: args.game,
            game,
            cursor: (0, 0),
            winner: None,
            ai,
            ai_only: args.ai_only,
            iterations: args.iterations,
            num_threads: args.num_threads,
            stats_interval_secs: args.stats_interval_secs,
            timeout_secs: args.timeout_secs,
        }
    }

    pub fn set_game(&mut self, index: usize) {
        self.game_type = self.titles[index].to_lowercase();
        self.game = match self.game_type.as_str() {
            "gomoku" => GameWrapper::Gomoku(GomokuState::new(19, 5)),
            "connect4" => GameWrapper::Connect4(Connect4State::new(7, 6, 4)), // 7 width, 6 height, 4 line_size
            "blokus" => GameWrapper::Blokus(BlokusState::new()),
            "othello" => GameWrapper::Othello(OthelloState::new(8)), // 8x8 board
            _ => panic!("Unknown game type"),
        };
        // Set cursor position based on game type
        self.cursor = match self.game_type.as_str() {
            "gomoku" => (9, 9), // Center of 19x19 board
            "connect4" => (0, 3), // Top row, center column
            "blokus" => (10, 10), // Center of 20x20 board
            "othello" => (3, 3), // Starting position for Othello (near center)
            _ => (0, 0),
        };
    }

    pub fn tick(&mut self) {
        if self.state == AppState::Playing && self.ai_only {
            if !self.game.is_terminal() {
                let mv = self.ai.search(&self.game, self.iterations, self.stats_interval_secs, self.timeout_secs);
                self.game.make_move(&mv);
                self.ai.advance_root(&mv);
                if self.game.is_terminal() {
                    self.winner = self.game.get_winner();
                    self.state = AppState::GameOver;
                }
            }
        }
    }

    pub fn next(&mut self) {
        self.index = (self.index + 1) % self.titles.len();
    }

    pub fn previous(&mut self) {
        if self.index > 0 {
            self.index -= 1;
        } else {
            self.index = self.titles.len() - 1;
        }
    }

    pub fn move_cursor_down(&mut self) {
        let board_size = self.game.get_board().len();
        if self.cursor.0 < board_size - 1 {
            self.cursor.0 += 1;
        }
    }

    pub fn move_cursor_up(&mut self) {
        if self.cursor.0 > 0 {
            self.cursor.0 -= 1;
        }
    }

    pub fn move_cursor_left(&mut self) {
        if self.cursor.1 > 0 {
            self.cursor.1 -= 1;
        }
    }

    pub fn move_cursor_right(&mut self) {
        let board_size = self.game.get_board().len();
        if self.cursor.1 < board_size - 1 {
            self.cursor.1 += 1;
        }
    }

    pub fn make_move(&mut self) {
        let (r, c) = self.cursor;
        if self.game.get_board()[r][c] == 0 {
            let player_move = match self.game {
                GameWrapper::Gomoku(_) => MoveWrapper::Gomoku(GomokuMove(r, c)),
                GameWrapper::Connect4(_) => MoveWrapper::Connect4(Connect4Move(c)),
                GameWrapper::Blokus(_) => {
                    // For Blokus, we'll use the first available piece and transformation as a placeholder
                    // This is a simplified move selection for UI purposes
                    MoveWrapper::Blokus(BlokusMove(0, 0, r, c))
                },
                GameWrapper::Othello(_) => MoveWrapper::Othello(OthelloMove(r, c)),
            };
            self.game.make_move(&player_move);
            self.ai.advance_root(&player_move);
            if self.game.is_terminal() {
                self.winner = self.game.get_winner();
                self.state = AppState::GameOver;
                return;
            }

            if !self.ai_only {
                let ai_move = self.ai.search(&self.game, self.iterations, self.stats_interval_secs, self.timeout_secs);
                self.game.make_move(&ai_move);
                self.ai.advance_root(&ai_move);
                if self.game.is_terminal() {
                    self.winner = self.game.get_winner();
                    self.state = AppState::GameOver;
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.state = AppState::Menu;
        self.game = match self.game_type.as_str() {
            "gomoku" => GameWrapper::Gomoku(GomokuState::new(19, 5)),
            "connect4" => GameWrapper::Connect4(Connect4State::new(7, 6, 4)),
            "blokus" => GameWrapper::Blokus(BlokusState::new()),
            "othello" => GameWrapper::Othello(OthelloState::new(8)),
            _ => panic!("Unknown game type"),
        };
        self.ai = MCTS::new(self.ai.get_exploration_parameter(), self.num_threads, self.ai.get_max_nodes());
        self.winner = None;
        self.cursor = match self.game_type.as_str() {
            "gomoku" => (9, 9), // Center of 19x19 board
            "connect4" => (0, 3), // Top row, center column
            "blokus" => (10, 10), // Center of 20x20 board
            "othello" => (3, 3), // Starting position for Othello
            _ => (0, 0),
        };
    }
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, default_value = "gomoku")]
    game: String,

    #[clap(short, long, default_value_t = 19)]
    board_size: usize,

    #[clap(short, long, default_value_t = 5)]
    line_size: usize,

    #[clap(short, long, default_value_t = 0)]
    num_threads: usize,

    #[clap(short = 'e', long, default_value_t = 4.0)]
    exploration_parameter: f64,

    #[clap(short = 'i', long, default_value_t = 1000000)]
    iterations: i32,

    #[clap(short = 'm', long, default_value_t = 100000)]
    max_nodes: usize,

    #[clap(long, default_value_t = 0)]
    stats_interval_secs: u64,

    #[clap(long, default_value_t = 0)]
    timeout_secs: u64,

    #[clap(long, action = clap::ArgAction::SetTrue)]
    ai_only: bool,

    #[clap(long, action = clap::ArgAction::SetTrue)]
    shared_tree: bool,
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let mut app = App::new(args);
    tui::run_tui(&mut app)
}
