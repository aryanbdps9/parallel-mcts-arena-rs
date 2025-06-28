use clap::Parser;
use mcts::{
    games::{blokus::{self, BlokusState}, connect4::Connect4State, gomoku::GomokuState, othello::OthelloState},
    GameState, MCTS,
};
use std::io;
use std::collections::HashMap;

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

fn print_board(board: &Vec<Vec<i32>>, _game: &str) {
    print!("   ");
    if !board.is_empty() {
        for i in 0..board[0].len() {
            print!("{:^3}", i);
        }
    }
    println!();
    for (i, row) in board.iter().enumerate() {
        print!("{:>2} ", i);
        for &cell in row {
            match cell {
                1 => print!("X  "),
                -1 => print!("O  "),
                2 => print!("A  "),
                3 => print!("B  "),
                4 => print!("C  "),
                _ => print!(".  "),
            }
        }
        println!();
    }
}

fn run_game<S: GameState>(mut state: S, args: Args, game: &str)
where
    S::Move: std::str::FromStr,
    <S::Move as std::str::FromStr>::Err: std::fmt::Debug,
{
    let mut mcts_map = HashMap::new();
    let mut single_mcts = MCTS::new(args.exploration_parameter, args.num_threads, args.max_nodes);

    while !state.is_terminal() {
        print_board(state.get_board(), game);
        let current_player = state.get_current_player();

        let is_human_turn = !args.ai_only && current_player == 1;

        let mv = if is_human_turn {
            // Human player
            let mut input = String::new();
            match game {
                "gomoku" | "othello" => {
                    println!("Enter your move as 'row,col' (e.g., '5,5'):");
                }
                "connect4" => {
                    println!("Enter the column to drop your piece (0-6):");
                }
                "blokus" => {
                    println!("Enter your move as '(piece_idx,trans_idx,row,col)' or 'pass':");
                }
                _ => {
                    println!("Enter your move:");
                }
            }
            io::stdin().read_line(&mut input).unwrap();
            if game == "blokus" && input.trim() == "pass" {
                // A special move to signify passing
                "(999,0,0,0)".parse().unwrap()
            } else {
                input.trim().parse().unwrap()
            }
        } else {
            // AI player
            println!("Player {} (AI) is thinking...", current_player);
            let mcts_instance = if args.shared_tree {
                &mut single_mcts
            } else {
                mcts_map.entry(current_player).or_insert_with(|| {
                    MCTS::new(args.exploration_parameter, args.num_threads, args.max_nodes)
                })
            };
            mcts_instance.search(
                &state,
                args.iterations,
                args.stats_interval_secs,
                args.timeout_secs,
            )
        };

        if !state.get_possible_moves().contains(&mv) {
            println!("Invalid move!");
            continue;
        }

        state.make_move(&mv);

        if args.shared_tree {
            single_mcts.advance_root(&mv);
        } else {
            // When not sharing, all players have their own tree, so we advance all of them
            for mcts_instance in mcts_map.values_mut() {
                mcts_instance.advance_root(&mv);
            }
            // Also advance the single_mcts instance if it's not a fully AI game
            if !args.ai_only {
                single_mcts.advance_root(&mv);
            }
        }
    }

    print_board(state.get_board(), game);
    match state.get_winner() {
        Some(1) => println!("Player 1 (X) wins!"),
        Some(-1) => println!("Player 2 (O) wins!"),
        Some(2) => println!("Player 2 (A) wins!"),
        Some(3) => println!("Player 3 (B) wins!"),
        Some(4) => println!("Player 4 (C) wins!"),
        _ => println!("It's a draw!"),
    }
}

fn main() {
    let args = Args::parse();
    let game = args.game.clone();

    match game.as_str() {
        "gomoku" => {
            let state = GomokuState::new(args.board_size, args.line_size);
            run_game(state, args, &game);
        }
        "othello" => {
            let board_size = 8;
            println!("Using default Othello settings: {}x{} board.", board_size, board_size);
            let state = OthelloState::new(board_size);
            run_game(state, args, &game);
        }
        "connect4" => {
            let width = 7;
            let height = 6;
            let line_size = 4;
            println!("Using default Connect4 settings: {}x{} board, {} in a row to win.", width, height, line_size);
            let state = Connect4State::new(width, height, line_size);
            run_game(state, args, &game);
        }
        "blokus" => {
            println!("Using default Blokus settings: 20x20 board.");
            println!("Available pieces and their transformation counts (trans_idx):");
            for (id, count) in blokus::get_piece_info() {
                println!("  Piece {} (piece_idx): {} transformations", id, count);
            }
            let state = BlokusState::new();
            run_game(state, args, &game);
        }
        _ => {
            println!("Unknown game: {}", args.game);
        }
    }
}
