use clap::Parser;
use mcts::{GameState, MCTS};
use std::io;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, default_value_t = 19)]
    board_size: usize,

    #[clap(short, long, default_value_t = 5)]
    line_size: usize,

    #[clap(short, long, default_value_t = 0)]
    num_threads: usize,
}

#[derive(Clone)]
struct GomokuState {
    board: Vec<Vec<i32>>,
    current_player: i32,
    board_size: usize,
    line_size: usize,
}

impl GameState for GomokuState {
    type Move = (usize, usize);

    fn get_possible_moves(&self) -> Vec<Self::Move> {
        let mut moves = vec![];
        let mut num_invalid = 0;
        let mut num_valid = 0;
        for r in 0..self.board_size {
            for c in 0..self.board_size {
                if self.board[r][c] == 0 {
                    moves.push((r, c));
                    num_valid += 1;
                } else {
                    num_invalid += 1;
                }
            }
        }
        let visited_cells = num_invalid + num_valid;

        assert_eq!(visited_cells, self.board_size * self.board_size, "Not all cells were visited! Invalid: {}, Valid: {}, Total: {}", num_invalid, num_valid, visited_cells);
        // print!("[get_possible_moves]: Possible moves: ");
        // for mv in &moves {
        //     print!("({},{}) ", mv.0, mv.1);
        // }
        // println!();
        moves
    }

    fn make_move(&mut self, mv: &Self::Move) {
        self.board[mv.0][mv.1] = self.current_player;
        self.current_player = -self.current_player;
    }

    fn is_terminal(&self) -> bool {
        self.get_winner().is_some() || self.get_possible_moves().is_empty()
    }

    fn get_winner(&self) -> Option<i32> {
        for r in 0..self.board_size {
            for c in 0..self.board_size {
                if self.board[r][c] != 0 {
                    let player = self.board[r][c];
                    // Check horizontal
                    if c + self.line_size <= self.board_size {
                        if (0..self.line_size).all(|i| self.board[r][c + i] == player) {
                            return Some(player);
                        }
                    }
                    // Check vertical
                    if r + self.line_size <= self.board_size {
                        if (0..self.line_size).all(|i| self.board[r + i][c] == player) {
                            return Some(player);
                        }
                    }
                    // Check diagonal (down-right)
                    if r + self.line_size <= self.board_size && c + self.line_size <= self.board_size {
                        if (0..self.line_size).all(|i| self.board[r + i][c + i] == player) {
                            return Some(player);
                        }
                    }
                    // Check diagonal (up-right)
                    if r >= self.line_size - 1 && c + self.line_size <= self.board_size {
                        if (0..self.line_size).all(|i| self.board[r - i][c + i] == player) {
                            return Some(player);
                        }
                    }
                }
            }
        }
        None
    }

    fn get_current_player(&self) -> i32 {
        self.current_player
    }
}

fn print_board(board: &Vec<Vec<i32>>) {
    for row in board {
        for &cell in row {
            print!(" ");
            match cell {
                1 => print!("X"),
                -1 => print!("O"),
                _ => print!("."),
            }
        }
        println!();
    }
}

fn main() {
    let args = Args::parse();

    let mut state = GomokuState {
        board: vec![vec![0; args.board_size]; args.board_size],
        current_player: 1,
        board_size: args.board_size,
        line_size: args.line_size,
    };

    let mut mcts = MCTS::new(1.414, args.num_threads);

    while !state.is_terminal() {
        print_board(&state.board);
        let mv = if state.current_player == 1 {
            // Human player
            let mut input = String::new();
            println!("Enter your move (row col):");
            io::stdin().read_line(&mut input).unwrap();
            let parts: Vec<usize> = input.trim().split_whitespace().map(|s| s.parse().unwrap()).collect();
            (parts[0], parts[1])
        } else {
            // AI player
            println!("AI is thinking...");
            let mv = mcts.search(&state, 100000);

            println!("AI move stats (value/wins/visits):");
            let stats = mcts.get_root_children_stats();
            let mut value_grid = vec![vec![0.0; state.board_size]; state.board_size];
            let mut wins_grid = vec![vec![0.0; state.board_size]; state.board_size];
            let mut visits_grid = vec![vec![0; state.board_size]; state.board_size];

            for ((r, c), (wins, visits)) in stats.iter() {
                if *visits > 0 {
                    value_grid[*r][*c] = wins / *visits as f64;
                }
                wins_grid[*r][*c] = *wins;
                visits_grid[*r][*c] = *visits;
            }

            println!("--- Values ---");
            for r in 0..state.board_size {
                for c in 0..state.board_size {
                    if state.board[r][c] == 1 {
                        print!("    X     ");
                    } else if state.board[r][c] == -1 {
                        print!("    O     ");
                    } else if visits_grid[r][c] > 0 {
                        print!("{:^10.2}", value_grid[r][c]);
                    } else {
                        print!("{:^10}", ".");
                    }
                }
                println!();
            }

            println!("\n--- Wins ---");
            for r in 0..state.board_size {
                for c in 0..state.board_size {
                    if state.board[r][c] == 1 {
                        print!("    X     ");
                    } else if state.board[r][c] == -1 {
                        print!("    O     ");
                    } else {
                        print!("{:^10.0}", wins_grid[r][c]);
                    }
                }
                println!();
            }

            println!("\n--- Visits ---");
            for r in 0..state.board_size {
                for c in 0..state.board_size {
                    if state.board[r][c] == 1 {
                        print!("    X     ");
                    } else if state.board[r][c] == -1 {
                        print!("    O     ");
                    } else {
                        print!("{:^10}", visits_grid[r][c]);
                    }
                }
                println!();
            }

            mv
        };

        // Print mv
        println!("[main]: Player {} moves to ({}, {})", state.current_player, mv.0, mv.1);
        if !state.get_possible_moves().contains(&mv) {
            println!("Invalid move! Try again.");
            continue;
        }

        state.make_move(&mv);
        mcts.advance_root(&mv);
    }

    print_board(&state.board);
    match state.get_winner() {
        Some(1) => println!("You win!"),
        Some(-1) => println!("AI wins!"),
        _ => println!("It's a draw!"),
    }
}
