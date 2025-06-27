use clap::Parser;
use colored::*;
use mcts::{GameState, MCTS};
use std::collections::HashSet;
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

    #[clap(short = 'e', long, default_value_t = 4.0)]
    exploration_parameter: f64,

    #[clap(short = 'i', long, default_value_t = 1000000)]
    iterations: i32,
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
    print!("   ");
    for i in 0..board.len() {
        print!("{:^3}", i);
    }
    println!();
    for (i, row) in board.iter().enumerate() {
        print!("{:>2} ", i);
        for &cell in row {
            match cell {
                1 => print!("X  "),
                -1 => print!("O  "),
                _ => print!(".  "),
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

    let mut mcts = MCTS::new(args.exploration_parameter, args.num_threads);

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
            let mv = mcts.search(&state, args.iterations);

            let root_stats = mcts.get_root_stats();
            // Normalize the root value to 0-1 range (since rewards are 0, 1, 2, we divide by 2)
            let root_value = if root_stats.1 > 0 { (root_stats.0 / root_stats.1 as f64) / 2.0 } else { 0.0 };
            println!("Root node value: {:.4}", root_value);

            println!("AI move stats (value/wins/visits):");
            let stats = mcts.get_root_children_stats();
            let mut value_grid = vec![vec![0.0; state.board_size]; state.board_size];
            let mut wins_grid = vec![vec![0.0; state.board_size]; state.board_size];
            let mut visits_grid = vec![vec![0; state.board_size]; state.board_size];

            for ((r, c), (wins, visits)) in stats.iter() {
                if *visits > 0 {
                    // Normalize the value to 0-1 range (since rewards are 0, 1, 2, we divide by 2)
                    value_grid[*r][*c] = (wins / *visits as f64) / 2.0;
                }
                wins_grid[*r][*c] = *wins;
                visits_grid[*r][*c] = *visits;
            }

            let mut top_values = stats.iter().filter(|(_, (_, v))| *v > 0).collect::<Vec<_>>();
            // Normalize values when sorting (divide by 2 since max reward is 2)
            top_values.sort_by(|a, b| ((b.1.0 / b.1.1 as f64) / 2.0).partial_cmp(&((a.1.0 / a.1.1 as f64) / 2.0)).unwrap());
            let top_5_value_moves: HashSet<_> = top_values.iter().take(5).map(|(m, _)| *m).collect();

            let mut top_wins = stats.iter().filter(|(_, (_, v))| *v > 0).collect::<Vec<_>>();
            top_wins.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            let top_5_win_moves: HashSet<_> = top_wins.iter().take(5).map(|(m, _)| *m).collect();

            let mut top_visits = stats.iter().collect::<Vec<_>>();
            top_visits.sort_by(|a, b| b.1.1.cmp(&a.1.1));
            let top_5_visit_moves: HashSet<_> = top_visits.iter().take(5).map(|(m, _)| *m).collect();

            println!("--- Values ---");
            for r in 0..state.board_size {
                for c in 0..state.board_size {
                    let text = if state.board[r][c] == 1 {
                        "    X     ".normal()
                    } else if state.board[r][c] == -1 {
                        "    O     ".normal()
                    } else if visits_grid[r][c] > 0 {
                        format!("{:^10.2}", value_grid[r][c]).normal()
                    } else {
                        format!("{:^10}", ".").normal()
                    };
                    if top_5_value_moves.contains(&(r, c)) {
                        print!("{}", text.red());
                    } else {
                        print!("{}", text);
                    }
                }
                println!();
            }

            println!("\n--- Wins ---");
            for r in 0..state.board_size {
                for c in 0..state.board_size {
                     let text = if state.board[r][c] == 1 {
                        "    X     ".normal()
                    } else if state.board[r][c] == -1 {
                        "    O     ".normal()
                    } else {
                        format!("{:^10.0}", wins_grid[r][c]).normal()
                    };
                    if top_5_win_moves.contains(&(r,c)) {
                        print!("{}", text.red());
                    } else {
                        print!("{}", text);
                    }
                }
                println!();
            }

            println!("\n--- Visits ---");
            for r in 0..state.board_size {
                for c in 0..state.board_size {
                    let text = if state.board[r][c] == 1 {
                        "    X     ".normal()
                    } else if state.board[r][c] == -1 {
                        "    O     ".normal()
                    } else {
                        format!("{:^10}", visits_grid[r][c]).normal()
                    };
                    if top_5_visit_moves.contains(&(r,c)) {
                        print!("{}", text.red());
                    } else {
                        print!("{}", text);
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
