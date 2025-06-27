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

    #[clap(short = 'm', long, default_value_t = 100000)]
    max_nodes: usize,
}

#[derive(Clone)]
struct GomokuState {
    board: Vec<Vec<i32>>,
    current_player: i32,
    board_size: usize,
    line_size: usize,
    last_move: Option<(usize, usize)>,
}

impl GameState for GomokuState {
    type Move = (usize, usize);

    fn get_possible_moves(&self) -> Vec<Self::Move> {
        (0..self.board_size)
            .flat_map(|r| (0..self.board_size).map(move |c| (r, c)))
            .filter(|&(r, c)| self.board[r][c] == 0)
            .collect()
    }

    fn make_move(&mut self, mv: &Self::Move) {
        self.board[mv.0][mv.1] = self.current_player;
        self.last_move = Some(*mv);
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
        last_move: None,
    };

    let mut mcts = MCTS::new(args.exploration_parameter, args.num_threads, args.max_nodes);

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
            println!("[DEBUG] Before search: {}", mcts.get_debug_info());
            let mv = mcts.search(&state, args.iterations);
            println!("[DEBUG] After search: {}", mcts.get_debug_info());

            let root_stats = mcts.get_root_stats();
            // Normalize the root value to 0-1 range (since rewards are 0, 1, 2, we divide by 2)
            let root_value = if root_stats.1 > 0 { (root_stats.0 / root_stats.1 as f64) / 2.0 } else { 0.0 };
            println!("Root node value: {:.4}", root_value);

            println!("AI move stats (value/wins/visits):");
            let stats = mcts.get_root_children_stats();
            let mut value_grid = vec![vec![0.0; state.board_size]; state.board_size];
            let mut wins_grid = vec![vec![0.0; state.board_size]; state.board_size];
            let mut visits_grid = vec![vec![0; state.board_size]; state.board_size];

            // Pre-allocate vectors for sorting
            let mut top_values = Vec::with_capacity(stats.len());
            let mut top_wins = Vec::with_capacity(stats.len());
            let mut top_visits = Vec::with_capacity(stats.len());

            for ((r, c), (wins, visits)) in stats.iter() {
                if *visits > 0 {
                    // Normalize the value to 0-1 range (since rewards are 0, 1, 2, we divide by 2)
                    value_grid[*r][*c] = (wins / *visits as f64) / 2.0;
                }
                wins_grid[*r][*c] = *wins;
                visits_grid[*r][*c] = *visits;
                
                if *visits > 0 {
                    top_values.push((*r, *c, (wins / *visits as f64) / 2.0));
                }
                top_wins.push((*r, *c, *wins));
                top_visits.push((*r, *c, *visits));
            }

            // Sort once and take top 5
            top_values.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            top_wins.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            top_visits.sort_by(|a, b| b.2.cmp(&a.2));

            let top_5_value_moves: HashSet<_> = top_values.iter().take(5).map(|(r, c, _)| (*r, *c)).collect();
            let top_5_win_moves: HashSet<_> = top_wins.iter().take(5).map(|(r, c, _)| (*r, *c)).collect();
            let top_5_visit_moves: HashSet<_> = top_visits.iter().take(5).map(|(r, c, _)| (*r, *c)).collect();

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

            // Now prune the tree after displaying statistics to prepare for the next search
            mcts.auto_prune();

            mv
        };

        // Print mv
        let current_player = state.current_player;
        if !state.get_possible_moves().contains(&mv) {
            println!("Invalid move [{}, {}]! Try again.", mv.0, mv.1);
            continue;
        }

        state.make_move(&mv);
        println!("[main]: Player {} made a move to ({}, {})", current_player, mv.0, mv.1);
        println!("[DEBUG] Before advancing root: {}", mcts.get_debug_info());
        mcts.advance_root(&mv);
        println!("[DEBUG] After advancing root: {}", mcts.get_debug_info());
        println!("[main]: MCTS root advanced to next state.");
    }

    print_board(&state.board);
    match state.get_winner() {
        Some(1) => println!("You win!"),
        Some(-1) => println!("AI wins!"),
        _ => println!("It's a draw!"),
    }
}
