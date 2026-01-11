/// Test comparing different exploration constants on the same position
/// This helps diagnose whether exploration mismatch explains the CPU/GPU difference

#[cfg(feature = "gpu")]
use mcts::gpu::{GpuContext, GpuConfig};
#[cfg(feature = "gpu")]
use mcts::gpu::mcts_othello::GpuOthelloMcts;
use std::sync::Arc;

fn board_from_string(s: &str) -> [i32; 64] {
    let mut board = [0i32; 64];
    let chars: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(chars.len(), 64);
    
    for (i, &ch) in chars.iter().enumerate() {
        board[i] = match ch {
            '.' => 0,
            'X' => 1,
            'O' => -1,
            _ => panic!("Invalid character"),
        };
    }
    board
}

fn compute_legal_moves(board: &[i32; 64], player: i32) -> Vec<(usize, usize)> {
    let mut moves = Vec::new();
    for y in 0..8 {
        for x in 0..8 {
            if is_valid_move(board, x, y, player) {
                moves.push((x, y));
            }
        }
    }
    moves
}

fn is_valid_move(board: &[i32; 64], x: usize, y: usize, player: i32) -> bool {
    if board[y * 8 + x] != 0 { return false; }
    let directions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];
    for (dx, dy) in &directions {
        if count_flips_in_direction(board, x as i32, y as i32, player, *dx, *dy) > 0 {
            return true;
        }
    }
    false
}

fn count_flips_in_direction(board: &[i32; 64], x: i32, y: i32, player: i32, dx: i32, dy: i32) -> usize {
    let mut count = 0;
    let mut cx = x + dx;
    let mut cy = y + dy;
    while cx >= 0 && cx < 8 && cy >= 0 && cy < 8 {
        let cell = board[(cy * 8 + cx) as usize];
        if cell == 0 { return 0; }
        else if cell == -player { count += 1; cx += dx; cy += dy; }
        else if cell == player { return count; }
    }
    0
}

#[cfg(feature = "gpu")]
#[test]
fn test_exploration_constant_comparison() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║        EXPLORATION CONSTANT COMPARISON TEST                    ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    
    // Early game position (move 2)
    let board = board_from_string("
        ........
        ........
        ........
        ...OX...
        ...XX...
        ....X...
        ........
        ........
    ");
    
    let current_player = 1;
    let legal_moves = compute_legal_moves(&board, current_player);
    
    println!("Position: Early game (move 2)");
    println!("Legal moves: {:?}", legal_moves);
    println!("Player {} to move\n", current_player);
    
    // Test different exploration constants
    let exploration_values = vec![0.1, 0.5, 1.0, 1.414, 2.0, 4.0];
    
    for &exploration in &exploration_values {
        println!("═══════════════════════════════════════════════════════════════");
        println!(" Testing with EXPLORATION = {:.3}", exploration);
        println!("═══════════════════════════════════════════════════════════════");
        
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
        
        let engine = Arc::new(GpuOthelloMcts::new(
            context.clone(),
            100_000,
            8192,
        ).expect("Failed to create GPU engine"));
        
        engine.init_tree(&board, current_player, &legal_moves);
        engine.dispatch_mcts_othello_kernel(1, exploration, 1.0, 0.06, 0.01, 42);
        
        // Run 10 batches
        for batch in 0..10 {
            engine.dispatch_mcts_othello_kernel(8192, exploration, 1.0, 0.06, 0.01, 42 + batch * 1000);
        }
        
        engine.flush_and_wait();
        
        let stats = engine.get_children_stats();
        let mut sorted = stats.clone();
        sorted.sort_by_key(|(_, _, v, _, _)| -(*v));
        
        let total_visits: i32 = stats.iter().map(|(_, _, v, _, _)| *v).sum();
        
        println!("\nResults (Total visits: {}):", total_visits);
        for (i, (x, y, visits, _wins, q)) in sorted.iter().take(3).enumerate() {
            let pct = 100.0 * (*visits as f64) / (total_visits as f64);
            println!("  {}. ({},{})  visits={:7} ({:5.1}%)  Q={:.4}", 
                     i + 1, x, y, visits, pct, q);
        }
        
        if sorted.len() >= 2 {
            let ratio = sorted[0].2 as f64 / sorted[1].2 as f64;
            let q_best = sorted[0].4;
            println!("\n  Visit concentration: {:.2}x", ratio);
            println!("  Best move Q-value: {:.4}", q_best);
        }
        
        println!();
    }
    
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                    ANALYSIS                                    ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!("\nExpectations:");
    println!("- Low exploration (0.1): Should exploit early, high visit concentration");
    println!("- High exploration (4.0): Should explore broadly, lower concentration");
    println!("- Q-values should become more accurate with higher exploration");
    println!();
}
