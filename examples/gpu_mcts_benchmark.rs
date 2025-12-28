//! GPU-Native MCTS Benchmark for Othello
//!
//! This example compares the GPU-native MCTS implementation with the hybrid approach.
//! Run with: cargo run --example gpu_mcts_benchmark --features gpu --release

use mcts::games::othello::OthelloState;
use mcts::{GameState, MCTS};
use std::time::Instant;

fn main() {
    println!("=== GPU-Native MCTS Benchmark for Othello ===\n");

    // Create initial Othello game
    let state = OthelloState::new(8);
    let legal_moves = state.get_possible_moves();
    let board_2d = state.get_board();
    
    // Flatten board for GPU
    let mut board = [0i32; 64];
    for (r, row) in board_2d.iter().enumerate() {
        for (c, &cell) in row.iter().enumerate() {
            board[r * 8 + c] = cell;
        }
    }
    
    // Convert moves to (x, y) format
    let legal_moves_xy: Vec<(usize, usize)> = legal_moves
        .iter()
        .map(|m| (m.1, m.0)) // OthelloMove is (row, col), GPU expects (x, y)
        .collect();
    
    println!("Initial position:");
    println!("{}", state);
    println!("Legal moves: {:?}\n", legal_moves);

    // Create MCTS engine with GPU
    let (mut mcts, gpu_info) = MCTS::<OthelloState>::with_gpu(
        1.4,   // exploration parameter
        8,     // num threads
        1_000_000, // max nodes
    );
    
    if let Some(info) = &gpu_info {
        println!("GPU Info: {}\n", info);
    }

    // Test 1: CPU-only MCTS (baseline)
    println!("--- Test 1: CPU-Only Search (3 seconds) ---");
    {
        let mut cpu_mcts = MCTS::<OthelloState>::new(1.4, 8, 100_000);
        let start = Instant::now();
        // Pass a large iteration count but let timeout control the search
        let (best_move, stats) = cpu_mcts.search(&state, 10_000_000, 0, 3); // 3 second timeout
        let elapsed = start.elapsed();
        
        println!("Best move: {:?}", best_move);
        println!("Root visits: {}", stats.root_visits);
        println!("Root Q: {:.4}", stats.root_value);
        println!("Time: {:.2}s", elapsed.as_secs_f64());
        println!("Simulations/sec: {:.0}\n", stats.root_visits as f64 / elapsed.as_secs_f64());
    }

    // Test 2: Hybrid GPU search (current implementation)
    println!("--- Test 2: Hybrid GPU Search (3 seconds) ---");
    let start = Instant::now();
    // Pass a large iteration count but let timeout control the search
    let (best_move, stats) = mcts.search(&state, 10_000_000, 0, 3); // 3 second timeout
    let elapsed = start.elapsed();
    
    println!("Best move: {:?}", best_move);
    println!("Root visits: {}", stats.root_visits);
    println!("Root Q: {:.4}", stats.root_value);
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!("Simulations/sec: {:.0}\n", stats.root_visits as f64 / elapsed.as_secs_f64());

    // Test 3: GPU-Native search (different batch sizes)
    #[cfg(feature = "gpu")]
    {
        for &iterations_per_batch in &[1024, 4096, 16384] {
            println!("--- Test 3: GPU-Native Search (batch={}) ---", iterations_per_batch);
            
            // Target roughly 3 seconds of iterations
            let target_iterations = 500_000u32;
            let num_batches = target_iterations / iterations_per_batch;
            
            let start = Instant::now();
            let result = mcts.search_gpu_native_othello(
                &board,
                state.get_current_player(),
                &legal_moves_xy,
                iterations_per_batch,
                num_batches,
                1.4,
                0, // No timeout, use batch limit
            );
            let elapsed = start.elapsed();
            
            if let Some(((x, y), visits, q, _children, total_nodes)) = result {
                let total_iterations = iterations_per_batch * num_batches;
                println!("Best move: ({}, {})", x, y);
                println!("Total iterations: {}", total_iterations);
                println!("Root visits: {}", visits);
                println!("Total nodes: {}", total_nodes);
                println!("Q value: {:.4}", q);
                println!("Time: {:.2}s", elapsed.as_secs_f64());
                println!("Iterations/sec: {:.0}\n", total_iterations as f64 / elapsed.as_secs_f64());
            } else {
                println!("GPU-native search returned no result\n");
            }
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature not enabled, skipping GPU-native test");
    }
}
