/// Test opening position to verify visit distribution matches Q-values

use std::sync::Arc;
use mcts::gpu::{GpuContext, GpuConfig};
use mcts::gpu::mcts_othello::GpuOthelloMcts;

#[test]
fn test_opening_position_visits() {
    // Create GPU context
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Standard Othello opening position
    let mut board = [0i32; 64];
    board[27] = -1; board[28] = 1;  // Row 3: O X
    board[35] = 1;  board[36] = -1; // Row 4: X O
    let player = 1;  // X to move
    
    // Legal moves in opening: (2,3), (3,2), (4,5), (5,4)
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    eprintln!("\n=== OPENING POSITION ===");
    eprintln!("  0 1 2 3 4 5 6 7");
    for y in 0..8 {
        eprint!("{} ", y);
        for x in 0..8 {
            let idx = y * 8 + x;
            let c = match board[idx] {
                1 => 'X',
                -1 => 'O',
                _ => '.',
            };
            eprint!("{} ", c);
        }
        eprintln!();
    }
    eprintln!("Player {} to move", player);
    eprintln!("Legal moves: {:?}", legal_moves);
    
    // Create MCTS engine
    let max_nodes = 50000;
    let batch_size = 2048;
    let engine = GpuOthelloMcts::new(context.clone(), max_nodes, batch_size)
        .expect("Failed to create MCTS engine");
    
    // Initialize tree
    engine.init_tree(&board, player, &legal_moves);
    
    // Run search with moderate parameters
    let num_threads = 2048;
    let num_steps = 200;
    let exploration = 1.414;
    let virtual_loss_weight = 0.2;
    let temperature = 0.06;  // Same as game
    let seed = 12345;
    
    eprintln!("\n=== RUNNING MCTS ===");
    eprintln!("Threads: {}, Steps: {}, Temp: {}", num_threads, num_steps, temperature);
    
    let telemetry = engine.run_incremental_mcts(
        num_threads,
        num_steps,
        exploration,
        virtual_loss_weight,
        temperature,
        seed,
        None,
        true,
    );
    
    eprintln!("\n=== SEARCH RESULTS ===");
    eprintln!("Total rollouts: {}", telemetry.diagnostics.rollouts);
    eprintln!("Expansion successes: {}", telemetry.diagnostics.expansion_success);
    eprintln!("Expansion locked: {}", telemetry.diagnostics.expansion_locked);
    eprintln!("Total children generated: {}", telemetry.diagnostics.total_children_gen);
    eprintln!("Total nodes in tree (CPU tracking): {}", engine.get_total_nodes());
    
    // Get root visits
    eprintln!("Root visits: {}", engine.get_root_visits());
    
    // Get children statistics
    let children_stats = engine.get_children_stats();
    
    // Sort by visits descending
    let mut sorted = children_stats.clone();
    sorted.sort_by_key(|(_, _, v, _, _)| -(*v));
    
    let total_visits: i32 = children_stats.iter().map(|c| c.2).sum();
    
    eprintln!("\nAll moves sorted by visits:");
    for (i, &(x, y, visits, wins, q)) in sorted.iter().enumerate() {
        let visit_pct = (visits as f64 / total_visits as f64) * 100.0;
        
        // Calculate PUCT
        let parent_visits = total_visits as f64;
        let sqrt_parent = (parent_visits + 1.0).sqrt();
        let prior = 1.0 / legal_moves.len() as f64;
        let effective_visits = visits as f64;
        let u = (exploration as f64) * prior * sqrt_parent / (1.0 + effective_visits);
        let puct = q + u;
        
        eprintln!("  #{}. ({},{}) visits={:6} ({:5.2}%), wins={:6}, Q={:.4}, U={:.4}, PUCT={:.4}", 
                 i+1, x, y, visits, visit_pct, wins, q, u, puct);
    }
    
    // Analysis
    eprintln!("\n=== ANALYSIS ===");
    
    // Check if visit distribution roughly follows softmax of Q-values
    // With temperature=0.06, better moves should get exponentially more visits
    if sorted.len() >= 2 {
        let best = sorted[0];
        let second = sorted[1];
        
        let q_diff = best.4 - second.4;
        let visit_ratio = best.2 as f64 / second.2.max(1) as f64;
        
        eprintln!("Best move: ({},{}) Q={:.4}, visits={}", best.0, best.1, best.4, best.2);
        eprintln!("2nd move:  ({},{}) Q={:.4}, visits={}", second.0, second.1, second.4, second.2);
        eprintln!("Q difference: {:.4}", q_diff);
        eprintln!("Visit ratio: {:.2}:1", visit_ratio);
        
        // With temp=0.06, if Q_diff=0.1, expected ratio ≈ exp(0.1/0.06) ≈ 5.3
        let expected_ratio = (q_diff / temperature as f64).exp();
        eprintln!("Expected visit ratio from softmax: {:.2}:1", expected_ratio);
        
        // Check if within reasonable bounds (softmax sampling has variance)
        let ratio_error = (visit_ratio / expected_ratio - 1.0).abs();
        if ratio_error > 0.5 {
            eprintln!("WARNING: Visit ratio deviates significantly from expected softmax!");
        } else {
            eprintln!("✓ Visit ratio roughly matches softmax expectations");
        }
    }
    
    eprintln!("\n=== Test Complete ===");
}
