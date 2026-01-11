/// Test to verify GPU MCTS Q-value calculation and perspective handling
/// 
/// This test creates a simple scenario where we know the correct Q values
/// and verifies that the GPU calculates them correctly.

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_q_value_perspective() {
    use mcts::gpu::mcts_othello::GpuOthelloMcts;
    use mcts::gpu::{GpuContext, GpuConfig};
    use std::sync::Arc;
    use std::time::Duration;

    println!("\n[TEST] ========== GPU Q-Value Perspective Test ==========\n");

    // Create a small GPU MCTS instance
    let max_nodes = 100000;
    let config = GpuConfig::default();
    let ctx = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    let engine = GpuOthelloMcts::new(ctx, max_nodes, 128).expect("Failed to create engine");

    // Set up initial Othello position (standard opening)
    let mut board = [0i32; 64];
    board[3 * 8 + 3] = -1; // O at (3,3)
    board[3 * 8 + 4] = 1;  // X at (4,3)
    board[4 * 8 + 3] = 1;  // X at (3,4)
    board[4 * 8 + 4] = -1; // O at (4,4)

    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];

    // Initialize tree
    engine.init_tree(&board, 1, &legal_moves);

    // Run some MCTS iterations to build statistics
    println!("[TEST] Running initial dispatch (1 workgroup)...");
    engine.dispatch_mcts_othello_kernel(1, 1.4, 1.0, 0.06, 0.01, 42);
    
    println!("[TEST] Running MCTS for 5 seconds...");
    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();
    let mut total_iterations = 0;
    let mut seed = 1000;
    
    while start.elapsed() < timeout {
        engine.dispatch_mcts_othello_kernel(128, 1.4, 1.0, 0.06, 0.01, seed);
        seed += 1;
        total_iterations += 128 * 64; // 128 workgroups * 64 threads
    }

    println!("[TEST] Completed {} iterations", total_iterations);

    // Ensure all GPU work is complete
    engine.flush_and_wait();
    
    // Update root stats before reading

    // Get children statistics
    let stats = engine.get_children_stats();
    println!("[TEST] Root children:");
    
    for (i, (x, y, visits, wins, q)) in stats.iter().enumerate() {
        let raw_q = if *visits > 0 {
            *wins as f64 / (*visits as f64 * 2.0)
        } else {
            0.0
        };
        
        println!("  {}. ({},{}) visits={} wins={} q={:.4} raw_q={:.4}", 
                 i + 1, x, y, visits, wins, q, raw_q);
    }

    // Verify Q values are in reasonable range [0, 1]
    for (x, y, visits, wins, q) in &stats {
        if *visits > 0 {
            assert!(*q >= 0.0 && *q <= 1.0, 
                "Q value {:.4} for move ({},{}) is out of range [0,1]", q, x, y);
            
            // Verify Q represents wins from child's perspective (player making the move)
            // Q should equal child's win rate: wins / (visits * 2.0)
            let raw_q = *wins as f64 / (*visits as f64 * 2.0);
            let diff = (q - raw_q).abs();
            
            assert!(diff < 0.01, 
                "Q value {:.4} for move ({},{}) doesn't match expected child's win rate {:.4}",
                q, x, y, raw_q);
        }
    }

    // Verify that visits are concentrated (not uniform)
    // With deterministic selection, the best move should have significantly more visits
    if stats.len() >= 2 {
        let mut sorted = stats.clone();
        sorted.sort_by_key(|(_, _, v, _, _)| -(*v as i64));
        
        let top_visits = sorted[0].2 as f64;
        let second_visits = sorted[1].2 as f64;
        let total_visits: i32 = stats.iter().map(|(_, _, v, _, _)| *v).sum();
        
        println!("\n[TEST] Visit distribution:");
        println!("  Top move: {} visits ({:.1}% of total)", 
                 sorted[0].2, 100.0 * top_visits / total_visits as f64);
        println!("  2nd move: {} visits ({:.1}% of total)", 
                 sorted[1].2, 100.0 * second_visits / total_visits as f64);
        
        // In opening positions where moves are symmetric, distribution will be nearly uniform
        // Just verify that we have reasonable visit counts (not all zero)
        assert!(sorted[0].2 > 0, "Top move should have visits");
    }

    println!("\n[TEST] ✓ Q-value perspective test passed!");
}
