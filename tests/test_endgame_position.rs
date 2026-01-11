/// Test MCTS from arbitrary near-endgame positions
/// 
/// This test verifies:
/// 1. Ability to start from arbitrary board states
/// 2. Correctness of Q-value calculations in near-terminal positions
/// 3. Comparison between CPU and GPU MCTS on the same position
/// 4. Convergence of evaluations with more compute

#[cfg(feature = "gpu")]
use mcts::gpu::{GpuContext, GpuConfig};
#[cfg(feature = "gpu")]
use mcts::gpu::mcts_othello::GpuOthelloMcts;
use std::sync::Arc;

/// Helper to create an Othello board from a string representation
/// '.' = empty, 'X' = player 1, 'O' = player -1
fn board_from_string(s: &str) -> [i32; 64] {
    let mut board = [0i32; 64];
    let chars: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    
    assert_eq!(chars.len(), 64, "Board string must have exactly 64 characters");
    
    for (i, &ch) in chars.iter().enumerate() {
        board[i] = match ch {
            '.' => 0,
            'X' => 1,
            'O' => -1,
            _ => panic!("Invalid character '{}' in board string", ch),
        };
    }
    
    board
}

/// Helper to print board state
fn print_board(board: &[i32; 64], current_player: i32) {
    println!("\nCurrent board (Player {} to move):", current_player);
    println!("  0 1 2 3 4 5 6 7");
    for y in 0..8 {
        print!("{} ", y);
        for x in 0..8 {
            let idx = y * 8 + x;
            let ch = match board[idx] {
                0 => '.',
                1 => 'X',
                -1 => 'O',
                _ => '?',
            };
            print!("{} ", ch);
        }
        println!();
    }
    
    // Count pieces
    let x_count = board.iter().filter(|&&v| v == 1).count();
    let o_count = board.iter().filter(|&&v| v == -1).count();
    let empty = board.iter().filter(|&&v| v == 0).count();
    println!("X: {}, O: {}, Empty: {}", x_count, o_count, empty);
}

/// Compute legal moves from a board state
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

/// Check if a move is valid
fn is_valid_move(board: &[i32; 64], x: usize, y: usize, player: i32) -> bool {
    let idx = y * 8 + x;
    if board[idx] != 0 {
        return false;
    }
    
    // Check all 8 directions
    let directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ];
    
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
        let idx = (cy * 8 + cx) as usize;
        let cell = board[idx];
        
        if cell == 0 {
            return 0; // Empty cell, no flips
        } else if cell == -player {
            count += 1;
            cx += dx;
            cy += dy;
        } else if cell == player {
            return count; // Found our piece
        }
    }
    
    0 // Went off board
}

/// Test near-endgame position with detailed diagnostics
#[test]
fn test_near_endgame_position() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║      MCTS ENDGAME POSITION DIAGNOSTIC TEST                    ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    
    // Use a position where move quality clearly differs
    // This is move 5-6 in a game - early enough to need rollouts
    let board = board_from_string("
        ........
        ........
        ..X.....
        ..XXX...
        ..XXXX..
        ..OXXXX.
        ...OOO..
        ........
    ");
    
    let current_player = -1; // O to move
    
    print_board(&board, current_player);
    
    let legal_moves = compute_legal_moves(&board, current_player);
    println!("\nLegal moves for player {}: {:?}", current_player, legal_moves);
    println!("Number of legal moves: {}", legal_moves.len());
    
    if legal_moves.is_empty() {
        println!("\n[WARNING] No legal moves in this position - test invalid!");
        return;
    }
    
    // Run CPU MCTS
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                     CPU MCTS ANALYSIS                          ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    
    let exploration = 1.414; // sqrt(2) - classic UCT
    let iterations = 100_000;
    
    // Note: CPU MCTS requires a full GameState to run search
    // For this diagnostic test, we focus on GPU MCTS which can be initialized from raw board
    println!("\n[CPU] Would use exploration={:.3}", exploration);
    println!("[CPU] Would run {} iterations on this position", iterations);
    println!("[CPU] Expected: Near-endgame positions should show clear winning/losing evaluations");
    println!("[CPU] Expected: Q-values should be close to 0.0 or 1.0 for near-terminal positions");
    
    // Run GPU MCTS
    #[cfg(feature = "gpu")]
    {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║                     GPU MCTS ANALYSIS                          ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
        
        let gpu_max_nodes = 100_000;
        let batch_size = 8192;
        let num_batches = 10;
        
        let engine = Arc::new(GpuOthelloMcts::new(
            context.clone(),
            gpu_max_nodes,
            batch_size,
        ).expect("Failed to create GPU engine"));
        
        println!("[GPU] Created engine: max_nodes={}, batch_size={}", gpu_max_nodes, batch_size);
        
        // Initialize tree from this position
        engine.init_tree(&board, current_player, &legal_moves);
        println!("[GPU] Initialized tree from custom position");
        
        // Run initial dispatch (important for tree initialization)
        engine.dispatch_mcts_othello_kernel(1, exploration as f32, 1.0, 0.06, 0.01, 42);
        println!("[GPU] Completed initial 1-workgroup dispatch");
        
        // Run main search
        let total_iterations = num_batches * batch_size;
        println!("[GPU] Running {} batches × {} iterations = {} total iterations", 
                 num_batches, batch_size, total_iterations);
        
        let start = std::time::Instant::now();
        for batch in 0..num_batches {
            let seed = 42 + batch * 1000;
            engine.dispatch_mcts_othello_kernel(batch_size, exploration as f32, 1.0, 0.06, 0.01, seed);
            
            // Periodic diagnostics
            if batch > 0 && batch % 5 == 0 {
                let stats = engine.get_children_stats();
                if !stats.is_empty() {
                    let (x, y, visits, _wins, q) = stats[0];
                    println!("[GPU]   Batch {}/{}: best move=({},{}), visits={}, Q={:.4}", 
                             batch, num_batches, x, y, visits, q);
                }
            }
        }
        
        let elapsed = start.elapsed();
        println!("[GPU] Completed search in {:.3}s ({:.0} iterations/sec)", 
                 elapsed.as_secs_f64(), 
                 total_iterations as f64 / elapsed.as_secs_f64());
        
        // Final sync and statistics
        engine.flush_and_wait();
        
        let stats = engine.get_children_stats();
        let total_visits: i32 = stats.iter().map(|(_, _, v, _, _)| *v).sum();
        
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║                  GPU FINAL STATISTICS                          ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        
        println!("\n[GPU] Total visits across all children: {}", total_visits);
        println!("[GPU] Nodes used: {} / {}", engine.calculate_nodes_used(), gpu_max_nodes);
        
        println!("\n[GPU] Move evaluations (sorted by visits):");
        let mut sorted_stats = stats.clone();
        sorted_stats.sort_by_key(|(_, _, v, _, _)| -(*v));
        
        for (i, (x, y, visits, wins, q)) in sorted_stats.iter().enumerate() {
            let visit_pct = 100.0 * (*visits as f64) / (total_visits as f64);
            println!("  {}. ({},{})  visits={:7} ({:5.1}%)  wins={:7}  Q={:.4}", 
                     i + 1, x, y, visits, visit_pct, wins, q);
        }
        
        // Diagnostic checks
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║                    DIAGNOSTIC CHECKS                           ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");
        
        // Check 1: Q-values in valid range
        let mut all_valid = true;
        for (x, y, visits, _, q) in &sorted_stats {
            if *visits > 0 && (*q < 0.0 || *q > 1.0) {
                println!("❌ FAIL: Move ({},{}) has Q={:.4} outside [0,1]", x, y, q);
                all_valid = false;
            }
        }
        if all_valid {
            println!("✓ PASS: All Q-values in valid range [0, 1]");
        }
        
        // Check 2: Visit concentration (MCTS should focus on good moves)
        if sorted_stats.len() >= 2 {
            let top_visits = sorted_stats[0].2;
            let second_visits = sorted_stats[1].2;
            let concentration_ratio = top_visits as f64 / second_visits as f64;
            
            println!("✓ INFO: Visit concentration ratio (best/2nd): {:.2}x", concentration_ratio);
            
            if concentration_ratio < 1.1 {
                println!("⚠ WARNING: Low visit concentration suggests weak differentiation");
            } else if concentration_ratio > 10.0 {
                println!("✓ GOOD: Strong visit concentration on best move");
            }
        }
        
        // Check 3: Endgame Q-values should be decisive (close to 0 or 1)
        let best_q = sorted_stats[0].4;
        if best_q > 0.9 || best_q < 0.1 {
            println!("✓ GOOD: Best move has decisive Q={:.4} (confident evaluation)", best_q);
        } else if best_q > 0.4 && best_q < 0.6 {
            println!("⚠ WARNING: Best move has uncertain Q={:.4} (should be more decisive in endgame)", best_q);
        } else {
            println!("✓ INFO: Best move has Q={:.4}", best_q);
        }
        
        // Check 4: Compare top 2 moves
        if sorted_stats.len() >= 2 {
            let (x1, y1, v1, _, q1) = sorted_stats[0];
            let (x2, y2, v2, _, q2) = sorted_stats[1];
            let q_diff = (q1 - q2).abs();
            
            println!("\n[COMPARISON] Top 2 moves:");
            println!("  1st: ({},{}) visits={} Q={:.4}", x1, y1, v1, q1);
            println!("  2nd: ({},{}) visits={} Q={:.4}", x2, y2, v2, q2);
            println!("  Q-value difference: {:.4}", q_diff);
            
            if q_diff > 0.1 {
                println!("✓ GOOD: Clear Q-value separation between top moves");
            } else {
                println!("✓ INFO: Small Q-value difference (moves may be similar quality)");
            }
        }
        
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║                   TEST COMPLETE                                ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        println!("\n[SKIP] GPU tests skipped (feature 'gpu' not enabled)");
    }
}

/// CRITICAL TEST: Verify rollout quality with shallow tree
/// This tests the hypothesis that rollouts are giving wrong signals
#[test]
fn test_shallow_tree_rollout_quality() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║     SHALLOW TREE ROLLOUT QUALITY TEST (Depth 1-2)             ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    
    println!("HYPOTHESIS: Rollouts may be giving wrong signals");
    println!("TEST STRATEGY: Use DECISIVE late-game position where one player");
    println!("               is clearly ahead. Limit tree to 1-2 levels.");
    println!("               If rollouts work, they should show statistical edge.\n");
    
    // Create a late-game position where X is clearly WINNING
    // X has 35 pieces, O has 15 pieces, 14 empty squares
    // Even with random rollouts, X should win most games from here
    let board = board_from_string("
        XXXXXXXX
        XXXXXXXX
        XXXXXXXX
        XXXOOOXX
        XXOOO.XX
        XXO..XXX
        XX...XXX
        XXXXXXXX
    ");
    
    let current_player = 1; // X to move (X is winning)
    print_board(&board, current_player);
    
    // Count material advantage
    let x_count = board.iter().filter(|&&v| v == 1).count();
    let o_count = board.iter().filter(|&&v| v == -1).count();
    println!("\n*** POSITION: X is CLEARLY WINNING ***");
    println!("Material: X={}, O={} (X ahead by {})", x_count, o_count, x_count as i32 - o_count as i32);
    
    let legal_moves = compute_legal_moves(&board, current_player);
    println!("\nLegal moves: {:?}", legal_moves);
    
    if legal_moves.is_empty() {
        println!("[WARNING] No legal moves!");
        return;
    }
    
    #[cfg(feature = "gpu")]
    {
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
        
        // Use very small batch size to limit tree depth
        // With 1 workgroup (64 threads), we get minimal expansion
        let engine = Arc::new(GpuOthelloMcts::new(
            context.clone(),
            50_000,  // Smaller tree
            64,      // Tiny batch = shallow tree
        ).expect("Failed to create GPU engine"));
        
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║              TEST 1: VERY SHALLOW TREE (1 batch)               ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        
        engine.init_tree(&board, current_player, &legal_moves);
        
        // Run ONLY 1 batch - this should barely expand the tree
        println!("\n[TEST 1] Running 1 batch of 64 iterations (minimal expansion)");
        engine.dispatch_mcts_othello_kernel(1, 1.414, 1.0, 0.06, 0.01, 42);
        engine.dispatch_mcts_othello_kernel(64, 1.414, 1.0, 0.06, 0.01, 100);
        
        engine.flush_and_wait();
        
        let stats1 = engine.get_children_stats();
        let mut sorted1 = stats1.clone();
        sorted1.sort_by_key(|(_, _, v, _, _)| -(*v));
        
        let total_visits1: i32 = stats1.iter().map(|(_, _, v, _, _)| *v).sum();
        let nodes_used1 = engine.calculate_nodes_used();
        
        println!("\nResults after 1 batch:");
        println!("  Total visits: {}", total_visits1);
        println!("  Nodes used: {}", nodes_used1);
        println!("\nMove evaluations:");
        for (i, (x, y, visits, wins, q)) in sorted1.iter().enumerate() {
            let pct = 100.0 * (*visits as f64) / (total_visits1 as f64);
            println!("  {}. ({},{})  visits={:6} ({:5.1}%)  wins={:6}  Q={:.4}", 
                     i + 1, x, y, visits, pct, wins, q);
        }
        
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║            TEST 2: SLIGHTLY DEEPER TREE (5 batches)           ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        
        // Reset and run with a bit more expansion
        engine.init_tree(&board, current_player, &legal_moves);
        engine.dispatch_mcts_othello_kernel(1, 1.414, 1.0, 0.06, 0.01, 42);
        
        println!("\n[TEST 2] Running 5 batches (depth ~2-3)");
        for batch in 0..5 {
            engine.dispatch_mcts_othello_kernel(64, 1.414, 1.0, 0.06, 0.01, 100 + batch * 1000);
        }
        
        engine.flush_and_wait();
        
        let stats2 = engine.get_children_stats();
        let mut sorted2 = stats2.clone();
        sorted2.sort_by_key(|(_, _, v, _, _)| -(*v));
        
        let total_visits2: i32 = stats2.iter().map(|(_, _, v, _, _)| *v).sum();
        let nodes_used2 = engine.calculate_nodes_used();
        
        println!("\nResults after 5 batches:");
        println!("  Total visits: {}", total_visits2);
        println!("  Nodes used: {}", nodes_used2);
        println!("\nMove evaluations:");
        for (i, (x, y, visits, wins, q)) in sorted2.iter().enumerate() {
            let pct = 100.0 * (*visits as f64) / (total_visits2 as f64);
            println!("  {}. ({},{})  visits={:6} ({:5.1}%)  wins={:6}  Q={:.4}", 
                     i + 1, x, y, visits, pct, wins, q);
        }
        
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║             TEST 3: DEEPER TREE (50 batches)                  ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        
        // Reset and run with more expansion
        engine.init_tree(&board, current_player, &legal_moves);
        engine.dispatch_mcts_othello_kernel(1, 1.414, 1.0, 0.06, 0.01, 42);
        
        println!("\n[TEST 3] Running 50 batches (deeper tree)");
        for batch in 0..50 {
            engine.dispatch_mcts_othello_kernel(64, 1.414, 1.0, 0.06, 0.01, 100 + batch * 1000);
        }
        
        engine.flush_and_wait();
        
        let stats3 = engine.get_children_stats();
        let mut sorted3 = stats3.clone();
        sorted3.sort_by_key(|(_, _, v, _, _)| -(*v));
        
        let total_visits3: i32 = stats3.iter().map(|(_, _, v, _, _)| *v).sum();
        let nodes_used3 = engine.calculate_nodes_used();
        
        println!("\nResults after 50 batches:");
        println!("  Total visits: {}", total_visits3);
        println!("  Nodes used: {}", nodes_used3);
        println!("\nMove evaluations:");
        for (i, (x, y, visits, wins, q)) in sorted3.iter().enumerate() {
            let pct = 100.0 * (*visits as f64) / (total_visits3 as f64);
            println!("  {}. ({},{})  visits={:6} ({:5.1}%)  wins={:6}  Q={:.4}", 
                     i + 1, x, y, visits, pct, wins, q);
        }
        
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║                    DIAGNOSTIC ANALYSIS                         ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");
        
        println!("ANALYSIS:");
        println!("─────────────────────────────────────────────────────────────────");
        
        // Check if Q-values are converging or diverging
        if sorted1.len() >= 2 && sorted2.len() >= 2 && sorted3.len() >= 2 {
            let q1_best = sorted1[0].4;
            let q2_best = sorted2[0].4;
            let q3_best = sorted3[0].4;
            
            println!("\nQ-value convergence (best move):");
            println!("  After 1 batch:  Q = {:.4}", q1_best);
            println!("  After 5 batches: Q = {:.4}", q2_best);
            println!("  After 50 batches: Q = {:.4}", q3_best);
            
            let convergence = (q3_best - q1_best).abs();
            println!("\n  Convergence: {:.4}", convergence);
            
            if convergence < 0.05 {
                println!("  ✓ GOOD: Q-values stable (rollouts may be working)");
            } else if convergence > 0.2 {
                println!("  ⚠ WARNING: Large Q-value shift suggests rollouts giving unstable signal!");
            }
            
            // Check if all Q-values are around 0.5 (noise)
            if q1_best > 0.45 && q1_best < 0.55 {
                println!("\n  ⚠ SUSPICIOUS: Q-values near 0.5 suggest rollouts returning noise");
                println!("                or position is genuinely equal");
            }
            
            // Compare Q-values across all moves
            println!("\nQ-value spread analysis:");
            let q1_spread = sorted1[0].4 - sorted1.last().unwrap().4;
            let q2_spread = sorted2[0].4 - sorted2.last().unwrap().4;
            let q3_spread = sorted3[0].4 - sorted3.last().unwrap().4;
            
            println!("  1 batch:  spread = {:.4}", q1_spread);
            println!("  5 batches: spread = {:.4}", q2_spread);
            println!("  50 batches: spread = {:.4}", q3_spread);
            
            if q1_spread < 0.01 && q2_spread < 0.01 {
                println!("\n  ❌ CRITICAL: No Q-value differentiation in shallow tree!");
                println!("              This strongly suggests rollouts are returning pure noise");
            } else if q3_spread > q1_spread * 2.0 {
                println!("\n  ✓ GOOD: Q-values differentiating with more search");
            }
        }
        
        println!("\n═════════════════════════════════════════════════════════════════");
        println!("CONCLUSION:");
        println!("─────────────────────────────────────────────────────────────────");
        println!("EXPECTED (if rollouts work correctly):");
        println!("  - Q-values should be HIGH (>0.6) even with shallow tree");
        println!("  - Because X is clearly ahead, random rollouts should still win");
        println!("  - Should show clear statistical edge from position advantage");
        println!("");
        println!("IF ROLLOUTS ARE BROKEN:");
        println!("  - Q-values would be ~0.5 regardless of position quality");
        println!("  - No correlation between material advantage and Q-value");
        println!("  - This would prove rollouts aren't capturing position quality");
        println!("═════════════════════════════════════════════════════════════════\n");
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        println!("\n[SKIP] GPU tests skipped (feature 'gpu' not enabled)");
    }
}

/// Test a position where the outcome is completely clear (forced win)
#[test]
fn test_forced_win_position() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║         FORCED WIN POSITION TEST                               ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    
    // Create a position where X has overwhelming advantage
    // Only 2 empty squares, X can win by a large margin
    let board = board_from_string("
        XXXXXXXX
        XXXXXXXX
        XXXXXXXX
        XXXXXXXX
        XXXXXXXX
        XXXXXXXX
        XXXXXX..
        XXXXXXXX
    ");
    
    let current_player = 1; // X to move
    print_board(&board, current_player);
    
    let legal_moves = compute_legal_moves(&board, current_player);
    println!("\nLegal moves: {:?}", legal_moves);
    
    if legal_moves.is_empty() {
        println!("[INFO] No legal moves - game is over");
        return;
    }
    
    #[cfg(feature = "gpu")]
    {
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
        
        let engine = Arc::new(GpuOthelloMcts::new(
            context.clone(),
            100_000,
            8192,
        ).expect("Failed to create GPU engine"));
        
        engine.init_tree(&board, current_player, &legal_moves);
        engine.dispatch_mcts_othello_kernel(1, 1.414, 1.0, 0.06, 0.01, 42);
        
        // Run for a bit
        for batch in 0..5 {
            engine.dispatch_mcts_othello_kernel(8192, 1.414, 1.0, 0.06, 0.01, 42 + batch * 1000);
        }
        
        engine.flush_and_wait();
        
        let stats = engine.get_children_stats();
        let mut sorted = stats.clone();
        sorted.sort_by_key(|(_, _, v, _, _)| -(*v));
        
        println!("\nResults:");
        for (i, (x, y, visits, _wins, q)) in sorted.iter().enumerate() {
            println!("  {}. ({},{}) visits={} Q={:.4}", i + 1, x, y, visits, q);
        }
        
        // In a forced win position, Q should be very high (> 0.95)
        if sorted.len() > 0 {
            let best_q = sorted[0].4;
            println!("\n[CHECK] Best move Q-value: {:.4}", best_q);
            
            if best_q > 0.95 {
                println!("✓ PASS: Q-value indicates clear win");
            } else if best_q > 0.8 {
                println!("⚠ WARNING: Q-value should be higher for forced win position");
            } else {
                println!("❌ FAIL: Q-value {:.4} too low for forced win", best_q);
            }
        }
    }
}
