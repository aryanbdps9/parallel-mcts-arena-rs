/// Diagnostic test to systematically verify GPU MCTS perspective semantics
/// 
/// This test will:
/// 1. Run a single MCTS iteration manually
/// 2. Trace exactly what happens in selection, expansion, rollout, backprop
/// 3. Verify the perspective at each step matches our expectations
/// 4. Print detailed diagnostics to identify the exact bug

use std::sync::Arc;

fn compute_othello_legal_moves(board: &[i32; 64], player: i32) -> Vec<(usize, usize)> {
    let mut moves = Vec::new();
    for y in 0..8 {
        for x in 0..8 {
            if is_valid_othello_move(board, x, y, player) {
                moves.push((x, y));
            }
        }
    }
    moves
}

fn is_valid_othello_move(board: &[i32; 64], x: usize, y: usize, player: i32) -> bool {
    if board[y * 8 + x] != 0 { return false; }
    let directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)];
    for (dx, dy) in directions.iter() {
        if would_flip(board, x as i32, y as i32, *dx, *dy, player) {
            return true;
        }
    }
    false
}

fn would_flip(board: &[i32; 64], x: i32, y: i32, dx: i32, dy: i32, player: i32) -> bool {
    let mut nx = x + dx;
    let mut ny = y + dy;
    let mut found_opponent = false;
    while nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
        let cell = board[(ny * 8 + nx) as usize];
        if cell == 0 { return false; }
        if cell == -player { found_opponent = true; }
        else if cell == player { return found_opponent; }
        nx += dx;
        ny += dy;
    }
    false
}

#[test]
fn test_gpu_perspective_diagnostic() {
    #[cfg(feature = "gpu")]
    {
        use mcts::gpu::{GpuContext, GpuConfig};
        
        println!("\n=== GPU PERSPECTIVE DIAGNOSTIC TEST ===\n");
        
        // Start from opening position
        let board = [0i32; 64];
        let mut board = board;
        board[27] = -1; // (3,3) white
        board[28] = 1;  // (4,3) black
        board[35] = 1;  // (3,4) black
        board[36] = -1; // (4,4) white
        
        let current_player = 1; // Black to move
        
        let legal_moves = compute_othello_legal_moves(&board, current_player);
        println!("Legal moves: {:?}", legal_moves);
        println!("Current player (to move): {}", current_player);
        
        // Initialize GPU
        let config = GpuConfig::default();
        let ctx = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
        
        // Create GPU MCTS engine
        let mcts = mcts::gpu::mcts_othello::GpuOthelloMcts::new(
            ctx,
            1_000_000,
            256,
        ).expect("Failed to create GPU MCTS");
        
        // Initialize with opening position
        mcts.init_tree(&board, current_player, &legal_moves);
        
        println!("\n--- STEP 1: Initial State ---");
        println!("Root player_at_node: {} (player to move)", current_player);
        
        // Dispatch kernel
        mcts.dispatch_mcts_othello_kernel(256, 1.4, 1.0, 0.06, 42);
        
        // Run exactly 256 iterations (one batch)
        println!("\n--- STEP 2: Running 256 iterations ---");
        mcts.run_iterations(256, 1.4, 1.0, 0.06, 42); // Use temp=0.06 like in actual game
        mcts.flush_and_wait();
        
        // Get children stats
        let children = mcts.get_children_stats();
        
        println!("\n--- STEP 3: After 64 iterations ---");
        println!("Root children ({} total):", children.len());
        for (x, y, visits, wins, q) in &children {
            let expected_q = if *visits > 0 {
                *wins as f64 / (*visits as f64 * 2.0)
            } else {
                0.0
            };
            println!("  ({},{}) visits={:5} wins={:5} Q={:.4} expected_Q={:.4}",
                x, y, visits, wins, q, expected_q);
        }
        
        // Now run many more iterations to see convergence
        println!("\n--- STEP 4: Running 4000 more iterations (16 batches) ---");
        for i in 0..16 {
            mcts.dispatch_mcts_othello_kernel(256, 1.4, 1.0, 0.06, 1000 + i);
            mcts.run_iterations(256, 1.4, 1.0, 0.06, 1000 + i); // Use temp=0.06
        }
        
        mcts.flush_and_wait();
        
        let children = mcts.get_children_stats();
        println!("\nAfter 4064 total iterations:");
        for (x, y, visits, wins, q) in &children {
            let wr = *wins as f64 / (*visits as f64 * 2.0);
            println!("  ({},{}) visits={:5} wins={:5} Q={:.4} win_rate={:.4}",
                x, y, visits, wins, q, wr);
        }
        
        // Calculate visit distribution variance
        let total_visits: i32 = children.iter().map(|(_, _, v, _, _)| v).sum();
        let avg_visits = total_visits as f64 / children.len() as f64;
        let variance: f64 = children.iter()
            .map(|(_, _, v, _, _)| {
                let diff = *v as f64 - avg_visits;
                diff * diff
            })
            .sum::<f64>() / children.len() as f64;
        let std_dev = variance.sqrt();
        let cv = std_dev / avg_visits; // coefficient of variation
        
        println!("\n--- STEP 5: Visit Distribution Analysis ---");
        println!("Total child visits: {}", total_visits);
        println!("Average visits per child: {:.1}", avg_visits);
        println!("Standard deviation: {:.1}", std_dev);
        println!("Coefficient of variation: {:.4}", cv);
        println!("");
        println!("Interpretation:");
        println!("  CV < 0.05: Nearly uniform (likely a bug)");
        println!("  CV 0.1-0.3: Good MCTS behavior (visits concentrate on best moves)");
        println!("  CV > 0.5: Extreme concentration (one move dominates)");
        
        println!("\n--- STEP 6: Q-value Analysis ---");
        let avg_q: f64 = children.iter().map(|(_, _, _, _, q)| q).sum::<f64>() / children.len() as f64;
        let min_q = children.iter().map(|(_, _, _, _, q)| q).fold(f64::INFINITY, |a, &b| a.min(b));
        let max_q = children.iter().map(|(_, _, _, _, q)| q).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        println!("Q-value range: [{:.4}, {:.4}]", min_q, max_q);
        println!("Q-value spread: {:.4}", max_q - min_q);
        println!("Average Q: {:.4}", avg_q);
        println!("");
        println!("Expected for balanced opening:");
        println!("  Average Q ≈ 0.50 (symmetric position)");
        println!("  Spread > 0.02 (moves have different values)");
        
        // Determine likely issue
        println!("\n=== DIAGNOSIS ===");
        if cv < 0.05 {
            println!("❌ ISSUE: Nearly uniform visits (CV={:.4})", cv);
            if max_q - min_q < 0.01 {
                println!("   → All Q-values are identical → Backpropagation bug");
                println!("   → Wins are not being accumulated correctly");
            } else {
                println!("   → Q-values differ but visits don't → Selection bug");
                println!("   → PUCT scores are not reflecting Q differences");
            }
        } else if (avg_q - 0.5).abs() > 0.1 {
            println!("❌ ISSUE: Average Q={:.4} far from 0.50", avg_q);
            println!("   → Perspective bug: wins accumulated for wrong player");
        } else {
            println!("✓ Visits and Q-values look reasonable");
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU features not enabled");
    }
}
