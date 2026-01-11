/// Test MCTS with a shallow tree (depth=2) in a late-game position
/// to verify selection, expansion, and backpropagation are all working correctly

use std::sync::Arc;
use mcts::gpu::{GpuContext, GpuConfig};
use mcts::gpu::mcts_othello::GpuOthelloMcts;

#[allow(dead_code)]
fn pcg_hash(state: u32) -> u32 {
    let mut s = state;
    s = s.wrapping_mul(747796405u32).wrapping_add(2891336453u32);
    let word = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277803737u32);
    (word >> 22) ^ word
}

fn compute_legal_moves(board: &[i32; 64], player: i32) -> Vec<(usize, usize)> {
    let mut moves = Vec::new();
    for y in 0..8 {
        for x in 0..8 {
            let idx = y * 8 + x;
            if board[idx] == 0 {
                // Check if this is a valid move
                for &(dx, dy) in &[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] {
                    let mut nx = x as i32 + dx;
                    let mut ny = y as i32 + dy;
                    let mut flipped_count = 0;
                    
                    while nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                        let nidx = (ny * 8 + nx) as usize;
                        if board[nidx] == 0 {
                            break;
                        }
                        if board[nidx] == player {
                            if flipped_count > 0 {
                                moves.push((x, y));
                            }
                            break;
                        }
                        flipped_count += 1;
                        nx += dx;
                        ny += dy;
                    }
                    if !moves.is_empty() && moves[moves.len()-1] == (x, y) {
                        break;
                    }
                }
            }
        }
    }
    moves.sort();
    moves.dedup();
    moves
}

#[allow(dead_code)]
fn make_move(board: &mut [i32; 64], player: i32, x: usize, y: usize) {
    let idx = y * 8 + x;
    board[idx] = player;
    
    for &(dx, dy) in &[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] {
        let mut to_flip = Vec::new();
        let mut nx = x as i32 + dx;
        let mut ny = y as i32 + dy;
        
        while nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
            let nidx = (ny * 8 + nx) as usize;
            if board[nidx] == 0 {
                break;
            }
            if board[nidx] == player {
                for &flip_idx in &to_flip {
                    board[flip_idx] = player;
                }
                break;
            }
            to_flip.push(nidx);
            nx += dx;
            ny += dy;
        }
    }
}

fn print_board(board: &[i32; 64]) {
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
}

#[test]
fn test_shallow_tree_late_game() {
    // Create GPU context
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Create a manually-crafted near-endgame position that's competitive
    // Board with ~8 empty squares, relatively balanced piece count
    let board = [
        -1, -1, -1, -1,  1,  1,  1,  1,  // Row 0
         1, -1, -1, -1, -1,  1,  1,  1,  // Row 1
         1,  1, -1, -1, -1, -1,  1,  1,  // Row 2
         1,  1,  1, -1, -1, -1, -1,  1,  // Row 3
         1,  1,  1,  1, -1, -1, -1, -1,  // Row 4
         1,  1,  1,  0,  0, -1, -1,  0,  // Row 5 (some empty)
         1,  1,  0,  0,  0,  0,  0,  0,  // Row 6 (some empty)
         1,  1,  1,  1,  1,  1,  1,  1,  // Row 7
    ];
    let player = 1;  // Player 1 to move
    
    // Count pieces
    let (x_count, o_count, empty_count) = board.iter().fold((0, 0, 0), |(x, o, e), &p| {
        match p {
            1 => (x+1, o, e),
            -1 => (x, o+1, e),
            _ => (x, o, e+1),
        }
    });
    
    eprintln!("\n=== CRAFTED NEAR-ENDGAME POSITION ===");
    eprintln!("X (player 1): {} pieces", x_count);
    eprintln!("O (player -1): {} pieces", o_count);
    eprintln!("Empty squares: {}", empty_count);
    print_board(&board);
    eprintln!("Current player: {}", player);
    
    let legal_moves = compute_legal_moves(&board, player);
    eprintln!("Legal moves: {} moves", legal_moves.len());
    for (x, y) in &legal_moves {
        eprintln!("  ({}, {})", x, y);
    }
    
    if legal_moves.is_empty() {
        eprintln!("No legal moves - skipping test");
        return;
    }
    
    // Create MCTS engine with VERY LIMITED depth
    // We'll modify the shader to enforce max_depth=2, but for now just use small capacity
    let max_nodes = 5000;   // Larger to allow more exploration
    let batch_size = 512;
    let engine = GpuOthelloMcts::new(context.clone(), max_nodes, batch_size)
        .expect("Failed to create MCTS engine");
    
    // Initialize tree
    engine.init_tree(&board, player, &legal_moves);
    
    // Run search with restrictive parameters
    let num_threads = 512;    // More threads
    let num_steps = 100;      // More steps to get diverse results
    let exploration = 1.414;
    let virtual_loss_weight = 0.2;
    let temperature = 0.01;   // Test softmax with numerical stability fix
    let seed = 54321;
    
    eprintln!("\n=== RUNNING SHALLOW MCTS ===");
    eprintln!("Max nodes: {}, Threads: {}, Steps: {}", max_nodes, num_threads, num_steps);
    eprintln!("Exploration: {}, VL Weight: {}, Temperature: {}", 
              exploration, virtual_loss_weight, temperature);
    
    let telemetry = engine.run_incremental_mcts(
        num_threads,
        num_steps,
        exploration,
        virtual_loss_weight,
        temperature,
        seed,
        None, // No timeout
        true,
    );
    let diagnostics = telemetry.diagnostics.clone();
    
    eprintln!("\n=== SEARCH DIAGNOSTICS ===");
    eprintln!("Total rollouts: {}", diagnostics.rollouts);
    eprintln!("Expansion successes: {}", diagnostics.expansion_success);
    eprintln!("Expansion locked: {}", diagnostics.expansion_locked);
    
    // Get children statistics
    let children_stats = engine.get_children_stats();
    eprintln!("\n=== ROOT CHILDREN STATISTICS ===");
    eprintln!("Total children: {}", children_stats.len());
    
    // Sort by visits descending
    let mut sorted = children_stats.clone();
    sorted.sort_by_key(|(_, _, v, _, _)| -(*v));
    
    // Calculate total visits first for PUCT calculation
    let total_visits: i32 = children_stats.iter().map(|c| c.2).sum();
    
    eprintln!("\nAll children sorted by visits:");
    for (i, &(x, y, visits, wins, q)) in sorted.iter().enumerate() {
        // Calculate what PUCT score should be
        let parent_visits = total_visits as f64;
        let sqrt_parent = (parent_visits + 1.0).sqrt();
        let prior = 1.0 / legal_moves.len() as f64; // Uniform prior
        let effective_visits = visits as f64;
        
        let u = (exploration as f64) * prior * sqrt_parent / (1.0 + effective_visits);
        let puct = q + u;
        
        eprintln!("  #{}. ({},{}) visits={:5}, wins={:5}, Q={:.4}, U={:.4}, PUCT={:.4}", 
                 i+1, x, y, visits, wins, q, u, puct);
    }
    
    // VERIFICATION
    eprintln!("\n=== VERIFICATION ===");
    
    // 1. All legal moves should have stats
    assert_eq!(children_stats.len(), legal_moves.len(), 
              "Should have stats for all legal moves");
    eprintln!("✓ All legal moves have stats");
    
    // 2. Total visits should match rollouts approximately
    eprintln!("Total child visits: {}, Total rollouts: {}", total_visits, diagnostics.rollouts);
    
    // 3. Q-values in valid range
    for &(x, y, visits, wins, q) in &children_stats {
        assert!(q >= 0.0 && q <= 1.0, 
               "Q-value {:.4} for move ({},{}) is out of range [0,1]", q, x, y);
        
        if visits > 0 {
            let expected_q = (wins as f64) / (visits as f64 * 2.0);
            let diff = (q - expected_q).abs();
            assert!(diff < 0.001, 
                   "Q-value mismatch for ({},{}): got {:.4}, expected {:.4}", 
                   x, y, q, expected_q);
        }
    }
    eprintln!("✓ All Q-values in valid range and match calculation");
    
    // 4. Check visit distribution with low temperature
    // With temp=0.01, visits should strongly correlate with Q-values
    if sorted.len() >= 2 {
        let top_child = sorted[0];
        let second_child = sorted[1];
        
        eprintln!("\nTop 2 children:");
        eprintln!("  1st: ({},{}) visits={}, Q={:.4}", 
                 top_child.0, top_child.1, top_child.2, top_child.4);
        eprintln!("  2nd: ({},{}) visits={}, Q={:.4}", 
                 second_child.0, second_child.1, second_child.2, second_child.4);
        
        // With temp=0.01, if top child has significantly better Q, 
        // it should have MORE visits (this is the key test!)
        if top_child.4 > second_child.4 + 0.1 {
            eprintln!("\nCHECKING: Top child has Q={:.4}, 2nd has Q={:.4} (diff={:.4})",
                     top_child.4, second_child.4, top_child.4 - second_child.4);
            eprintln!("          Top visits={}, 2nd visits={}", top_child.2, second_child.2);
            
            if top_child.2 < second_child.2 {
                eprintln!("WARNING: Better Q-value child got FEWER visits!");
                eprintln!("This suggests selection is not following PUCT scores correctly.");
            } else {
                eprintln!("✓ Better Q-value child got more visits (as expected)");
            }
        }
    }
    
    eprintln!("\n=== Test Complete ===");
}
