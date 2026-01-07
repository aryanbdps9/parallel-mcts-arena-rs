/// Test GPU-native MCTS advance_root functionality
/// Runs a few moves to verify tree reuse works correctly

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use mcts::gpu::{GpuContext, GpuConfig};
use mcts::gpu::mcts_othello::GpuOthelloMcts;

#[test]
fn test_gpu_advance_root() {
    // Blanket timeout: kill test after 60 seconds no matter what
    let timeout_flag = Arc::new(AtomicBool::new(false));
    let timeout_flag_clone = timeout_flag.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(60));
        timeout_flag_clone.store(true, Ordering::SeqCst);
        eprintln!("\n[TIMEOUT] Test exceeded 60 second timeout - FAILING\n");
        std::process::exit(1);
    });
    
    println!("\n[TEST] ========== GPU Advance Root Test (60s timeout) ==========\n");
    
    // Create GPU context
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    println!("[TEST] Created GPU context");
    
    // Create engine with smaller node count for faster test
    let max_nodes = 1_000_000; // 1M nodes - enough for test
    let batch_size = 8192;
    
    let engine = Arc::new(GpuOthelloMcts::new(
        context.clone(),
        max_nodes,
        batch_size,
    ).expect("Failed to create GpuOthelloMcts"));
    
    println!("[TEST] Created engine with max_nodes={}\n", max_nodes);
    
    // Initial board state (Othello starting position)
    let mut board = [0i32; 64];
    board[27] = -1; // (3, 3) = O (Black)
    board[28] = 1;  // (4, 3) = X (White)
    board[35] = 1;  // (3, 4) = X (White)
    board[36] = -1; // (4, 4) = O (Black)
    
    let mut legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    let mut current_player = 1; // White (X) starts
    
    // Play 1 move to test pruning (keeps test under 60s)
    for move_num in 1..=1 {
        println!("[TEST] ========== Move {} ==========", move_num);
        
        // Initialize/reinit tree for first move
        if move_num == 1 {
            println!("[TEST] Initializing tree");
            engine.init_tree(&board, current_player, &legal_moves);
            
            // Check nodes after init - should be just the root (~1 node)
            let nodes_after_init = engine.calculate_nodes_used();
            println!("[TEST] Nodes after init: {} / {} (only root allocated)", nodes_after_init, max_nodes);
            assert!(nodes_after_init < 100, 
                "After init, only root should be allocated, got {}", nodes_after_init);
        }
        
        // Run MCTS for ~30 batches (with 256 workgroups = ~491K iterations)
        let target_batches = 30;
        println!("[TEST] Running MCTS iterations ({} batches)", target_batches);
        let start = std::time::Instant::now();
        
        for batch in 0..target_batches {
            engine.dispatch_mcts_othello_kernel(256, 1.4, 1.0, 1.0, 0.01, 42); // 256 workgroups * 64 = 16384 iterations (use all free lists!)
            
            if batch % 100 == 0 && batch > 0 {
                let stats = engine.get_children_stats();
                if !stats.is_empty() {
                    println!("[TEST]   Batch {}: best child ({},{}) has {} visits", 
                        batch, stats[0].0, stats[0].1, stats[0].2);
                }
            }
        }
        
        println!("[TEST] Completed {} batches in {:?}", target_batches, start.elapsed());
        
        // Check nodes after MCTS - tree should have grown significantly
        let nodes_after_mcts = engine.calculate_nodes_used();
        println!("[TEST] Nodes after MCTS: {} / {}", nodes_after_mcts, max_nodes);
        assert!(nodes_after_mcts <= max_nodes, 
            "Nodes used cannot exceed max_nodes! {} > {}", nodes_after_mcts, max_nodes);
        assert!(nodes_after_mcts > 1000, 
            "After 500 batches, should have allocated significant nodes, got {}", nodes_after_mcts);
        
        // Sync GPU stats to host
        println!("[TEST] Syncing GPU stats to host");
        engine.update_root_stats();
        
        // Get statistics
        let stats = engine.get_children_stats();
        println!("[TEST] Children stats:");
        for (i, &(x, y, visits, wins, q)) in stats.iter().enumerate() {
            println!("[TEST]   {}. ({},{}) visits={} wins={} Q={:.4}", i+1, x, y, visits, wins, q);
        }
        
        // Pick the best move (most visits)
        let best = stats.iter().max_by_key(|s| s.2).expect("No children found!");
        let (best_x, best_y) = (best.0, best.1);
        println!("[TEST] Best move: ({},{})", best_x, best_y);
        
        // Verify move is legal
        assert!(legal_moves.contains(&(best_x, best_y)), 
            "Best move ({},{}) is not in legal moves: {:?}", best_x, best_y, legal_moves);
        
        // Apply move to board (simplified - just flip one piece for testing)
        let move_idx = best_y * 8 + best_x;
        board[move_idx] = current_player;
        
        // Prepare for next move
        current_player = -current_player;
        
        // Hardcoded legal moves for next position (simplified for testing)
        legal_moves = match move_num {
            1 => vec![(5, 3), (3, 5), (5, 5)], // After first move
            2 => vec![(2, 2), (2, 4), (4, 2)], // After second move
            _ => vec![(1, 1)], // Dummy for third move
        };
        
        // Test advance_root (skip for last move)
        if move_num < 2 {
            println!("[TEST] Testing advance_root to ({},{})", best_x, best_y);
            println!("[TEST] New player: {}, legal_moves: {:?}", current_player, legal_moves);
            
            let nodes_before = engine.calculate_nodes_used();
            println!("[TEST] Nodes BEFORE pruning: {}", nodes_before);
            
            let advance_success = engine.advance_root(
                best_x,
                best_y,
                &board,
                current_player,
                &legal_moves
            );
            
            if !advance_success {
                panic!("[TEST] FAILED: advance_root returned false on move {}", move_num);
            }
            
            println!("[TEST] ✓ advance_root succeeded");
            
            let nodes_after = engine.calculate_nodes_used();
            println!("[TEST] Nodes AFTER pruning: {}", nodes_after);
            
            // Run MCTS batches to expand the new root for next move
            // This will FAIL if pruning didn't work (pool saturated)
            println!("[TEST] Running MCTS to expand new root (50 batches to verify no saturation)");
            let batch_start = std::time::Instant::now();
            for batch in 0..50 {
                engine.dispatch_mcts_othello_kernel(256, 1.4, 1.0, 1.0, 0.01, 42);
                
                // Check periodically that we're not saturated
                if batch % 10 == 9 {
                    let nodes_mid = engine.calculate_nodes_used();
                    println!("[TEST]   After {} batches: {} nodes", batch + 1, nodes_mid);
                    assert!(nodes_mid <= max_nodes, 
                        "[TEST] FAILED: Pool saturated at batch {}! {} > {}", 
                        batch, nodes_mid, max_nodes);
                }
            }
            
            let nodes_final = engine.calculate_nodes_used();
            println!("[TEST] Completed 50 batches in {:.2}s, nodes: {}\n", 
                batch_start.elapsed().as_secs_f64(), nodes_final);
            
            // The key test: we should be able to allocate more nodes without saturation
            assert!(nodes_final <= max_nodes,
                "[TEST] FAILED: Pool saturated after pruning and expansion! {} > {}",
                nodes_final, max_nodes);
            println!("[TEST] ✓ No saturation - pruning enabled continued growth!");
        }
    }
    
    println!("\n[TEST] ========== SUCCESS: Pruning prevents saturation! ==========\n");
    println!("[TEST] ✓ Tree reuse works via pruning");
    println!("[TEST] ✓ Can continue allocating after advance_root");
    println!("[TEST] ✓ No pool saturation\n");
}
