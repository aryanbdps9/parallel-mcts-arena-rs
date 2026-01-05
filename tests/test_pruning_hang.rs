/// Test to reproduce the pruning hang issue
/// This test simulates the exact sequence that causes the hang:
/// 1. Create GPU-native engine with 8M nodes
/// 2. Run MCTS to populate the tree
/// 3. Call advance_root (first pruning - returns wrong result)
/// 4. Call advance_root again (second pruning - hangs)

use std::sync::Arc;
use std::time::Duration;
use mcts::gpu::{GpuContext, GpuConfig};
use mcts::gpu::mcts_othello::GpuOthelloMcts;

#[test]
fn test_pruning_hang() {
    
    println!("[TEST] Starting pruning hang test");
    
    // Create GPU context
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    
    println!("[TEST] Created GPU context");
    
    // Create engine with 8M nodes (same as the failing case)
    let max_nodes = 8_000_000;
    let batch_size = 8192;
    
    let engine = Arc::new(GpuOthelloMcts::new(
        context.clone(),
        max_nodes,
        batch_size,
    ).expect("Failed to create GpuOthelloMcts"));
    
    println!("[TEST] Created engine with max_nodes={}", max_nodes);
    
    // Initial board state (Othello starting position) - EXACT same as game
    let mut board = [0i32; 64];
    board[27] = -1; // (3, 3) = O (Black)
    board[28] = 1;  // (4, 3) = X (White)
    board[35] = 1;  // (3, 4) = X (White)
    board[36] = -1; // (4, 4) = O (Black)
    
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    // Initialize tree
    println!("[TEST] Initializing tree");
    engine.init_tree(&board, 1, &legal_moves);
    
    // Run MCTS for a few iterations to populate the tree
    // Match the game: ~2500 batches of 8192 iterations = ~20M iterations
    println!("[TEST] Running MCTS iterations (target: ~2500 batches like the game)");
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(5);
    let target_batches = 2500;
    let mut batch_count = 0;
    
    while start.elapsed() < timeout && batch_count < target_batches {
        engine.dispatch_mcts_othello_kernel(128, 1.4, 1.0, 1.0, 42); // 128 workgroups * 64 threads = 8192 iterations per batch
        batch_count += 1;
        
        // Log progress every 500 batches
        if batch_count % 500 == 0 {
            let stats = engine.get_children_stats();
            if !stats.is_empty() {
                println!("[TEST] Batch {}: best child has {} visits", batch_count, stats[0].2);
            }
        }
    }
    
    println!("[TEST] Completed {} batches in {:?}", batch_count, start.elapsed());
    
    println!("[TEST] MCTS complete, testing first advance_root");
    
    // First advance_root - this should work but may return wrong root
    let mut new_board = board.clone();
    new_board[28] = -1; // Apply move (4, 5) -> index 28
    new_board[29] = -1;
    
    let new_legal_moves = vec![(5, 3), (3, 5), (5, 5)];
    
    // Spawn first advance_root in a thread with timeout
    let engine_clone = engine.clone();
    let handle1 = std::thread::spawn(move || {
        println!("[TEST] Thread 1: calling advance_root for move (4, 5)");
        let result = engine_clone.advance_root(4, 5, &new_board, -1, &new_legal_moves);
        println!("[TEST] Thread 1: advance_root returned {}", result);
        result
    });
    
    // Wait with timeout
    match handle1.join() {
        Ok(result) => {
            println!("[TEST] First advance_root completed: {}", result);
        }
        Err(_) => {
            panic!("[TEST] First advance_root panicked!");
        }
    }
    
    // Give it a moment
    std::thread::sleep(Duration::from_millis(100));
    
    println!("[TEST] Testing second advance_root - THIS MAY HANG");
    
    // Second advance_root - this is where it hangs
    let mut board2 = new_board.clone();
    board2[37] = -1; // Apply move (5, 3)
    board2[38] = -1;
    board2[39] = -1;
    
    let legal_moves2 = vec![(2, 2), (3, 2), (4, 2), (5, 2), (6, 2)];
    
    // Spawn in thread with timeout to detect hang
    let engine_clone2 = engine.clone();
    let handle2 = std::thread::spawn(move || {
        println!("[TEST] Thread 2: calling advance_root for move (5, 3)");
        let start = std::time::Instant::now();
        let result = engine_clone2.advance_root(5, 3, &board2, 1, &legal_moves2);
        let elapsed = start.elapsed();
        println!("[TEST] Thread 2: advance_root returned {} in {:?}", result, elapsed);
        (result, elapsed)
    });
    
    // Wait with 10 second timeout
    let wait_start = std::time::Instant::now();
    let timeout = Duration::from_secs(10);
    
    loop {
        if handle2.is_finished() {
            match handle2.join() {
                Ok((result, elapsed)) => {
                    println!("[TEST] Second advance_root completed: {} in {:?}", result, elapsed);
                    break;
                }
                Err(_) => {
                    panic!("[TEST] Second advance_root panicked!");
                }
            }
        }
        
        if wait_start.elapsed() > timeout {
            panic!("[TEST FAILED] Second advance_root HUNG for more than 10 seconds! This reproduces the bug.");
        }
        
        std::thread::sleep(Duration::from_millis(100));
    }
    
    println!("[TEST] Test completed successfully - no hang detected");
}
