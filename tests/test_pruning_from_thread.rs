/// Test pruning from a separate thread (like the AI worker does)
/// This may expose thread-safety issues with WGPU buffer mapping

use std::sync::Arc;
use std::time::Duration;
use mcts::gpu::{GpuContext, GpuConfig};
use mcts::gpu::mcts_othello::GpuOthelloMcts;

#[test]
fn test_pruning_from_separate_thread() {
    println!("[TEST] Starting pruning from separate thread test");
    
    // Create GPU context
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    
    // Create engine with 8M nodes (same as game)
    let max_nodes = 8_000_000;
    let batch_size = 8192;
    
    let engine = Arc::new(GpuOthelloMcts::new(
        context.clone(),
        max_nodes,
        batch_size,
    ).expect("Failed to create GpuOthelloMcts"));
    
    println!("[TEST] Created engine");
    
    // Initial board
    let mut board = [0i32; 64];
    board[27] = -1;
    board[28] = 1;
    board[35] = 1;
    board[36] = -1;
    
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    engine.init_tree(&board, 1, &legal_moves);
    println!("[TEST] Tree initialized");
    
    // Run ~1700 batches like the game does
    println!("[TEST] Running 1700 batches");
    for i in 0..1700 {
        engine.dispatch_mcts_othello_kernel(128, 1.4, 1.0, 1.0, 0.01, 42);
        if i % 500 == 0 {
            println!("[TEST] Completed {} batches", i);
        }
    }
    
    println!("[TEST] MCTS complete, spawning thread for advance_root");
    
    // Apply move
    let mut new_board = board.clone();
    new_board[28] = -1;
    new_board[29] = -1;
    let new_legal_moves = vec![(5, 3), (3, 5), (5, 5)];
    
    // Call advance_root from a SEPARATE THREAD (like AI worker)
    // Use Builder to set larger stack size (default 2MB might not be enough)
    let engine_clone = engine.clone();
    let handle = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024) // 8MB stack
        .spawn(move || {
            println!("[TEST] Thread: calling advance_root for move (4, 5)");
            let result = engine_clone.advance_root(4, 5, &new_board, -1, &new_legal_moves);
            println!("[TEST] Thread: advance_root returned {}", result);
            result
        })
        .expect("Failed to spawn thread");
    
    // Wait with timeout
    let start = std::time::Instant::now();
    loop {
        if handle.is_finished() {
            match handle.join() {
                Ok(result) => {
                    println!("[TEST] advance_root completed: {}", result);
                    assert!(result, "advance_root should succeed when called from separate thread");
                    
                    // Give GPU time to fully complete and clean up before test exits
                    // This prevents crashes during cleanup
                    std::thread::sleep(Duration::from_millis(100));
                    
                    return;
                }
                Err(e) => {
                    panic!("[TEST] advance_root panicked: {:?}", e);
                }
            }
        }
        
        if start.elapsed() > Duration::from_secs(10) {
            panic!("[TEST FAILED] advance_root HUNG for more than 10 seconds when called from separate thread!");
        }
        
        std::thread::sleep(Duration::from_millis(100));
    }
}
