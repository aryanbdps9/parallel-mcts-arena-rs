#[cfg(feature = "gpu")]
#[test]
fn test_max_temp_boost_resets_between_moves() {
    use std::sync::Arc;
    use mcts::gpu::{GpuContext, GpuConfig};
    use mcts::gpu::mcts_othello::GpuOthelloMcts;
    
    // Create GPU context and engine
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    let max_nodes = 10000;
    let mcts = GpuOthelloMcts::new(context, max_nodes, 8192).expect("Failed to create GpuOthelloMcts");
    
    // Setup first position
    let mut board1 = [0i32; 64];
    board1[3 * 8 + 3] = 1;
    board1[3 * 8 + 4] = -1;
    board1[4 * 8 + 3] = -1;
    board1[4 * 8 + 4] = 1;
    let root_player = 1;
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    mcts.init_tree(&board1, root_player, &legal_moves);
    
    // Run iterations to build up virtual loss and trigger temperature boost
    mcts.dispatch_mcts_othello_kernel(1, 1.4, 5.0, 1.0, 0.01, 42);
    for _ in 0..3 {
        mcts.dispatch_mcts_othello_kernel(16, 1.4, 5.0, 1.0, 0.01, 42);
    }
    mcts.flush_and_wait();
    
    // Read diagnostics after first position
    let telemetry1 = mcts.run_iterations(1, 1.4, 5.0, 1.0, 42);
    let max_boost1 = telemetry1.diagnostics.max_temp_boost;
    
    println!("First position: max_temp_boost = {} (as f32: {:.3})", 
        max_boost1, max_boost1 as f32 / 1000.0);
    
    assert!(max_boost1 > 0, "max_temp_boost should be > 0 after running iterations");
    
    // Setup second position (different board)
    let mut board2 = [0i32; 64];
    board2[3 * 8 + 3] = 1;
    board2[3 * 8 + 4] = -1;
    board2[4 * 8 + 3] = -1;
    board2[4 * 8 + 4] = 1;
    board2[2 * 8 + 3] = 1;  // Different from board1
    let legal_moves2 = vec![(3, 2), (4, 5), (5, 4)];
    
    mcts.init_tree(&board2, root_player, &legal_moves2);
    
    // Read diagnostics immediately after reset (before any iterations)
    let telemetry_after_reset = mcts.run_iterations(1, 1.4, 5.0, 1.0, 42);
    let max_boost_after_reset = telemetry_after_reset.diagnostics.max_temp_boost;
    
    println!("After reset (before new iterations): max_temp_boost = {} (as f32: {:.3})", 
        max_boost_after_reset, max_boost_after_reset as f32 / 1000.0);
    
    assert_eq!(max_boost_after_reset, 0, 
        "max_temp_boost should be reset to 0 after init_tree, but got {}", 
        max_boost_after_reset);
    
    // Run iterations on second position
    mcts.dispatch_mcts_othello_kernel(1, 1.4, 5.0, 1.0, 0.01, 100);
    for _ in 0..3 {
        mcts.dispatch_mcts_othello_kernel(16, 1.4, 5.0, 1.0, 0.01, 100);
    }
    mcts.flush_and_wait();
    
    // Read diagnostics after second position
    let telemetry2 = mcts.run_iterations(1, 1.4, 5.0, 1.0, 100);
    let max_boost2 = telemetry2.diagnostics.max_temp_boost;
    
    println!("Second position: max_temp_boost = {} (as f32: {:.3})", 
        max_boost2, max_boost2 as f32 / 1000.0);
    
    assert!(max_boost2 > 0, "max_temp_boost should be > 0 after running iterations on second position");
    
    // The values might be similar but they should be independently calculated
    println!("\nSummary:");
    println!("  Position 1: {:.3}", max_boost1 as f32 / 1000.0);
    println!("  After reset: {:.3}", max_boost_after_reset as f32 / 1000.0);
    println!("  Position 2: {:.3}", max_boost2 as f32 / 1000.0);
}
#[cfg(feature = "gpu")]
#[test]
fn test_max_temp_boost_resets_with_advance_root() {
    use std::sync::Arc;
    use mcts::gpu::{GpuContext, GpuConfig};
    use mcts::gpu::mcts_othello::GpuOthelloMcts;
    
    // Create GPU context and engine
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    let max_nodes = 10000;
    let mcts = GpuOthelloMcts::new(context, max_nodes, 8192).expect("Failed to create GpuOthelloMcts");
    
    // Setup first position
    let mut board1 = [0i32; 64];
    board1[3 * 8 + 3] = 1;
    board1[3 * 8 + 4] = -1;
    board1[4 * 8 + 3] = -1;
    board1[4 * 8 + 4] = 1;
    let root_player = 1;
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    mcts.init_tree(&board1, root_player, &legal_moves);
    
    // Run iterations to build up virtual loss and trigger temperature boost
    // Need to actually expand the tree so we have children to advance to
    mcts.dispatch_mcts_othello_kernel(1, 1.4, 5.0, 1.0, 0.01, 42);
    for i in 0..5 {
        mcts.dispatch_mcts_othello_kernel(32, 1.4, 5.0, 1.0, 0.01, 42 + i);
    }
    mcts.flush_and_wait();
    
    // Read diagnostics after first position
    let telemetry1 = mcts.run_iterations(1, 1.4, 5.0, 1.0, 42);
    let max_boost1 = telemetry1.diagnostics.max_temp_boost;
    
    println!("Move 1: max_temp_boost = {} (as f32: {:.3})", 
        max_boost1, max_boost1 as f32 / 1000.0);
    
    assert!(max_boost1 > 0, "max_temp_boost should be > 0 after first move");
    
    // Simulate making a move using advance_root (tree reuse path)
    // Move at (2,3) should exist in the tree now
    let mut board2 = board1.clone();
    board2[2 * 8 + 3] = 1;  // Make move at (2,3)
    board2[3 * 8 + 3] = 1;  // Flip piece at (3,3)
    let legal_moves2 = vec![(2, 2), (2, 4), (4, 2)]; // Legal moves from new position
    
    let success = mcts.advance_root(2, 3, &board2, -1, &legal_moves2);
    if !success {
        println!("Warning: advance_root failed, test will use init_tree fallback path");
    }
    
    // Read diagnostics after advance_root (before running new iterations)
    let telemetry_after_advance = mcts.run_iterations(1, 1.4, 5.0, 1.0, 100);
    let max_boost_after_advance = telemetry_after_advance.diagnostics.max_temp_boost;
    
    println!("After advance_root (before new iterations): max_temp_boost = {} (as f32: {:.3})", 
        max_boost_after_advance, max_boost_after_advance as f32 / 1000.0);
    
    assert_eq!(max_boost_after_advance, 0, 
        "max_temp_boost should be reset to 0 after advance_root, but got {}", 
        max_boost_after_advance);
    
    // Run iterations on second position
    mcts.dispatch_mcts_othello_kernel(1, 1.4, 5.0, 1.0, 0.01, 200);
    for _ in 0..3 {
        mcts.dispatch_mcts_othello_kernel(16, 1.4, 5.0, 1.0, 0.01, 200);
    }
    mcts.flush_and_wait();
    
    // Read diagnostics after second position
    let telemetry2 = mcts.run_iterations(1, 1.4, 5.0, 1.0, 200);
    let max_boost2 = telemetry2.diagnostics.max_temp_boost;
    
    println!("Move 2: max_temp_boost = {} (as f32: {:.3})", 
        max_boost2, max_boost2 as f32 / 1000.0);
    
    assert!(max_boost2 > 0, "max_temp_boost should be > 0 after second move");
    
    // The values should be independent
    println!("\nSummary (advance_root path):");
    println!("  Move 1: {:.3}", max_boost1 as f32 / 1000.0);
    println!("  After advance_root: {:.3}", max_boost_after_advance as f32 / 1000.0);
    println!("  Move 2: {:.3}", max_boost2 as f32 / 1000.0);
}