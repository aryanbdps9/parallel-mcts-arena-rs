/// Test that params buffer works with STORAGE instead of UNIFORM
#[cfg(feature = "gpu")]
#[test]
fn test_params_storage_buffer_works() {
    use mcts::gpu::{GpuContext, GpuConfig};
    use mcts::gpu::mcts_othello::GpuOthelloMcts;
    use std::sync::Arc;

    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    let mcts = GpuOthelloMcts::new(context, 10000, 128).expect("Failed to create GpuOthelloMcts");
    
    // Standard Othello starting position
    // Center 4 squares: (3,3)=W (4,4)=W (3,4)=B (4,3)=B
    // B=1, W=-1
    let mut board = [0i32; 64];
    board[3 * 8 + 3] = -1; // (3,3) White
    board[3 * 8 + 4] = 1;  // (3,4) Black
    board[4 * 8 + 3] = 1;  // (4,3) Black
    board[4 * 8 + 4] = -1; // (4,4) White
    
    let root_player = 1; // Black to move
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    mcts.init_tree(&board, root_player, &legal_moves);
    
    // Dispatch kernel multiple times to allow memory visibility
    // First dispatch: expand root
    mcts.dispatch_mcts_othello_kernel(1, 1.4, 1.0, 0.06, 0.01, 42);
    
    // Subsequent dispatches: do rollouts with expanded tree
    for _ in 0..3 {
        mcts.dispatch_mcts_othello_kernel(16, 1.4, 1.0, 0.06, 0.01, 42);
    }
    
    // Small delay to let urgent events propagate
    std::thread::sleep(std::time::Duration::from_millis(100));
    
    // Read root node info directly
    let root_info = mcts.debug_get_node_info(0);
    assert!(root_info.num_children > 0, "Root should have children after expansion");
    
    // Read telemetry
    let telemetry = mcts.run_iterations(0, 1.4, 1.0, 0.06, 42);
    
    println!("Total nodes: {}", mcts.get_total_nodes());
    println!("Rollouts: {} Expansions: {}/{}", 
        telemetry.diagnostics.rollouts,
        telemetry.diagnostics.expansion_success,
        telemetry.diagnostics.expansion_attempts);
    
    // Verify we got some visits
    let children = mcts.get_children_stats();
    let total_visits: i32 = children.iter().map(|(_, _, v, _, _)| *v).sum();
    
    println!("Total child visits: {}", total_visits);
    for (x, y, visits, wins, q) in &children {
        println!("  ({},{}) visits={} wins={} Q={:.4}", x, y, visits, wins, q);
    }
    
    assert!(total_visits > 0, "Should have recorded visits after multiple dispatches");
    assert!(root_info.num_children == 4, "Root should have 4 children (standard Othello starting position)");
}
