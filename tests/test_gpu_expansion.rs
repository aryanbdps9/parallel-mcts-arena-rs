/// Simple test to verify GPU tree expansion is working
#[cfg(feature = "gpu")]
#[test]
fn test_gpu_basic_expansion() {
    use mcts::gpu::mcts_othello::GpuOthelloMcts;
    use mcts::gpu::{GpuContext, GpuConfig};
    use std::sync::Arc;

    println!("\n[TEST] ========== GPU Basic Expansion Test ==========\n");

    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    let max_nodes = 10000;
    let batch_size = 128;
    
    let engine = GpuOthelloMcts::new(
        context.clone(),
        max_nodes,
        batch_size,
    ).expect("Failed to create GpuOthelloMcts");

    // Initial board
    let mut board = [0i32; 64];
    board[27] = -1; board[28] = 1;
    board[35] = 1; board[36] = -1;
    
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    println!("[TEST] Initializing tree");
    engine.init_tree(&board, 1, &legal_moves);
    
    let nodes_init = engine.calculate_nodes_used();
    println!("[TEST] Nodes after init: {}", nodes_init);
    
    // Run multiple batches for tree growth
    println!("[TEST] Running initial batch (1 workgroup)");
    engine.dispatch_mcts_othello_kernel(1, 1.4, 1.0, 1.0, 42);
    
    println!("[TEST] Running follow-up batches (16 workgroups x 3)");
    for i in 0..3 {
        engine.dispatch_mcts_othello_kernel(16, 1.4, 1.0, 1.0, 1000 + i);
    }
    
    let nodes_after = engine.calculate_nodes_used();
    println!("[TEST] Nodes after 1 batch: {}", nodes_after);
    
    // Get diagnostics
    engine.update_root_stats();
    let stats = engine.get_children_stats();
    
    println!("[TEST] Root children:");
    for (x, y, visits, wins, q) in &stats {
        println!("  ({},{}) visits={} wins={} q={:.4}", x, y, visits, wins, q);
    }
    
    assert!(nodes_after > nodes_init, "Tree should expand after running MCTS");
    assert!(stats.iter().any(|(_, _, v, _, _)| *v > 0), "At least one child should have visits");
    
    println!("\n[TEST] ✓ Basic expansion test passed");
}
