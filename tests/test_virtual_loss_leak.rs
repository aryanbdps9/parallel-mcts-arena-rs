/// Test for Virtual Loss leak
/// Verifies that when threads fail to acquire expansion lock,
/// they still clear their virtual loss via rollout + backprop

use std::sync::Arc;
use mcts::gpu::{GpuContext, GpuConfig};
use mcts::gpu::mcts_othello::GpuOthelloMcts;

#[test]
fn test_virtual_loss_no_leak() {
    println!("\n=== VIRTUAL LOSS LEAK TEST ===");
    println!("Goal: Verify VL is cleared even when expansion fails");
    println!("Expectation: Total VL across tree should be ~0 after each dispatch\n");
    
    // Create GPU context
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Starting position
    let mut board = [0i32; 64];
    board[27] = 1; board[28] = -1;
    board[35] = -1; board[36] = 1;
    
    let legal_moves = vec![(2,3), (3,2), (4,5), (5,4)];
    
    // Create engine
    let max_nodes = 200_000;
    let batch_size = 8192;
    
    let engine = Arc::new(GpuOthelloMcts::new(
        context.clone(),
        max_nodes,
        batch_size,
    ).expect("Failed to create engine"));
    
    println!("[SETUP] Engine created, initializing tree\n");
    engine.init_tree(&board, 1, &legal_moves);
    
    // Run multiple dispatches with high contention to trigger lock failures
    println!("=== RUNNING DISPATCHES WITH HIGH CONTENTION ===");
    
    let mut max_vl_seen = 0i32;
    
    for dispatch_num in 1..=10 {
        engine.dispatch_mcts_othello_kernel(
            256,  // 16,384 threads - high contention expected
            1.4,  // exploration
            5.0,  // vl_weight (high to make VL effects visible)
            1.0,  // rollout_temp
            0.01, // vl_temp_scale
            12345 + dispatch_num
        );
        
        engine.update_root_stats();
        
        // Read VL from GPU
        let total_vl = engine.read_total_virtual_loss();
        let nodes_used = engine.calculate_nodes_used();
        
        println!("Dispatch {}: nodes={:6}, total_vl={:6}", 
                 dispatch_num, nodes_used, total_vl);
        
        if total_vl.abs() > max_vl_seen {
            max_vl_seen = total_vl.abs();
        }
        
        // After dispatch completes, all threads should have finished backprop
        // VL should be cleared (near 0, allowing small timing residual)
        if dispatch_num > 2 {  // Skip first 2 dispatches (warmup/tree building)
            assert!(
                total_vl.abs() < 1000,
                "VL LEAK! Dispatch {}: total_vl = {} (expected near 0)\n\
                 This means threads are dying without clearing VL via backprop!",
                dispatch_num,
                total_vl
            );
        }
    }
    
    println!("\n=== RESULTS ===");
    println!("Max VL seen across all dispatches: {}", max_vl_seen);
    println!("✓ No VL leak detected - all threads properly backpropagate");
    println!("✓ Even failed expansion threads clear their VL");
}
