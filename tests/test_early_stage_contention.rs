/// Test to analyze early-stage contention (first few dispatches)
/// This should show if contention is worst at the beginning when the tree is shallow

use std::sync::Arc;
use mcts::gpu::{GpuContext, GpuConfig};
use mcts::gpu::mcts_othello::GpuOthelloMcts;

#[test]
fn test_early_stage_contention() {
    println!("\n=== EARLY STAGE CONTENTION TEST ===");
    println!("Analyzing contention in first few dispatches\n");
    
    // Create GPU context
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Starting position (4 legal moves)
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
    
    println!("[SETUP] batch_size={}, max_nodes={}\n", batch_size, max_nodes);
    engine.init_tree(&board, 1, &legal_moves);
    
    println!("=== DISPATCH-BY-DISPATCH ANALYSIS ===");
    println!("{:<10} {:<15} {:<20} {:<12}", 
             "Dispatch", "Nodes Created", "Nodes Per Thread", "Efficiency");
    println!("{}", "-".repeat(60));
    
    let mut prev_nodes = 0;
    
    for dispatch_num in 1..=20 {
        engine.dispatch_mcts_othello_kernel(
            256,  // 256 * 64 = 16,384 threads
            1.4,  // exploration
            5.0,  // vl_weight
            1.0,  // rollout_temp
            0.01, // vl_temp_scale
            12345 + dispatch_num
        );
        
        let current_nodes = engine.calculate_nodes_used();
        let new_nodes = current_nodes - prev_nodes;
        let efficiency = new_nodes as f64 / (256 * 64) as f64;
        
        println!("{:<10} {:<15} {:<20.4} {:<12.1}%", 
                 dispatch_num, 
                 new_nodes,
                 efficiency,
                 efficiency * 100.0);
        
        prev_nodes = current_nodes;
    }
    
    println!("\n=== INTERPRETATION ===");
    println!("If efficiency is:");
    println!("  - Very low at start (~5%), then HIGH at end (~50%+):");
    println!("    → CAPACITY issue: Early tree is too shallow");
    println!("  - Consistently low throughout (~30%):");
    println!("    → TEMPORAL/ATOMIC issue: Not dependent on tree size");
    println!("  - Decreasing over time:");
    println!("    → Tree getting deeper helps, but other factors limit");
}
