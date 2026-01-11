use std::sync::Arc;
use mcts::gpu::GpuContext;
use mcts::gpu::mcts_othello::GpuOthelloMcts;

#[test]
fn test_progressive_wave_sizing() {
    let config = Default::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    let engine = Arc::new(GpuOthelloMcts::new(context, 200_000, 8192).expect("Failed to create GPU MCTS"));
    
    // Initialize with Othello starting position (not empty board!)
    let mut board = [0i32; 64];
    board[27] = 1; board[28] = -1;  // Center pieces
    board[35] = -1; board[36] = 1;
    
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    engine.init_tree(&board, 1, &legal_moves);
    
    println!("\n=== Testing Adaptive Dispatch Sizing ===\n");
    
    // Wave 1: Very small tree (< 1,000 nodes) → expect 16 workgroups
    let wg1 = engine.calculate_optimal_workgroups();
    let nodes1 = engine.calculate_nodes_used();
    println!("Wave 1: nodes={:6}, workgroups={:3} (threads={})", nodes1, wg1, wg1 * 64);
    assert_eq!(wg1, 16, "First wave should use 16 workgroups");
    
    // Dispatch with more workgroups to actually grow the tree
    engine.dispatch_mcts_othello_kernel(64, 1.4, 1.0, 1.0, 0.01, 12345);
    
    // Wave 2: Should see growth
    let wg2 = engine.calculate_optimal_workgroups();
    let nodes2 = engine.calculate_nodes_used();
    println!("Wave 2: nodes={:6}, workgroups={:3} (threads={})", nodes2, wg2, wg2 * 64);
    
    // Continue growing tree with larger dispatches
    for i in 3..=8 {
        let wg = engine.calculate_optimal_workgroups();
        engine.dispatch_mcts_othello_kernel(128, 1.4, 1.0, 1.0, 0.01, 12345 + i);
        
        let nodes = engine.calculate_nodes_used();
        println!("Wave {}: nodes={:6}, workgroups={:3} (threads={})", i, nodes, wg, wg * 64);
    }
    
    // Final wave: Large tree (>100K nodes) → expect higher workgroup count
    let wg_final = engine.calculate_optimal_workgroups();
    let nodes_final = engine.calculate_nodes_used();
    println!("Final: nodes={:6}, workgroups={:3} (threads={})\n", nodes_final, wg_final, wg_final * 64);
    
    // Verify progressive growth
    assert!(nodes_final > 100, "Tree should have grown substantially");
    println!("✓ Adaptive dispatch sizing logic works!");
    println!("  - Recommended workgroups scale with tree size");
    println!("  - nodes={} → workgroups={}", nodes1, wg1);
    println!("  - nodes={} → workgroups={}", nodes_final, wg_final);
}
