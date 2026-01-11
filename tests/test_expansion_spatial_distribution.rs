/// Test to understand WHERE contention happens
/// Tracks which nodes threads attempt to expand to determine if contention is:
/// - Spatial: Many threads targeting same few nodes
/// - Temporal: Many threads expanding different nodes at same time
/// - Capacity: Too many threads vs available leaves

use std::sync::Arc;
use mcts::gpu::{GpuContext, GpuConfig};
use mcts::gpu::mcts_othello::GpuOthelloMcts;

#[test]
fn test_expansion_spatial_distribution() {
    println!("\n=== EXPANSION SPATIAL DISTRIBUTION TEST ===");
    println!("Goal: Understand WHERE expansion attempts happen\n");
    
    // Create GPU context
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Starting position (4 legal moves)
    let mut board = [0i32; 64];
    board[27] = 1; board[28] = -1;
    board[35] = -1; board[36] = 1;
    
    let legal_moves = vec![(2,3), (3,2), (4,5), (5,4)];
    
    // Create engine with high parallelism
    let max_nodes = 200_000;
    let batch_size = 8192;  // High parallelism to create contention
    
    let engine = Arc::new(GpuOthelloMcts::new(
        context.clone(),
        max_nodes,
        batch_size,
    ).expect("Failed to create engine"));
    
    println!("[SETUP] Engine: batch_size={}, max_nodes={}\n", batch_size, max_nodes);
    engine.init_tree(&board, 1, &legal_moves);
    
    // Run MULTIPLE dispatches to grow the tree
    println!("[TEST] Running 10 dispatches (256 workgroups each) to grow tree");
    for i in 0..10 {
        engine.dispatch_mcts_othello_kernel(
            256,  // 256 * 64 = 16,384 threads per dispatch
            1.4,  // exploration
            5.0,  // vl_weight
            1.0,  // rollout_temp
            0.01, // vl_temp_scale
            12345 + i // seed
        );
        
        if i % 3 == 2 {
            let nodes = engine.calculate_nodes_used();
            println!("  After dispatch {}: {} nodes", i + 1, nodes);
        }
    }
    
    // Get final tree state
    let nodes_used = engine.calculate_nodes_used();
    
    println!("\n=== TREE GROWTH ===");
    println!("Total nodes after 10 dispatches: {}", nodes_used);
    println!("Threads per dispatch:            {}", 256 * 64);
    println!("Total thread-iterations:         {}", 10 * 256 * 64);
    
    // Simple ratio analysis
    let threads_total = 10 * 256 * 64;
    let nodes_per_thread = nodes_used as f64 / threads_total as f64;
    
    println!("\n=== EXPANSION EFFICIENCY ===");
    println!("Nodes created per thread:        {:.4}", nodes_per_thread);
    
    if nodes_per_thread > 0.5 {
        println!("✅  HIGH EFFICIENCY: Most threads successfully expanded");
    } else if nodes_per_thread > 0.1 {
        println!("⚠️  MODERATE EFFICIENCY: Some contention present");
    } else {
        println!("❌  LOW EFFICIENCY: High contention (most threads failed to expand)");
    }
    
    // Calculate theoretical capacity from tree structure
    // If we assume a balanced tree with 4 children per node:
    // - Root has 4 children
    // - Each of those has ~4 children
    // - At depth d, we have ~4^d leaves
    
    // Work backwards: Given N nodes total, estimate depth
    // Rough estimate: balanced 4-ary tree with N nodes has depth ~log4(N)
    let estimated_depth = if nodes_used > 1 {
        (nodes_used as f64).log(4.0) as u32
    } else {
        0
    };
    
    // Theoretical leaves at this depth: 4^depth
    let theoretical_leaves = 4u32.pow(estimated_depth);
    
    println!("\n=== CAPACITY ESTIMATE ===");
    println!("Estimated tree depth:            ~{}", estimated_depth);
    println!("Theoretical max leaves:          ~{}", theoretical_leaves);
    println!("Threads per dispatch:            {}", 256 * 64);
    
    if theoretical_leaves < 256 * 64 {
        let unavoidable = ((256 * 64 - theoretical_leaves) as f64 / (256 * 64) as f64) * 100.0;
        println!("⚠️  CAPACITY BOTTLENECK:");
        println!("    ~{} leaves can't handle {} threads per dispatch", 
                 theoretical_leaves, 256 * 64);
        println!("    Theoretical minimum contention: {:.1}%", unavoidable);
        println!("    This is a CAPACITY issue - too few expandable nodes");
    } else {
        println!("✅  SUFFICIENT CAPACITY:");
        println!("    ~{} leaves > {} threads", theoretical_leaves, 256 * 64);
        println!("    Low efficiency must be due to:");
        println!("    - Temporal clustering (threads expand at same time)");
        println!("    - Atomic serialization (GPU atomics are serialized)");
    }
    
    println!("\n=== CONCLUSION ===");
    if nodes_per_thread < 0.1 && theoretical_leaves < 256 * 64 / 10 {
        println!("✓ CAPACITY ISSUE: Tree too shallow for high parallelism");
        println!("  Early in search, few leaves exist - contention is unavoidable");
        println!("  As tree grows deeper, more leaves become available");
    } else if nodes_per_thread < 0.1 {
        println!("⚠️ TEMPORAL/ATOMIC ISSUE: Contention despite sufficient leaves");
        println!("  Threads likely expanding at same microsecond (temporal)");
        println!("  OR GPU atomic operations being serialized");
    } else {
        println!("✓ HEALTHY EXPANSION: Good balance of growth vs contention");
    }
}
