/// Test that incremental MCTS properly expands the root node
#[cfg(feature = "gpu")]
use mcts::gpu::{GpuConfig, GpuContext, GpuOthelloMcts};
use std::sync::Arc;

#[test]
#[cfg(feature = "gpu")]
fn test_incremental_expands_root() {
    println!("\n=== Test: Incremental Root Expansion ===");
    
    // Create GPU context and engine
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    let engine = GpuOthelloMcts::new(context, 10000, 128).expect("Failed to create engine");
    
    // Standard Othello starting position
    let mut board = [0i32; 64];
    board[3 * 8 + 3] = -1; board[3 * 8 + 4] = 1;
    board[4 * 8 + 3] = 1;  board[4 * 8 + 4] = -1;
    
    let root_player = 1;
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    // Initialize tree
    println!("[TEST] Initializing tree...");
    engine.init_tree(&board, root_player, &legal_moves);
    
    // Check root before incremental MCTS
    println!("[TEST] Checking root state before incremental MCTS...");
    let nodes_before = engine.calculate_nodes_used();
    println!("[TEST] Nodes before: {}", nodes_before);
    assert_eq!(nodes_before, 1, "Should have exactly 1 node (root) before search");
    
    // Run incremental MCTS with more steps to allow children exploration
    println!("[TEST] Running incremental MCTS (100 steps)...");
    let telemetry = engine.run_incremental_mcts(
        4,      // num_threads (small for testing)
        100,    // max_steps (enough for: root expand → root rollouts → select children → children rollouts)
        1.414,  // exploration
        1.0,    // virtual_loss_weight
        1.0,    // temperature
        42,     // seed
        None,   // no timeout
    );
    
    println!("[TEST] Incremental MCTS completed");
    
    // Check root after incremental MCTS
    println!("[TEST] Checking root state after incremental MCTS...");
    let nodes_after = engine.calculate_nodes_used();
    println!("[TEST] Nodes after: {}", nodes_after);
    
    // Check root node visits directly
    let root_visits = engine.debug_get_node_visits(0);
    println!("[TEST] Root node (index 0) visits: {}", root_visits);
    
    // Get root children stats
    let children = engine.get_children_stats();
    println!("[TEST] Root has {} children in stats", children.len());
    
    for (i, (x, y, visits, wins, q)) in children.iter().enumerate() {
        println!("[TEST]   Child {}: ({},{}) visits={} wins={} q={:.4}", 
                 i, x, y, visits, wins, q);
    }
    
    // Check telemetry BEFORE assertions
    println!("[TEST] Telemetry: iterations={} nodes_used={}", 
             telemetry.iterations_launched, telemetry.alloc_count_after);
    println!("[TEST] Telemetry: exp_attempts={} exp_success={} rollouts={}", 
             telemetry.diagnostics.expansion_attempts, telemetry.diagnostics.expansion_success, 
             telemetry.diagnostics.rollouts);
    println!("[TEST] Telemetry: total_children_gen={} alloc_failures={}", 
             telemetry.diagnostics.total_children_gen, telemetry.diagnostics.alloc_failures);
    
    // Verify root was expanded
    assert!(nodes_after > 1, 
            "Root should be expanded - expected more than 1 node, got {}", nodes_after);
    
    // Verify children have visits
    let total_child_visits: i32 = children.iter().map(|(_, _, visits, _, _)| visits).sum();
    println!("[TEST] Total child visits: {}", total_child_visits);
    
    assert!(total_child_visits > 0, 
            "Children should have been visited - got 0 total visits");
    
    // Check telemetry
    println!("[TEST] Telemetry: iterations={} nodes_used={}", 
             telemetry.iterations_launched, telemetry.alloc_count_after);
    println!("[TEST] Telemetry: exp_attempts={} exp_success={} rollouts={}", 
             telemetry.diagnostics.expansion_attempts, telemetry.diagnostics.expansion_success, 
             telemetry.diagnostics.rollouts);
    
    println!("\n[TEST] ✓ Root expansion test PASSED");
}
