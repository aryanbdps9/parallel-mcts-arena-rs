use mcts::gpu::mcts_othello::GpuOthelloMcts;
use mcts::gpu::{GpuContext, GpuConfig};
use std::sync::Arc;

#[test]
fn test_dynamic_free_list_allocation() {
    // This test verifies that workgroups can dynamically claim multiple free lists
    // Previously, each workgroup was limited to its own free list, causing premature
    // memory exhaustion when only a few workgroups were active.
    
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Create MCTS with 10,000 nodes (distributed across 256 free lists = ~39 nodes per list)
    let max_nodes = 10000;
    let iterations_per_batch = 8192;
    let mcts = GpuOthelloMcts::new(context, max_nodes, iterations_per_batch).expect("Failed to create GPU MCTS");
    
    // Initialize with a simple board position
    let mut board = [0i32; 64];
    board[27] = 1;  // E4
    board[28] = -1; // D4
    board[35] = -1; // E5
    board[36] = 1;  // D5
    
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)]; // Standard opening moves
    
    mcts.init_tree(&board, 1, &legal_moves);
    
    // Run MCTS with only 1 workgroup initially
    // This workgroup should be able to claim multiple free lists as it needs them
    println!("[TEST] Running MCTS with 1 workgroup to test dynamic allocation");
    mcts.dispatch_mcts_othello_kernel(1, 1.4, 1.0, 1.0, 0.01, 42);
    
    let telemetry = mcts.run_iterations(1, 1.4, 1.0, 1.0, 42);
    
    println!("[TEST] After 1 workgroup batch:");
    println!("  Nodes allocated: {}", telemetry.alloc_count_after);
    println!("  Nodes freed: {}", telemetry.free_count_after);
    println!("  Capacity: {}", telemetry.node_capacity);
    println!("  Saturated: {}", telemetry.saturated);
    
    // Now dispatch with more workgroups (10) to use more of the tree
    println!("[TEST] Running MCTS with 10 workgroups");
    for i in 0..20 {
        mcts.dispatch_mcts_othello_kernel(10, 1.4, 1.0, 1.0, 0.01, 42 + i * 1000);
        let tel = mcts.run_iterations(1, 1.4, 1.0, 1.0, 42 + i * 1000);
        
        println!("[TEST] Batch {}: nodes={}/{}, saturated={}", 
                 i + 1, tel.alloc_count_after, tel.node_capacity, tel.saturated);
        
        // Should be able to use a significant portion of capacity
        // Before the fix, would saturate at ~4% (1/256 * 10 workgroups)
        // After the fix, should grow to much higher utilization
        if tel.saturated {
            println!("[TEST] WARNING: Saturated at batch {} with {}% utilization", 
                     i + 1, (tel.alloc_count_after * 100) / tel.node_capacity);
        }
    }
    
    let final_telemetry = mcts.run_iterations(1, 1.4, 1.0, 1.0, 99999);
    let utilization_pct = (final_telemetry.alloc_count_after * 100) / final_telemetry.node_capacity;
    
    println!("[TEST] Final utilization: {}% ({}/{})", 
             utilization_pct, final_telemetry.alloc_count_after, final_telemetry.node_capacity);
    
    // With dynamic allocation, we should be able to use at least 50% of capacity
    // (Previously would fail at ~4% with 10 workgroups)
    assert!(utilization_pct >= 50, 
            "Expected at least 50% utilization, got {}%. Dynamic free list allocation may not be working.",
            utilization_pct);
    
    println!("[TEST] ✓ Dynamic free list allocation working - achieved {}% utilization", utilization_pct);
}

#[test]
fn test_workgroup_can_claim_multiple_lists() {
    // More targeted test: verify a single workgroup can claim and use multiple free lists
    
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Small capacity: 2000 nodes across 256 lists = ~8 nodes per list
    // With only 1 workgroup active, it should claim multiple lists to continue allocation
    let max_nodes = 2000;
    let iterations_per_batch = 8192;
    let mcts = GpuOthelloMcts::new(context, max_nodes, iterations_per_batch).expect("Failed to create GPU MCTS");
    
    let mut board = [0i32; 64];
    board[27] = 1;
    board[28] = -1;
    board[35] = -1;
    board[36] = 1;
    
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    mcts.init_tree(&board, 1, &legal_moves);
    
    // Run with just 1 workgroup - it should claim multiple free lists
    println!("[TEST] Running single workgroup to force claiming multiple lists");
    
    for i in 0..50 {
        mcts.dispatch_mcts_othello_kernel(1, 1.4, 1.0, 1.0, 0.01, 1000 + i * 100);
        let tel = mcts.run_iterations(1, 1.4, 1.0, 1.0, 1000 + i * 100);
        
        if i % 10 == 0 {
            let pct = (tel.alloc_count_after * 100) / tel.node_capacity;
            println!("[TEST] Batch {}: {}% utilization", i, pct);
        }
        
        if tel.saturated {
            let pct = (tel.alloc_count_after * 100) / tel.node_capacity;
            println!("[TEST] Saturated at batch {} with {}% utilization", i, pct);
            break;
        }
    }
    
    let final_tel = mcts.run_iterations(1, 1.4, 1.0, 1.0, 99999);
    let final_pct = (final_tel.alloc_count_after * 100) / final_tel.node_capacity;
    
    println!("[TEST] Final: {}% utilization with single workgroup", final_pct);
    
    // Single workgroup should be able to use most of the capacity by claiming lists
    // Without dynamic allocation, it could only use 1/256 ≈ 0.4%
    assert!(final_pct >= 70,
            "Single workgroup should achieve >70% utilization by claiming multiple lists, got {}%",
            final_pct);
    
    println!("[TEST] ✓ Single workgroup successfully claimed multiple free lists - {}% utilization", final_pct);
}
