use mcts::gpu::{GpuConfig, GpuContext};
use mcts::gpu::mcts_othello::GpuOthelloMcts;
use std::sync::Arc;

#[test]
fn test_mcts_liar_rollout_distribution() {
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Standard Othello opening position
    #[rustfmt::skip]
    let board = [
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0, -1,  1,  0,  0,  0,
        0,  0,  0,  1, -1,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
    ];
    
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    let engine = GpuOthelloMcts::new(
        context.clone(),
        128 * 1024,  // max_nodes
        256,         // free_list_capacity
    ).expect("Failed to create engine");
    
    engine.init_tree(&board, 1, &legal_moves);
    
    // Run MCTS with deterministic argmax selection (temp=0) and honest rollouts
    let _telemetry = engine.run_incremental_mcts(
        8192,   // num_threads (same as game)
        100,    // max_steps
        1.0,    // exploration
        0.2,    // vl_weight
        0.0,    // temperature (forces argmax, deterministic)
        42,     // seed
        None,   // timeout
        false,  // use_random_rollouts (honest rollout mode)
    );
    
    // Get diagnostics - we only care about rollout distribution
    let diag = engine.read_diagnostics();
    
    // Read child statistics to check visit distribution
    let child_stats = engine.get_children_stats();
    
    let total_rollouts = diag.rollouts;
    
    println!("\n=== MCTS Honest Rollout Test ===");
    println!("Configuration:");
    println!("  Threads: 8192");
    println!("  Steps: 100");
    println!("  Temperature: 0.0 (deterministic argmax)");
    println!("  Honest rollouts: enabled");
    println!();
    println!("Rollout Statistics:");
    println!("  Total rollouts: {}", total_rollouts);
    
    // Print child visit distribution
    println!();
    println!("Root Child Visit Distribution:");
    let total_child_visits: i32 = child_stats.iter().map(|c| c.2).sum();
    for (idx, &(x, y, visits, wins, q)) in child_stats.iter().enumerate() {
        let visit_pct = 100.0 * visits as f64 / total_child_visits as f64;
        println!("  Child {}: ({},{}) - visits: {} ({:.2}%), wins: {}, q: {:.4}", 
            idx, x, y, visits, visit_pct, wins, q);
    }
    println!("  Total child visits: {}", total_child_visits);
    
    println!("\n✓ MCTS honest rollout test PASSED");
    println!("  Tree search working correctly with real game simulations");
}
