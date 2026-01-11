/// Test that selection remains balanced after tree reuse (advance_root)
/// 
/// This reproduces the stampede bug where the first search is balanced,
/// but after advancing the root, subsequent searches give 98% of visits
/// to the child with LOWEST PUCT.

use mcts::gpu::{GpuConfig, GpuContext};
use mcts::gpu::mcts_othello::GpuOthelloMcts;
use std::sync::Arc;

#[test]
fn test_no_stampede_after_tree_reuse() {
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
        128 * 1024,
        256,
    ).expect("Failed to create engine");
    
    engine.init_tree(&board, 1, &legal_moves);
    
    println!("\n=== FIRST SEARCH (Fresh Tree) ===");
    
    // First search - should be balanced
    let _telemetry = engine.run_incremental_mcts(
        8192,
        100,
        1.0,
        0.2,
        0.06,
        42,
        None,
        false,
    );
    
    let first_stats = engine.get_children_stats();
    
    let total_visits_1: i32 = first_stats.iter().map(|s| s.2).sum();
    let max_visits_1 = first_stats.iter().map(|s| s.2).max().unwrap();
    let max_ratio_1 = max_visits_1 as f64 / total_visits_1 as f64;
    
    println!("\nFirst search results:");
    for &(x, y, visits, _wins, q) in &first_stats {
        if visits > 0 {
            let ratio = visits as f64 / total_visits_1 as f64;
            println!("  ({},{}) - visits: {} ({:.1}%), Q: {:.4}", x, y, visits, ratio * 100.0, q);
        }
    }
    println!("Max visit ratio: {:.1}%", max_ratio_1 * 100.0);
    
    assert!(max_ratio_1 < 0.60, 
        "First search: One child has {:.1}% of visits. Should be reasonably balanced.",
        max_ratio_1 * 100.0);
    
    // Now simulate making a move: advance root to position after player 1 plays (3,2)
    // and opponent responds with (2,2)
    println!("\n=== ADVANCING ROOT (Simulating moves) ===");
    
    // After (3,2) by player 1
    #[rustfmt::skip]
    let board_after_move1 = [
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  1,  0,  0,  0,  0,
        0,  0,  0,  1,  1,  0,  0,  0,
        0,  0,  0,  1, -1,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
    ];
    
    let legal_moves_1 = vec![(2, 2), (4, 2), (2, 4)];
    
    engine.advance_root(3, 2, &board_after_move1, -1, &legal_moves_1);
    
    // After opponent plays (2,2)
    #[rustfmt::skip]
    let board_after_move2 = [
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0, -1,  1,  0,  0,  0,  0,
        0,  0,  0, -1,  1,  0,  0,  0,
        0,  0,  0,  1, -1,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
    ];
    
    // The actual children that will exist in the tree after advancing root twice
    // (These were the children of the node we advanced to)
    let legal_moves_2 = vec![(1, 2), (2, 3), (4, 5), (5, 4)];
    
    engine.advance_root(2, 2, &board_after_move2, 1, &legal_moves_2);
    
    println!("Moved to position after (3,2) and (2,2)");
    println!("New root has {} legal moves", legal_moves_2.len());
    
    // DEBUG: Check actual state of children after tree reuse
    engine.debug_print_root_children();
    
    // CRITICAL TEST: Second search after tree reuse
    println!("\n=== SECOND SEARCH (After Tree Reuse) ===");
    
    // Reset diagnostics to get accurate counters for this search only
    engine.reset_diagnostics_gpu();
    
    let _telemetry2 = engine.run_incremental_mcts(
        8192,
        100,
        1.0,
        0.2,
        0.06,
        43,  // Different seed
        None,
        false,
    );
    
    // DEBUG: Check state AFTER search
    println!("\n=== AFTER SECOND SEARCH ===");
    engine.debug_print_root_children();
    
    // DEBUG: Check expansion diagnostics
    let diag = engine.read_diagnostics();
    println!("\nExpansion diagnostics:");
    println!("  Expansion attempts: {}", diag.expansion_attempts);
    println!("  Expansion success: {}", diag.expansion_success);
    println!("  Selection no_children: {}", diag.selection_no_children);
    println!("  Selection invalid_child: {}", diag.selection_invalid_child);
    
    let second_stats = engine.get_children_stats();
    
    let total_visits_2: i32 = second_stats.iter().map(|s| s.2).sum();
    let max_visits_2 = second_stats.iter().map(|s| s.2).max().unwrap();
    let min_visits_2 = second_stats.iter().filter(|s| s.2 > 0).map(|s| s.2).min().unwrap();
    let max_ratio_2 = max_visits_2 as f64 / total_visits_2 as f64;
    
    println!("\nSecond search results:");
    for &(x, y, visits, _wins, q) in &second_stats {
        if visits > 0 {
            let ratio = visits as f64 / total_visits_2 as f64;
            println!("  ({},{}) - visits: {} ({:.1}%), Q: {:.4}", x, y, visits, ratio * 100.0, q);
        }
    }
    println!("Max visit ratio: {:.1}%", max_ratio_2 * 100.0);
    println!("Min visits: {}, Max visits: {}", min_visits_2, max_visits_2);
    
    // THIS IS WHERE THE BUG MANIFESTS: After tree reuse, one child gets 90%+ of visits
    assert!(max_ratio_2 < 0.90, 
        "STAMPEDE BUG DETECTED: After tree reuse, one child has {:.1}% of visits! \
         This means 8192 threads are all selecting the SAME child (likely the one with LOWEST PUCT). \
         Expected: reasonably balanced distribution like the first search.",
        max_ratio_2 * 100.0);
    
    // Also check that visit distribution didn't get drastically worse
    let ratio_change = max_ratio_2 / max_ratio_1;
    assert!(ratio_change < 2.0,
        "Visit concentration increased {:.1}x after tree reuse (from {:.1}% to {:.1}%). \
         Selection appears to break after advance_root.",
        ratio_change, max_ratio_1 * 100.0, max_ratio_2 * 100.0);
    
    println!("\n✓ Tree reuse test PASSED");
    println!("  No extreme stampede after advancing root");
}

#[test]
fn test_multiple_tree_reuse_cycles() {
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Standard Othello opening
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
        128 * 1024,
        256,
    ).expect("Failed to create engine");
    
    engine.init_tree(&board, 1, &legal_moves);
    
    println!("\n=== MULTIPLE TREE REUSE CYCLES TEST ===");
    
    for cycle in 0..3 {
        println!("\n--- Cycle {} ---", cycle);
        
        let _telemetry = engine.run_incremental_mcts(
            8192,
            50,  // Shorter searches for multiple cycles
            1.0,
            0.2,
            0.06,
            42 + cycle as u32,
            None,
            false,
        );
        
        let stats = engine.get_children_stats();
        let total_visits: i32 = stats.iter().map(|s| s.2).sum();
        
        if total_visits == 0 {
            println!("No visits recorded in cycle {}", cycle);
            continue;
        }
        
        let max_visits = stats.iter().map(|s| s.2).max().unwrap();
        let max_ratio = max_visits as f64 / total_visits as f64;
        
        println!("Visit distribution:");
        for &(x, y, visits, _wins, q) in &stats {
            if visits > 0 {
                let ratio = visits as f64 / total_visits as f64;
                println!("  ({},{}) - {} visits ({:.1}%), Q={:.4}", x, y, visits, ratio * 100.0, q);
            }
        }
        
        assert!(max_ratio < 0.85,
            "Cycle {}: Extreme stampede detected - {:.1}% to one child",
            cycle, max_ratio * 100.0);
        
        // Advance to first child for next cycle
        if let Some(&(x, y, _, _, _)) = stats.iter().find(|s| s.2 > 0) {
            // Just use same board and moves for simplicity (not a real game)
            engine.advance_root(x, y, &board, -1, &legal_moves);
            println!("Advanced to child ({}, {})", x, y);
        }
    }
    
    println!("\n✓ Multiple cycle test PASSED");
}
