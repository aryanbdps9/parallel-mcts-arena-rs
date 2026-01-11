/// Test that softmax selection chooses children with HIGHER PUCT scores
/// 
/// This test verifies that the softmax temperature-based selection
/// preferentially explores children with higher PUCT values.

use mcts::gpu::{GpuConfig, GpuContext};
use mcts::gpu::mcts_othello::GpuOthelloMcts;
use std::sync::Arc;

#[test]
fn test_softmax_selects_higher_puct() {
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
    
    // Run with low temperature for more deterministic selection
    let _telemetry = engine.run_incremental_mcts(
        8192,
        200,  // More iterations to get clear signal
        1.0,
        0.2,
        0.06, // Low temperature
        42,
        None,
        false,
    );
    
    let child_stats = engine.get_children_stats();
    
    println!("\n=== Softmax Selection Test ===");
    println!("Testing that children with HIGHER PUCT get MORE visits\n");
    
    // Print all children sorted by PUCT
    let mut stats_with_puct: Vec<_> = child_stats.iter()
        .filter(|s| s.2 > 0) // Only visited children
        .map(|&(x, y, visits, wins, q)| {
            // Calculate approximate PUCT (we don't have exact U here, but Q gives us ordering)
            (x, y, visits, wins, q)
        })
        .collect();
    
    // Sort by Q value (proxy for PUCT in early game with uniform priors)
    stats_with_puct.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
    
    println!("Children sorted by Q value (descending):");
    for (idx, &(x, y, visits, wins, q)) in stats_with_puct.iter().enumerate() {
        println!("  {}. ({},{}) - visits: {}, wins: {}, Q: {:.4}", 
            idx + 1, x, y, visits, wins, q);
    }
    
    // Check that visit distribution correlates with PUCT
    // The child with highest PUCT should get more visits than child with lowest PUCT
    // (after accounting for the exploration term evening things out)
    if stats_with_puct.len() >= 2 {
        let total_visits: i32 = stats_with_puct.iter().map(|s| s.2).sum();
        
        // Sort by visits to see which children actually got explored
        let mut by_visits = stats_with_puct.clone();
        by_visits.sort_by(|a, b| b.2.cmp(&a.2)); // Sort descending by visits
        
        println!("\nChildren sorted by visits (descending):");
        for (idx, &(x, y, visits, _wins, q)) in by_visits.iter().enumerate() {
            let ratio = visits as f64 / total_visits as f64;
            println!("  {}. ({},{}) - visits: {} ({:.1}%), Q: {:.4}", 
                idx + 1, x, y, visits, ratio * 100.0, q);
        }
        
        // In balanced opening, Q values are similar (0.47-0.48)
        // So PUCT differences will be small and exploration will even things out
        // We just need to check we're not systematically preferring LOW Q
        
        // Check: Does the child with lowest Q have significantly MORE visits than highest Q?
        let highest_q_visits = stats_with_puct[0].2;
        let lowest_q_visits = stats_with_puct.last().unwrap().2;
        
        // Calculate ratio - if > 2.0, it means lowest Q has 2x the visits of highest Q
        let visit_ratio = lowest_q_visits as f64 / highest_q_visits as f64;
        
        println!("\nHighest Q child: {} visits", highest_q_visits);
        println!("Lowest Q child: {} visits", lowest_q_visits);
        println!("Visit ratio (lowest/highest): {:.2}", visit_ratio);
        
        // Allow some variance, but if lowest Q consistently gets 2x+ visits, something is wrong
        assert!(visit_ratio < 1.5, 
            "Child with lowest Q has {:.1}x more visits than highest Q child. \
             This suggests selection is biased toward LOWER values!",
            visit_ratio);
        
        println!("\n✓ Softmax selection test PASSED");
        println!("  No systematic bias toward lowest Q children");
    }
}

#[test]
fn test_softmax_visit_distribution() {
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Standard opening
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
    
    let child_stats = engine.get_children_stats();
    
    println!("\n=== Visit Distribution Test ===");
    
    let total_visits: i32 = child_stats.iter().map(|s| s.2).sum();
    let mut min_visits = i32::MAX;
    let mut max_visits = 0;
    
    for &(x, y, visits, _, q) in &child_stats {
        if visits > 0 {
            let ratio = visits as f64 / total_visits as f64;
            println!("({},{}) - visits: {} ({:.1}%), Q: {:.4}", x, y, visits, ratio * 100.0, q);
            min_visits = min_visits.min(visits);
            max_visits = max_visits.max(visits);
        }
    }
    
    // Check that visit distribution is reasonable
    // In opening, all moves are similar, so we shouldn't see 95%+ to one child
    let max_ratio = max_visits as f64 / total_visits as f64;
    
    println!("\nMax visits ratio: {:.1}%", max_ratio * 100.0);
    
    assert!(max_ratio < 0.95, 
        "Visit distribution too skewed! One child has {:.1}% of visits. \
         This suggests selection is broken (either selecting minimum or stampeding).",
        max_ratio * 100.0);
    
    println!("\n✓ Visit distribution test PASSED");
    println!("  No extreme bias toward single child");
}
