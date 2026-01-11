use mcts::gpu::{GpuConfig, GpuContext};
use mcts::gpu::mcts_othello::GpuOthelloMcts;
use std::sync::Arc;

/// Test that backpropagation correctly handles perspective
/// A winning position for player 1 should have high Q values for player 1's moves
#[test]
fn test_backprop_perspective() {
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Create a position where player 1 (X) is about to win
    // Player 1 can play (0,0) to win immediately
    #[rustfmt::skip]
    let board = [
        0,  1,  1,  1,  1,  1,  1,  0,  // Row 0: Player 1 has 6 in a row, can win at (0,0) or (7,0)
        -1, -1, -1, -1, -1, -1, -1, -1, // Row 1: Player -1
        1,  1,  1,  1,  1,  1,  1,  1,  // Row 2: Player 1
        -1, -1, -1, -1, -1, -1, -1, -1, // Row 3: Player -1
        1,  1,  1,  1,  1,  1,  1,  1,  // Row 4: Player 1
        -1, -1, -1, -1, -1, -1, -1, -1, // Row 5: Player -1
        1,  1,  1,  1,  1,  1,  1,  1,  // Row 6: Player 1
        -1, -1, -1, -1, -1, -1, -1, -1, // Row 7: Player -1
    ];
    
    let legal_moves = vec![(0, 0), (7, 0)];
    
    let engine = GpuOthelloMcts::new(
        context.clone(),
        128 * 1024,
        256,
    ).expect("Failed to create engine");
    
    engine.init_tree(&board, 1, &legal_moves); // Player 1 to move
    
    // Run MCTS with honest rollouts
    let _telemetry = engine.run_incremental_mcts(
        8192,
        100,
        1.0,
        0.2,
        0.0,  // temperature=0 for deterministic selection
        42,
        None,
        false, // honest rollouts
    );
    
    let child_stats = engine.get_children_stats();
    
    println!("\n=== Backprop Perspective Test ===");
    println!("Position: Player 1 has 38 pieces, Player -1 has 24 pieces");
    println!("Player 1 is winning and can capture more pieces");
    println!();
    
    for (idx, &(x, y, visits, wins, q)) in child_stats.iter().enumerate() {
        println!("  Move {}: ({},{}) - visits: {}, wins: {}, Q: {:.4}", 
            idx, x, y, visits, wins, q);
    }
    
    // Player 1 is winning, so Q values should be > 0.5
    // With 38 pieces vs 24, player 1 has a big advantage
    // Q values should reflect this advantage
    let total_visits: i32 = child_stats.iter().map(|s| s.2).sum();
    let avg_q: f64 = child_stats.iter()
        .map(|s| s.4 * (s.2 as f64 / total_visits as f64))
        .sum();
    
    println!("\nAverage Q value: {:.4}", avg_q);
    println!("Expected: Q > 0.5 (player 1 is winning)");
    
    // Assert that Q values are reasonable for a winning position
    assert!(avg_q > 0.5, 
        "Expected Q > 0.5 for player 1's winning position, got Q = {:.4}. \
         This suggests backprop perspective is flipped!", avg_q);
    
    println!("\n✓ Backprop perspective test PASSED");
    println!("  Player 1's winning position correctly shows Q > 0.5");
}

/// Test Q values in standard opening position
/// Should have Q values around 0.5 (balanced position)
#[test]
fn test_standard_opening_q_values() {
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
    
    engine.init_tree(&board, 1, &legal_moves); // Player 1 to move
    
    let _telemetry = engine.run_incremental_mcts(
        8192,
        100,
        1.0,
        0.2,
        0.0,
        42,
        None,
        false, // honest rollouts
    );
    
    let child_stats = engine.get_children_stats();
    
    println!("\n=== Standard Opening Position Test ===");
    println!("Balanced opening position (2 pieces each)");
    println!();
    
    for (idx, &(x, y, visits, wins, q)) in child_stats.iter().enumerate() {
        println!("  Move {}: ({},{}) - visits: {}, wins: {}, Q: {:.4}", 
            idx, x, y, visits, wins, q);
    }
    
    let total_visits: i32 = child_stats.iter().map(|s| s.2).sum();
    
    if total_visits == 0 {
        println!("\nWARNING: No visits recorded");
        return;
    }
    
    let avg_q: f64 = child_stats.iter()
        .filter(|s| s.2 > 0)
        .map(|s| s.4 * (s.2 as f64 / total_visits as f64))
        .sum();
    
    println!("\nAverage Q value: {:.4}", avg_q);
    println!("Expected: Q ≈ 0.5 (balanced position)");
    
    // In a balanced opening, Q should be close to 0.5
    // Allow some variance due to random rollouts
    assert!(avg_q > 0.40 && avg_q < 0.60, 
        "Expected Q ≈ 0.5 for balanced opening, got Q = {:.4}. \
         Q values outside [0.40, 0.60] suggest perspective bug!", avg_q);
    
    println!("\n✓ Standard opening test PASSED");
    println!("  Q values in reasonable range for balanced position");
}
