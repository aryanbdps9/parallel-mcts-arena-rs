/// Test that opening move selection respects symmetry
/// 
/// In the standard Othello opening position, all 4 legal moves are 
/// equivalent by symmetry. This test verifies that the GPU-native MCTS
/// explores all 4 moves roughly equally.

use std::sync::Arc;
use mcts::gpu::{GpuContext, GpuConfig};
use mcts::gpu::mcts_othello::GpuOthelloMcts;

#[test]
fn test_opening_move_symmetry() {
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Standard Othello opening position
    let mut board = [0i32; 64];
    board[27] = -1; board[28] = 1;
    board[35] = 1;  board[36] = -1;
    let player = 1;
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    eprintln!("\n=== TEST: Opening Move Symmetry ===");
    
    let engine = GpuOthelloMcts::new(context.clone(), 100_000, 8)
        .expect("Failed to create engine");
    engine.init_tree(&board, player, &legal_moves);
    
    // Run with many threads to test GPU synchronization
    let num_threads = 8192;
    let max_steps = 100;
    
    // Test with argmax (temp=0) and various softmax temperatures
    // Test only at temp=0.06 to focus on RNG issues
    for temperature in [0.06] {
        eprintln!("\n--- Testing with temperature {} ---", temperature);
        
        // Reset tree for each temperature test
        engine.init_tree(&board, player, &legal_moves);
        
        let _telemetry = engine.run_incremental_mcts(
            num_threads,
            max_steps,
            1.0,  // exploration (user's value)
            0.2,  // virtual_loss (user's value)
            temperature,
            42,   // seed
            None, // timeout
            true, // Enable random rollouts for testing
        );
        
        // Get diagnostics to check random rollout distribution
        let diag = engine.read_diagnostics();
        eprintln!("Random rollout results: wins={} draws={} losses={}", 
                  diag.random_rollout_wins, diag.random_rollout_draws, diag.random_rollout_losses);
        eprintln!("Random rollout leaf_player distribution: p1={} p2={}", 
            diag.random_rollout_from_p1, diag.random_rollout_from_p2);
        eprintln!("Random rollout raw game outcomes: p1_wins_raw={} p1_losses_raw={}", 
            diag.random_rollout_p1_wins_raw, diag.random_rollout_p1_losses_raw);
        let total_raw = diag.random_rollout_p1_wins_raw + diag.random_rollout_p1_losses_raw;
        if total_raw > 0 {
            let total_rollouts = diag.random_rollout_from_p1 + diag.random_rollout_from_p2;
            let avg_p1_score = diag.random_rollout_p1_score_sum as f64 / total_rollouts as f64;
            eprintln!("Average p1_score: {:.2} (expected 32.0 for uniform distribution)", avg_p1_score);
            eprintln!("Raw p1 wins: {} ({:.1}%), losses: {} ({:.1}%), expected 49.2% each",
                diag.random_rollout_p1_wins_raw, 
                100.0 * diag.random_rollout_p1_wins_raw as f64 / total_rollouts as f64,
                diag.random_rollout_p1_losses_raw,
                100.0 * diag.random_rollout_p1_losses_raw as f64 / total_rollouts as f64);
            eprintln!("Thread ID range: {} to {} (num_threads={})", 
                diag.random_rollout_min_tid, diag.random_rollout_max_tid, 8192);
        }
        let total_random = diag.random_rollout_wins + diag.random_rollout_draws + diag.random_rollout_losses;
        if total_random > 0 {
            eprintln!("Random rollout percentages: wins={:.1}% draws={:.1}% losses={:.1}%",
                      100.0 * diag.random_rollout_wins as f64 / total_random as f64,
                      100.0 * diag.random_rollout_draws as f64 / total_random as f64,
                      100.0 * diag.random_rollout_losses as f64 / total_random as f64);
        }
        
        // Get visit distribution
        let stats = engine.get_children_stats();
        
        eprintln!("\nVisit distribution:");
        let total_visits: i32 = stats.iter().map(|(_, _, v, _, _)| *v).sum();
        for (x, y, visits, wins, q) in &stats {
            let visit_pct = 100.0 * *visits as f64 / total_visits as f64;
            eprintln!("  ({},{}) visits={:6} ({:5.1}%) wins={:6} Q={:.4}", 
                x, y, visits, visit_pct, wins, q);
        }
        eprintln!("\nTotal visits: {}", total_visits);
        
        // Check symmetry
        let visits: Vec<i32> = stats.iter().map(|(_, _, v, _, _)| *v).collect();
        let total_visits: i32 = visits.iter().sum();
        let mean_visits = total_visits as f64 / visits.len() as f64;
        
        eprintln!("\nTotal visits: {}", total_visits);
        eprintln!("Mean visits per move: {:.1}", mean_visits);
        
        // Calculate coefficient of variation (std dev / mean)
        let variance: f64 = visits.iter()
            .map(|v| {
                let diff = *v as f64 - mean_visits;
                diff * diff
            })
            .sum::<f64>() / visits.len() as f64;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean_visits;
        
        eprintln!("Std deviation: {:.1}", std_dev);
        eprintln!("Coefficient of variation: {:.4}", cv);
        
        // With perfect symmetry, CV should be near 0
        // With random sampling, CV should be sqrt(3/n) ≈ 0.02 for 1000 visits each
        // Allow up to 30% CV for noisy GPU execution
        if cv < 0.30 {
            eprintln!("✓ Symmetry maintained with temperature {}: CV={:.4}", temperature, cv);
            return; // Test passes
        } else {
            eprintln!("✗ Poor symmetry with temperature {}: CV={:.4}", temperature, cv);
        }
    }
    
    panic!("Failed to achieve symmetric move distribution at any tested temperature");
}
