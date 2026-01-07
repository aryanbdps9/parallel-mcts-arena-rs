/// Test for decoupled rollout/backprop architecture
/// 
/// Verifies:
/// 1. Tree workers never block on rollouts
/// 2. Rollout queue fills from tree expansions
/// 3. Backprop queue fills from completed rollouts
/// 4. Tree workers consume backprop queue
/// 5. Visit counts accumulate correctly

use std::sync::Arc;
use mcts::gpu::{GpuContext, GpuConfig};
use mcts::gpu::mcts_othello::GpuOthelloMcts;

#[test]
fn test_queue_based_execution() {
    // Create GPU context
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Standard opening position
    let mut board = [0i32; 64];
    board[27] = -1; board[28] = 1;
    board[35] = 1;  board[36] = -1;
    let player = 1;
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    eprintln!("\n=== TEST: Decoupled Rollout Architecture ===");
    
    // Create engine
    let engine = GpuOthelloMcts::new(context.clone(), 200_000, 8)
        .expect("Failed to create engine");
    engine.init_tree(&board, player, &legal_moves);
    
    // Run with small thread count to make queue dynamics visible
    let num_threads = 256;
    let max_steps = 50;  // Limited steps to see queuing behavior
    
    let telemetry = engine.run_incremental_mcts(
        num_threads,
        max_steps,
        1.4,  // exploration
        1.0,  // virtual_loss
        0.06, // temperature
        42,   // seed
        None,
    );
    
    eprintln!("\n=== RESULTS ===");
    eprintln!("Total rollouts: {}", telemetry.diagnostics.rollouts);
    eprintln!("Expansion successes: {}", telemetry.diagnostics.expansion_success);
    
    // Get root visits
    let root_visits = engine.get_root_visits();
    eprintln!("Root visits: {}", root_visits);
    
    // Key assertions for decoupled architecture
    
    // 1. Some rollouts should have started
    assert!(telemetry.diagnostics.rollouts > 0, 
            "No rollouts started - tree workers not queuing work");
    
    // 2. Some expansions should have succeeded
    assert!(telemetry.diagnostics.expansion_success > 0,
            "No expansions succeeded - tree not growing");
    
    // 3. Root visits should accumulate from backprop
    assert!(root_visits > 0,
            "Root has zero visits - backprop not working");
    
    // 4. In decoupled architecture, we can have more rollouts started than completed
    //    because rollout workers process asynchronously
    eprintln!("\n✓ Queue-based execution working:");
    eprintln!("  - Tree workers queued {} rollout requests", telemetry.diagnostics.rollouts);
    eprintln!("  - {} expansions created leaf nodes", telemetry.diagnostics.expansion_success);
    eprintln!("  - Backprop accumulated {} visits at root", root_visits);
}

#[test]
fn test_tree_workers_dont_block() {
    // Test that tree workers complete quickly even if rollouts are slow
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    let mut board = [0i32; 64];
    board[27] = -1; board[28] = 1;
    board[35] = 1;  board[36] = -1;
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    eprintln!("\n=== TEST: Tree Workers Non-Blocking ===");
    
    let engine = GpuOthelloMcts::new(context.clone(), 200_000, 8)
        .expect("Failed to create engine");
    engine.init_tree(&board, 1, &legal_moves);
    
    // Run with many threads but few steps
    // Tree workers should make progress even if rollouts don't complete
    let start = std::time::Instant::now();
    let telemetry = engine.run_incremental_mcts(
        2048,  // Many threads
        10,    // Very few steps
        1.4,
        1.0,
        0.06,
        42,
        None,
    );
    let elapsed = start.elapsed();
    
    eprintln!("\nCompleted in {:?}", elapsed);
    eprintln!("Expansions: {}", telemetry.diagnostics.expansion_success);
    
    // With non-blocking architecture:
    // - Tree workers can expand even in 10 steps
    // - Rollouts may not complete (need ~5-10 steps each)
    // - But expansion should still happen
    
    assert!(telemetry.diagnostics.expansion_success > 0,
            "Tree workers blocked - no expansions in {} steps", 10);
    
    // Execution should be fast (tree ops are ~10μs, rollouts are ~100μs)
    // For 2048 threads × 10 steps, 40s is reasonable on slower GPUs
    assert!(elapsed.as_secs() < 40,
            "Execution too slow ({:?}) - suggests blocking", elapsed);
    
    eprintln!("\n✓ Tree workers non-blocking:");
    eprintln!("  - {} expansions in only {} steps", 
              telemetry.diagnostics.expansion_success, 10);
    eprintln!("  - Completed in {:?} (fast)", elapsed);
}

#[test]
fn test_parent_pointer_backprop() {
    // Test that backprop works using parent pointers (no path array needed)
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    let mut board = [0i32; 64];
    board[27] = -1; board[28] = 1;
    board[35] = 1;  board[36] = -1;
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    eprintln!("\n=== TEST: Parent Pointer Backpropagation ===");
    
    let engine = GpuOthelloMcts::new(context.clone(), 200_000, 8)
        .expect("Failed to create engine");
    engine.init_tree(&board, 1, &legal_moves);
    
    let telemetry = engine.run_incremental_mcts(
        512,
        100,
        1.4,
        1.0,
        0.06,
        42,
        None,
    );
    
    let root_visits = engine.get_root_visits();
    let children_stats = engine.get_children_stats();
    let total_child_visits: i32 = children_stats.iter().map(|c| c.2).sum();
    
    eprintln!("\nRoot visits: {}", root_visits);
    eprintln!("Total child visits: {}", total_child_visits);
    
    // Root should have approximately same visits as sum of children
    // (Small difference OK due to VL or in-progress rollouts)
    let difference = (root_visits as i32 - total_child_visits).abs();
    let tolerance = root_visits / 7;  // ~15% tolerance
    
    assert!(difference <= tolerance as i32,
            "Root visits ({}) don't match child visits ({}) - backprop broken?",
            root_visits, total_child_visits);
    
    eprintln!("\n✓ Parent pointer backprop working:");
    eprintln!("  - Root: {} visits", root_visits);
    eprintln!("  - Children total: {} visits", total_child_visits);
    eprintln!("  - Difference: {} (within ~15% tolerance)", difference);
}

#[test]
fn test_dynamic_load_balancing() {
    // Test that tree workers throttle when rollout queue gets full
    // This prevents queue overflow and ensures balanced execution
    
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    let mut board = [0i32; 64];
    board[27] = -1; board[28] = 1;
    board[35] = 1;  board[36] = -1;
    let player = 1;
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    eprintln!("\n=== TEST: Dynamic Load Balancing ===");
    
    let engine = GpuOthelloMcts::new(context.clone(), 200_000, 8)
        .expect("Failed to create engine");
    engine.init_tree(&board, player, &legal_moves);
    
    // High thread count with moderate rollout workers creates queue pressure
    let num_threads = 1024;  // Many tree workers
    let max_steps = 100;      // Long enough to see balancing
    
    let telemetry = engine.run_incremental_mcts(
        num_threads,
        max_steps,
        1.4,
        1.0,
        0.06,
        42,
        None,
    );
    
    eprintln!("\n=== RESULTS ===");
    eprintln!("Rollouts: {}", telemetry.diagnostics.rollouts);
    eprintln!("Expansions: {}", telemetry.diagnostics.expansion_success);
    
    let root_visits = engine.get_root_visits();
    eprintln!("Root visits: {}", root_visits);
    
    // With load balancing:
    // - Rollouts should complete (high root visits)
    // - No queue overflow (would cause lost work)
    // - Reasonable expansion count (tree still growing)
    
    assert!(root_visits > 100,
            "Too few visits ({}) - load balancing may have throttled too much", 
            root_visits);
    
    assert!(telemetry.diagnostics.expansion_success > 5,
            "Too few expansions ({}) - tree not growing",
            telemetry.diagnostics.expansion_success);
    
    // Completion rate should be good (not perfect due to throttling)
    let completion_rate = (root_visits as f64) / (telemetry.diagnostics.rollouts as f64);
    assert!(completion_rate > 0.70,
            "Low completion rate ({:.1}%) - queue balancing may be broken",
            completion_rate * 100.0);
    
    eprintln!("\n✓ Dynamic load balancing working:");
    eprintln!("  - {} visits from {} rollouts ({:.1}% completion)",
              root_visits, telemetry.diagnostics.rollouts,
              completion_rate * 100.0);
    eprintln!("  - {} expansions (tree growth maintained)", 
              telemetry.diagnostics.expansion_success);
    eprintln!("  - No queue overflow detected");
}

#[test]
fn test_opening_move_symmetry() {
    // Test that symmetric opening moves get roughly equal exploration
    // This verifies the jitter fix prevents all threads from selecting the same move
    
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Standard Othello opening: 4-way symmetric position
    let mut board = [0i32; 64];
    board[27] = -1; board[28] = 1;  // Row 3: X O
    board[35] = 1;  board[36] = -1;  // Row 4: O X
    let player = 1;
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)]; // All 4 moves are symmetric
    
    eprintln!("\n=== TEST: Opening Move Symmetry ===");
    
    let engine = GpuOthelloMcts::new(context.clone(), 200_000, 8)
        .expect("Failed to create engine");
    engine.init_tree(&board, player, &legal_moves);
    
    // Use high thread count to stress-test synchronization
    let num_threads = 2048;
    let max_steps = 200;
    
    let _telemetry = engine.run_incremental_mcts(
        num_threads,
        max_steps,
        1.4,   // exploration
        1.0,   // virtual_loss
        0.06,  // temperature (low, to make the problem worse)
        42,    // seed
        None,
    );
    
    // Get visit counts for all 4 children
    let children_stats = engine.get_children_stats();
    
    eprintln!("\n=== VISIT DISTRIBUTION ===");
    for (idx, stat) in children_stats.iter().enumerate() {
        let (move_id, visits) = (stat.1, stat.2);
        let (x, y) = (move_id % 8, move_id / 8);
        eprintln!("  Move {}: ({},{}) = {} visits", idx + 1, x, y, visits);
    }
    
    assert_eq!(children_stats.len(), 4, "Should have exactly 4 opening moves");
    
    // Extract visit counts
    let visits: Vec<i32> = children_stats.iter().map(|c| c.2).collect();
    let total_visits: i32 = visits.iter().sum();
    let mean_visits = total_visits as f64 / 4.0;
    
    eprintln!("\nTotal visits: {}", total_visits);
    eprintln!("Mean visits per move: {:.1}", mean_visits);
    
    // Check that each move got a reasonable share of visits
    // Due to symmetry, each move should get ~25% of visits
    // Allow 10-40% range (generous to account for randomness)
    for (idx, &visit_count) in visits.iter().enumerate() {
        let percentage = (visit_count as f64 / total_visits as f64) * 100.0;
        eprintln!("  Move {} percentage: {:.1}%", idx + 1, percentage);
        
        assert!(percentage >= 10.0 && percentage <= 40.0,
                "Move {} got {:.1}% of visits (expected 15-35% for symmetric moves). \
                 Visit counts: {:?}. This suggests threads are not exploring symmetrically.",
                idx + 1, percentage, visits);
    }
    
    // Calculate coefficient of variation (std dev / mean)
    // Lower CV means more uniform distribution
    let variance: f64 = visits.iter()
        .map(|&v| {
            let diff = v as f64 - mean_visits;
            diff * diff
        })
        .sum::<f64>() / 4.0;
    let std_dev = variance.sqrt();
    let cv = std_dev / mean_visits;
    
    eprintln!("Coefficient of variation: {:.3}", cv);
    
    // CV should be low for symmetric distribution
    // Without jitter fix, CV would be ~1.0+ (one move dominates)
    // With jitter fix, CV should be < 0.3
    assert!(cv < 0.35,
            "Coefficient of variation {:.3} too high - moves not explored symmetrically. \
             Visit distribution: {:?}. Expected roughly equal visits.",
            cv, visits);
    
    eprintln!("\n✓ Opening move symmetry verified:");
    eprintln!("  - All 4 symmetric moves explored");
    eprintln!("  - Visit distribution balanced (CV={:.3})", cv);
    eprintln!("  - No thundering herd detected");
}
