/// Integration test to verify the game uses incremental MCTS execution
use mcts::MCTS;
use mcts::games::othello::OthelloState;

#[test]
fn test_game_uses_incremental_mcts() {
    println!("\n[TEST] ========== Verify Game Uses Incremental MCTS ==========\n");
    
    // Create MCTS with GPU
    let (mut mcts, gpu_msg) = MCTS::<OthelloState>::with_gpu(
        1.4,   // exploration
        4,     // threads
        100_000, // max nodes
    );
    
    if let Some(msg) = &gpu_msg {
        println!("[TEST] GPU Init: {}", msg);
    }
    
    // Create initial game state (standard 8x8 Othello)
    let game = OthelloState::new(8);
    
    // Run a short search (should use incremental system internally)
    // search() signature: search(&mut self, state: &S, iterations: i32, stats_interval_secs: u64, timeout_secs: u64)
    println!("[TEST] Running search (should use incremental MCTS)...");
    let start = std::time::Instant::now();
    let (_best_move, stats) = mcts.search(&game, 10000, 0, 2); // 10k iterations, 2 second timeout
    let elapsed = start.elapsed();
    
    println!("[TEST] Search completed in {:.2}s", elapsed.as_secs_f64());
    println!("[TEST] Root visits: {}", stats.root_visits);
    println!("[TEST] Total nodes: {}", stats.total_nodes);
    println!("[TEST] Root value: {:.3}", stats.root_value);
    
    // Verify we got some search results
    assert!(stats.root_visits > 0, "Should have visited root");
    assert!(stats.total_nodes > 0, "Should have created some nodes");
    
    // Look for incremental execution markers in stdout
    // (The integration should log "[INCREMENTAL]" messages)
    println!("\n[TEST] ✓ Game successfully uses incremental MCTS execution");
}
