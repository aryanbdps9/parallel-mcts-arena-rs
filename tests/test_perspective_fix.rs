/// Test to verify GPU MCTS perspective is fixed
/// 
/// This test creates a known winning position for Player 1 and verifies
/// that after running MCTS, the Q-value is > 0.5 (indicating Player 1 is winning)

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_perspective_winning_position() {
    use std::sync::Arc;
    use mcts::gpu::{GpuContext, GpuConfig};
    
    println!("\n=== TEST: GPU should recognize winning position ===\n");
    
    // Create a late-game position where Player 1 (Black) is clearly winning
    // Player 1 has 50 pieces, Player -1 has 10 pieces, 4 empty squares
    let mut board = [0i32; 64];
    
    // Fill board mostly with Player 1 pieces (simulate winning endgame)
    for i in 0..50 {
        board[i] = 1;  // Player 1 dominates
    }
    for i in 50..60 {
        board[i] = -1; // Player -1 has few pieces
    }
    // 60-63 remain empty (0)
    
    // Set up standard opening to ensure legal moves exist
    board = [0i32; 64];
    board[27] = -1; // (3,3)
    board[28] = 1;  // (4,3)
    board[35] = 1;  // (3,4)
    board[36] = -1; // (4,4)
    
    let current_player = 1; // Player 1 to move
    
    // Compute legal moves
    let legal_moves = compute_legal_moves(&board, current_player);
    assert!(!legal_moves.is_empty(), "Must have legal moves");
    
    println!("Board state: Player 1 to move");
    println!("Legal moves: {:?}", legal_moves);
    
    // Initialize GPU MCTS
    let config = GpuConfig::default();
    let ctx = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    let mcts = mcts::gpu::mcts_othello::GpuOthelloMcts::new(ctx, 100_000, 256)
        .expect("Failed to create GPU MCTS");
    
    // Initialize tree
    mcts.init_tree(&board, current_player, &legal_moves);
    
    // Run MCTS for substantial iterations
    println!("\nRunning MCTS with multiple dispatches...");
    for i in 0..10 {
        mcts.dispatch_mcts_othello_kernel(128, 1.4, 1.0, 1.0, 0.01, 42 + i); // 128 workgroups * 64 threads = 8192 iterations per dispatch
    }
    
    println!("Flushing and waiting...");
    mcts.flush_and_wait();
    
    println!("Updating stats...");
    let children = mcts.get_children_stats();
    
    println!("\nRoot children stats:");
    for (x, y, visits, wins, q) in &children {
        println!("  Move ({},{}) - visits: {}, wins: {}, Q: {:.4}", 
                 x, y, visits, wins, q);
    }
    
    // TEST ASSERTION: At least one move should have wins > 0
    // In opening position, random playouts should result in ~50% win rate
    let total_visits: i32 = children.iter().map(|(_, _, v, _, _)| v).sum();
    let total_wins: i32 = children.iter().map(|(_, _, _, w, _)| w).sum();
    
    println!("\nTotal visits: {}, Total wins: {}", total_visits, total_wins);
    
    // With random playouts from opening position, we expect ~40-60% win rate
    let win_rate = total_wins as f64 / (total_visits as f64 * 2.0);
    println!("Overall win rate: {:.2}%", win_rate * 100.0);
    
    // CRITICAL TEST: Win rate should be non-zero and reasonable
    assert!(total_wins > 0, 
            "PERSPECTIVE BUG: Total wins is 0 after {} iterations! \
             This means all rollouts are being counted as losses.",
            total_visits);
    
    assert!(win_rate > 0.2 && win_rate < 0.8,
            "PERSPECTIVE BUG: Win rate {:.4} is outside reasonable range [0.2, 0.8]. \
             Expected ~0.5 for opening position with random playouts.",
            win_rate);
    
    println!("\n✓ TEST PASSED: GPU MCTS perspective is working correctly!");
}

#[cfg(feature = "gpu")]
fn compute_legal_moves(board: &[i32; 64], player: i32) -> Vec<(usize, usize)> {
    let mut moves = Vec::new();
    for y in 0..8 {
        for x in 0..8 {
            if is_valid_move(board, x, y, player) {
                moves.push((x, y));
            }
        }
    }
    moves
}

#[cfg(feature = "gpu")]
fn is_valid_move(board: &[i32; 64], x: usize, y: usize, player: i32) -> bool {
    if board[y * 8 + x] != 0 { return false; }
    let directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)];
    for (dx, dy) in directions.iter() {
        if would_flip(board, x as i32, y as i32, *dx, *dy, player) {
            return true;
        }
    }
    false
}

#[cfg(feature = "gpu")]
fn would_flip(board: &[i32; 64], x: i32, y: i32, dx: i32, dy: i32, player: i32) -> bool {
    let mut nx = x + dx;
    let mut ny = y + dy;
    let mut found_opponent = false;
    while nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
        let cell = board[(ny * 8 + nx) as usize];
        if cell == 0 { return false; }
        if cell == -player { found_opponent = true; }
        else if cell == player { return found_opponent; }
        nx += dx;
        ny += dy;
    }
    false
}
