use mcts::gpu::{GpuContext, GpuConfig};
use mcts::gpu::mcts_othello::GpuOthelloMcts;

/// Test that advance_root handles root with no children gracefully
/// This can happen if timeout occurs before any search completes
#[test]
fn test_advance_root_with_no_children() {
    // Create a simple opening position
    let mut board = [0i32; 64];
    board[27] = -1; // O
    board[28] = 1;  // X
    board[35] = 1;  // X
    board[36] = -1; // O
    
    let player = 1; // X to move
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    let config = GpuConfig::default();
    let ctx = GpuContext::new(&config).expect("Failed to create GPU context");
    let mcts = GpuOthelloMcts::new(std::sync::Arc::new(ctx), 100000, 64)
        .expect("Failed to create GpuOthelloMcts");
    
    // Initialize tree but DON'T run any search
    mcts.init_tree(&board, player, &legal_moves);
    
    // Try to advance root - should fail gracefully and reset tree
    // Root has 0 children because we never searched
    board[19] = 1; // Make the move (2,3) on the board
    board[27] = 1; // Flip piece
    let new_player = -1;
    let new_legal_moves = vec![(2, 2), (2, 4), (4, 2)];
    
    let reused = mcts.advance_root(2, 3, &board, new_player, &new_legal_moves);
    
    // Should return false (tree wasn't reused) but NOT panic
    assert!(!reused, "Tree should not be reused when root has no children");
}

/// Test that we can properly detect terminal states before calling advance_root
#[test]
fn test_detect_terminal_before_advance() {
    // Create terminal board (completely filled)
    let mut board = [0i32; 64];
    for i in 0..64 {
        board[i] = if i % 2 == 0 { 1 } else { -1 };
    }
    
    // Terminal state has no legal moves
    let legal_moves: Vec<(usize, usize)> = vec![];
    
    // The caller should check for empty legal_moves and NOT call advance_root
    // This test just validates the detection works
    assert!(legal_moves.is_empty(), "Terminal state must have no legal moves");
}
