/// Test GPU-native MCTS handling of Othello pass moves
use mcts::MCTS;
use mcts::games::othello::OthelloState;

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_native_pass_move_included() {
    // Create a position where white has no regular moves but black does
    // Use a realistic late-game position
    
    #[rustfmt::skip]
    let board = [
         1,  1,  1,  1,  1,  1,  1,  1,   // Row 0: all black
         1,  1,  1,  1,  1,  1,  1,  1,   // Row 1: all black
         1,  1,  1,  1,  1,  1,  1,  0,   // Row 2: mostly black, one empty at (7,2)
         1,  1,  1,  1,  1,  1, -1,  0,   // Row 3: mostly black, one white at (6,3), empty at (7,3)
         1,  1,  1,  1,  1, -1, -1,  0,   // Row 4: mostly black, white at (5,4) and (6,4), empty at (7,4)
         1,  1,  1,  1,  1,  1,  0,  0,   // Row 5: mostly black, empty at (6,5) and (7,5)
         1,  1,  1,  1,  1,  1,  1,  0,   // Row 6: mostly black, empty at (7,6)
         1,  1,  1,  1,  1,  1,  1,  0,   // Row 7: mostly black, empty at (7,7)
    ];
    
    // White (-1) to move - white pieces are trapped and have no moves
    // But black has moves (can play at the empty squares)
    let current_player = -1;
    
    // Create MCTS engine
    let mcts: MCTS<OthelloState> = MCTS::new(1.0, 1, 1000);
    
    // Call with the pass move provided
    let result = mcts.search_gpu_native_othello(
        &board,
        current_player,
        &[(usize::MAX, usize::MAX)], // Provide pass move
        8192,
        1,
        1.0,
        1.0,
        0.06,
        0.01,
        1,
        Some(10000),
    );
    
    // Should succeed and return the pass move
    assert!(result.is_some(), "GPU should accept pass move when player has no regular moves");
    
    let ((x, y), _visits, _q, children, _nodes, _telemetry) = result.unwrap();
    
    // Verify it's the pass move
    assert_eq!(x, usize::MAX, "Should return pass move x");
    assert_eq!(y, usize::MAX, "Should return pass move y");
    assert_eq!(children.len(), 1, "Should have one child (the pass move)");
    
    println!("✓ GPU correctly handles pass moves");
}

#[test]
#[cfg(feature = "gpu")]
#[should_panic(expected = "search called with no legal moves (terminal position)")]
fn test_gpu_native_terminal_position_panics() {
    // Create a terminal board (all squares filled)
    #[rustfmt::skip]
    let board = [
        1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,
    ];
    
    let current_player = -1;
    let mcts: MCTS<OthelloState> = MCTS::new(1.0, 1, 1000);
    
    // This should panic because the game is terminal
    let _ = mcts.search_gpu_native_othello(
        &board,
        current_player,
        &[], // No moves (terminal)
        8192,
        1,
        1.0,
        1.0,
        0.06,
        0.01,
        1,
        Some(10000),
    );
}
