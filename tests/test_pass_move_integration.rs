/// Integration test for pass move handling in actual gameplay
use mcts::games::othello::OthelloState;
use mcts::GameState;

#[test]
fn test_pass_move_integration() {
    // Simulate a game scenario where a pass occurs
    let mut state = OthelloState::new(8);
    
    // Set up a late-game position where white will have no moves
    let board = vec![
        1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1, -1,  0,  1,  1,
        1,  1,  1, -1, -1, -1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,
    ];
    
    state.set_board_for_test(board, -1); // White to move
    
    // 1. White has no moves, must pass
    let white_moves = state.get_possible_moves();
    assert_eq!(white_moves.len(), 1);
    assert!(white_moves[0].is_pass());
    
    // 2. Make the pass move
    let pass_move = white_moves[0].clone();
    state.make_move(&pass_move);
    
    // 3. Now it's black's turn
    assert_eq!(state.get_current_player(), 1);
    
    // 4. Black should have at least one legal move
    let black_moves = state.get_possible_moves();
    assert!(!black_moves.is_empty());
    assert!(black_moves.iter().any(|m| !m.is_pass()), "Black should have non-pass moves");
    
    // 5. Game should not be terminal yet
    assert!(!state.is_terminal());
}

#[test]
fn test_consecutive_passes_lead_to_terminal() {
    // Create a completely filled board
    let mut state = OthelloState::new(8);
    let board = vec![1; 64];
    state.set_board_for_test(board, 1);
    
    // Should be terminal (no moves for either player)
    assert!(state.is_terminal());
    
    // Should have no moves at all
    let moves = state.get_possible_moves();
    assert!(moves.is_empty(), "Terminal state should have no moves, not even pass");
}
