use mcts::games::othello::{OthelloState, OthelloMove};
use mcts::GameState;

#[test]
fn test_pass_move_when_no_legal_moves() {
    // Create a late-game position where white has no legal moves but black does
    let mut state = OthelloState::new(8);
    
    // Set up a board where white (-1) is surrounded and has no moves
    // but black still has moves (so not terminal)
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
    
    // White should have no legal moves (can't place anywhere that would flip pieces)
    // But game isn't terminal (black can still move at position (3,5))
    let moves = state.get_possible_moves();
    
    println!("Moves for white: {:?}", moves);
    println!("Is terminal: {}", state.is_terminal());
    
    // Should return exactly one move: the pass move
    assert_eq!(moves.len(), 1, "Should have exactly one move (pass), got {:?}", moves);
    assert!(moves[0].is_pass(), "The only move should be a pass move");
    
    // Pass should be legal
    assert!(state.is_legal(&moves[0]), "Pass move should be legal");
    
    // Make the pass move
    state.make_move(&moves[0]);
    
    // Current player should switch to black
    assert_eq!(state.get_current_player(), 1, "Should switch to black after pass");
    
    // Black should now have legal moves
    let black_moves = state.get_possible_moves();
    assert!(!black_moves.is_empty(), "Black should have moves available");
    assert!(black_moves.iter().any(|m| !m.is_pass()), "Black should have non-pass moves");
}

#[test]
fn test_terminal_when_both_players_have_no_moves() {
    let mut state = OthelloState::new(8);
    
    // Create a completely filled board
    let board = vec![1; 64]; // All black
    state.set_board_for_test(board, 1);
    
    // Should be terminal
    assert!(state.is_terminal(), "Completely filled board should be terminal");
    
    // Should have no moves (not even pass)
    let moves = state.get_possible_moves();
    assert!(moves.is_empty(), "Terminal state should have no moves");
}

#[test]
fn test_normal_moves_when_available() {
    // Standard opening position
    let state = OthelloState::new(8);
    
    let moves = state.get_possible_moves();
    
    // Should have 4 standard opening moves, no pass
    assert_eq!(moves.len(), 4, "Opening should have 4 moves");
    assert!(moves.iter().all(|m| !m.is_pass()), "No move should be a pass in opening");
}

#[test]
fn test_pass_move_serialization() {
    let pass = OthelloMove::pass();
    assert!(pass.is_pass(), "Pass move should be identified as pass");
    assert_eq!(pass.0, usize::MAX);
    assert_eq!(pass.1, usize::MAX);
}
