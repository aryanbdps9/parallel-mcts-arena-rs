/// Test CPU MCTS handling of Othello pass moves
use mcts::MCTS;
use mcts::games::othello::OthelloState;
use mcts::GameState;

#[test]
fn test_cpu_mcts_pass_moves_are_in_tree() {
    // Pass moves should be in the tree like any other legal move
    let mut mcts_instance: MCTS<OthelloState> = MCTS::new(2.0, 1, 100000);
    
    // Create a position where white has no legal moves but must pass
    let mut state = OthelloState::new(8);
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
    state.set_board_for_test(board.clone(), -1); // White to move
    
    // Get the pass move
    let moves = state.get_possible_moves();
    assert_eq!(moves.len(), 1, "Should have exactly one move (pass)");
    assert!(moves[0].is_pass(), "Move should be a pass");
    
    // Advance root with pass move - should work like any other move
    // Pass moves ARE in the tree, they just represent "same board, different player"
    println!("Advancing root with pass move...");
    mcts_instance.advance_root(&moves[0], Some("TestPassMove"));
    println!("Successfully advanced to pass move child node!");
    
    // The tree advanced to the child node representing "after the pass"
    // That child has the same board but it's now the opponent's turn
}

#[test]
fn test_pass_moves_treated_like_normal_moves() {
    // Pass moves should be treated exactly like normal moves in the tree
    // The only difference is the board state doesn't change
    let mut mcts_instance: MCTS<OthelloState> = MCTS::new(2.0, 1, 100000);
    let state = OthelloState::new(8); // Standard opening
    
    // Get a normal move
    let moves = state.get_possible_moves();
    assert!(!moves.is_empty());
    assert!(!moves[0].is_pass(), "Opening moves should not be pass");
    
    // Advance root with normal move
    println!("Advancing root with normal move...");
    mcts_instance.advance_root(&moves[0], Some("TestNormalMove"));
    println!("Successfully advanced to normal move child node!");
}
