#[cfg(feature = "gpu")]
#[test]
#[ignore] // TODO: Fix this test - needs refactoring to avoid tree resets between searches
fn test_same_iteration_softmax_diversity() {
    use mcts::MCTS;
    use mcts::games::othello::OthelloState;
    use mcts::GameState;
    
    // Create MCTS instance
    let mut mcts: MCTS<OthelloState> = MCTS::new(1.4, 1, 1000);
    
    // Set up ASYMMETRIC starting position (a few moves into the game)
    // This creates unequal child values for more interesting softmax behavior
    let mut game = OthelloState::new(8);
    // Make a couple moves to create asymmetry
    let moves_to_make = vec![(3, 2), (2, 2), (2, 3)];
    for &(row, col) in &moves_to_make {
        if let Some(m) = game.get_possible_moves().iter().find(|m| m.0 == row && m.1 == col) {
            game.make_move(m);
        }
    }
    
    let board_2d = game.get_board();
    let mut board = [0i32; 64];
    for (r, row) in board_2d.iter().enumerate() {
        for (c, &cell) in row.iter().enumerate() {
            board[r * 8 + c] = cell;
        }
    }
    
    let legal_moves = game.get_possible_moves();
    let legal_moves_xy: Vec<(usize, usize)> = legal_moves
        .iter()
        .map(|m| (m.1, m.0))
        .collect();
    
    let current_player = game.get_current_player();
    
    println!("Asymmetric position has {} legal moves: {:?}", legal_moves_xy.len(), legal_moves_xy);
    
    // Initialize GPU MCTS
    mcts.init_gpu_native_othello(&board, current_player, &legal_moves_xy, 8192);
    
    // FIRST: Build an established tree with many iterations
    println!("\n=== BUILDING TREE (many iterations) ===");
    mcts.search_gpu_native_othello(
        &board,
        current_player,
        &legal_moves_xy,
        8192,  // iterations_per_batch (threads)
        10,    // num_batches (more iterations for deeper tree)
        1.4,   // exploration
        1.0,   // virtual_loss_weight
        1.0,   // temperature
        0.01,  // vl_temp_scale
        30,    // timeout_secs (enough time to build tree)
        None,  // gpu_max_nodes
    );
    println!("Tree established with root expanded and children visited");
    
    // THEN: Test diversity with minimal iterations on the established tree
    // Using just a few threads to see if they select different children via softmax
    println!("\n=== SAME ITERATION TEST (8 threads, sufficient steps on existing tree) ===");
    if let Some(((_x, _y), _visits, _q, children_stats, _total_nodes, telemetry)) = mcts.search_gpu_native_othello(
        &board,
        current_player,
        &legal_moves_xy,
        8,     // iterations_per_batch = 8 threads (small number)
        10,    // num_batches (enough for ~100 total steps per thread)
        1.4,   // exploration
        1.0,   // virtual_loss_weight  
        1.0,   // temperature (softmax diversity)
        0.01,  // vl_temp_scale
        30,    // timeout_secs
        None,  // gpu_max_nodes
    ) {
        println!("\nMinimal iterations on existing tree:");
        println!("  Total rollouts: {}", telemetry.diagnostics.rollouts);
        
        // Analyze children visit distribution from MINIMAL iterations
        println!("\nChildren visit distribution (4 threads on existing tree):");
        let mut children_with_visits = 0;
        let total_visits: i32 = children_stats.iter().map(|(_, _, v, _, _)| v).sum();
        
        for (i, &(x, y, visits, wins, q)) in children_stats.iter().enumerate() {
            let visit_pct = if total_visits > 0 {
                (visits as f64 / total_visits as f64) * 100.0
            } else {
                0.0
            };
            println!("  Child {}: ({},{}) visits={} ({:.1}%) wins={} Q={:.4}", 
                     i+1, x, y, visits, visit_pct, wins, q);
            
            if visits > 0 {
                children_with_visits += 1;
            }
        }
        
        println!("\nSame-iteration diversity metrics:");
        println!("  Children with visits: {}/{}", children_with_visits, children_stats.len());
        println!("  Total visits: {}", total_visits);
        println!("  Rollouts per child avg: {:.1}", total_visits as f64 / children_with_visits as f64);
        
        // With only 2 iterations (workgroup dispatches), if we see multiple children,
        // it means threads within THE SAME workgroup are selecting different children
        
        // Calculate entropy
        let entropy: f64 = children_stats.iter()
            .map(|(_, _, v, _, _)| {
                if *v > 0 && total_visits > 0 {
                    let p = *v as f64 / total_visits as f64;
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();
        
        let max_entropy = (children_stats.len() as f64).log2();
        let normalized_entropy = if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 };
        
        println!("  Entropy: {:.3} / {:.3} (normalized: {:.1}%)", 
                 entropy, max_entropy, normalized_entropy * 100.0);
        
        // With an established tree and softmax sampling, different threads should
        // explore different children even in the same iteration
        // At minimum, we should see at least 2 different children explored
        assert!(
            children_with_visits >= 2,
            "BUG: With 4 threads on established tree, only {} child(ren) received visits! \
             Threads should select different children via softmax. \
             Children: {:?}",
            children_with_visits,
            children_stats
        );
        
        println!("\nSUCCESS: Threads select different children via softmax!");
        println!("  ✓ {} children explored by 4 threads", children_with_visits);
        if children_with_visits >= 3 {
            println!("  ✓ Strong softmax diversity (3+ children)");
        }
        
    } else {
        panic!("search_gpu_native_othello returned None");
    }
}
