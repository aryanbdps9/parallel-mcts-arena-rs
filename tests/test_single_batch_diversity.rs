#[cfg(feature = "gpu")]
#[test]
fn test_single_batch_softmax_diversity() {
    use mcts::MCTS;
    use mcts::games::othello::OthelloState;
    use mcts::GameState;
    
    // Create MCTS instance
    let mut mcts: MCTS<OthelloState> = MCTS::new(1.4, 1, 1000);
    
    // Set up initial board position
    let game = OthelloState::new(8);
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
    
    println!("Initial position has {} legal moves: {:?}", legal_moves_xy.len(), legal_moves_xy);
    
    // Initialize GPU MCTS
    mcts.init_gpu_native_othello(&board, current_player, &legal_moves_xy, 8192);
    
    // Run ONLY 1 BATCH to see within-batch diversity
    println!("\n=== SINGLE BATCH TEST ===");
    if let Some(((_x, _y), _visits, _q, children_stats, _total_nodes, telemetry)) = mcts.search_gpu_native_othello(
        &board,
        current_player,
        &legal_moves_xy,
        8192,  // iterations per batch
        1,     // num_batches = 1 (SINGLE BATCH ONLY)
        1.4,   // exploration
        5.0,   // virtual_loss_weight
        1.0,   // temperature
        0.01,  // vl_temp_scale
        10,    // timeout_secs
        None,  // gpu_max_nodes
    ) {
        println!("\nSingle batch completed:");
        println!("  Total rollouts: {}", telemetry.diagnostics.rollouts);
        
        // Analyze children visit distribution FROM SINGLE BATCH
        println!("\nChildren visit distribution (single batch):");
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
        
        println!("\nSingle-batch diversity metrics:");
        println!("  Children with visits: {}/{}", children_with_visits, children_stats.len());
        println!("  Total visits: {}", total_visits);
        
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
        let normalized_entropy = entropy / max_entropy;
        
        println!("  Entropy: {:.3} / {:.3} (normalized: {:.1}%)", 
                 entropy, max_entropy, normalized_entropy * 100.0);
        
        // CRITICAL TEST: Within a single batch, do different threads select different children?
        assert!(
            children_with_visits >= 2,
            "BUG: Within a single batch, only {} child(ren) received visits! \
             Threads in the SAME batch should be selecting different children via softmax. \
             This would mean all threads picked the same child. Children: {:?}",
            children_with_visits,
            children_stats
        );
        
        // If only 2 children explored, entropy should still be reasonable
        // (not all visits on one child)
        assert!(
            normalized_entropy > 0.2,
            "BUG: Within single batch, entropy too low ({:.1}%)! \
             Threads should diversify via softmax sampling even in one batch. \
             Children: {:?}",
            normalized_entropy * 100.0,
            children_stats
        );
        
        println!("\nSUCCESS: Within a single batch, threads select different children!");
        println!("  ✓ {} children explored in one batch", children_with_visits);
        println!("  ✓ Entropy: {:.1}% (threads are diversifying)", normalized_entropy * 100.0);
        
    } else {
        panic!("search_gpu_native_othello returned None");
    }
}
