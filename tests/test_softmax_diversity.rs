#[cfg(feature = "gpu")]
#[test]
fn test_threads_select_different_children() {
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
    
    // Run a search with multiple batches to accumulate visits
    if let Some(((_x, _y), _visits, _q, children_stats, _total_nodes, telemetry)) = mcts.search_gpu_native_othello(
        &board,
        current_player,
        &legal_moves_xy,
        8192,  // iterations per batch - enough threads to see diversity
        5,     // num_batches - multiple batches to accumulate visits
        1.4,   // exploration
        5.0,   // virtual_loss_weight
        1.0,   // temperature - using 1.0 to get good diversity
        0.01,  // vl_temp_scale
        10,    // timeout_secs
        None,  // gpu_max_nodes
    ) {
        println!("\nSearch completed:");
        println!("  Total iterations: {}", telemetry.diagnostics.rollouts);
        println!("  Max temp boost: {:.3}", telemetry.diagnostics.max_temp_boost as f32 / 1000.0);
        
        // Analyze children visit distribution
        println!("\nChildren visit distribution:");
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
        
        println!("\nDiversity metrics:");
        println!("  Children with visits: {}/{}", children_with_visits, children_stats.len());
        println!("  Total visits distributed: {}", total_visits);
        
        // ASSERTION: Multiple children should have visits (softmax diversity working)
        assert!(
            children_with_visits >= 2,
            "BUG: Only {} child(ren) received visits! Threads should be selecting different children via softmax sampling. \
             All {} children: {:?}",
            children_with_visits,
            children_stats.len(),
            children_stats
        );
        
        // ASSERTION: No single child should have ALL the visits (that would indicate deterministic selection)
        let max_visits = children_stats.iter().map(|(_, _, v, _, _)| v).max().unwrap_or(&0);
        let max_visit_pct = if total_visits > 0 {
            (*max_visits as f64 / total_visits as f64) * 100.0
        } else {
            0.0
        };
        
        assert!(
            max_visit_pct < 95.0,
            "BUG: One child received {:.1}% of all visits! Softmax sampling should distribute visits across multiple children. \
             Children: {:?}",
            max_visit_pct,
            children_stats
        );
        
        // Calculate entropy of visit distribution as a diversity measure
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
        
        assert!(
            normalized_entropy > 0.3,
            "BUG: Visit distribution entropy too low ({:.1}%)! Softmax should create diverse selections. \
             Children: {:?}",
            normalized_entropy * 100.0,
            children_stats
        );
        
        println!("\nSUCCESS: Threads are selecting different children via softmax sampling!");
        println!("  ✓ {} different children explored", children_with_visits);
        println!("  ✓ No child dominates (max {:.1}%)", max_visit_pct);
        println!("  ✓ Good entropy diversity ({:.1}%)", normalized_entropy * 100.0);
        
    } else {
        panic!("search_gpu_native_othello returned None");
    }
}
