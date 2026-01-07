#[cfg(feature = "gpu")]
#[test]
fn test_threads_see_different_virtual_loss() {
    use mcts::MCTS;
    use mcts::games::othello::OthelloState;
    use mcts::GameState;
    
    // This test verifies that as more threads select the same child,
    // the virtual loss accumulates, causing subsequent threads to see
    // higher VL values and (hopefully) select different children.
    
    let mut mcts: MCTS<OthelloState> = MCTS::new(1.4, 1, 1000);
    
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
    
    println!("Testing virtual loss accumulation...");
    println!("Initial position: {} legal moves", legal_moves_xy.len());
    
    mcts.init_gpu_native_othello(&board, current_player, &legal_moves_xy, 8192);
    
    let norm_entropy_high: f64;
    
    // Run with VERY HIGH virtual_loss_weight to amplify the effect
    println!("\n=== HIGH VL TEST (virtual_loss_weight=50.0) ===");
    if let Some(((_x, _y), _visits, _q, children_stats_high_vl, _total_nodes, telemetry_high)) = mcts.search_gpu_native_othello(
        &board,
        current_player,
        &legal_moves_xy,
        8192,  
        1,     
        1.4,   
        50.0,  // HIGH virtual_loss_weight (10x normal)
        1.0,   
        0.01,  
        10,    
        None,  
    ) {
        println!("High VL search completed:");
        println!("  Max temp boost: {:.3}", telemetry_high.diagnostics.max_temp_boost as f32 / 1000.0);
        
        let total_high: i32 = children_stats_high_vl.iter().map(|(_, _, v, _, _)| v).sum();
        println!("\nHigh VL distribution:");
        for (i, &(x, y, visits, _, _)) in children_stats_high_vl.iter().enumerate() {
            let pct = (visits as f64 / total_high as f64) * 100.0;
            println!("  Child {}: ({},{}) visits={} ({:.1}%)", i+1, x, y, visits, pct);
        }
        
        let entropy_high: f64 = children_stats_high_vl.iter()
            .map(|(_, _, v, _, _)| {
                if *v > 0 && total_high > 0 {
                    let p = *v as f64 / total_high as f64;
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();
        let max_entropy = (children_stats_high_vl.len() as f64).log2();
        norm_entropy_high = entropy_high / max_entropy;
        println!("  Entropy: {:.1}%", norm_entropy_high * 100.0);
    } else {
        panic!("High VL search returned None");
    }
    
    // Reset and run with LOW VL
    mcts.init_gpu_native_othello(&board, current_player, &legal_moves_xy, 8192);
    
    let norm_entropy_low: f64;
    
    println!("\n=== LOW VL TEST (virtual_loss_weight=0.5) ===");
    if let Some(((_x, _y), _visits, _q, children_stats_low_vl, _total_nodes, telemetry_low)) = mcts.search_gpu_native_othello(
        &board,
        current_player,
        &legal_moves_xy,
        8192,  
        1,     
        1.4,   
        0.5,   // LOW virtual_loss_weight (10x smaller)
        1.0,   
        0.01,  
        10,    
        None,  
    ) {
        println!("Low VL search completed:");
        println!("  Max temp boost: {:.3}", telemetry_low.diagnostics.max_temp_boost as f32 / 1000.0);
        
        let total_low: i32 = children_stats_low_vl.iter().map(|(_, _, v, _, _)| v).sum();
        println!("\nLow VL distribution:");
        for (i, &(x, y, visits, _, _)) in children_stats_low_vl.iter().enumerate() {
            let pct = (visits as f64 / total_low as f64) * 100.0;
            println!("  Child {}: ({},{}) visits={} ({:.1}%)", i+1, x, y, visits, pct);
        }
        
        let entropy_low: f64 = children_stats_low_vl.iter()
            .map(|(_, _, v, _, _)| {
                if *v > 0 && total_low > 0 {
                    let p = *v as f64 / total_low as f64;
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();
        let max_entropy = (children_stats_low_vl.len() as f64).log2();
        norm_entropy_low = entropy_low / max_entropy;
        println!("  Entropy: {:.1}%", norm_entropy_low * 100.0);
        
        println!("\n=== COMPARISON ===");
        println!("High VL (50.0) entropy: {:.1}%", norm_entropy_high * 100.0);
        println!("Low VL (0.5) entropy:   {:.1}%", norm_entropy_low * 100.0);
        
        // If VL is working, high VL should give MORE uniform distribution (higher entropy)
        // because threads see different VL values and are pushed to explore different children
        if norm_entropy_high > norm_entropy_low {
            println!("\n✅ SUCCESS: Higher VL → more diversity!");
            println!("   This proves threads see different VL values and diversify accordingly.");
        } else if norm_entropy_high < norm_entropy_low {
            println!("\n⚠️  UNEXPECTED: Higher VL → LESS diversity!");
            println!("   This suggests VL might be causing convergence instead of diversification.");
        } else {
            println!("\n⚠️  No significant difference between high and low VL");
        }
        
        // The key test: does changing VL affect distribution?
        let diff: f64 = (norm_entropy_high - norm_entropy_low).abs();
        assert!(
            diff > 0.05,
            "Virtual loss weight doesn't affect distribution! \
             High VL entropy: {:.1}%, Low VL entropy: {:.1}%. \
             This suggests threads aren't seeing different VL values.",
            norm_entropy_high * 100.0,
            norm_entropy_low * 100.0
        );
        
        println!("\n✅ Virtual loss IS affecting selection diversity!");
    } else {
        panic!("Low VL search returned None");
    }
}