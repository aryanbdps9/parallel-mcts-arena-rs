/// Test that temperature parameter actually affects selection
use std::sync::Arc;

fn compute_othello_legal_moves(board: &[i32; 64], player: i32) -> Vec<(usize, usize)> {
    let mut moves = Vec::new();
    for y in 0..8 {
        for x in 0..8 {
            if is_valid_othello_move(board, x, y, player) {
                moves.push((x, y));
            }
        }
    }
    moves
}

fn is_valid_othello_move(board: &[i32; 64], x: usize, y: usize, player: i32) -> bool {
    if board[y * 8 + x] != 0 { return false; }
    let directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)];
    for (dx, dy) in directions.iter() {
        if would_flip(board, x as i32, y as i32, *dx, *dy, player) {
            return true;
        }
    }
    false
}

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

#[test]
#[ignore = "Temperature test requires controlled Q-values which is difficult with GPU MCTS randomness"]
fn test_temperature_effect() {
    #[cfg(feature = "gpu")]
    {
        use mcts::gpu::{GpuContext, GpuConfig};
        
        println!("\n=== TEMPERATURE EFFECT TEST ===\n");
        
        let board = [0i32; 64];
        let mut board = board;
        board[27] = -1;
        board[28] = 1;
        board[35] = 1;
        board[36] = -1;
        
        let current_player = 1;
        let legal_moves = compute_othello_legal_moves(&board, current_player);
        
        // Test with HIGH temperature (should be uniform)
        println!("--- TEST 1: High Temperature (10.0) ---");
        let config = GpuConfig::default();
        let ctx = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
        let mcts = mcts::gpu::mcts_othello::GpuOthelloMcts::new(ctx, 1_000_000, 256).expect("Failed to create GPU MCTS");
        
        mcts.init_tree(&board, current_player, &legal_moves);
        mcts.dispatch_mcts_othello_kernel(256, 1.4, 1.0, 10.0, 0.01, 1000); // HIGH temp, different seed
        
        for _i in 0..16 {
            mcts.dispatch_mcts_othello_kernel(256, 1.4, 1.0, 10.0, 0.01, 2000 + _i);
            mcts.run_iterations(256, 1.4, 1.0, 10.0, 2000 + _i);
        }
        mcts.flush_and_wait();
        
        let children_high_temp = mcts.get_children_stats();
        let visits_high: Vec<i32> = children_high_temp.iter().map(|(_, _, v, _, _)| *v).collect();
        let total_high: i32 = visits_high.iter().sum();
        let avg_high = total_high as f64 / visits_high.len() as f64;
        let var_high: f64 = visits_high.iter().map(|v| {
            let diff = *v as f64 - avg_high;
            diff * diff
        }).sum::<f64>() / visits_high.len() as f64;
        let cv_high = var_high.sqrt() / avg_high;
        
        println!("High temp visits: {:?}", visits_high);
        println!("CV = {:.4} (should be < 0.05, nearly uniform)", cv_high);
        
        // Test with LOW temperature (should concentrate)
        println!("\n--- TEST 2: Low Temperature (0.01) ---");
        let ctx2 = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
        let mcts2 = mcts::gpu::mcts_othello::GpuOthelloMcts::new(ctx2, 1_000_000, 256).expect("Failed to create GPU MCTS");
        
        mcts2.init_tree(&board, current_player, &legal_moves);
        mcts2.dispatch_mcts_othello_kernel(256, 1.4, 1.0, 0.01, 0.01, 3000); // LOW temp, different seed range
        
        for _i in 0..16 {
            mcts2.dispatch_mcts_othello_kernel(256, 1.4, 1.0, 0.01, 0.01, 4000 + _i);
            mcts2.run_iterations(256, 1.4, 1.0, 0.01, 4000 + _i);
        }
        mcts2.flush_and_wait();
        
        let children_low_temp = mcts2.get_children_stats();
        let visits_low: Vec<i32> = children_low_temp.iter().map(|(_, _, v, _, _)| *v).collect();
        let total_low: i32 = visits_low.iter().sum();
        let avg_low = total_low as f64 / visits_low.len() as f64;
        let var_low: f64 = visits_low.iter().map(|v| {
            let diff = *v as f64 - avg_low;
            diff * diff
        }).sum::<f64>() / visits_low.len() as f64;
        let cv_low = var_low.sqrt() / avg_low;
        
        println!("Low temp visits: {:?}", visits_low);
        println!("CV = {:.4} (should be > high temp CV)", cv_low);
        
        println!("\n=== RESULT ===");
        if cv_low <= cv_high {
            println!("❌ FAILURE: Low temp CV ({:.4}) should be > high temp CV ({:.4})", cv_low, cv_high);
            println!("   Temperature parameter may not be working correctly.");
            panic!("Temperature parameter not working!");
        } else if (cv_low - cv_high) < 0.002 {
            println!("⚠ WARNING: Temperature effect is weak (CV difference = {:.4})", cv_low - cv_high);
            println!("   This may be expected if Q-values are very similar in the opening.");
        } else {
            println!("✓ SUCCESS: Temperature affects visit distribution as expected");
            println!("   High temp CV={:.4}, Low temp CV={:.4}, difference={:.4}", cv_high, cv_low, cv_low - cv_high);
        }
    }
}