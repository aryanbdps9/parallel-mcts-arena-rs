use std::sync::Arc;
use mcts::gpu::{GpuContext, GpuConfig, OthelloNodeInfo};
use mcts::gpu::mcts_othello::GpuOthelloMcts;

#[test]
fn test_endgame_detailed_diagnostics() {
    // Create GPU context
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    
    // Start with initial position and do a rollout to get an endgame position
    let mut board = [0i32; 64];
    board[27] = -1; board[28] = 1;
    board[35] = 1;  board[36] = -1;
    let mut player = 1;
    
    // Simple random rollout to endgame (many pieces placed)
    let mut rng_state = 12345u32;
    for _step in 0..40 { // Place ~40 more pieces to get to endgame
        let legal_moves = compute_legal_moves(&board, player);
        if legal_moves.is_empty() {
            player = -player;
            let legal_moves_other = compute_legal_moves(&board, player);
            if legal_moves_other.is_empty() {
                break; // Both players pass = game over
            }
            continue;
        }
        
        // Pick random move
        rng_state = pcg_hash(rng_state);
        let idx = (rng_state as usize) % legal_moves.len();
        let (x, y) = legal_moves[idx];
        make_move(&mut board, player, x, y);
        player = -player;
    }
    
    // Now we have an endgame position - print it
    eprintln!("\n=== ENDGAME POSITION ===");
    print_board(&board);
    eprintln!("Current player: {}", player);
    
    let legal_moves = compute_legal_moves(&board, player);
    eprintln!("Legal moves: {} moves", legal_moves.len());
    for (x, y) in &legal_moves {
        eprintln!("  ({}, {})", x, y);
    }
    
    if legal_moves.is_empty() {
        eprintln!("No legal moves - skipping test");
        return;
    }
    
    // Create MCTS engine with small capacity for endgame
    let max_nodes = 10000;
    let batch_size = 1024;
    let engine = GpuOthelloMcts::new(context.clone(), max_nodes, batch_size)
        .expect("Failed to create MCTS engine");
    
    // Initialize tree
    engine.init_tree(&board, player, &legal_moves);
    
    // Run search with detailed parameters
    let num_threads = 1024;  // Moderate thread count
    let num_steps = 50;      // Enough to build a small tree
    let exploration = 1.414;
    let virtual_loss_weight = 1.0;
    let temperature = 0.0;   // Zero temperature = deterministic max PUCT selection
    let seed = 54321;
    
    eprintln!("\n=== RUNNING MCTS ===");
    eprintln!("Threads: {}, Steps: {}", num_threads, num_steps);
    eprintln!("Exploration: {}, VL Weight: {}, Temperature: {}", 
              exploration, virtual_loss_weight, temperature);
    
    let telemetry = engine.run_incremental_mcts(
        num_threads,
        num_steps,
        exploration,
        virtual_loss_weight,
        temperature,
        seed,
        None, // No timeout
        true,
    );
    let diagnostics = telemetry.diagnostics.clone();
    
    eprintln!("\n=== SEARCH DIAGNOSTICS ===");
    eprintln!("Total rollouts: {}", diagnostics.rollouts);
    eprintln!("Nodes created: {}", diagnostics.init_nodes_count);
    eprintln!("Expansion successes: {}", diagnostics.expansion_success);
    
    // Get children statistics
    let children_stats = engine.get_children_stats();
    eprintln!("\n=== CHILDREN STATISTICS (from root) ===");
    eprintln!("Total children: {}", children_stats.len());
    
    // Sort by visits descending
    let mut sorted = children_stats.clone();
    sorted.sort_by_key(|(_, _, v, _, _)| -(*v));
    
    for (i, (x, y, visits, wins, q)) in sorted.iter().enumerate() {
        eprintln!("\nChild #{}: Move ({}, {})", i + 1, x, y);
        eprintln!("  Visits: {}", visits);
        eprintln!("  Wins: {} (raw, using 0-2 reward encoding)", wins);
        eprintln!("  Q-value: {:.4} (normalized: wins/(visits*2) = {:.4})", q, if *visits > 0 { *wins as f64 / (*visits as f64 * 2.0) } else { 0.0 });
        
        if *visits > 0 {
            // Calculate what PUCT components would be during selection
            let root_visits: i32 = sorted.iter().map(|(_, _, v, _, _)| v).sum();
            
            // Q is ALREADY from parent's perspective (wins are counted from parent's view in GPU)
            // NO NEED to invert it!
            let q_for_puct = *q;
            
            // Prior probability (uniform for now - we don't have neural net)
            let prior = 1.0 / legal_moves.len() as f64;
            
            // Exploration term: C * P * sqrt(N_parent) / (1 + N_child)
            let exploration_term = exploration as f64 * prior * (root_visits as f64).sqrt() / (1.0 + *visits as f64);
            
            // PUCT score
            let puct = q_for_puct + exploration_term;
            
            eprintln!("  --- PUCT Analysis ---");
            eprintln!("  Root total visits: {}", root_visits);
            eprintln!("  Q (for PUCT): {:.4}", q_for_puct);
            eprintln!("  Exploration term: {:.4}", exploration_term);
            eprintln!("  PUCT score: {:.4}", puct);
            eprintln!("  Prior P: {:.4}", prior);
        }
    }
    
    // Now let's examine this from GPU's perspective - read raw node data
    eprintln!("\n=== GPU BUFFER DIAGNOSTICS ===");
    
    // Try to read root node info
    if let Some(root_info) = try_read_root_node(&engine) {
        eprintln!("Root node (index 0):");
        eprintln!("  Player at node: {}", root_info.player_at_node);
        eprintln!("  Num children: {}", root_info.num_children);
        eprintln!("  Parent idx: {}", root_info.parent_idx);
        eprintln!("  Move ID: {}", root_info.move_id);
        eprintln!("  Flags: 0x{:08x}", root_info.flags);
        
        // Note: Visit/win counts are in separate buffers (visits[] and wins[] arrays)
        // indexed by board position, not node index
        eprintln!("  (Visit/win counts are in position-indexed buffers, not node struct)");
    }
    
    eprintln!("\n=== CONCLUSION ===");
    eprintln!("Examine the PUCT scores above.");
    eprintln!("High-visit children should have high PUCT scores explaining why they were selected.");
    eprintln!("If a low-Q child has high visits, its exploration term must be very large.");
}

// Helper: Simple legal move computation
fn compute_legal_moves(board: &[i32; 64], player: i32) -> Vec<(usize, usize)> {
    let mut moves = Vec::new();
    for y in 0..8 {
        for x in 0..8 {
            let idx = y * 8 + x;
            if board[idx] != 0 {
                continue; // Occupied
            }
            if is_legal_move(board, player, x, y) {
                moves.push((x, y));
            }
        }
    }
    moves
}

fn is_legal_move(board: &[i32; 64], player: i32, x: usize, y: usize) -> bool {
    let directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ];
    
    for (dx, dy) in &directions {
        if check_direction(board, player, x as i32, y as i32, *dx, *dy) {
            return true;
        }
    }
    false
}

fn check_direction(board: &[i32; 64], player: i32, x: i32, y: i32, dx: i32, dy: i32) -> bool {
    let mut nx = x + dx;
    let mut ny = y + dy;
    let mut found_opponent = false;
    
    while nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
        let idx = (ny * 8 + nx) as usize;
        let cell = board[idx];
        
        if cell == 0 {
            return false; // Empty cell
        } else if cell == -player {
            found_opponent = true; // Opponent's piece
        } else {
            return found_opponent; // Our piece - valid if we found opponent
        }
        
        nx += dx;
        ny += dy;
    }
    false
}

fn make_move(board: &mut [i32; 64], player: i32, x: usize, y: usize) {
    let idx = y * 8 + x;
    board[idx] = player;
    
    let directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ];
    
    for (dx, dy) in &directions {
        flip_direction(board, player, x as i32, y as i32, *dx, *dy);
    }
}

fn flip_direction(board: &mut [i32; 64], player: i32, x: i32, y: i32, dx: i32, dy: i32) {
    let mut to_flip = Vec::new();
    let mut nx = x + dx;
    let mut ny = y + dy;
    
    while nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
        let idx = (ny * 8 + nx) as usize;
        let cell = board[idx];
        
        if cell == 0 {
            return; // Empty - no flips
        } else if cell == -player {
            to_flip.push(idx);
        } else {
            // Found our piece - do the flips
            for flip_idx in to_flip {
                board[flip_idx] = player;
            }
            return;
        }
        
        nx += dx;
        ny += dy;
    }
}

fn print_board(board: &[i32; 64]) {
    eprintln!("  0 1 2 3 4 5 6 7");
    for y in 0..8 {
        eprint!("{} ", y);
        for x in 0..8 {
            let idx = y * 8 + x;
            let symbol = match board[idx] {
                1 => 'X',
                -1 => 'O',
                _ => '.',
            };
            eprint!("{} ", symbol);
        }
        eprintln!();
    }
    
    let x_count = board.iter().filter(|&&v| v == 1).count();
    let o_count = board.iter().filter(|&&v| v == -1).count();
    eprintln!("Piece count: X={} O={}", x_count, o_count);
}

fn pcg_hash(mut state: u32) -> u32 {
    state = state.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

fn try_read_root_node(engine: &GpuOthelloMcts) -> Option<OthelloNodeInfo> {
    // Try to use debug method if available
    Some(engine.debug_get_node_info(0))
}
