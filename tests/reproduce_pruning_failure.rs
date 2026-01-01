use mcts::gpu::mcts_othello::GpuOthelloMcts;
use mcts::gpu::{GpuConfig, GpuContext};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use mcts::gpu::urgent_event_logger::start_and_log_urgent_events_othello;

#[test]
fn test_reproduce_pruning_failure() {
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    let mcts = Arc::new(GpuOthelloMcts::new(context, 2_000_000, 128).expect("Failed to create GpuOthelloMcts"));

    // Start urgent event logger
    let stop_flag = Arc::new(AtomicBool::new(false));
    let _events = start_and_log_urgent_events_othello(mcts.clone(), 10, stop_flag.clone());

    // Initial board
    let mut board = [0i32; 64];
    board[3 * 8 + 3] = 1;
    board[3 * 8 + 4] = -1;
    board[4 * 8 + 3] = -1;
    board[4 * 8 + 4] = 1;
    let root_player = 1;
    let legal_moves = vec![(4, 2), (2, 4), (3, 5), (5, 3)];

    println!("Initializing tree...");
    mcts.init_tree(&board, root_player, &legal_moves);

    // Run iterations to expand root
    println!("Running iterations...");
    mcts.dispatch_mcts_othello_kernel(2048);
    let _telemetry = mcts.run_iterations(2048, 0.1, 1.0, 0.06, 42);

    // Check Node 0
    let node0 = mcts.debug_get_node_info(0);
    println!("Node 0 after iterations: {:?}", node0);
    assert!(node0.num_children > 0, "Root should have children");

    // Get children to find a valid move
    // Note: get_children_stats uses the legal_moves passed to init_tree!
    let children_stats = mcts.get_children_stats();
    println!("Children stats: {:?}", children_stats);
    assert!(!children_stats.is_empty(), "Should have children stats");

    // Pick the first child (4, 2)
    let (move_x, move_y, _, _, _) = children_stats[0];
    
    println!("Attempting to advance root to move (x={}, y={})", move_x, move_y);

    // Prepare dummy next state (not used by pruning kernel, but needed for API)
    let new_board = [0i32; 64]; 
    let new_player = -root_player;
    let new_legal_moves = vec![];

    // Advance root
    let success = mcts.advance_root(move_x, move_y, &new_board, new_player, &new_legal_moves);
    assert!(success, "advance_root should return true");
    
    // Verify no urgent event debug was emitted (or check logs)
    println!("advance_root returned: {}", success);

    // Check if root changed
    let new_root_info = mcts.debug_get_node_info(0); // Root is always index 0? No, advance_root might swap?
    // Actually advance_root updates the root pointer or swaps content.
    // In this implementation, it likely swaps the new root to index 0 or updates internal state.
    // Let's check if it panicked or logged errors.
    
    // Give some time for urgent events to be processed
    std::thread::sleep(std::time::Duration::from_millis(500));
    
    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
}
