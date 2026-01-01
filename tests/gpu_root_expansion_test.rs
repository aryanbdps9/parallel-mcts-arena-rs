use mcts::gpu::mcts_othello::GpuOthelloMcts;
use mcts::gpu::{GpuConfig, GpuContext};
use std::sync::Arc;

#[test]
fn test_reproduce_root_expansion_failure() {
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    let mcts = GpuOthelloMcts::new(context, 2_000_000, 128).expect("Failed to create GpuOthelloMcts");

    // Initial board
    let mut board = [0i32; 64];
    board[3 * 8 + 3] = 1;
    board[3 * 8 + 4] = -1;
    board[4 * 8 + 3] = -1;
    board[4 * 8 + 4] = 1;
    let root_player = 1;
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];

    println!("Initializing tree...");
    mcts.init_tree(&board, root_player, &legal_moves);

    // Check Node 0
    let node0 = mcts.debug_get_node_info(0);
    println!("Node 0 after init: {:?}", node0);
    assert_eq!(node0.num_children, 0, "Root should have 0 children initially");
    assert_eq!(node0.flags, 0, "Root flags should be 0");

    // Check Free Tops
    let free_tops = mcts.debug_get_free_tops();
    println!("Free tops (first 10): {:?}", &free_tops[0..10]);
    let total_free: u32 = free_tops.iter().sum();
    println!("Total free nodes in lists: {}", total_free);
    
    // We expect free lists to be populated.
    // 2,000,000 nodes. 256 lists. ~7812 per list.
    // free_tops should be around 7812.
    assert!(total_free > 1_000_000, "Free lists should be populated");

    // Run 1 batch of iterations
    println!("Running iterations...");
    // Dispatch kernel first
    mcts.dispatch_mcts_othello_kernel(2048);
    
    let telemetry = mcts.run_iterations(2048, 0.1, 1.0, 0.06, 42);
    println!("Telemetry: {:?}", telemetry);

    // Check Node 0 again
    let node0_after = mcts.debug_get_node_info(0);
    println!("Node 0 after iterations: {:?}", node0_after);

    // Check diagnostics
    if telemetry.diagnostics.selection_no_children > 0 {
        println!("WARNING: selection_no_children = {}", telemetry.diagnostics.selection_no_children);
    }
    if telemetry.diagnostics.expansion_attempts > 0 {
        println!("Expansion attempts: {}", telemetry.diagnostics.expansion_attempts);
    }
    if telemetry.diagnostics.expansion_success > 0 {
        println!("Expansion success: {}", telemetry.diagnostics.expansion_success);
    }
    if telemetry.diagnostics.alloc_failures > 0 {
        println!("Alloc failures: {}", telemetry.diagnostics.alloc_failures);
    }

    assert!(node0_after.num_children > 0, "Root should have children after iterations");
}
