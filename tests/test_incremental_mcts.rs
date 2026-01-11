/// Tests for incremental MCTS execution with thread states
#[cfg(feature = "gpu")]
#[test]
fn test_thread_state_buffer_allocation() {
    use mcts::gpu::mcts_othello::GpuOthelloMcts;
    use mcts::gpu::{GpuContext, GpuConfig};
    use std::sync::Arc;

    println!("\n[TEST] ========== Thread State Buffer Allocation Test ==========\n");

    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    let max_nodes = 10000;
    let batch_size = 128;
    
    let engine = GpuOthelloMcts::new(
        context.clone(),
        max_nodes,
        batch_size,
    ).expect("Failed to create GpuOthelloMcts");

    // Test buffer allocation
    let num_threads = 1024u32;
    println!("[TEST] Allocating thread states buffer for {} threads", num_threads);
    
    let buffer = engine.ensure_thread_states_buffer(num_threads);
    println!("[TEST] Buffer allocated: size = {} bytes", buffer.size());
    
    // Expected size: ~816 bytes per thread
    let expected_min = num_threads as u64 * 800;
    let expected_max = num_threads as u64 * 850;
    
    assert!(buffer.size() >= expected_min, 
        "Buffer too small: {} < {}", buffer.size(), expected_min);
    assert!(buffer.size() <= expected_max,
        "Buffer too large: {} > {}", buffer.size(), expected_max);
    
    // Test that calling again returns same buffer
    let buffer2 = engine.ensure_thread_states_buffer(num_threads);
    assert!(Arc::ptr_eq(&buffer, &buffer2), 
        "Should return same buffer on second call");
    
    println!("[TEST] ✓ Thread state buffer allocation test passed");
}

#[cfg(feature = "gpu")]
#[test]
fn test_incremental_phase_initialization() {
    use mcts::gpu::mcts_othello::GpuOthelloMcts;
    use mcts::gpu::{GpuContext, GpuConfig};
    use std::sync::Arc;

    println!("\n[TEST] ========== Incremental Phase Initialization Test ==========\n");

    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    let max_nodes = 10000;
    let batch_size = 128;
    
    let engine = GpuOthelloMcts::new(
        context.clone(),
        max_nodes,
        batch_size,
    ).expect("Failed to create GpuOthelloMcts");

    // Initial board
    let mut board = [0i32; 64];
    board[27] = -1; board[28] = 1;
    board[35] = 1; board[36] = -1;
    
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    println!("[TEST] Initializing tree");
    engine.init_tree(&board, 1, &legal_moves);
    
    let nodes_init = engine.calculate_nodes_used();
    println!("[TEST] Nodes after init: {} (root only)", nodes_init);
    
    // After init, only root exists (children created during first simulations)
    assert_eq!(nodes_init, 1, "Should have only root node after init");
    
    // Run a few simulations to trigger root expansion
    println!("[TEST] Running simulations to expand root");
    engine.dispatch_mcts_othello_kernel(1, 1.414, 1.0, 1.0, 1.0, 42);
    
    let nodes_after = engine.calculate_nodes_used();
    println!("[TEST] Nodes after simulations: {} (root + children)", nodes_after);
    assert!(nodes_after > 1, "Should have expanded root with children");
    
    println!("[TEST] ✓ Incremental phase initialization test passed");
}

/// Test that verifies thread states can be initialized and kernels compile
#[cfg(feature = "gpu")]
#[test]
fn test_incremental_kernel_exists() {
    use mcts::gpu::mcts_othello::GpuOthelloMcts;
    use mcts::gpu::{GpuContext, GpuConfig};
    use std::sync::Arc;

    println!("\n[TEST] ========== Incremental Kernel Exists Test ==========\n");

    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    let max_nodes = 10000;
    let batch_size = 128;
    
    println!("[TEST] Creating GPU MCTS engine");
    let engine = GpuOthelloMcts::new(
        context.clone(),
        max_nodes,
        batch_size,
    ).expect("Failed to create GpuOthelloMcts");

    // Check that shader compiled successfully (constructor didn't panic)
    println!("[TEST] Shader compilation successful");
    
    // Allocate thread states buffer
    let num_threads = 64u32;
    let buffer = engine.ensure_thread_states_buffer(num_threads);
    println!("[TEST] Thread states buffer allocated: {} bytes for {} threads", 
        buffer.size(), num_threads);
    
    // Verify buffer size is reasonable
    assert!(buffer.size() >= num_threads as u64 * 800, "Buffer size reasonable");
    
    println!("[TEST] ✓ Incremental kernel compilation test passed");
}

#[cfg(feature = "gpu")]
#[test]
fn test_incremental_dispatch_flow() {
    use mcts::gpu::mcts_othello::GpuOthelloMcts;
    use mcts::gpu::{GpuContext, GpuConfig};
    use std::sync::Arc;

    println!("\n[TEST] ========== Incremental Dispatch Flow Test ==========\n");

    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GPU context"));
    let max_nodes = 10000;
    let batch_size = 128;
    
    println!("[TEST] Creating GPU MCTS engine");
    let engine = GpuOthelloMcts::new(
        context.clone(),
        max_nodes,
        batch_size,
    ).expect("Failed to create GpuOthelloMcts");

    // Initialize tree with starting position
    let mut board = [0i32; 64];
    board[27] = -1; board[28] = 1;
    board[35] = 1; board[36] = -1;
    
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    println!("[TEST] Initializing tree");
    engine.init_tree(&board, 1, &legal_moves);
    
    // Run a few regular iterations first to create tree structure
    println!("[TEST] Running standard MCTS to build tree");
    engine.dispatch_mcts_othello_kernel(2, 1.414, 1.0, 1.0, 1.0, 42);
    
    let nodes_before = engine.calculate_nodes_used();
    println!("[TEST] Tree size before incremental: {} nodes", nodes_before);
    
    // Run incremental MCTS (with stub implementations, just verifies it doesn't crash)
    println!("[TEST] Running incremental MCTS");
    let telemetry = engine.run_incremental_mcts(
        64,    // num_threads
        5,     // max_steps (small for testing)
        1.414, // exploration
        1.0,   // vl_weight
        0.0,   // temperature
        123,   // seed
        None,  // timeout
        true,
    );
    
    println!("[TEST] Incremental MCTS completed");
    println!("[TEST] Telemetry: {} nodes used, {} iterations", 
        telemetry.alloc_count_after, telemetry.iterations_launched);
    
    println!("[TEST] ✓ Incremental dispatch flow test passed");
}