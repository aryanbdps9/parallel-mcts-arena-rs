//! Test that no unexpected early exit events are emitted by the main MCTS kernel

// use std::sync::Arc;
// use std::sync::atomic::AtomicBool;

// Import from the local crate (assumes lib.rs exposes gpu mod)
// use parallel_mcts_arena::gpu::{GpuOthelloMcts, assert_no_early_exit_events, start_and_log_urgent_events_othello};

// #[test]
// fn test_no_early_exit_in_mcts_kernel() {
//     // Setup a minimal GpuOthelloMcts engine (reuse your real setup code)
//     // let context = Arc::new(parallel_mcts_arena::gpu::GpuContext::new_default());
// //     let mut engine = GpuOthelloMcts::new(context.clone(), 1024, 128, 8, 8);
//     engine.init_tree(1, &vec![(0, 1.0), (1, 1.0)]);
// //     engine.create_bind_groups(context.device());
//     let engine_arc = Arc::new(engine);
//     let stop_flag = Arc::new(AtomicBool::new(false));
// //     let events = start_and_log_urgent_events_othello(engine_arc.clone(), 10, stop_flag.clone());
//
//     // Dispatch the main kernel (simulate one turn)
// //     engine_arc.dispatch_mcts_othello_kernel(128);
//     engine_arc.flush_and_wait();
//
//     // Wait for logger to poll events
//     std::thread::sleep(std::time::Duration::from_millis(100));
//
//     // Assert no early exit events were emitted
// //     assert_no_early_exit_events(&events);
// }
