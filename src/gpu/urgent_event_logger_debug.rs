use std::sync::{Arc, atomic::AtomicBool};
use super::mcts_gpu::GpuMctsEngine;

/// Debug version: just print entry and return empty vec
pub fn start_and_log_urgent_events_debug(val: u64, flag: Arc<AtomicBool>, engine: Arc<GpuMctsEngine>) {
    println!("[DIAG] start_and_log_urgent_events_debug: ENTER (very first line), val={}, flag ptr={:p}, engine ptr={:p}", val, Arc::as_ptr(&flag), Arc::as_ptr(&engine));
}
