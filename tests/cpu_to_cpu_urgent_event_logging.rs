//! Test for CPU-to-CPU urgent event logging pipeline using the new log_urgent_event_from_cpu API

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::Duration;
use mcts::gpu::urgent_event_logger::start_and_log_urgent_events;
use mcts::gpu::mcts_gpu::GpuMctsEngine;
use mcts::gpu::GpuContext;

#[test]
fn test_cpu_to_cpu_urgent_event_logging() {
    let config = mcts::gpu::GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    let engine = GpuMctsEngine::new(context.clone(), 1024, 128, 8, 8);

    let engine_arc = Arc::new(engine);
    // Log a test string from the CPU
    let test_str = "Hello from CPU logging!";
    let event_type = 99;
    let timestamp = 123456789u64;
    engine_arc.log_urgent_event_from_cpu(event_type, timestamp, test_str);
    // Wait briefly to ensure buffer is unmapped before polling thread starts
    std::thread::sleep(Duration::from_millis(20));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let events_arc = start_and_log_urgent_events(engine_arc.clone(), 10, stop_flag.clone());

    // Wait for the event to be polled and appear in the queue
    let max_wait_ms = 2000;
    let poll_interval = 50;
    let mut waited = 0;
    let mut found = false;
    while waited < max_wait_ms {
        if let Some(ev) = events_arc.pop() {
            if ev.event_type == event_type && u64::from(ev.timestamp) == timestamp {
                // Decode the string from the payload
                let mut bytes = Vec::new();
                for word in ev.payload.iter() {
                    bytes.extend_from_slice(&word.to_le_bytes());
                }
                let msg = String::from_utf8_lossy(&bytes);
                assert!(msg.contains(test_str));
                found = true;
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(poll_interval));
        waited += poll_interval;
    }
    stop_flag.store(true, Ordering::Relaxed);
    assert!(found, "Test urgent event was not received from CPU within {} ms", max_wait_ms);
}
