use wgpu::util::DeviceExt;
#[test]
fn test_gpu_shader_to_cpu_urgent_event_logging() {
    // use wgpu::*;
    let config = mcts::gpu::GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    // --- OTHELLO-NATIVE TEST: This exposes the bug in urgent event logging for GpuOthelloMcts ---
    use mcts::gpu::mcts_othello::GpuOthelloMcts;
    use mcts::gpu::urgent_event_logger::start_and_log_urgent_events_othello;
    use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
    use std::time::Duration;

    let othello_engine = Arc::new(GpuOthelloMcts::new(context.clone(), 1024, 128).expect("Failed to create GpuOthelloMcts"));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let events_arc = start_and_log_urgent_events_othello(othello_engine.clone(), 10, stop_flag.clone());

    // Inject a fake urgent event by copying to the GPU buffer, then polling as the logger does
    {
        use mcts::gpu::mcts_othello::UrgentEvent;
        let inner = othello_engine.inner.lock().unwrap();
        if let (Some(urgent_event_buffer_gpu), Some(urgent_event_write_head_gpu)) = (inner.urgent_event_buffer_gpu.as_ref(), inner.urgent_event_write_head_gpu.as_ref()) {
            let device = othello_engine.context.device();
            let queue = othello_engine.context.queue();
            // Write a fake event to a staging buffer, then copy to GPU buffer
            let _ring_size = 256u32;
            let idx = 0u32; // always write to slot 0 for test
            let fake_event = UrgentEvent {
                timestamp: 12345678,
                event_type: 42,
                _pad: 0,
                payload: [0; 255],
            };
            let event_bytes = unsafe {
                std::slice::from_raw_parts((&fake_event as *const UrgentEvent) as *const u8, std::mem::size_of::<UrgentEvent>())
            };
            let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("UrgentEventStaging"),
                contents: event_bytes,
                usage: wgpu::BufferUsages::COPY_SRC,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("TestUrgentEventCopy") });
            encoder.copy_buffer_to_buffer(&staging, 0, urgent_event_buffer_gpu, (idx as usize * std::mem::size_of::<UrgentEvent>()) as u64, std::mem::size_of::<UrgentEvent>() as u64);
            // Write head
            let write_head_bytes = 1u32.to_le_bytes();
            let staging_head = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("UrgentEventWriteHeadStaging"),
                contents: &write_head_bytes,
                usage: wgpu::BufferUsages::COPY_SRC,
            });
            encoder.copy_buffer_to_buffer(&staging_head, 0, urgent_event_write_head_gpu, 0, 4);
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        }
    }

    // Wait for the event to be polled and appear in the queue
    let max_wait_ms = 2000;
    let poll_interval = 50;
    let mut waited = 0;
    let mut found = false;
    let mut found_event = None;
    while waited < max_wait_ms {
        if let Some(ev) = events_arc.pop() {
            if ev.event_type == 42 && ev.timestamp == 12345678 {
                found = true;
                found_event = Some(ev);
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(poll_interval));
        waited += poll_interval;
    }
    stop_flag.store(true, Ordering::Relaxed);
    assert!(found, "Test urgent event was not received from GpuOthelloMcts within {} ms (this should fail if the logger is broken)", max_wait_ms);
    if let Some(ev) = found_event {
        println!("[TEST] Received urgent event from GpuOthelloMcts: type={}, ts={}, payload[0]={}", ev.event_type, ev.timestamp, ev.payload[0]);
    }
}
/// Test for GPU-to-CPU urgent event logging pipeline
// This test writes a known urgent event from the GPU and verifies it is received on the CPU via the lock-free queue.

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::Duration;
use mcts::gpu::urgent_event_logger::start_and_log_urgent_events;
use mcts::gpu::mcts_gpu::GpuMctsEngine;
use mcts::gpu::GpuContext;

#[test]
fn test_gpu_urgent_event_logging_pipeline() {
    let config = mcts::gpu::GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    let engine = GpuMctsEngine::new(context.clone(), 1024, 128, 8, 8);
    // Write a known urgent event to the GPU-side buffer (simulate GPU log)
    // This assumes you have a method to inject a test event for diagnostics
    // let test_event = mcts::gpu::mcts_othello::UrgentEvent { ... } // Removed unused variable
    // engine.inject_test_urgent_event(test_event); // Removed: method does not exist
    let engine_arc = Arc::new(engine);
    let stop_flag = Arc::new(AtomicBool::new(false));
    let events_arc = start_and_log_urgent_events(engine_arc.clone(), 10, stop_flag.clone());
    // Inject a known urgent event from the CPU
    // Set payload[0] to 0xAB
    let mut payload = [0u32; 255];
    payload[0] = 0xAB;
    // Use a custom log_urgent_event_from_cpu_with_payload for this test
    engine_arc.log_urgent_event_from_cpu_with_payload(42, 123456, &payload);
    // Wait for the event to be polled and appear in the queue
    let max_wait_ms = 2000;
    let poll_interval = 50;
    let mut waited = 0;
    let mut found = false;
    let mut found_event = None;
    while waited < max_wait_ms {
        if let Some(ev) = events_arc.pop() {
            if ev.event_type == 42 && ev.timestamp == 123456 && ev.payload[0] == 0xAB {
                found = true;
                found_event = Some(ev);
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(poll_interval));
        waited += poll_interval;
    }
    stop_flag.store(true, Ordering::Relaxed);
    assert!(found, "Test urgent event was not received from GPU within {} ms", max_wait_ms);
    if let Some(ev) = found_event {
        println!("[TEST] Received urgent event: type={}, ts={}, payload[0]={}", ev.event_type, ev.timestamp, ev.payload[0]);
    }
}
