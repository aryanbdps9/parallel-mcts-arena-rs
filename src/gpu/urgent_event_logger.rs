use super::mcts_othello::GpuOthelloMcts;
/// Starts the urgent event polling thread for GpuOthelloMcts and prints events every interval.


pub fn start_and_log_urgent_events_othello(
    _gpu_engine: Arc<GpuOthelloMcts>,
    poll_interval_ms: u64,
    stop_flag: Arc<AtomicBool>,
) -> Arc<SegQueue<UrgentEvent>> {
    let events = Arc::new(SegQueue::new());
    // let events_clone = Arc::clone(&events);
    // let engine_clone = gpu_engine.clone();
    let stop_flag_clone = Arc::clone(&stop_flag);
    thread::spawn(move || {
        while !stop_flag_clone.load(Ordering::Relaxed) {
            // No urgent event polling for dummy engine, but you can add polling logic here if needed
            // let raw_events: Vec<[u8; 1024]> = ...
            // for raw in raw_events { ... }
            std::thread::sleep(Duration::from_millis(poll_interval_ms));
        }
    });
    events
}
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use crossbeam_queue::SegQueue;
use std::thread;
use std::time::Duration;
use super::mcts_othello::UrgentEvent;
use super::mcts_gpu::GpuMctsEngine;

/// Starts the urgent event polling thread and prints events every interval.
pub fn start_and_log_urgent_events(
    gpu_engine: Arc<GpuMctsEngine>,
    poll_interval_ms: u64,
    stop_flag: Arc<AtomicBool>,
) -> Arc<SegQueue<UrgentEvent>> {
    // println!("[DIAG] start_and_log_urgent_events: ENTER (very first line)");
    let events = Arc::new(SegQueue::new());
    let events_clone = Arc::clone(&events);
    let engine_clone = gpu_engine.clone();
    let stop_flag_clone = Arc::clone(&stop_flag);
    thread::spawn(move || {
        let mut last_seen_write_head: u32 = 0;
        let ring_size = 256u32;
        while !stop_flag_clone.load(Ordering::Relaxed) {
            // Poll urgent events and get the current write head directly
            let urgent_event_buffers = &engine_clone.urgent_event_buffers;
            // Copy write head to staging
            let device = engine_clone.context.device();
            let queue = engine_clone.context.queue();
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("UrgentEventWriteHeadRead"),
            });
            encoder.copy_buffer_to_buffer(
                &urgent_event_buffers.urgent_event_write_head_buffer,
                0,
                &urgent_event_buffers.urgent_event_write_head_staging,
                0,
                4,
            );
            queue.submit(Some(encoder.finish()));
            let slice = urgent_event_buffers.urgent_event_write_head_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range();
            let current_write_head = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            println!("[DIAG] last_seen_write_head: {}, current_write_head: {}", last_seen_write_head, current_write_head);
            drop(data);
            urgent_event_buffers.urgent_event_write_head_staging.unmap();

            let events_vec = engine_clone.poll_urgent_events();
            let mut idx = last_seen_write_head;
            while idx != current_write_head {
                let slot = (idx % ring_size) as usize;
                if slot < events_vec.len() {
                    let raw = &events_vec[slot];
                    let first_bytes: Vec<u8> = raw.iter().take(16).cloned().collect();
                    println!("[URGENT EVENT] GPU event[{}] first 16 bytes: {:?}", slot, first_bytes);
                    let event: UrgentEvent = unsafe { std::ptr::read(raw.as_ptr() as *const UrgentEvent) };
                    events_clone.push(event);
                }
                idx = idx.wrapping_add(1) % ring_size;
                if idx == 0 && current_write_head < last_seen_write_head {
                    break;
                }
            }
            last_seen_write_head = current_write_head;
            std::thread::sleep(Duration::from_millis(poll_interval_ms));
        }
    });
    events
}
