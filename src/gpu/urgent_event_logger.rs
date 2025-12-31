use super::mcts_othello::GpuOthelloMcts;
/// Starts the urgent event polling thread for GpuOthelloMcts and prints events every interval.


pub fn start_and_log_urgent_events_othello(
    gpu_engine: Arc<GpuOthelloMcts>,
    poll_interval_ms: u64,
    stop_flag: Arc<AtomicBool>,
) -> Arc<SegQueue<UrgentEvent>> {
    let events = Arc::new(SegQueue::new());
    let events_clone = Arc::clone(&events);
    let engine_clone = gpu_engine.clone();
    let stop_flag_clone = Arc::clone(&stop_flag);
    thread::spawn(move || {
        println!("[URGENT LOGGER] Thread started for GpuOthelloMcts");
        let mut last_seen_write_head: u32 = 0;
        let ring_size = 256u32;
        let mut poll_count = 0;
        while !stop_flag_clone.load(Ordering::Relaxed) {
            // Lock and get buffer handles
            let (urgent_event_buffer_host, urgent_event_write_head_host, urgent_event_buffer_gpu, urgent_event_write_head_gpu) = {
                let inner = engine_clone.inner.lock().unwrap();
                (
                    inner.urgent_event_buffer_host.as_ref().cloned(),
                    inner.urgent_event_write_head_host.as_ref().cloned(),
                    inner.urgent_event_buffer_gpu.as_ref().cloned(),
                    inner.urgent_event_write_head_gpu.as_ref().cloned(),
                )
            };
            if urgent_event_buffer_host.is_none() || urgent_event_write_head_host.is_none() || urgent_event_buffer_gpu.is_none() || urgent_event_write_head_gpu.is_none() {
                if poll_count % 10 == 0 {
                    println!("[URGENT LOGGER] Waiting for buffers to be ready...");
                }
                std::thread::sleep(Duration::from_millis(poll_interval_ms));
                poll_count += 1;
                continue;
            }
                        if poll_count % 20 == 0 {
                            println!("[URGENT LOGGER] Polling for urgent events (last_seen_write_head={})", last_seen_write_head);
                        }
                        poll_count += 1;
            let urgent_event_buffer_host = urgent_event_buffer_host.unwrap();
            let urgent_event_write_head_host = urgent_event_write_head_host.unwrap();
            let urgent_event_buffer_gpu = urgent_event_buffer_gpu.unwrap();
            let urgent_event_write_head_gpu = urgent_event_write_head_gpu.unwrap();



            // (Do not unmap here; only unmap after reading. wgpu will error if you unmap an unmapped buffer.)

            // Copy GPU buffer to host-mapped buffer
            let device = engine_clone.context.device();
            let queue = engine_clone.context.queue();
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("UrgentEventCopyToHost"),
            });
            encoder.copy_buffer_to_buffer(&urgent_event_buffer_gpu, 0, &urgent_event_buffer_host, 0, (ring_size as usize * std::mem::size_of::<UrgentEvent>()) as u64);
            encoder.copy_buffer_to_buffer(&urgent_event_write_head_gpu, 0, &urgent_event_write_head_host, 0, 4);
            queue.submit(Some(encoder.finish()));

            // Map and read write head
            let write_head_slice = urgent_event_write_head_host.slice(..);
            write_head_slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let write_head_data = write_head_slice.get_mapped_range();
            let current_write_head = u32::from_le_bytes([write_head_data[0], write_head_data[1], write_head_data[2], write_head_data[3]]);

            drop(write_head_data);
            urgent_event_write_head_host.unmap();
            device.poll(wgpu::Maintain::Wait); // Ensure unmap is complete before next copy

            // Map and read urgent event buffer
            let buffer_slice = urgent_event_buffer_host.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let buffer_data = buffer_slice.get_mapped_range();
            let event_bytes = &buffer_data[..];
            // Print the first 32 bytes of the urgent event buffer for diagnostics only if new events are found
            let mut found_any = false;
            let mut pruning_start_printed = false;
            let mut pruning_end_printed = false;
            for idx in last_seen_write_head..current_write_head {
                let slot = (idx % ring_size) as usize;
                let offset = slot * std::mem::size_of::<UrgentEvent>();
                if offset + std::mem::size_of::<UrgentEvent>() <= event_bytes.len() {
                    let event_ptr = event_bytes[offset..offset + std::mem::size_of::<UrgentEvent>()].as_ptr() as *const UrgentEvent;
                    let event: UrgentEvent = unsafe { std::ptr::read_unaligned(event_ptr) };
                    match event.event_type {
                        10 => println!("[URGENT EVENT] REROOT_START (root_idx: {})", event.payload[0]),
                        11 => println!("[URGENT EVENT] REROOT_END (root_idx: {})", event.payload[0]),
                        12 => if !pruning_start_printed {
                            println!("[URGENT EVENT] PRUNING_START");
                            pruning_start_printed = true;
                        },
                        13 => if !pruning_end_printed {
                            println!("[URGENT EVENT] PRUNING_END");
                            pruning_end_printed = true;
                        },
                        14 => println!("[URGENT EVENT] MEMORY_PRESSURE (node_idx: {})", event.payload[0]),
                        1 => (),
                        2 => println!("[URGENT EVENT] HALT"),
                        _ => println!("[URGENT EVENT] type {} payload[0..4]: {:?}", event.event_type, &event.payload[0..4]),
                    }
                    events_clone.push(event);
                    found_any = true;
                }
            }
            if found_any {
                println!("[URGENT LOGGER DIAG] urgent_event_buffer_host[0..32]: {:?}", &event_bytes[..32.min(event_bytes.len())]);
                // Print the first 16 bytes of the write head buffer for diagnostics only if new events are found
                let write_head_slice = urgent_event_write_head_host.slice(..16);
                write_head_slice.map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::Maintain::Wait);
                let write_head_data = write_head_slice.get_mapped_range();
                println!("[URGENT LOGGER DIAG] urgent_event_write_head_host[0..16]: {:?}", &write_head_data[..16.min(write_head_data.len())]);
                drop(write_head_data);
                urgent_event_write_head_host.unmap();
                println!("[URGENT LOGGER] Found urgent event(s) in this poll cycle");
            }
            drop(buffer_data);
            urgent_event_buffer_host.unmap();
            device.poll(wgpu::Maintain::Wait); // Ensure unmap is complete before next copy

            last_seen_write_head = current_write_head;
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
                    let event: UrgentEvent = unsafe { std::ptr::read(raw.as_ptr() as *const UrgentEvent) };
                    // Print/log new urgent event types with clear messages
                    match event.event_type {
                        10 => println!("[URGENT EVENT] REROOT_START (root_idx: {})", event.payload[0]),
                        11 => println!("[URGENT EVENT] REROOT_END (root_idx: {})", event.payload[0]),
                        12 => println!("[URGENT EVENT] PRUNING_START"),
                        13 => println!("[URGENT EVENT] PRUNING_END"),
                        14 => println!("[URGENT EVENT] MEMORY_PRESSURE (node_idx: {})", event.payload[0]),
                        1 => (), // URGENT_EVENT_START: skip or print if desired
                        2 => println!("[URGENT EVENT] HALT"),
                        _ => println!("[URGENT EVENT] type {} payload[0..4]: {:?}", event.event_type, &event.payload[0..4]),
                    }
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
