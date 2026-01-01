use super::mcts_othello::GpuOthelloMcts;
use std::sync::Mutex;
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
    // Mutex to guard all buffer map/unmap/submit operations for urgent event buffers
    // This ensures only one thread can access these buffers at a time, preventing mapped buffer validation errors
    static URGENT_EVENT_BUFFER_MUTEX: once_cell::sync::Lazy<Mutex<()>> = once_cell::sync::Lazy::new(|| Mutex::new(()));
    thread::spawn(move || {
        println!("[URGENT LOGGER] Thread started for GpuOthelloMcts");
        let mut last_seen_write_head: u32 = 0;
        let ring_size = 256u32;
        let mut poll_count = 0;
        while !stop_flag_clone.load(Ordering::Relaxed) {
            // Lock and get buffer handles
            let _buffer_guard = URGENT_EVENT_BUFFER_MUTEX.lock().unwrap();
            let (urgent_event_buffer_gpu, urgent_event_write_head_gpu, urgent_event_staging, urgent_event_write_head_staging) = {
                let inner = engine_clone.inner.lock().unwrap();
                (
                    inner.urgent_event_buffer_gpu.as_ref().cloned(),
                    inner.urgent_event_write_head_gpu.as_ref().cloned(),
                    inner.urgent_event_staging.as_ref().cloned(),
                    inner.urgent_event_write_head_staging.as_ref().cloned(),
                )
            };
            if urgent_event_buffer_gpu.is_none() || urgent_event_write_head_gpu.is_none() || urgent_event_staging.is_none() || urgent_event_write_head_staging.is_none() {
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
            let urgent_event_buffer_gpu = urgent_event_buffer_gpu.unwrap();
            let urgent_event_write_head_gpu = urgent_event_write_head_gpu.unwrap();
            let urgent_event_staging = urgent_event_staging.unwrap();
            let urgent_event_write_head_staging = urgent_event_write_head_staging.unwrap();

            // Copy GPU buffer to staging buffer
            let device = engine_clone.context.device();
            let queue = engine_clone.context.queue();
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("UrgentEventCopyToStaging"),
            });
            encoder.copy_buffer_to_buffer(&urgent_event_buffer_gpu, 0, &urgent_event_staging, 0, (ring_size as usize * std::mem::size_of::<UrgentEvent>()) as u64);
            encoder.copy_buffer_to_buffer(&urgent_event_write_head_gpu, 0, &urgent_event_write_head_staging, 0, 4);
            queue.submit(Some(encoder.finish()));

            // Map and read write head from staging
            let write_head_slice = urgent_event_write_head_staging.slice(..);
            write_head_slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let write_head_data = write_head_slice.get_mapped_range();
            let current_write_head = u32::from_le_bytes([write_head_data[0], write_head_data[1], write_head_data[2], write_head_data[3]]);
            drop(write_head_data);
            urgent_event_write_head_staging.unmap();
            device.poll(wgpu::Maintain::Wait);

            // Map and read urgent event buffer from staging
            let buffer_slice = urgent_event_staging.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let buffer_data = buffer_slice.get_mapped_range();
            let event_bytes = &buffer_data[..];

            // Handle write head reset (e.g. new kernel dispatch)
            if current_write_head < last_seen_write_head {
                // println!("[URGENT LOGGER] Write head reset detected ({} -> {}). Resetting last_seen.", last_seen_write_head, current_write_head);
                last_seen_write_head = 0;
            }

            // Handle buffer overflow (host too slow)
            if current_write_head > last_seen_write_head + ring_size {
                println!("[URGENT LOGGER] Buffer overflow! Skipped {} events.", current_write_head - (last_seen_write_head + ring_size));
                last_seen_write_head = current_write_head - ring_size;
            }

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
                        10 => println!("[URGENT EVENT] REROOT_START (idx={}) payload[0..4]: {:?}", idx, &event.payload[0..4]),
                        11 => {
                            // REROOT_END: payload[0]=turn_number, payload[1]=prev (atomic), payload[2..]=...
                            println!("[URGENT EVENT] REROOT_END   (idx={}) turn={} prev={} payload[2..4]: {:?}", idx, event.payload[0], event.payload[1], &event.payload[2..4]);
                            if event.payload[1] != 0 {
                                println!("[DIAG] REROOT_END prev value is not zero! prev={}", event.payload[1]);
                            }
                        }
                        12 => if !pruning_start_printed {
                            println!("[URGENT EVENT] PRUNING_START");
                            pruning_start_printed = true;
                        },
                        13 => if !pruning_end_printed {
                            println!("[URGENT EVENT] PRUNING_END");
                            pruning_end_printed = true;
                        },
                        14 => println!("[URGENT EVENT] MEMORY_PRESSURE (node_idx: {})", event.payload[0]),
                        15 => println!("[URGENT EVENT] EARLY_EXIT (thread_idx: {})", event.payload[0]),
                        1 => (),
                        2 => println!("[URGENT EVENT] HALT"),
                        _ => println!("[URGENT EVENT] type {} payload[0..4]: {:?}", event.event_type, &event.payload[0..4]),
                    }
                    events_clone.push(event);
                    found_any = true;
                }
            }

            if found_any {
                println!("[URGENT LOGGER DIAG] urgent_event_staging[0..32]: {:?}", &event_bytes[..32.min(event_bytes.len())]);
                // Print the first 16 bytes of the write head buffer for diagnostics only if new events are found
                let write_head_slice = urgent_event_write_head_staging.slice(..16);
                write_head_slice.map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::Maintain::Wait);
                let write_head_data = write_head_slice.get_mapped_range();
                println!("[URGENT LOGGER DIAG] urgent_event_write_head_staging[0..16]: {:?}", &write_head_data[..16.min(write_head_data.len())]);
                drop(write_head_data);
                urgent_event_write_head_staging.unmap();
                println!("[URGENT LOGGER] Found urgent event(s) in this poll cycle");
            }

            drop(buffer_data);
            urgent_event_staging.unmap();
            device.poll(wgpu::Maintain::Wait);
            // Mutex guard drops here, allowing next access

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
            
            // Acquire lock
            while urgent_event_buffers.urgent_event_buffer_in_use.swap(true, Ordering::Acquire) {
                std::thread::yield_now();
            }

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
            drop(data);
            urgent_event_buffers.urgent_event_write_head_staging.unmap();

            // Release lock
            urgent_event_buffers.urgent_event_buffer_in_use.store(false, Ordering::Release);

            let events_vec = engine_clone.poll_urgent_events();
            let mut idx = last_seen_write_head;
            while idx != current_write_head {
                let slot = (idx % ring_size) as usize;
                if slot < events_vec.len() {
                    let raw = &events_vec[slot];
                    let event: UrgentEvent = unsafe { std::ptr::read(raw.as_ptr() as *const UrgentEvent) };
                    // Print/log new urgent event types with clear messages
                    match event.event_type {
                        10 => println!("[URGENT EVENT] BATCH_START (root_idx: {})", event.payload[0]),
                        11 => println!("[URGENT EVENT] BATCH_END (root_idx: {})", event.payload[0]),
                        12 => println!("[URGENT EVENT] PRUNING_START"),
                        13 => println!("[URGENT EVENT] PRUNING_END"),
                        14 => println!("[URGENT EVENT] MEMORY_PRESSURE (node_idx: {})", event.payload[0]),
                        15 => println!("[URGENT EVENT] REROOT_OP_START"),
                        16 => println!("[URGENT EVENT] REROOT_OP_END"),
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
