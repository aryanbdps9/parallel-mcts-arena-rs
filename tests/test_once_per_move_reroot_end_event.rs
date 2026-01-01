//! Integration test: Only one REROOT_END event per kernel dispatch

use std::sync::Arc;
use mcts::gpu::mcts_othello::{GpuOthelloMcts, UrgentEvent};
use mcts::gpu::GpuContext;
use wgpu;

#[test]
fn test_once_per_move_reroot_end_event() {
    // Setup GPU context and engine
    let config = mcts::gpu::GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    let engine = GpuOthelloMcts::new(context.clone(), 1024, 128).expect("Failed to create engine");

    // Prepare legal moves as (row, col)
    let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
    
    // Initial board
    let mut board = [0i32; 64];
    board[3 * 8 + 3] = 1;
    board[4 * 8 + 4] = 1;
    board[3 * 8 + 4] = -1;
    board[4 * 8 + 3] = -1;

    engine.init_tree(&board, 1, &legal_moves);

    // Run the kernel with MANY workgroups to stress test the atomic coordination
    engine.dispatch_mcts_othello_kernel(128);

    // Poll urgent events manually
    let device = context.device();
    let queue = context.queue();
    
    let (urgent_event_buffer_gpu, urgent_event_write_head_gpu, urgent_event_staging, urgent_event_write_head_staging) = {
        let inner = engine.inner.lock().unwrap();
        (
            inner.urgent_event_buffer_gpu.as_ref().cloned().unwrap(),
            inner.urgent_event_write_head_gpu.as_ref().cloned().unwrap(),
            inner.urgent_event_staging.as_ref().cloned().unwrap(),
            inner.urgent_event_write_head_staging.as_ref().cloned().unwrap(),
        )
    };

    // Copy GPU buffer to staging buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("UrgentEventCopyToStaging"),
    });
    let ring_size = 256u64;
    encoder.copy_buffer_to_buffer(&urgent_event_buffer_gpu, 0, &urgent_event_staging, 0, ring_size * std::mem::size_of::<UrgentEvent>() as u64);
    encoder.copy_buffer_to_buffer(&urgent_event_write_head_gpu, 0, &urgent_event_write_head_staging, 0, 4);
    queue.submit(Some(encoder.finish()));

    // Map and read write head
    let write_head_slice = urgent_event_write_head_staging.slice(..);
    write_head_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let write_head_data = write_head_slice.get_mapped_range();
    let write_head = u32::from_le_bytes([write_head_data[0], write_head_data[1], write_head_data[2], write_head_data[3]]);
    drop(write_head_data);
    urgent_event_write_head_staging.unmap();

    // Map and read events
    let buffer_slice = urgent_event_staging.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let buffer_data = buffer_slice.get_mapped_range();
    let event_bytes = &buffer_data[..];

    let mut reroot_end_count = 0;
    // Handle ring buffer wrapping: only read the last 256 events if write_head > 256
    let start_idx = if write_head > ring_size as u32 { write_head - ring_size as u32 } else { 0 };
    
    for idx in start_idx..write_head {
        let slot = (idx as u64 % ring_size) as usize;
        let offset = slot * std::mem::size_of::<UrgentEvent>();
        let event_ptr = event_bytes[offset..offset + std::mem::size_of::<UrgentEvent>()].as_ptr() as *const UrgentEvent;
        let event: UrgentEvent = unsafe { std::ptr::read_unaligned(event_ptr) };
        
        if event.event_type == 11 { // REROOT_END
            reroot_end_count += 1;
            println!("Found REROOT_END event at idx {}", idx);
        }
    }
    drop(buffer_data);
    urgent_event_staging.unmap();
    
    assert_eq!(reroot_end_count, 1, "Expected exactly one REROOT_END event, got {}", reroot_end_count);
}
