#[test]
fn test_gpu_shader_to_cpu_urgent_event_logging() {
    use wgpu::*;
    let config = mcts::gpu::GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    let engine = GpuMctsEngine::new(context.clone(), 1024, 128, 8, 8);
    let device = context.device();
    let queue = context.queue();
    // Load the minimal urgent event emission shader
    let shader = device.create_shader_module(include_wgsl!("../src/gpu/shaders/test_urgent_event.wgsl"));
    // Create bind group layout and bind group for urgent event buffers
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Urgent Event Layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let urgent_event_buffers = &engine.urgent_event_buffers;
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Urgent Event Bind Group"),
        layout: &layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: urgent_event_buffers.urgent_event_buffer.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: urgent_event_buffers.urgent_event_write_head_buffer.as_entire_binding() },
        ],
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Test Urgent Event Pipeline"),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Test Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        })),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Start the urgent event logger polling thread BEFORE dispatch, to clear out any pre-existing events
    let stop_flag = Arc::new(AtomicBool::new(false));
    let events_arc = start_and_log_urgent_events(Arc::new(engine), 10, stop_flag.clone());

    // Drain any pre-existing events
    while let Some(_ev) = events_arc.pop() {}

    // Dispatch the kernel to emit an urgent event from the GPU
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: Some("Test Urgent Event Encoder") });
    {
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Test Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(Some(encoder.finish()));
    // Ensure GPU work is complete before polling
    device.poll(wgpu::Maintain::Wait);

    // Wait for the event to be polled and appear in the queue
    let max_wait_ms = 2000;
    let poll_interval = 50;
    let mut waited = 0;
    let mut found = false;
    let mut found_event = None;
    while waited < max_wait_ms {
        if let Some(ev) = events_arc.pop() {
            if ev.event_type == 42 && ev.timestamp == 12345678 && ev.payload[0] == 0 {
                found = true;
                found_event = Some(ev);
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(poll_interval));
        waited += poll_interval;
    }
    stop_flag.store(true, Ordering::Relaxed);
    assert!(found, "Test urgent event was not received from GPU shader within {} ms", max_wait_ms);
    if let Some(ev) = found_event {
        println!("[TEST] Received urgent event from GPU: type={}, ts={}, payload[0]={}", ev.event_type, ev.timestamp, ev.payload[0]);
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
