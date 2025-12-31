//! Minimal test for urgent event emission from a compute shader
use wgpu::*;

#[test]
fn test_minimal_urgent_event_kernel() {
    // Setup WGPU device/queue (use your context setup if available)
    let instance = Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).expect("No adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: None,
        required_features: Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: Default::default(),
    }, None)).expect("No device");

    // Create urgent event buffer and write head
    // Each UrgentEvent is 4 + 4 + 4 + 255*4 = 1028 bytes, but std430 alignment pads to 1032 bytes (multiple of 16)
    let urgent_event_struct_size = 1032;
    let urgent_event_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Urgent Event Buffer"),
        size: (256 * urgent_event_struct_size) as u64, // 264192
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let urgent_event_write_head = device.create_buffer(&BufferDescriptor {
        label: Some("Urgent Event Write Head"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let urgent_event_write_head_staging = device.create_buffer(&BufferDescriptor {
        label: Some("Urgent Event Write Head Staging"),
        size: 4,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let urgent_event_staging = device.create_buffer(&BufferDescriptor {
        label: Some("Urgent Event Staging"),
        size: (256 * urgent_event_struct_size) as u64, // 264192
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Load shader
    let shader = device.create_shader_module(include_wgsl!("shaders/test_urgent_event.wgsl"));
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
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Urgent Event Bind Group"),
        layout: &layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: urgent_event_buffer.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: urgent_event_write_head.as_entire_binding() },
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

    // Dispatch kernel
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
    // Copy buffers for readback
    encoder.copy_buffer_to_buffer(&urgent_event_write_head, 0, &urgent_event_write_head_staging, 0, 4);
        encoder.copy_buffer_to_buffer(&urgent_event_buffer, 0, &urgent_event_staging, 0, urgent_event_struct_size as u64);
    queue.submit(Some(encoder.finish()));

    // Read back write head
    {
        let slice = urgent_event_write_head_staging.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(Maintain::Wait);
        let data = slice.get_mapped_range();
        let val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        println!("[TEST] urgent_event_write_head = {}", val);
        assert!(val > 0, "Kernel did not write urgent event");
    }
    // Read back urgent event
    {
        let slice = urgent_event_staging.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(Maintain::Wait);
        let data = slice.get_mapped_range();
        let timestamp = u32::from_le_bytes([data[0],data[1],data[2],data[3]]);
        let event_type = u32::from_le_bytes([data[4],data[5],data[6],data[7]]);
        println!("[TEST] urgent_event[0] timestamp = {} type = {}", timestamp, event_type);
        assert_eq!(timestamp, 12345678u32, "Timestamp mismatch");
        assert_eq!(event_type, 42u32, "Event type mismatch");
    }
}
