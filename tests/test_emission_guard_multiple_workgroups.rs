//! Test emission guard with multiple workgroups and edge cases

#[cfg(test)]
mod tests {
    use mcts::{GpuContext, GpuConfig};
    use wgpu::util::DeviceExt;

    #[test]
    fn test_emission_guard_multiple_workgroups() {
        let config = GpuConfig::default();
        let context = GpuContext::new(&config).expect("Failed to create GpuContext");
        let device = context.device();
        let queue = context.queue();
        let workgroup_size = 32u32;
        let num_workgroups = 4u32;
        let total_threads = workgroup_size * num_workgroups;

        // Create a buffer for the atomic (initialized to total_threads)
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Emission Guard Buffer"),
            contents: &total_threads.to_le_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        // Output buffer for event emission (1 u32, incremented by each emitter)
        let outbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Emission OutBuf"),
            contents: &[0, 0, 0, 0],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // WGSL kernel: only one thread should write 1 to outbuf[0]
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Test Emission Guard Kernel"),
            source: wgpu::ShaderSource::Wgsl(format!(r#"
@group(5) @binding(0) var<storage, read_write> global_atomic: atomic<u32>;
@group(5) @binding(1) var<storage, read_write> outbuf: array<u32, 1>;
@compute @workgroup_size({})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let prev = atomicSub(&global_atomic, 1u);
    if (prev == 1u) {{
        outbuf[0] = 1u;
    }}
}}
"#, workgroup_size).into()),
        });

        // Bind group layout and bind group for group 5
        let group5_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Test Group5 Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let group5_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Test Group5 BindGroup"),
            layout: &group5_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }, wgpu::BindGroupEntry {
                binding: 1,
                resource: outbuf.as_entire_binding(),
            }],
        });

        // Dummy layouts for groups 0-4
        let dummy_layouts: Vec<_> = (0..5).map(|i| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("Dummy Layout {}", i)),
            entries: &[],
        })).collect();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Test Pipeline Layout"),
            bind_group_layouts: &[
                &dummy_layouts[0],
                &dummy_layouts[1],
                &dummy_layouts[2],
                &dummy_layouts[3],
                &dummy_layouts[4],
                &group5_layout,
            ],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Test Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Dispatch kernel
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Test ComputePass"),
                timestamp_writes: None,
            });
            for i in 0..5 {
                pass.set_bind_group(i as u32, &device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Dummy BindGroup {}", i)),
                    layout: &dummy_layouts[i],
                    entries: &[],
                }), &[]);
            }
            pass.set_bind_group(5, &group5_bind_group, &[]);
            pass.set_pipeline(&pipeline);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Read back outbuf
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Staging Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(&outbuf, 0, &staging, 0, 4);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(val, 1, "Emission guard failed with multiple workgroups: more than one or no thread emitted!");
    }
}
