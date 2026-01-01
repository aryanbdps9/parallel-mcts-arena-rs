// Regression test: pipeline layout and bind group must match WGSL for urgent event logging (prune kernel)
// This test intentionally creates a pipeline with a missing binding to ensure validation fails.
// It should panic with a wgpu validation error if the root cause is not fixed.
// NOTE: Due to Rust's unwind safety rules, we cannot catch the panic from wgpu::Device pipeline creation.
// If the pipeline layout is invalid, this test will panic and fail (as desired).

#[test]
fn test_prune_kernel_pipeline_layout_validation() {
    use wgpu::*;
    let instance = Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).expect("No adapter");
    let (device, _queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: None,
        required_features: Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: Default::default(),
    }, None)).expect("No device");

    // Create dummy buffers for all groups
    let _dummy = device.create_buffer(&BufferDescriptor {
        label: Some("Dummy"),
        size: 4096,
        usage: BufferUsages::STORAGE | BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    // Create group 3 layout with only binding 0 (should fail if shader expects binding 1)
    let group3_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Group3 Layout (regression test, missing binding 1)"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    // Other dummy layouts
    let dummy_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Dummy Layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let group4_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Group4 Layout (dummy)"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let urgent_event_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Urgent Event Layout (correct)"),
        entries: &[BindGroupLayoutEntry {
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
        }],
    });
    // Compose pipeline layout with group 3 missing binding 1
    let bind_group_layouts = vec![&dummy_layout, &dummy_layout, &dummy_layout, &group3_layout, &group4_layout, &urgent_event_layout];
    let shader = device.create_shader_module(include_wgsl!("../src/gpu/shaders/mcts_othello.wgsl"));
    // This should panic if the pipeline layout does not match the shader (root cause regression)
    let _pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Regression Prune Pipeline (should fail)"),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Regression Prune Pipeline Layout (should fail)"),
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: &[],
        })),
        module: &shader,
        entry_point: Some("prune_unreachable_topdown"),
        cache: None,
        compilation_options: Default::default(),
    });
}
