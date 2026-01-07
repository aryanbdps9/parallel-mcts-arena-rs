use mcts::gpu::{GpuConfig, GpuContext};

#[test]
fn test_rng_uniformity() {
    let config = GpuConfig::default();
    let context = GpuContext::new(&config).expect("Failed to create GPU context");
    let device = context.device();
    let queue = context.queue();
    
    // Simple shader that just generates random p1_scores
    let shader_source = r#"
struct Diagnostics {
    p1_score_sum: atomic<u32>,
    p1_wins: atomic<u32>,
    p1_losses: atomic<u32>,
    p1_draws: atomic<u32>,
    total_samples: atomic<u32>,
}

struct RolloutJob {
    leaf_player: i32,
    position_hash: u32,  // Hash of board position - different for each leaf
}

@group(0) @binding(0) var<storage, read_write> diagnostics: Diagnostics;
@group(0) @binding(1) var<storage, read_write> rng_states: array<u32>;
@group(0) @binding(2) var<storage, read_write> rollout_queue: array<RolloutJob>;
@group(0) @binding(3) var<storage, read_write> queue_head: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> queue_tail: atomic<u32>;

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Separate kernel for selection phase (like MCTS incremental_mcts_step)
@compute @workgroup_size(256)
fn selection_phase(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    if (thread_id >= 8192u) { return; }
    
    var rng_state = rng_states[thread_id];
    let num_selections = thread_id % 6u;
    for (var i = 0u; i < num_selections; i++) {
        rng_state = pcg_hash(rng_state);
    }
    rng_states[thread_id] = rng_state;
    
    // Enqueue rollout jobs
    if (thread_id % 10u == 0u) {
        let job_idx = atomicAdd(&queue_tail, 1u);
        if (job_idx < 8192u) {
            rollout_queue[job_idx].leaf_player = 1;
            rollout_queue[job_idx].position_hash = pcg_hash(thread_id * 7919u + job_idx);
        }
    }
}

// Separate kernel for rollout phase (like MCTS rollout_chunk_kernel)
@compute @workgroup_size(256)
fn rollout_phase(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    if (thread_id >= 8192u) { return; }
    
    let my_job_idx = atomicAdd(&queue_head, 1u);
    if (my_job_idx < atomicLoad(&queue_tail)) {
        let job = rollout_queue[my_job_idx];
        var rng_state = rng_states[thread_id];
        rng_state = rng_state ^ job.position_hash;
        rng_state = pcg_hash(rng_state);
        let p1_score = rng_state % 65u;
        rng_states[thread_id] = rng_state;
        
        atomicAdd(&diagnostics.p1_score_sum, p1_score);
        atomicAdd(&diagnostics.total_samples, 1u);
        
        if (p1_score > 32u) {
            atomicAdd(&diagnostics.p1_wins, 1u);
        } else if (p1_score < 32u) {
            atomicAdd(&diagnostics.p1_losses, 1u);
        } else {
            atomicAdd(&diagnostics.p1_draws, 1u);
        }
    }
}
"#;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("RNG Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    
    // Create diagnostics buffer
    let diagnostics_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Diagnostics"),
        size: 20, // 5 u32s
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    // Create RNG states buffer (8192 u32s = 32768 bytes)
    let rng_states_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("RNG States"),
        size: 8192 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    // Initialize RNG states (same as MCTS: seed + thread_id, NOT hashed)
    let init_states: Vec<u32> = (0..8192).map(|tid| 42 + tid).collect();
    queue.write_buffer(&rng_states_buf, 0, bytemuck::cast_slice(&init_states));
    
    // Create rollout queue buffer
    let queue_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Rollout Queue"),
        size: 8192 * 8, // 8192 jobs * 8 bytes (2 u32s per job)
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    // Create queue head/tail atomics
    let queue_head_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Queue Head"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let queue_tail_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Queue Tail"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    // Create explicit bind group layout (shared between pipelines)
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("RNG Test Bind Group Layout"),
        entries: &[
            // Binding 0: diagnostics
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 1: rng_states
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 2: rollout_queue
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 3: queue_head
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 4: queue_tail
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("RNG Test Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    // Create pipelines (separate for selection and rollout, like MCTS)
    let selection_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Selection Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("selection_phase"),
        compilation_options: Default::default(),
        cache: None,
    });
    
    let rollout_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Rollout Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("rollout_phase"),
        compilation_options: Default::default(),
        cache: None,
    });
    
    // Create bind group (shared between both pipelines)
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("RNG Test Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: diagnostics_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rng_states_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: queue_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: queue_head_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: queue_tail_buf.as_entire_binding(),
            },
        ],
    });
    
    // Zero the buffers
    queue.write_buffer(&diagnostics_buf, 0, &[0u8; 20]);
    
    // Dispatch kernel 1000 times (like MCTS running 1000 steps)
    for _iter in 0..1000 {
        // Reset queue head/tail for each iteration
        queue.write_buffer(&queue_head_buf, 0, &0u32.to_le_bytes());
        queue.write_buffer(&queue_tail_buf, 0, &0u32.to_le_bytes());
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RNG Test Encoder"),
        });
        
        // PHASE 1: Selection (like incremental_mcts_step)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Selection Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&selection_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(32, 1, 1);
        }
        
        // PHASE 2: Rollout (like rollout_chunk_kernel) - separate dispatch!
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Rollout Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&rollout_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(32, 1, 1);
        }
        
        queue.submit(Some(encoder.finish()));
    }
    
    device.poll(wgpu::Maintain::Wait);
    
    // Read back results
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Copy Encoder"),
    });
    
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging"),
        size: 20,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    encoder.copy_buffer_to_buffer(&diagnostics_buf, 0, &staging_buf, 0, 20);
    queue.submit(Some(encoder.finish()));
    
    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    
    let data = slice.get_mapped_range();
    let results: &[u32] = bytemuck::cast_slice(&data);
    
    let p1_score_sum = results[0];
    let p1_wins = results[1];
    let p1_losses = results[2];
    let p1_draws = results[3];
    let total_samples = results[4];
    
    println!("\n=== RNG Uniformity Test (MCTS-style) ===");
    println!("Threads: 8192, iterations: 1000");
    println!("Total samples: {}", total_samples);
    println!("Initial RNG seeds: 42 + thread_id (NOT hashed, like MCTS)");
    println!("Variable RNG advancement: each thread does (thread_id % 6) selections");
    println!("Job queue: ~10% enqueue jobs, ANY thread can dequeue ANY job");
    println!("Position variance: each job has different position_hash mixed into RNG");
    println!("Separate kernels: selection_phase then rollout_phase (2 dispatches)");
    println!();
    println!("p1_score sum: {}", p1_score_sum);
    println!("Average p1_score: {:.4} (expected 32.0)", p1_score_sum as f64 / total_samples as f64);
    println!();
    println!("P1 wins (score > 32): {} ({:.2}%, expected 49.23%)", p1_wins, 100.0 * p1_wins as f64 / total_samples as f64);
    println!("P1 losses (score < 32): {} ({:.2}%, expected 49.23%)", p1_losses, 100.0 * p1_losses as f64 / total_samples as f64);
    println!("Draws (score == 32): {} ({:.2}%, expected 1.54%)", p1_draws, 100.0 * p1_draws as f64 / total_samples as f64);
    println!();
    
    // Statistical test: average should be within 0.1 of 32.0
    let avg = p1_score_sum as f64 / total_samples as f64;
    assert!((avg - 32.0).abs() < 0.1, "Average p1_score {} too far from 32.0", avg);
    
    // Win rate should be within 1% of 49.23%
    let win_rate = 100.0 * p1_wins as f64 / total_samples as f64;
    assert!((win_rate - 49.23).abs() < 1.0, "Win rate {}% too far from 49.23%", win_rate);
    
    println!("✓ RNG uniformity test PASSED");
}
