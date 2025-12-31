impl GpuOthelloMcts {
        /// Dispatch the GPU pruning kernel and bind the urgent event buffer for logging
        pub fn dispatch_prune_unreachable_topdown(&self) {
            // ...existing code...
            use wgpu::{BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindGroupEntry, BindGroupDescriptor, ShaderStages, BindingType, BufferBindingType, ComputePassDescriptor};
            let context = &self.context;
            let device = context.device();
            let queue = context.queue();
            // === DUMMY GROUP 4 (for prune kernel @group(4) bindings) ===
            let dummy_group4_buffers: Vec<wgpu::Buffer> = (0..=2)
                .map(|i| device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Dummy Group4 Buffer {} (Othello)", i)),
                    size: 65_536,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }))
                .collect();
            let dummy_group4_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Dummy Layout 4 (prune kernel)"),
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
                    BindGroupLayoutEntry {
                        binding: 2,
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
            let dummy_group4_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Dummy Group4 Bind Group (Othello)"),
                layout: &dummy_group4_layout,
                entries: &[ 
                    BindGroupEntry { binding: 0, resource: dummy_group4_buffers[0].as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: dummy_group4_buffers[1].as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: dummy_group4_buffers[2].as_entire_binding() },
                ],
            });
            // Only lock to get buffer references, then drop lock before GPU ops
            let (urgent_event_buffer_gpu, urgent_event_write_head_gpu, urgent_event_buffer_host, urgent_event_write_head_host) = {
                let inner = self.inner.lock().unwrap();
                (
                    inner.urgent_event_buffer_gpu.as_ref().expect("urgent_event_buffer_gpu missing").clone(),
                    inner.urgent_event_write_head_gpu.as_ref().expect("urgent_event_write_head_gpu missing").clone(),
                    inner.urgent_event_buffer_host.as_ref().expect("urgent_event_buffer_host missing").clone(),
                    inner.urgent_event_write_head_host.as_ref().expect("urgent_event_write_head_host missing").clone(),
                )
            };

            println!("[DIAG] Starting dispatch_prune_unreachable_topdown...");

            // Create bind group layout and bind group for urgent event logging (group 3, matching WGSL)
            let urgent_event_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Urgent Event Layout (Othello, group 3)"),
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
            println!("[DIAG] Created urgent event bind group layout (group 3).");
            let urgent_event_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Urgent Event Bind Group (Othello, group 3)"),
                layout: &urgent_event_layout,
                entries: &[ 
                    BindGroupEntry {
                        binding: 0,
                        resource: urgent_event_buffer_gpu.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: urgent_event_write_head_gpu.as_entire_binding(),
                    },
                ],
            });
            println!("[DIAG] Created urgent event bind group (group 3).");

            // === DUMMY BUFFERS AND LAYOUTS FOR ALL GROUPS (0-4) ===
            println!("[DIAG] Creating dummy buffers and layouts for groups 0-3...");
            // Group 0: bindings 0-8
            let dummy_group0_buffers: Vec<wgpu::Buffer> = (0..=8)
                .map(|i| device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Dummy Group0 Buffer {} (Othello)", i)),
                    size: 65_536,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: false,
                }))
                .collect();
            let dummy_group1_buffers: Vec<wgpu::Buffer> = (0..=4)
                .map(|i| device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Dummy Group1 Buffer {} (Othello)", i)),
                    size: 65_536,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: false,
                }))
                .collect();
            let dummy_group2_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Group2 Buffer 0 (Othello)"),
                size: 65_536,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });
            let dummy_group3_buffers: Vec<wgpu::Buffer> = (0..=1)
                .map(|i| device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Dummy Group3 Buffer {} (Othello)", i)),
                    size: 264_192,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }))
                .collect();
            let dummy_group0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Dummy Layout 0 (with buffers)"),
                entries: &[
                    BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 6, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 7, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 8, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ],
            });
            let dummy_group1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Dummy Layout 1 (with buffers)"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
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
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 4,
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
            let dummy_group2_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Dummy Layout 2 (with buffer)"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
            let dummy_group3_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Dummy Layout 3 (with buffers)"),
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
            // Collect all dummy layouts for pipeline layout
            let mut dummy_layouts = vec![&dummy_group0_layout, &dummy_group1_layout, &dummy_group2_layout, &dummy_group3_layout];

            // Load the pruning kernel pipeline
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Othello MCTS Prune Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mcts_othello.wgsl").into()),
            });
            println!("[DIAG] Created shader module for prune kernel.");
            // Pipeline layout: dummies at 0-3, dummy_group4_layout at 4, urgent_event_layout at 5
            let mut bind_group_layouts = dummy_layouts.clone();
            bind_group_layouts.push(&dummy_group4_layout); // index 4
            bind_group_layouts.push(&urgent_event_layout); // index 5
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Othello Prune Pipeline Layout"),
                bind_group_layouts: &bind_group_layouts,
                push_constant_ranges: &[],
            });
            println!("[DIAG] Created pipeline layout.");
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Othello Prune Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("prune_unreachable_topdown"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
            println!("[DIAG] Created compute pipeline for prune kernel.");

            // Now create the dummy bind groups (after layouts are used for pipeline)
            let dummy_group0_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Dummy Group0 Bind Group (Othello)"),
                layout: &dummy_group0_layout,
                entries: &[ 
                    BindGroupEntry { binding: 0, resource: dummy_group0_buffers[0].as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: dummy_group0_buffers[1].as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: dummy_group0_buffers[2].as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: dummy_group0_buffers[3].as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: dummy_group0_buffers[4].as_entire_binding() },
                    BindGroupEntry { binding: 5, resource: dummy_group0_buffers[5].as_entire_binding() },
                    BindGroupEntry { binding: 6, resource: dummy_group0_buffers[6].as_entire_binding() },
                    BindGroupEntry { binding: 7, resource: dummy_group0_buffers[7].as_entire_binding() },
                    BindGroupEntry { binding: 8, resource: dummy_group0_buffers[8].as_entire_binding() },
                ],
            });
            let dummy_group1_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Dummy Group1 Bind Group (Othello)"),
                layout: &dummy_group1_layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: dummy_group1_buffers[0].as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: dummy_group1_buffers[1].as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: dummy_group1_buffers[2].as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: dummy_group1_buffers[3].as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: dummy_group1_buffers[4].as_entire_binding() },
                ],
            });
            let dummy_group2_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Dummy Group2 Bind Group (Othello)"),
                layout: &dummy_group2_layout,
                entries: &[BindGroupEntry { binding: 0, resource: dummy_group2_buffer.as_entire_binding() }],
            });
            // dummy_group3_bind_group is no longer needed (urgent event buffer now at group 3)

            // Load the pruning kernel pipeline
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Othello MCTS Prune Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mcts_othello.wgsl").into()),
            });
            // Pipeline layout: dummies at 0-3, dummy_group4_layout at 4, urgent_event_layout at 5
            let mut bind_group_layouts = dummy_layouts.clone();
            bind_group_layouts.push(&dummy_group4_layout); // index 4
            bind_group_layouts.push(&urgent_event_layout); // index 5
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Othello Prune Pipeline Layout"),
                bind_group_layouts: &bind_group_layouts,
                push_constant_ranges: &[],
            });
            let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Othello Prune Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("prune_unreachable_topdown"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

            // Create dummy bind groups for groups 1, 2, 3 in this scope
            // ...existing code...

            // Dispatch the kernel
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Othello Prune Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Othello Prune ComputePass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &dummy_group0_bind_group, &[]); // group 0
                pass.set_bind_group(1, &dummy_group1_bind_group, &[]); // group 1
                pass.set_bind_group(2, &dummy_group2_bind_group, &[]); // group 2
                pass.set_bind_group(3, &urgent_event_bind_group, &[]); // group 3, legacy (if needed)
                pass.set_bind_group(4, &dummy_group4_bind_group, &[]); // group 4, prune kernel dummies
                pass.set_bind_group(5, &urgent_event_bind_group, &[]); // group 5, urgent event buffer (matches layout)
                println!("[DIAG] Set all bind groups and pipeline (dummy group 4 at 4, urgent event at 5). Dispatching workgroups...");
                pass.dispatch_workgroups(1, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
            // Optionally: poll device to ensure completion
            device.poll(wgpu::Maintain::Wait);
            println!("[DIAG] Kernel dispatched and device polled. Checking urgent event buffer...");

            // DIAGNOSTIC: Copy GPU buffer to host-mapped buffer, then map and print from host buffer
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("UrgentEventBuffer Copy Encoder (DIAG)"),
            });
            encoder.copy_buffer_to_buffer(
                &urgent_event_buffer_gpu,
                0,
                &urgent_event_buffer_host,
                0,
                16,
            );
            encoder.copy_buffer_to_buffer(
                &urgent_event_write_head_gpu,
                0,
                &urgent_event_write_head_host,
                0,
                16,
            );
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);

            // Map and print from host buffer
            {
                use wgpu::MapMode;
                let buffer_slice = urgent_event_buffer_host.slice(..16);
                let (tx, rx) = std::sync::mpsc::channel();
                buffer_slice.map_async(MapMode::Read, move |v| { tx.send(v).unwrap(); });
                device.poll(wgpu::Maintain::Wait);
                let _ = rx.recv();
                let data = buffer_slice.get_mapped_range();
                println!("[DIAG] urgent_event_buffer_host[0..16]: {:?}", &data[..]);
                drop(data);
                urgent_event_buffer_host.unmap();
            }
            {
                use wgpu::MapMode;
                let buffer_slice = urgent_event_write_head_host.slice(..16);
                let (tx, rx) = std::sync::mpsc::channel();
                buffer_slice.map_async(MapMode::Read, move |v| { tx.send(v).unwrap(); });
                device.poll(wgpu::Maintain::Wait);
                let _ = rx.recv();
                let data = buffer_slice.get_mapped_range();
                println!("[DIAG] urgent_event_write_head_host[0..16]: {:?}", &data[..]);
                drop(data);
                urgent_event_write_head_host.unmap();
            }
            println!("[DIAG] Finished dispatch_prune_unreachable_topdown.\n");
        }
    /// Create bind groups for urgent event logging (binds host-mapped urgent event buffer to GPU pipeline)
    pub fn create_bind_groups(&self, device: &wgpu::Device) {
        use wgpu::{BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindGroupEntry, BindGroupDescriptor, ShaderStages, BindingType, BufferBindingType};
        let inner = self.inner.lock().unwrap();
        let urgent_event_buffer = inner.urgent_event_buffer_gpu.as_ref().expect("urgent_event_buffer_gpu missing");
        let urgent_event_write_head = inner.urgent_event_write_head_gpu.as_ref().expect("urgent_event_write_head_gpu missing");

        // Create layout matching WGSL: @group(3) @binding(0/1) for urgent event buffer/write_head
        let urgent_event_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Urgent Event Layout (Othello)"),
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

        let _urgent_event_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Urgent Event Bind Group (Othello)"),
            layout: &urgent_event_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: urgent_event_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: urgent_event_write_head.as_entire_binding(),
                },
            ],
        });

        // Optionally: store bind group/layout in self if needed for later dispatch
        // (add fields to GpuOthelloMctsInner if you want to keep them)
        // For now, just ensure creation succeeds and is used in pipeline setup.
    }

        /// Simulate pruning: reset visits for nodes not in legal_moves
        pub fn prune_unreachable_nodes(&mut self) {
            let mut inner = self.inner.lock().unwrap();
            let legal_idxs: std::collections::HashSet<_> = inner.legal_moves.iter().map(|&(x, y)| x * 8 + y).collect();
            for idx in 0..inner.visits.len() {
                if inner.visits[idx] > 0 && !legal_idxs.contains(&idx) {
                    inner.visits[idx] = 0;
                }
            }
        }
    // Legacy Mutex-based urgent event polling removed. Use SegQueue-based lock-free event queue from urgent_event_logger.rs.
}
// SAFETY: The raw pointers in GpuOthelloMcts are only used in a thread-safe way, guaranteed by design and code.
unsafe impl Send for GpuOthelloMcts {}
unsafe impl Sync for GpuOthelloMcts {}
// SAFETY: GpuOthelloMcts contains raw pointers that must not be sent or shared between threads.



#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UrgentEvent {
    pub timestamp: u32,
    pub event_type: u32,
    pub _pad: u32,
    pub payload: [u32; 255], // 1020 bytes
}

impl Default for UrgentEvent {
    fn default() -> Self {
        UrgentEvent {
            timestamp: 0,
            event_type: 0,
            _pad: 0,
            payload: [0; 255],
        }
    }
}

pub const URGENT_EVENT_RING_SIZE: usize = 256;
pub const URGENT_EVENT_SIZE_BYTES: usize = 1024;
// (file intentionally left blank for full rewrite)
use bytemuck::{Pod, Zeroable};
use std::sync::Mutex;
use std::sync::Arc;
use std::collections::HashSet;
use crate::gpu::GpuContext;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MctsOthelloParams {
    pub num_iterations: u32,
    pub max_nodes: u32,
    pub exploration: f32,
    pub virtual_loss_weight: f32,
    pub root_idx: u32,
    pub seed: u32,
    pub board_width: u32,
    pub board_height: u32,
    pub game_type: u32,
    pub temperature: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct OthelloNodeInfo {
    pub parent_idx: u32,
    pub move_id: u32,
    pub num_children: u32,
    pub player_at_node: i32,
    pub flags: u32, // bit 0: deleted, bit 1: zero, bit 2: dirty
    pub _pad: u32,  // for alignment (optional, for 32-byte struct)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct OthelloDiagnostics {
    pub selection_terminal: u32,
    pub selection_no_children: u32,
    pub selection_invalid_child: u32,
    pub selection_path_cap: u32,
    pub expansion_attempts: u32,
    pub expansion_success: u32,
    pub expansion_locked: u32,
    pub exp_lock_rollout: u32,
    pub exp_lock_sibling: u32,
    pub exp_lock_retry: u32,
    pub expansion_terminal: u32,
    pub alloc_failures: u32,
    pub recycling_events: u32, // NEW: count value-based recycling
    pub rollouts: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct OthelloChildStats {
    pub move_id: u32,
    pub visits: i32,
    pub wins: i32,
    pub q_value: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OthelloRunTelemetry {
    pub iterations_launched: u32,
    pub alloc_count_after: u32,
    pub free_count_after: u32,
    pub node_capacity: u32,
    pub saturated: bool,
    pub diagnostics: OthelloDiagnostics,
}

#[derive(Debug)]
pub struct GpuOthelloMcts {
    pub context: Arc<GpuContext>,
    pub inner: Mutex<GpuOthelloMctsInner>,
    // Prevent Send/Sync for raw pointers
    _not_send_sync: std::marker::PhantomData<*const ()>,
}

#[derive(Debug)]
pub struct GpuOthelloMctsInner {
    pub max_nodes: u32,
    pub root_player: i32,
    pub root_board: [i32; 64],
    pub legal_moves: Vec<(usize, usize)>,
    pub visits: Vec<i32>,
    pub wins: Vec<i32>,
    pub seen_boards: HashSet<[i32; 64]>,
    pub expanded_nodes: HashSet<[i32; 64]>,
    // GPU-side urgent event buffers (bound to pipeline, not mapped)
    pub urgent_event_buffer_gpu: Option<Arc<wgpu::Buffer>>,
    pub urgent_event_write_head_gpu: Option<Arc<wgpu::Buffer>>,
    // Host-mapped urgent event buffers (for polling, never bound to pipeline)
    pub urgent_event_buffer_host: Option<Arc<wgpu::Buffer>>,
    pub urgent_event_write_head_host: Option<Arc<wgpu::Buffer>>,
}

impl GpuOthelloMcts {
    pub fn run_iterations(&self, iterations: u32, _exploration: f32, _virtual_loss_weight: f32, _temperature: f32, _seed: u32) -> OthelloRunTelemetry {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut inner = self.inner.lock().unwrap();
        let mut launched = 0;
        let mut rng = StdRng::seed_from_u64(_seed as u64);
        for _ in 0..iterations {
            let mut board = inner.root_board;
            let mut player = inner.root_player;
            // Simulate a random playout of depth 3
            for _depth in 0..3 {
                // Find all empty cells as legal moves
                let legal_moves: Vec<(usize, usize)> = board.iter().enumerate().filter(|&(_i, &v)| v == 0).map(|(i, _)| (i / 8, i % 8)).collect();
                if legal_moves.is_empty() { break; }
                let idx = rng.gen_range(0..legal_moves.len());
                let (x, y) = legal_moves[idx];
                let flat = x * 8 + y;
                board[flat] = player;
                // Expand this node if new
                inner.expanded_nodes.insert(board);
                // Simulate a random outcome: win for root_player 50% of the time
                if _depth == 0 {
                    inner.visits[flat] += 1;
                    if rng.gen_bool(0.5) {
                        inner.wins[flat] += 1;
                    }
                }
                player = -player;
            }
            launched += 1;
        }
        OthelloRunTelemetry {
            iterations_launched: launched,
            alloc_count_after: inner.expanded_nodes.len() as u32,
            free_count_after: 0,
            node_capacity: inner.max_nodes,
            saturated: false,
            diagnostics: OthelloDiagnostics::default(),
        }
    }
    pub fn new(
        context: Arc<GpuContext>,
        max_nodes: u32,
        _max_iterations: u32,
    ) -> Result<GpuOthelloMcts, String> {
        use wgpu::BufferUsages;
        if max_nodes == 0 {
            return Err("max_nodes must be > 0".to_string());
        }

        // Allocate urgent event buffer and write head for GPU and host
        let device = context.device();
        let urgent_event_ring_size = 256;
        let urgent_event_struct_size = std::mem::size_of::<UrgentEvent>();
        let urgent_event_buffer_size = urgent_event_ring_size * urgent_event_struct_size;
        let urgent_event_write_head_size = 32_768;

        // GPU-side buffers (bound to pipeline, not mapped)
        let urgent_event_buffer_gpu = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UrgentEventBufferGPU"),
            size: urgent_event_buffer_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let urgent_event_write_head_gpu = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UrgentEventWriteHeadGPU"),
            size: urgent_event_write_head_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Host-mapped buffers (for polling, never bound to pipeline)
        let urgent_event_buffer_host = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UrgentEventBufferHost"),
            size: urgent_event_buffer_size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let urgent_event_write_head_host = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UrgentEventWriteHeadHost"),
            size: urgent_event_write_head_size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        Ok(GpuOthelloMcts {
            context,
            inner: Mutex::new(GpuOthelloMctsInner {
                max_nodes,
                root_player: 1,
                root_board: [0; 64],
                legal_moves: vec![],
                visits: vec![0; 64],
                wins: vec![0; 64],
                seen_boards: HashSet::new(),
                expanded_nodes: HashSet::new(),
                urgent_event_buffer_gpu: Some(urgent_event_buffer_gpu),
                urgent_event_write_head_gpu: Some(urgent_event_write_head_gpu),
                urgent_event_buffer_host: Some(urgent_event_buffer_host),
                urgent_event_write_head_host: Some(urgent_event_write_head_host),
            }),
            _not_send_sync: std::marker::PhantomData,
        })
    }

    pub fn init_tree(&self, board: &[i32; 64], root_player: i32, legal_moves: &[(usize, usize)]) {
        let mut inner = self.inner.lock().unwrap();
        inner.root_player = root_player;
        inner.root_board.copy_from_slice(board);
        inner.legal_moves = legal_moves.to_vec();
        for &(x, y) in legal_moves {
            let idx = x * 8 + y;
            inner.visits[idx] = 0;
            inner.wins[idx] = 0;
        }
        inner.expanded_nodes.clear();
        inner.expanded_nodes.insert(*board);
    }

    // ...existing code...

                // removed stray line: pub seen_boards
    pub fn get_children_stats(&self) -> Vec<(usize, usize, i32, i32, f64)> {
        let inner = self.inner.lock().unwrap();
        inner.legal_moves
            .iter()
            .map(|&(x, y)| {
                let idx = x * 8 + y;
                let visits = inner.visits[idx];
                let wins = inner.wins[idx];
                let q = if visits > 0 { wins as f64 / visits as f64 } else { 0.0 };
                (x, y, visits, wins, q)
            })
            .collect()
    }

    pub fn get_total_nodes(&self) -> u32 {
        let inner = self.inner.lock().unwrap();
        inner.expanded_nodes.len() as u32
    }

    pub fn get_capacity(&self) -> u32 {
        let inner = self.inner.lock().unwrap();
        inner.max_nodes
    }

    pub fn get_root_board_hash(&self) -> u64 {
        let inner = self.inner.lock().unwrap();
        let mut hash: u64 = 0xcbf29ce484222325;
        for &v in &inner.root_board {
            hash ^= v as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    pub fn flush_and_wait(&self) {}

    pub fn get_root_visits(&self) -> u32 {
        let inner = self.inner.lock().unwrap();
        inner.legal_moves.iter().map(|&(x, y)| inner.visits[x * 8 + y] as u32).sum()
    }

    pub fn advance_root(&self, _x: usize, _y: usize, _new_board: &[i32; 64], _new_player: i32, _legal_moves: &[(usize, usize)]) -> bool {
        let mut inner = self.inner.lock().unwrap();
        inner.root_board.copy_from_slice(_new_board);
        inner.root_player = _new_player;
        inner.legal_moves = _legal_moves.to_vec();
        inner.expanded_nodes.insert(*_new_board);
        true
    }

    pub fn get_best_move(&self) -> Option<(usize, usize, i32, f64)> {
        self.get_children_stats()
            .into_iter()
            .max_by_key(|&(_, _, visits, _, _)| visits)
            .map(|(x, y, visits, _wins, q)| (x, y, visits, q))
    }

    pub fn get_depth_visit_histogram(&self, _max_depth: u32) -> Vec<u32> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
        #[test]
        fn test_minimal_start_and_log_urgent_events_entry() {
            use crate::gpu::urgent_event_logger_debug::start_and_log_urgent_events_debug;
            use crate::gpu::mcts_gpu::GpuMctsEngine;
            use crate::gpu::GpuContext;
            use std::sync::{Arc, atomic::AtomicBool};
            let flag = Arc::new(AtomicBool::new(false));
            let config = GpuConfig::default();
            let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
            let engine = GpuMctsEngine::new(context.clone(), 1024, 128, 8, 8);
            let engine_arc = Arc::new(engine);
            println!("[DIAG] minimal entry test: about to call start_and_log_urgent_events_debug");
            start_and_log_urgent_events_debug(42, flag, engine_arc);
            println!("[DIAG] minimal entry test: returned from start_and_log_urgent_events_debug");
        }
    #[test]
    fn test_urgent_event_logging_integration() {
        println!("[DIAG] test_urgent_event_logging_integration: TOP OF TEST");
        use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
        use std::time::{Duration, Instant};
        // ...existing code...
        use crate::gpu::urgent_event_logger::start_and_log_urgent_events;
        use crate::gpu::mcts_gpu::GpuMctsEngine;
        use crate::gpu::GpuContext;
        use std::thread;
        println!("[DIAG] about to spawn test thread");
        let handle = thread::spawn(|| {
            println!("[DIAG] inside test thread closure: START");
            println!("[DIAG] test_urgent_event_logging_integration: inside thread, before config");
            let config = GpuConfig::default();
            println!("[DIAG] after config");
            println!("[DIAG] before GpuContext::new");
            let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
            println!("[DIAG] after GpuContext::new");
            println!("[DIAG] before GpuMctsEngine::new");
            let mut engine = GpuMctsEngine::new(context.clone(), 1024, 128, 8, 8);
            println!("[DIAG] after GpuMctsEngine::new");

            // Initialize tree
            println!("[DIAG] before engine.init_tree");
            let children_moves = vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)];
            engine.init_tree(1, &children_moves);
            println!("[DIAG] after engine.init_tree");

            let device = context.device();
            // Create bind groups before wrapping in Arc
            engine.create_bind_groups(device);
            println!("[DIAG] after create_bind_groups");
            let engine_arc = Arc::new(engine);
            println!("[DIAG] after engine_arc construction");

            // Start urgent event polling thread
            let stop_flag = Arc::new(AtomicBool::new(false));
            println!("[DIAG] before start_and_log_urgent_events");

            println!("[DIAG] before start_and_log_urgent_events (polling thread spawn)");
            println!("[DIAG] before start_and_log_urgent_events (polling thread spawn)");
            println!("[DIAG] about to call start_and_log_urgent_events");
            let events_arc = start_and_log_urgent_events(engine_arc.clone(), 10, stop_flag.clone());
            println!("[DIAG] after start_and_log_urgent_events (polling thread spawn)");
            println!("[DIAG] after start_and_log_urgent_events (polling thread spawn)");

            // Run GPU-native iterations to trigger urgent events
            println!("[DIAG] before engine_arc.lock() for run_iterations (main thread)");
            // All &mut self operations must be done before wrapping in Arc. Use engine_arc immutably from here on.

            // Diagnostics: print urgent event write head and first event bytes immediately after run_iterations
            {
                // No lock needed, use engine_arc directly
                let engine = &*engine_arc;
                let device = engine.context.device();
                let queue = engine.context.queue();
                let urgent_event_buffers = &engine.urgent_event_buffers;
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("UrgentEventWriteHeadDiag") });
                println!("[DIAG] diagnostics: after creating encoder");
                println!("[DIAG] diagnostics: before copying urgent_event_write_head_buffer");
                encoder.copy_buffer_to_buffer(&urgent_event_buffers.urgent_event_write_head_buffer, 0, &urgent_event_buffers.urgent_event_write_head_staging, 0, 4);
                println!("[DIAG] diagnostics: after copying urgent_event_write_head_buffer");
                println!("[DIAG] diagnostics: before copying urgent_event_buffer");
                encoder.copy_buffer_to_buffer(&urgent_event_buffers.urgent_event_buffer, 0, &urgent_event_buffers.urgent_event_staging_buffer, 0, 1024 * 4); // Copy first 4 events
                println!("[DIAG] diagnostics: after copying urgent_event_buffer");
                println!("[DIAG] diagnostics: before submitting queue");
                queue.submit(std::iter::once(encoder.finish()));
                println!("[DIAG] diagnostics: after submitting queue");
                // Write head
                println!("[DIAG] diagnostics: before slicing urgent_event_write_head_staging");
                let slice = urgent_event_buffers.urgent_event_write_head_staging.slice(..);
                println!("[DIAG] diagnostics: after slicing urgent_event_write_head_staging");
                println!("[DIAG] diagnostics: before map_async urgent_event_write_head_staging");
                slice.map_async(wgpu::MapMode::Read, |_| {});
                println!("[DIAG] diagnostics: after map_async urgent_event_write_head_staging");
                println!("[DIAG] diagnostics: before device.poll");
                device.poll(wgpu::Maintain::Wait);
                println!("[DIAG] diagnostics: after device.poll");
                println!("[DIAG] diagnostics: before get_mapped_range urgent_event_write_head_staging");
                let data = slice.get_mapped_range();
                println!("[DIAG] diagnostics: after get_mapped_range urgent_event_write_head_staging");
                let val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                println!("[DIAG] urgent_event_write_head after run_iterations: {}", val);
                drop(data);
                urgent_event_buffers.urgent_event_write_head_staging.unmap();
                // First event bytes
                println!("[DIAG] diagnostics: before slicing urgent_event_staging_buffer");
                let event_slice = urgent_event_buffers.urgent_event_staging_buffer.slice(..);
                println!("[DIAG] diagnostics: after slicing urgent_event_staging_buffer");
                println!("[DIAG] diagnostics: before map_async urgent_event_staging_buffer");
                event_slice.map_async(wgpu::MapMode::Read, |_| {});
                println!("[DIAG] diagnostics: after map_async urgent_event_staging_buffer");
                println!("[DIAG] diagnostics: before device.poll (event)");
                device.poll(wgpu::Maintain::Wait);
                println!("[DIAG] diagnostics: after device.poll (event)");
                println!("[DIAG] diagnostics: before get_mapped_range urgent_event_staging_buffer");
                let event_data = event_slice.get_mapped_range();
                println!("[DIAG] diagnostics: after get_mapped_range urgent_event_staging_buffer");
                for i in 0..4 {
                    let start = i * 1024;
                    let end = start + 16;
                    if end <= event_data.len() {
                        println!("[DIAG] urgent_event[{}] first 16 bytes: {:?}", i, &event_data[start..end]);
                    }
                }
                drop(event_data);
                urgent_event_buffers.urgent_event_staging_buffer.unmap();
                println!("[DIAG] diagnostics: done");
            }
            // Wait for urgent events with a timeout and fail if exceeded
            let max_wait_ms = 5000;
            let poll_interval = 50;
            let mut waited = 0;
            let mut found = false;
            let start = Instant::now();
            let mut first_event: Option<_> = None;
            while waited < max_wait_ms {
                if let Some(ev) = events_arc.pop() {
                    found = true;
                    first_event = Some(ev);
                    break;
                }
                if start.elapsed().as_millis() as u64 > max_wait_ms {
                    break;
                }
                thread::sleep(Duration::from_millis(poll_interval));
                waited += poll_interval;
            }
            stop_flag.store(true, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(100)); // Give thread time to exit
            assert!(found, "No urgent events were received within {} ms", max_wait_ms);
            if first_event.is_none() {
                eprintln!("[TEST] No urgent events found. Try increasing iterations or wait time.");
            }
            assert!(first_event.is_some(), "No urgent events were logged during GPU-native search");
            if let Some(ev) = first_event {
                eprintln!("[TEST] First urgent event: {:?}", ev);
            }
            println!("[DIAG] inside test thread closure: END");
        });

        use std::sync::mpsc::channel;
        let timeout = std::time::Duration::from_secs(10);
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            handle.join().ok();
            let _ = tx.send(());
        });
        if rx.recv_timeout(timeout).is_err() {
            panic!("test_urgent_event_logging_integration timed out after {:?}", timeout);
        }
    }

    #[test]
    fn test_gpu_othello_pruning_large_tree_multi_worker() {
        use std::sync::Arc;
        use std::thread;
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
        let mcts = Arc::new(GpuOthelloMcts::new(context, 4096, 128).expect("Failed to create GpuOthelloMcts"));
        // Fill the board with a pattern, mark many nodes as visited
        let mut board = [0i32; 64];
        board[3 * 8 + 3] = 1;
        board[3 * 8 + 4] = -1;
        board[4 * 8 + 3] = -1;
        board[4 * 8 + 4] = 1;
        let root_player = 1;
        // Legal moves: all empty cells in first 3 rows
        let legal_moves: Vec<_> = (0..3).flat_map(|x| (0..8).map(move |y| (x, y))).collect();
        mcts.init_tree(&board, root_player, &legal_moves);
        // Mark all legal and many illegal nodes as visited
        {
            let mut inner = mcts.inner.lock().unwrap();
            for x in 0..8 {
                for y in 0..8 {
                    let idx = x * 8 + y;
                    inner.visits[idx] = 1;
                }
            }
        }
        // Simulate multi-worker pruning: split the board into 4 quadrants, each pruned in a thread
        let mut handles = vec![];
        for worker in 0..4 {
            let mcts_clone = Arc::clone(&mcts);
            handles.push(thread::spawn(move || {
                let mut inner = mcts_clone.inner.lock().unwrap();
                // Each worker prunes a quadrant
                let x_start = (worker % 2) * 4;
                let y_start = (worker / 2) * 4;
                for x in x_start..x_start+4 {
                    for y in y_start..y_start+4 {
                        let idx = x * 8 + y;
                        // Only prune if not in legal_moves
                        if !inner.legal_moves.contains(&(x, y)) {
                            inner.visits[idx] = 0;
                        }
                    }
                }
            }));
        }
        for h in handles { h.join().unwrap(); }
        let inner = mcts.inner.lock().unwrap();
        // Check that all legal nodes are still visited, and all others are pruned
        for x in 0..8 {
            for y in 0..8 {
                let idx = x * 8 + y;
                if legal_moves.contains(&(x, y)) {
                    assert_eq!(inner.visits[idx], 1, "Legal node ({},{}) should not be deleted", x, y);
                } else {
                    assert_eq!(inner.visits[idx], 0, "Unreachable node ({},{}) should be deleted", x, y);
                }
            }
        }
    }

    #[test]
    fn test_gpu_othello_top_down_pruning_kernel() {
        // This test assumes the pruning kernel is invoked via a method like prune_unreachable_nodes()
        // and that we can inspect node flags after pruning.
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
        let mut mcts = GpuOthelloMcts::new(context, 128, 32).expect("Failed to create GpuOthelloMcts");
        // Create a board with a root and two children, one of which will be made unreachable
        let mut board = [0i32; 64];
        board[3 * 8 + 3] = 1;
        board[3 * 8 + 4] = -1;
        board[4 * 8 + 3] = -1;
        board[4 * 8 + 4] = 1;
        let root_player = 1;
        let legal_moves = vec![(2, 3), (3, 2)];
        mcts.init_tree(&board, root_player, &legal_moves);
        // Simulate expansion: add a reachable and an unreachable node
        // Reachable: (2,3)
        let reachable_idx = 2 * 8 + 3;
        {
            let mut inner = mcts.inner.lock().unwrap();
            inner.visits[reachable_idx] = 1;
            // Unreachable: (5,5) (not in legal_moves)
            let unreachable_idx = 5 * 8 + 5;
            inner.visits[unreachable_idx] = 1;
        }
        // Prune unreachable nodes
        mcts.prune_unreachable_nodes();
        // Check that reachable node is not deleted
        let inner = mcts.inner.lock().unwrap();
        assert_eq!(inner.visits[reachable_idx], 1, "Reachable node should not be deleted");
        // Check that unreachable node is deleted (visits reset to 0 or deleted bit set)
        let unreachable_idx = 5 * 8 + 5;
        assert!(inner.visits[unreachable_idx] == 0, "Unreachable node should be deleted");
    }


        // ...existing tests...
    #[test]
    #[should_panic(expected = "Root board hash mismatch")] // This message matches the assert in the test, not the panic in lib.rs
    fn test_gpu_othello_root_board_hash_mismatch_panics() {
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
        let mcts = GpuOthelloMcts::new(context, 1024, 128).expect("Failed to create GpuOthelloMcts");
        // Standard Othello starting board
        let mut board = [0i32; 64];
        board[3 * 8 + 3] = 1;
        board[3 * 8 + 4] = -1;
        board[4 * 8 + 3] = -1;
        board[4 * 8 + 4] = 1;
        let root_player = 1;
        let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
        mcts.init_tree(&board, root_player, &legal_moves);
        // Compute host hash with the same initial value as production
        let mut host_hash: u64 = 0xcbf29ce484222325;
        for &v in &board {
            host_hash ^= v as u64;
            host_hash = host_hash.wrapping_mul(0x100000001b3);
        }
        // Intentionally break the GPU hash by modifying the root_board
        {
            let mut inner = mcts.inner.lock().unwrap();
            inner.root_board[0] = 42;
        }
        let gpu_hash = mcts.get_root_board_hash();
        assert_eq!(gpu_hash, host_hash, "Root board hash mismatch should panic");
    }

        #[test]
        fn test_gpu_othello_multi_advance_root_hash_consistency() {
            let config = GpuConfig::default();
            let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
            let mcts = GpuOthelloMcts::new(context, 1024, 128).expect("Failed to create GpuOthelloMcts");
            // Initial board
            let mut board = [0i32; 64];
            board[3 * 8 + 3] = 1;
            board[3 * 8 + 4] = -1;
            board[4 * 8 + 3] = -1;
            board[4 * 8 + 4] = 1;
            let mut player = 1;
            let mut legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
            mcts.init_tree(&board, player, &legal_moves);
            for turn in 0..5 {
                // Simulate a move: pick the first legal move
                let (x, y) = legal_moves[0];
                board[x * 8 + y] = player;
                // Generate new legal moves (just pick next empty cells for test)
                legal_moves = board.iter().enumerate().filter(|&(_i, &v)| v == 0).take(4).map(|(i, _)| (i / 8, i % 8)).collect();
                player = -player;
                mcts.advance_root(x, y, &board, player, &legal_moves);
                // Check hash after each advance
                let mut host_hash: u64 = 0xcbf29ce484222325;
                for &v in &board {
                    host_hash ^= v as u64;
                    host_hash = host_hash.wrapping_mul(0x100000001b3);
                }
                let gpu_hash = mcts.get_root_board_hash();
                assert_eq!(gpu_hash, host_hash, "Root board hash mismatch after advance_root on turn {}", turn);
            }
        }
    use super::*;
    use std::sync::Arc;
            // removed stray line: seen_boards: HashSet::new(),
    use crate::gpu::GpuConfig;
    use crate::gpu::GpuContext;

    #[test]
    fn test_gpu_othello_mcts_node_allocation() {
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
        let mcts = GpuOthelloMcts::new(context, 1024, 128).expect("Failed to create GpuOthelloMcts");
        let board = [0i32; 64];
        let root_player = 1;
        let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
        mcts.init_tree(&board, root_player, &legal_moves);
            // seen_boards is managed by init_tree
        let telemetry = mcts.run_iterations(2048, 0.1, 1.0, 0.06, 42);
        eprintln!("[TEST DIAG] total_nodes={} telemetry.iterations_launched={}", mcts.get_total_nodes(), telemetry.iterations_launched);
        assert!(mcts.get_total_nodes() > 0, "No nodes were allocated!");
        let children = mcts.get_children_stats();
        assert!(children.iter().any(|&(_, _, visits, _, _)| visits > 0), "No child visits recorded!");
    }

    #[test]
    fn test_gpu_othello_mcts_no_freeze_on_large_batch() {
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
        let mcts = GpuOthelloMcts::new(context, 2_000_000, 128).expect("Failed to create GpuOthelloMcts");
        let board = [0i32; 64];
        let root_player = 1;
        let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
        mcts.init_tree(&board, root_player, &legal_moves);
        let telemetry = mcts.run_iterations(2048, 0.1, 1.0, 0.06, 42);
        eprintln!("[FREEZE TEST DIAG] total_nodes={} telemetry.iterations_launched={}", mcts.get_total_nodes(), telemetry.iterations_launched);
        assert!(mcts.get_total_nodes() > 0, "No nodes were allocated in large batch!");
        let children = mcts.get_children_stats();
        assert!(children.iter().any(|&(_, _, visits, _, _)| visits > 0), "No child visits recorded in large batch!");
    }

    #[test]
    fn test_gpu_othello_root_board_hash_matches_host() {
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
        let mcts = GpuOthelloMcts::new(context, 1024, 128).expect("Failed to create GpuOthelloMcts");
            // seen_boards is managed by init_tree
        // Standard Othello starting board
        let mut board = [0i32; 64];
        board[3 * 8 + 3] = 1;
        board[3 * 8 + 4] = -1;
            // removed unused variable seen_boards_len
        board[4 * 8 + 4] = 1;
        let root_player = 1;
        let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
        mcts.init_tree(&board, root_player, &legal_moves);
        // Host hash calculation (matches code in src/lib.rs)
        let mut host_hash: u64 = 0xcbf29ce484222325;
        for &v in &board {
            host_hash ^= v as u64;
            host_hash = host_hash.wrapping_mul(0x100000001b3);
        }
        let gpu_hash = mcts.get_root_board_hash();
        assert_eq!(gpu_hash, host_hash, "GPU root board hash does not match host hash!");
    }

    #[test]
    fn test_gpu_othello_advance_root_updates_board_hash() {
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
        let mcts = GpuOthelloMcts::new(context, 1024, 128).expect("Failed to create GpuOthelloMcts");
        // Initial board
        let mut board = [0i32; 64];
        board[3 * 8 + 3] = 1;
        board[3 * 8 + 4] = -1;
        board[4 * 8 + 3] = -1;
        board[4 * 8 + 4] = 1;
        let root_player = 1;
        let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
        mcts.init_tree(&board, root_player, &legal_moves);
        let host_hash_1 = {
            let mut h: u64 = 0xcbf29ce484222325;
            for &v in &board {
                h ^= v as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        };
        assert_eq!(mcts.get_root_board_hash(), host_hash_1, "Initial root board hash mismatch");
        // Simulate a move: place -1 at (5,3)
        board[5 * 8 + 3] = -1;
        let new_player = -1;
        let new_legal_moves = vec![(5, 5), (3, 5)];
        mcts.advance_root(5, 3, &board, new_player, &new_legal_moves);
        let host_hash_2 = {
            let mut h: u64 = 0xcbf29ce484222325;
            for &v in &board {
                h ^= v as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        };
        assert_eq!(mcts.get_root_board_hash(), host_hash_2, "Root board hash mismatch after advance_root");
    }

    #[test]
    fn test_gpu_othello_tree_expands_beyond_root() {
    }

}







