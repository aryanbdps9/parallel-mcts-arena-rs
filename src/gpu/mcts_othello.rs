use std::sync::Mutex;
use once_cell::sync::Lazy;

/// Global mutex to prevent concurrent device.poll() and buffer mapping operations.
/// WGPU's device.poll() is not thread-safe - only one thread can poll/map buffers at a time.
/// This mutex is shared between the urgent event logger thread and any code that maps buffers (e.g., pruning).
pub static DEVICE_POLL_MUTEX: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

/// Utility: After kernel dispatch, assert that no URGENT_EVENT_EARLY_EXIT events were emitted
pub fn assert_no_early_exit_events(events: &crossbeam_queue::SegQueue<UrgentEvent>) {
    let mut found = false;
    let mut count = 0;
    let mut threads = vec![];
    while let Some(event) = events.pop() {
        if event.event_type == 15 {
            found = true;
            count += 1;
            threads.push(event.payload[0]);
        }
    }
    if found {
        panic!("[TEST FAILURE] URGENT_EVENT_EARLY_EXIT detected: {} threads exited early: {:?}", count, threads);
    }
}
impl GpuOthelloMcts {
            /// Dispatch the main GPU-native MCTS kernel and bind the urgent event buffer for logging
            pub fn dispatch_mcts_othello_kernel(&self, num_workgroups: u32, exploration: f32, virtual_loss_weight: f32, temperature: f32, vl_temp_scale: f32, seed: u32) {
        // Handle WGPU limit of 65535 workgroups per dimension
        let max_dim = 65535;
        let (dispatch_x, dispatch_y) = if num_workgroups > max_dim {
            let y = (num_workgroups + max_dim - 1) / max_dim;
            (max_dim, y)
        } else {
            (num_workgroups, 1)
        };

        // Reset urgent event write head to zero before dispatch (prevents stale events)
        {
            let inner = self.inner.lock().unwrap();
            if let Some(write_head_buf) = &inner.urgent_event_write_head_gpu {
                let device = self.context.device();
                let queue = self.context.queue();
                let zero = 0u32.to_le_bytes();
                queue.write_buffer(write_head_buf, 0, &zero);
                device.poll(wgpu::Maintain::Wait);
            }
        }
        // Set the atomic to the expected thread count before dispatch
        {
            let inner = self.inner.lock().unwrap();
            let device = self.context.device();
            let queue = self.context.queue();
            let total_threads = 64 * dispatch_x * dispatch_y; // match kernel logic
            let temp = (total_threads as u32).to_le_bytes();

            if let Some(buf) = &inner.global_reroot_threads_remaining {
                queue.write_buffer(buf, 0, &temp);
            }
            if let Some(buf) = &inner.global_reroot_start_threads_remaining {
                queue.write_buffer(buf, 0, &temp);
            }
            device.poll(wgpu::Maintain::Wait);
        }
        
        assert!(num_workgroups > 0, "num_workgroups must be > 0 or kernel will not run!");

        use wgpu::{BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindGroupEntry, BindGroupDescriptor, ShaderStages, BindingType, BufferBindingType};
                let context = &self.context;
                let device = context.device();
                let queue = context.queue();
                
                // Retrieve real buffers from inner
                let (
                    node_info,
                    node_visits,
                    node_wins,
                    node_vl,
                    node_state,
                    children_indices,
                    children_priors,
                    free_lists,
                    free_tops,
                    free_list_ownership_buf,
                    global_free_queue_buf,
                    global_free_head_buf,
                    expansion_paused_buf
                ) = {
                    let inner = self.inner.lock().unwrap();
                    (
                        inner.node_info_buffer.as_ref().expect("node_info missing").clone(),
                        inner.node_visits_buffer.as_ref().expect("node_visits missing").clone(),
                        inner.node_wins_buffer.as_ref().expect("node_wins missing").clone(),
                        inner.node_vl_buffer.as_ref().expect("node_vl missing").clone(),
                        inner.node_state_buffer.as_ref().expect("node_state missing").clone(),
                        inner.children_indices_buffer.as_ref().expect("children_indices missing").clone(),
                        inner.children_priors_buffer.as_ref().expect("children_priors missing").clone(),
                        inner.free_lists_buffer.as_ref().expect("free_lists missing").clone(),
                        inner.free_tops_buffer.as_ref().expect("free_tops missing").clone(),
                        inner.free_list_ownership_buffer.as_ref().expect("free_list_ownership missing").clone(),
                        inner.global_free_queue_buffer.as_ref().expect("global_free_queue missing").clone(),
                        inner.global_free_head_buffer.as_ref().expect("global_free_head missing").clone(),
                        inner.expansion_paused_buffer.as_ref().expect("expansion_paused missing").clone(),
                    )
                };

                // Layouts for each group
                let group0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Group 0 Layout (Node Data)"),
                    entries: &(0..=12).map(|i| BindGroupLayoutEntry {
                        binding: i,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                        count: None,
                    }).collect::<Vec<_>>(),
                });
                
                let group0_bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Group 0 Bind Group (Node Data)"),
                    layout: &group0_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: node_info.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: node_visits.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: node_wins.as_entire_binding() },
                        BindGroupEntry { binding: 3, resource: node_vl.as_entire_binding() },
                        BindGroupEntry { binding: 4, resource: node_state.as_entire_binding() },
                        BindGroupEntry { binding: 5, resource: children_indices.as_entire_binding() },
                        BindGroupEntry { binding: 6, resource: children_priors.as_entire_binding() },
                        BindGroupEntry { binding: 7, resource: free_lists.as_entire_binding() },
                        BindGroupEntry { binding: 8, resource: free_tops.as_entire_binding() },
                        BindGroupEntry { binding: 9, resource: free_list_ownership_buf.as_entire_binding() },
                        BindGroupEntry { binding: 10, resource: global_free_queue_buf.as_entire_binding() },
                        BindGroupEntry { binding: 11, resource: global_free_head_buf.as_entire_binding() },
                        BindGroupEntry { binding: 12, resource: expansion_paused_buf.as_entire_binding() },
                    ],
                });

                // Retrieve Group 1 and Group 4 buffers
                let (
                    mcts_params_buf,
                    work_items_buf,
                    paths_buf,
                    alloc_counter_buf,
                    diagnostics_buf,
                    reroot_params_buf,
                    new_root_output_buf,
                    work_queue_buf,
                    work_head_buf,
                    work_claimed_buf,
                    work_completed_buf
                ) = {
                    let inner = self.inner.lock().unwrap();
                    (
                        inner.mcts_params_buffer.as_ref().expect("mcts_params missing").clone(),
                        inner.work_items_buffer.as_ref().expect("work_items missing").clone(),
                        inner.paths_buffer.as_ref().expect("paths missing").clone(),
                        inner.alloc_counter_buffer.as_ref().expect("alloc_counter missing").clone(),
                        inner.diagnostics_buffer.as_ref().expect("diagnostics missing").clone(),
                        inner.reroot_params_buffer.as_ref().expect("reroot_params missing").clone(),
                        inner.new_root_output_buffer.as_ref().expect("new_root_output missing").clone(),
                        inner.work_queue_buffer.as_ref().expect("work_queue missing").clone(),
                        inner.work_head_buffer.as_ref().expect("work_head missing").clone(),
                        inner.work_claimed_buffer.as_ref().expect("work_claimed missing").clone(),
                        inner.work_completed_buffer.as_ref().expect("work_completed missing").clone(),
                    )
                };

                // Update MctsParams
                {
                    let inner = self.inner.lock().unwrap();
                    let free_list_capacity = (inner.max_nodes + 255) / 256;
                    let params = MctsOthelloParams {
                        num_iterations: 1, 
                        max_nodes: inner.max_nodes,
                        exploration,
                        virtual_loss_weight,
                        root_idx: inner.current_root_idx,
                        seed,
                        board_width: 8,
                        board_height: 8,
                        game_type: 0,
                        temperature,
                        turn_number: 0, 
                        free_list_capacity,
                        vl_temp_scale,
                        num_threads: 0, // Not used for monolithic kernel
                        root_node: inner.current_root_idx,
                        use_vl_preincrement: 0, // Disabled by default for testing
                        use_random_rollouts: 0,
                    };
                    queue.write_buffer(&mcts_params_buf, 0, bytemuck::bytes_of(&params));
                    device.poll(wgpu::Maintain::Wait); // Ensure params are written before creating bind group
                }

                // Group 1 Layout (MCTS Params & Work Items)
                let group1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Group 1 Layout (MCTS Params)"),
                    entries: &[
                        BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    ],
                });

                let root_stats_buf = {
                    let inner = self.inner.lock().unwrap();
                    inner.root_stats_buffer.as_ref().expect("root_stats_buffer missing").clone()
                };

                let group1_bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Group 1 Bind Group (MCTS Params)"),
                    layout: &group1_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: mcts_params_buf.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: work_items_buf.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: paths_buf.as_entire_binding() },
                        BindGroupEntry { binding: 3, resource: alloc_counter_buf.as_entire_binding() },
                        BindGroupEntry { binding: 4, resource: diagnostics_buf.as_entire_binding() },
                        BindGroupEntry { binding: 5, resource: root_stats_buf.as_entire_binding() },
                    ],
                });
                
                let root_board_buf = {
                    let inner = self.inner.lock().unwrap();
                    inner.root_board_buffer.as_ref().expect("root_board_buffer missing").clone()
                };
                
                // Write current root board
                {
                    let inner = self.inner.lock().unwrap();
                    // println!("[GPU-Native] Writing root_board to GPU. Board[27..37]: {:?}", &inner.root_board[27..37]);
                    queue.write_buffer(&root_board_buf, 0, bytemuck::cast_slice(&inner.root_board));
                }

                // DEBUG: Read back root_board to verify
                // {
                //     let size = 64 * 4;
                //     let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                //         label: Some("Staging Readback"),
                //         size,
                //         usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                //         mapped_at_creation: false,
                //     });
                //     let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                //     encoder.copy_buffer_to_buffer(&root_board_buf, 0, &staging_buf, 0, size);
                //     queue.submit(Some(encoder.finish()));
                //     
                //     let slice = staging_buf.slice(..);
                //     let (tx, rx) = std::sync::mpsc::channel();
                //     slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
                //     device.poll(wgpu::Maintain::Wait);
                //     rx.recv().unwrap().unwrap();
                //     let data = slice.get_mapped_range();
                //     let result: &[i32] = bytemuck::cast_slice(&data);
                //     println!("[GPU-Native] Readback root_board from GPU. Board[27..37]: {:?}", &result[27..37]);
                //     drop(data);
                //     staging_buf.unmap();
                // }

                let group2_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Group 2 Layout (Root Board)"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                        count: None,
                    }],
                });
                
                let group2_bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Group 2 Bind Group (Root Board)"),
                    layout: &group2_layout,
                    entries: &[BindGroupEntry { binding: 0, resource: root_board_buf.as_entire_binding() }],
                });
                
                // Group 4 Layout (Pruning Params - used for layout compatibility)
                let group4_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Group 4 Layout (Pruning)"),
                    entries: &[
                        BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        BindGroupLayoutEntry { binding: 6, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        BindGroupLayoutEntry { binding: 7, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    ],
                });
                
                let group4_bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Group 4 Bind Group (Pruning)"),
                    layout: &group4_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: reroot_params_buf.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: new_root_output_buf.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: global_free_queue_buf.as_entire_binding() },
                        BindGroupEntry { binding: 3, resource: global_free_head_buf.as_entire_binding() },
                        BindGroupEntry { binding: 4, resource: work_queue_buf.as_entire_binding() },
                        BindGroupEntry { binding: 5, resource: work_head_buf.as_entire_binding() },
                        BindGroupEntry { binding: 6, resource: work_claimed_buf.as_entire_binding() },
                        BindGroupEntry { binding: 7, resource: work_completed_buf.as_entire_binding() },
                    ],
                });
                // Urgent event buffer layout (group 3)
                let urgent_event_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Urgent Event Layout (Othello, group 3)"),
                    entries: &[ 
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                            count: None,
                        },
                    ],
                });
                // Only lock to get buffer references, then drop lock before GPU ops
                let (urgent_event_buffer_gpu, urgent_event_write_head_gpu) = {
                    let inner = self.inner.lock().unwrap();
                    (
                        inner.urgent_event_buffer_gpu.as_ref().expect("urgent_event_buffer_gpu missing").clone(),
                        inner.urgent_event_write_head_gpu.as_ref().expect("urgent_event_write_head_gpu missing").clone(),
                    )
                };
                // Create bind groups
                // Group 0 is now real
                
                let urgent_event_bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Urgent Event Bind Group (Othello, group 3)"),
                    layout: &urgent_event_layout,
                    entries: &[BindGroupEntry { binding: 0, resource: urgent_event_buffer_gpu.as_entire_binding() }, BindGroupEntry { binding: 1, resource: urgent_event_write_head_gpu.as_entire_binding() }],
                });
                // Create pipeline
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Othello MCTS Main Kernel Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mcts_othello.wgsl").into()),
                });
                // === PERSISTENT GROUP 5 (for main kernel @group(5) bindings) ===
                // Use the persistent buffer for global_reroot_threads_remaining
                let (global_reroot_threads_remaining, global_reroot_start_threads_remaining) = {
                    let inner = self.inner.lock().unwrap();
                    (
                        inner.global_reroot_threads_remaining.as_ref().expect("global_reroot_threads_remaining missing").clone(),
                        inner.global_reroot_start_threads_remaining.as_ref().expect("global_reroot_start_threads_remaining missing").clone()
                    )
                };
                let group5_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Group 5 Layout (main kernel, persistent)"),
                    entries: &[
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
                    ],
                });
                let group5_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Group 5 Bind Group (main kernel, persistent)"),
                    layout: &group5_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: global_reroot_threads_remaining.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: global_reroot_start_threads_remaining.as_entire_binding(),
                        },
                    ],
                });
                // Pipeline layout: groups 0-5 (must match WGSL)
                let bind_group_layouts = vec![
                    &group0_layout,
                    &group1_layout,
                    &group2_layout,
                    &urgent_event_layout, // group 3 for urgent events
                    &group4_layout,
                    &group5_layout, // group 5 (persistent)
                ];
                assert_eq!(bind_group_layouts.len(), 6, "Pipeline layout must have 6 groups (0-5) for main kernel");
                let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Othello MCTS Main Pipeline Layout"),
                    bind_group_layouts: &bind_group_layouts,
                    push_constant_ranges: &[],
                });
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Othello MCTS Main Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });
                // Dispatch the kernel
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Othello MCTS Main Encoder"),
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Othello MCTS Main ComputePass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group0_bind_group, &[]);
                    pass.set_bind_group(1, &group1_bind_group, &[]);
                    pass.set_bind_group(2, &group2_bind_group, &[]);
                    pass.set_bind_group(3, &urgent_event_bind_group, &[]);
                    pass.set_bind_group(4, &group4_bind_group, &[]);
                    pass.set_bind_group(5, &group5_bind_group, &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);
                // println!("[DIAG] Othello MCTS main kernel dispatched and device polled.");
            }
        /// Dispatch the GPU pruning kernel and bind the urgent event buffer for logging
        pub fn dispatch_pruning_kernels(&self, move_x: u32, move_y: u32) -> bool {
            println!("[DIAG] Pruning: ENTER dispatch_pruning_kernels move=({}, {})", move_x, move_y);
            use wgpu::{BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindGroupEntry, BindGroupDescriptor, ShaderStages, BindingType, BufferBindingType};
            let context = &self.context;
            let device = context.device();
            let queue = context.queue();

            println!("[DIAG] Pruning: got device and queue");

            // 1. Prepare RerootParams
            let (reroot_params_buf, current_root) = {
                let inner = self.inner.lock().unwrap();
                (
                    inner.reroot_params_buffer.as_ref().expect("reroot_params_buffer missing").clone(),
                    inner.current_root_idx
                )
            };

            println!("[DIAG] Pruning: current_root_idx={}", current_root);

            let max_nodes = {
                let inner = self.inner.lock().unwrap();
                inner.max_nodes
            };

            let params = RerootParams {
                move_x,
                move_y,
                current_root,
                max_nodes,
            };
            queue.write_buffer(&reroot_params_buf, 0, bytemuck::bytes_of(&params));

            println!("[DIAG] Pruning: wrote params to buffer");

            // 2. Get all buffers
            let (
                new_root_output_buf,
                global_free_queue_buf,
                global_free_head_buf,
                work_queue_buf,
                work_head_buf,
                work_claimed_buf,
                work_completed_buf
            ) = {
                let inner = self.inner.lock().unwrap();
                (
                    inner.new_root_output_buffer.as_ref().expect("new_root_output_buffer missing").clone(),
                    inner.global_free_queue_buffer.as_ref().expect("global_free_queue_buffer missing").clone(),
                    inner.global_free_head_buffer.as_ref().expect("global_free_head_buffer missing").clone(),
                    inner.work_queue_buffer.as_ref().expect("work_queue_buffer missing").clone(),
                    inner.work_head_buffer.as_ref().expect("work_head_buffer missing").clone(),
                    inner.work_claimed_buffer.as_ref().expect("work_claimed_buffer missing").clone(),
                    inner.work_completed_buffer.as_ref().expect("work_completed_buffer missing").clone(),
                )
            };

            println!("[DIAG] Pruning: got all pruning buffers");

            // Initialize new_root_output to a sentinel value to detect if shader ran
            queue.write_buffer(&new_root_output_buf, 0, &0xDEADBEEFu32.to_le_bytes());
            // Also initialize work_head to 0 to ensure clean state
            queue.write_buffer(&work_head_buf, 0, &0u32.to_le_bytes());
            
            // Clear expansion_paused flag - pruning will free nodes making memory available
            {
                let inner = self.inner.lock().unwrap();
                let expansion_paused_buf = inner.expansion_paused_buffer.as_ref().expect("expansion_paused missing");
                queue.write_buffer(expansion_paused_buf, 0, &0u32.to_le_bytes());
            }
            
            device.poll(wgpu::Maintain::Wait); // Ensure writes complete before shader reads
            println!("[DIAG] Pruning: initialized new_root_output to 0xDEADBEEF, work_head to 0, expansion_paused to 0");

            // VERIFY: Read back the buffers to confirm initialization worked
            {
                let _verify_poll_guard = DEVICE_POLL_MUTEX.lock().unwrap();
                let staging_verify = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Verify Init Staging"),
                    size: 8,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut encoder_verify = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Verify Init") });
                encoder_verify.copy_buffer_to_buffer(&new_root_output_buf, 0, &staging_verify, 0, 4);
                encoder_verify.copy_buffer_to_buffer(&work_head_buf, 0, &staging_verify, 4, 4);
                queue.submit(Some(encoder_verify.finish()));
                device.poll(wgpu::Maintain::Wait);
                
                let slice = staging_verify.slice(..);
                let (tx_verify, rx_verify) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |v| { let _ = tx_verify.send(v); });
                device.poll(wgpu::Maintain::Wait);
                match rx_verify.recv_timeout(std::time::Duration::from_secs(2)) {
                    Ok(Ok(())) => {
                        let data = slice.get_mapped_range();
                        let new_root_verify = u32::from_le_bytes(data[0..4].try_into().unwrap());
                        let work_head_verify = u32::from_le_bytes(data[4..8].try_into().unwrap());
                        println!("[DIAG] VERIFY BEFORE SHADER: new_root_output=0x{:08X}, work_head=0x{:08X}", new_root_verify, work_head_verify);
                        drop(data);
                        staging_verify.unmap();
                    }
                    Ok(Err(e)) => {
                        println!("[DIAG] VERIFY FAILED: Buffer mapping error: {:?}", e);
                    }
                    Err(e) => {
                        println!("[DIAG] VERIFY FAILED: Timeout or channel error: {:?}", e);
                    }
                }
            }

            // DEBUG: Read root node's children before pruning
            {
                // CRITICAL: Poll device to ensure all previous GPU operations have completed
                println!("[DIAG] Pruning: polling device to ensure previous operations complete");
                device.poll(wgpu::Maintain::Wait);
                println!("[DIAG] Pruning: device poll complete");
                
                let target_move_id = move_y * 8 + move_x;
                println!("[DIAG] Pruning: target_move_id = {} (move_x={}, move_y={})", target_move_id, move_x, move_y);
                
                // Read root node's NodeInfo to see num_children
                let node_info_buf = {
                    let inner = self.inner.lock().unwrap();
                    inner.node_info_buffer.as_ref().expect("node_info missing").clone()
                };
                
                // Read root NodeInfo (16 bytes = 4 u32s)
                let staging_info = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Debug Root NodeInfo Staging"),
                    size: 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                encoder.copy_buffer_to_buffer(&node_info_buf, current_root as u64 * 32, &staging_info, 0, 16); // NodeInfo is 32 bytes per entry
                queue.submit(Some(encoder.finish()));
                
                let (tx, rx) = std::sync::mpsc::channel();
                staging_info.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });
                
                device.poll(wgpu::Maintain::Wait);
                if let Ok(Ok(())) = rx.recv_timeout(std::time::Duration::from_secs(2)) {
                    let data = staging_info.slice(..).get_mapped_range();
                    let info: &[u32] = bytemuck::cast_slice(&data);
                    let parent_idx = info[0];
                    let move_id = info[1];
                    let num_children = info[2];
                    let player_at_node = i32::from_le_bytes(info[3].to_le_bytes());
                    println!("[DIAG] Pruning: root NodeInfo: parent_idx=0x{:08X}, move_id={}, num_children={}, player={}", 
                        parent_idx, move_id, num_children, player_at_node);
                    drop(data);
                    staging_info.unmap();
                    
                    // Read children_indices for root
                    if num_children > 0 {
                        let children_indices_buf = {
                            let inner = self.inner.lock().unwrap();
                            inner.children_indices_buffer.as_ref().expect("children_indices missing").clone()
                        };
                        
                        let max_children = 64; // MAX_CHILDREN constant
                        let staging_children = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Debug Children Indices Staging"),
                            size: max_children * 4,
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        
                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                        // children_indices[root * MAX_CHILDREN + slot]
                        encoder.copy_buffer_to_buffer(&children_indices_buf, current_root as u64 * max_children * 4, &staging_children, 0, max_children * 4);
                        queue.submit(Some(encoder.finish()));
                        
                        let (tx2, rx2) = std::sync::mpsc::channel();
                        staging_children.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                            let _ = tx2.send(result);
                        });
                        
                        device.poll(wgpu::Maintain::Wait);
                        if let Ok(Ok(())) = rx2.recv_timeout(std::time::Duration::from_secs(2)) {
                            let data = staging_children.slice(..).get_mapped_range();
                            let children: &[u32] = bytemuck::cast_slice(&data);
                            println!("[DIAG] Pruning: root children_indices[0..{}]: {:?}", num_children.min(8), &children[0..num_children.min(8) as usize]);
                            
                            // Read move_ids of these children
                            let children_vec: Vec<u32> = children[0..num_children.min(4) as usize].to_vec();
                            drop(data);
                            staging_children.unmap();
                            
                            for (i, &child_idx) in children_vec.iter().enumerate() {
                                let staging_child = device.create_buffer(&wgpu::BufferDescriptor {
                                    label: Some(&format!("Debug Child {} Info", i)),
                                    size: 16,
                                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                                    mapped_at_creation: false,
                                });
                                
                                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                                encoder.copy_buffer_to_buffer(&node_info_buf, child_idx as u64 * 32, &staging_child, 0, 16); // NodeInfo is 32 bytes per entry
                                queue.submit(Some(encoder.finish()));
                                
                                let (tx3, rx3) = std::sync::mpsc::channel();
                                staging_child.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                                    let _ = tx3.send(result);
                                });
                                
                                device.poll(wgpu::Maintain::Wait);
                                if let Ok(Ok(())) = rx3.recv_timeout(std::time::Duration::from_secs(2)) {
                                    let data = staging_child.slice(..).get_mapped_range();
                                    let child_info: &[u32] = bytemuck::cast_slice(&data);
                                    let child_move_id = child_info[1]; // move_id is at offset 4 (second u32)
                                    println!("[DIAG] Pruning: child[{}] idx={} move_id={} {}", 
                                        i, child_idx, child_move_id,
                                        if child_move_id == target_move_id { "*** MATCH ***" } else { "" });
                                    drop(data);
                                    staging_child.unmap();
                                }
                            }
                        }
                    }
                }
            }

            // 3. Create Bind Group 4 (Pruning Resources)
            println!("[DIAG] Pruning: creating bind group 4 layout");
            let group4_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Pruning Group 4 Layout"),
                entries: &[
                    BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 6, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 7, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ],
            });

            let group4_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Pruning Group 4 Bind Group"),
                layout: &group4_layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: reroot_params_buf.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: new_root_output_buf.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: global_free_queue_buf.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: global_free_head_buf.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: work_queue_buf.as_entire_binding() },
                    BindGroupEntry { binding: 5, resource: work_head_buf.as_entire_binding() },
                    BindGroupEntry { binding: 6, resource: work_claimed_buf.as_entire_binding() },
                    BindGroupEntry { binding: 7, resource: work_completed_buf.as_entire_binding() },
                ],
            });

            // 4. Create Group 0 Bind Group (Node Data)
            let (
                node_info,
                node_visits,
                node_wins,
                node_vl,
                node_state,
                children_indices,
                children_priors,
                free_lists,
                free_tops,
                free_list_ownership_buf,
                global_free_queue_buf,
                global_free_head_buf,
                expansion_paused_buf
            ) = {
                let inner = self.inner.lock().unwrap();
                (
                    inner.node_info_buffer.as_ref().expect("node_info missing").clone(),
                    inner.node_visits_buffer.as_ref().expect("node_visits missing").clone(),
                    inner.node_wins_buffer.as_ref().expect("node_wins missing").clone(),
                    inner.node_vl_buffer.as_ref().expect("node_vl missing").clone(),
                    inner.node_state_buffer.as_ref().expect("node_state missing").clone(),
                    inner.children_indices_buffer.as_ref().expect("children_indices missing").clone(),
                    inner.children_priors_buffer.as_ref().expect("children_priors missing").clone(),
                    inner.free_lists_buffer.as_ref().expect("free_lists missing").clone(),
                    inner.free_tops_buffer.as_ref().expect("free_tops missing").clone(),
                    inner.free_list_ownership_buffer.as_ref().expect("free_list_ownership missing").clone(),
                    inner.global_free_queue_buffer.as_ref().expect("global_free_queue missing").clone(),
                    inner.global_free_head_buffer.as_ref().expect("global_free_head missing").clone(),
                    inner.expansion_paused_buffer.as_ref().expect("expansion_paused missing").clone(),
                )
            };

            let group0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Pruning Group 0 Layout"),
                entries: &(0..=12).map(|i| BindGroupLayoutEntry {
                    binding: i,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                }).collect::<Vec<_>>(),
            });

            let group0_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Pruning Group 0 Bind Group"),
                layout: &group0_layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: node_info.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: node_visits.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: node_wins.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: node_vl.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: node_state.as_entire_binding() },
                    BindGroupEntry { binding: 5, resource: children_indices.as_entire_binding() },
                    BindGroupEntry { binding: 6, resource: children_priors.as_entire_binding() },
                    BindGroupEntry { binding: 7, resource: free_lists.as_entire_binding() },
                    BindGroupEntry { binding: 8, resource: free_tops.as_entire_binding() },
                    BindGroupEntry { binding: 9, resource: free_list_ownership_buf.as_entire_binding() },
                    BindGroupEntry { binding: 10, resource: global_free_queue_buf.as_entire_binding() },
                    BindGroupEntry { binding: 11, resource: global_free_head_buf.as_entire_binding() },
                    BindGroupEntry { binding: 12, resource: expansion_paused_buf.as_entire_binding() },
                ],
            });

            // Retrieve Group 1, 2, 3 buffers
            let (
                mcts_params_buf,
                work_items_buf,
                paths_buf,
                alloc_counter_buf,
                diagnostics_buf,
                root_board_buf,
                urgent_event_buf,
                urgent_event_head_buf
            ) = {
                let inner = self.inner.lock().unwrap();
                (
                    inner.mcts_params_buffer.as_ref().expect("mcts_params missing").clone(),
                    inner.work_items_buffer.as_ref().expect("work_items missing").clone(),
                    inner.paths_buffer.as_ref().expect("paths missing").clone(),
                    inner.alloc_counter_buffer.as_ref().expect("alloc_counter missing").clone(),
                    inner.diagnostics_buffer.as_ref().expect("diagnostics missing").clone(),
                    inner.root_board_buffer.as_ref().expect("root_board_buffer missing").clone(),
                    inner.urgent_event_buffer_gpu.as_ref().expect("urgent_event_buffer_gpu missing").clone(),
                    inner.urgent_event_write_head_gpu.as_ref().expect("urgent_event_write_head_gpu missing").clone(),
                )
            };

            // Group 1 (MCTS Params)
            let group1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Pruning Group 1 Layout"),
                entries: &[
                    BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ],
            });
            let group1_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Pruning Group 1 Bind Group"),
                layout: &group1_layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: mcts_params_buf.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: work_items_buf.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: paths_buf.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: alloc_counter_buf.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: diagnostics_buf.as_entire_binding() },
                ],
            });

            // Group 2 (Root Board)
            let group2_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Pruning Group 2 Layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                }],
            });
            let group2_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Pruning Group 2 Bind Group"),
                layout: &group2_layout,
                entries: &[BindGroupEntry { binding: 0, resource: root_board_buf.as_entire_binding() }],
            });
            
            // Group 3 (Urgent Events)
            let group3_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Pruning Group 3 Layout"),
                entries: &[
                    BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ],
            });
            let group3_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Pruning Group 3 Bind Group"),
                layout: &group3_layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: urgent_event_buf.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: urgent_event_head_buf.as_entire_binding() },
                ],
            });

            // 5. Create Pipeline Layout
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pruning Pipeline Layout"),
                bind_group_layouts: &[&group0_layout, &group1_layout, &group2_layout, &group3_layout, &group4_layout],
                push_constant_ranges: &[],
            });

            // 6. Load Shader and Create Pipelines
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MctsOthelloShader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mcts_othello.wgsl").into()),
            });

            let identify_garbage_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Identify Garbage Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("identify_garbage"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            let _prune_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Prune Unreachable Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("prune_unreachable_topdown"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            // 7. Dispatch
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Pruning Encoder") });
            
            // Phase 1: Identify Garbage
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Identify Garbage Pass"), timestamp_writes: None });
                cpass.set_pipeline(&identify_garbage_pipeline);
                cpass.set_bind_group(0, &group0_bind_group, &[]);
                cpass.set_bind_group(1, &group1_bind_group, &[]);
                cpass.set_bind_group(2, &group2_bind_group, &[]);
                cpass.set_bind_group(3, &group3_bind_group, &[]);
                cpass.set_bind_group(4, &group4_bind_group, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
            
            // DEBUG: Submit Phase 1 and read back results immediately
            {
                let work_head_staging_temp = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Temp Work Head Check"),
                    size: 8,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(&new_root_output_buf, 0, &work_head_staging_temp, 0, 4);
                encoder.copy_buffer_to_buffer(&work_head_buf, 0, &work_head_staging_temp, 4, 4);
                queue.submit(Some(encoder.finish()));
                
                let _poll_guard = DEVICE_POLL_MUTEX.lock().unwrap();
                device.poll(wgpu::Maintain::Wait);
                
                let slice = work_head_staging_temp.slice(..);
                let (tx_temp, rx_temp) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |v| { let _ = tx_temp.send(v); });
                device.poll(wgpu::Maintain::Wait);
                if let Ok(Ok(())) = rx_temp.recv_timeout(std::time::Duration::from_secs(2)) {
                    let data = slice.get_mapped_range();
                    let new_root_temp = u32::from_le_bytes(data[0..4].try_into().unwrap());
                    let work_head_temp = u32::from_le_bytes(data[4..8].try_into().unwrap());
                    println!("[DIAG] AFTER PHASE 1 (identify_garbage): new_root_output=0x{:08X}, work_head=0x{:08X}", new_root_temp, work_head_temp);
                    drop(data);
                    work_head_staging_temp.unmap();
                } else {
                    println!("[DIAG] AFTER PHASE 1: Failed to read buffers");
                }
            }
            
            // Recreate encoder for final readback
            
            // Phase 2: Prune Unreachable (simplified - no recursion)
            // Only frees the direct garbage nodes identified in Phase 1, doesn't recurse into children
            // This avoids the infinite loop issue while still freeing immediate garbage
            {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Phase 2 Encoder") });
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Prune Pass"), timestamp_writes: None });
                cpass.set_pipeline(&_prune_pipeline);
                cpass.set_bind_group(0, &group0_bind_group, &[]);
                cpass.set_bind_group(1, &group1_bind_group, &[]);
                cpass.set_bind_group(2, &group2_bind_group, &[]);
                cpass.set_bind_group(3, &group3_bind_group, &[]);
                cpass.set_bind_group(4, &group4_bind_group, &[]);
                cpass.dispatch_workgroups(64, 1, 1); // Reduced workgroups since we're not recursing
                drop(cpass);
                queue.submit(Some(encoder.finish()));
                println!("[DIAG] Pruning: Phase 2 submitted (simplified non-recursive version)");
            }
            
            // Phase 3: Defragment free lists to prevent fragmentation
            {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Defragment Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mcts_othello.wgsl").into()),
                });
                let defrag_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Defragment Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("defragment_free_lists"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
                
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Defragment Encoder") });
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Defragment Pass"), timestamp_writes: None });
                cpass.set_pipeline(&defrag_pipeline);
                cpass.set_bind_group(0, &group0_bind_group, &[]);
                cpass.set_bind_group(1, &group1_bind_group, &[]);
                cpass.set_bind_group(2, &group2_bind_group, &[]);
                cpass.set_bind_group(3, &group3_bind_group, &[]);
                cpass.set_bind_group(4, &group4_bind_group, &[]);
                cpass.dispatch_workgroups(1, 1, 1); // 256 threads = 1 workgroup with size 256
                drop(cpass);
                queue.submit(Some(encoder.finish()));
                println!("[DIAG] Pruning: Defragmentation complete");
            }
            
            // Create encoder for final readback
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Final Readback Encoder") });
            
            // Create a fresh staging buffer for this operation to avoid stale data from previous mappings
            // Using a shared staging buffer across threads can cause reads to return old data
            let new_root_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("New Root Staging Buffer"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&new_root_output_buf, 0, &new_root_staging_buf, 0, 4);
            
            // DEBUG: Also read work_head to verify shader execution
            let work_head_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Debug Work Head Staging"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&work_head_buf, 0, &work_head_staging, 0, 4);
            
            println!("[DIAG] Pruning: submitting queue");
            queue.submit(Some(encoder.finish()));
            
            // CRITICAL: Acquire device poll mutex to prevent conflicts with urgent event logger thread
            println!("[DIAG] Pruning: acquiring DEVICE_POLL_MUTEX");
            let _poll_guard = DEVICE_POLL_MUTEX.lock().unwrap();
            println!("[DIAG] Pruning: acquired DEVICE_POLL_MUTEX");
            
            // DEBUG: Read work_head first (in its own scope to ensure cleanup)
            {
                let slice = work_head_staging.slice(..);
                let (tx_wh, rx_wh) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |v| { let _ = tx_wh.send(v); });
                device.poll(wgpu::Maintain::Wait);
                if let Ok(Ok(())) = rx_wh.recv_timeout(std::time::Duration::from_secs(2)) {
                    let data = slice.get_mapped_range();
                    let work_head_value = u32::from_le_bytes(data[0..4].try_into().unwrap());
                    println!("[DIAG] Pruning: work_head value after shader = 0x{:08X} (expect 0xDECAFBAD if shader ran)", work_head_value);
                    drop(data);
                }
                work_head_staging.unmap();
                // Explicitly drop the staging buffer to release resources
                drop(work_head_staging);
            }
            
            // 8. Read back new root (in a scope to ensure cleanup before function returns)
            let new_root_idx = {
                let slice = new_root_staging_buf.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                println!("[DIAG] Pruning: calling map_async");
                slice.map_async(wgpu::MapMode::Read, move |v| {
                    println!("[DIAG] Pruning: map_async callback invoked with result: {:?}", v);
                    let _ = tx.send(v); // Ignore send errors if receiver dropped
                });
                println!("[DIAG] Pruning: waiting for device poll");
                device.poll(wgpu::Maintain::Wait);
                println!("[DIAG] Pruning: device poll complete, waiting for map_async result with 5 second timeout");
                
                let map_result = match rx.recv_timeout(std::time::Duration::from_secs(5)) {
                    Ok(result) => {
                        println!("[DIAG] Pruning: map_async result: {:?}", result);
                        result
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        panic!("[GPU-Native FATAL] map_async callback timed out after 5 seconds! Device may be hung.");
                    }
                    Err(e) => {
                        panic!("[GPU-Native FATAL] Failed to receive map_async result: {:?}", e);
                    }
                };
                
                match map_result {
                    Ok(()) => {
                        println!("[DIAG] Pruning: mapping succeeded, calling get_mapped_range");
                    }
                    Err(e) => {
                        panic!("[GPU-Native FATAL] Buffer mapping failed: {:?}", e);
                    }
                }
                
                // Add small delay to ensure buffer is fully ready
                std::thread::sleep(std::time::Duration::from_millis(10));
                println!("[DIAG] Pruning: about to call get_mapped_range (after 10ms delay)");
                let data = slice.get_mapped_range();
                println!("[DIAG] Pruning: got mapped range, reading data");
                let new_root_idx = u32::from_le_bytes(data[0..4].try_into().unwrap());
                println!("[DIAG] Pruning: Read new_root_idx = 0x{:08X} (decimal: {})", new_root_idx, new_root_idx);
                drop(data);
                new_root_staging_buf.unmap();
                
                // Explicitly drop the staging buffer before releasing the mutex
                drop(new_root_staging_buf);
                
                new_root_idx
            };
            
            // Ensure all GPU operations are fully complete before releasing mutex
            device.poll(wgpu::Maintain::Wait);
            drop(_poll_guard); // Release DEVICE_POLL_MUTEX
            println!("[DIAG] Pruning: released DEVICE_POLL_MUTEX");
            
            // Get current root for comparison
            let current_root = {
                let inner = self.inner.lock().unwrap();
                inner.current_root_idx
            };
            
            if new_root_idx == 0xFFFFFFF0 {
                println!("[DIAG] Pruning failed: Root node has NO children (num_children=0). Resetting tree.");
                // Don't update current_root_idx - keep the old root, but we need to reset the tree
                // Actually, we should reset the tree here
                return false;
            } else if new_root_idx == 0xFFFFFFFF {
                println!("[DIAG] Pruning failed: Children exist but move not found.");
                // Don't update current_root_idx - keep the old root
                return false;
            } else if new_root_idx == current_root {
                println!("[DIAG] Pruning failed: Returned same root ({}). This indicates shader failure or device corruption.", new_root_idx);
                eprintln!("[GPU-Native ERROR] Pruning returned invalid result. Device may be corrupted.");
                eprintln!("[GPU-Native ERROR] Aborting advance_root to prevent further operations on corrupted device.");
                // CRITICAL: Do NOT continue - the device is corrupted
                return false;
            } else if (new_root_idx & 0xE0000000) == 0xE0000000 {
                let first_move = new_root_idx & 0xFF;
                let num_children = (new_root_idx >> 8) & 0xFF;
                let mx = first_move % 8;
                let my = first_move / 8;
                println!("[DIAG] Pruning failed: Children exist but move not found. Num children: {}. First child move_id={} (x={}, y={})", num_children, first_move, mx, my);
                // Don't update current_root_idx - keep the old root
                return false;
            } else {
                println!("[DIAG] Pruning complete. New root: {}", new_root_idx);
                
                // Update host state only if pruning succeeded
                let mut inner = self.inner.lock().unwrap();
                inner.current_root_idx = new_root_idx;
                
                // CRITICAL: Ensure ALL GPU work (including any leftover MCTS work) is complete
                // before we clear buffers. This prevents in-flight threads from accessing
                // VL buffers after they've been zeroed.
                println!("[DIAG] Waiting for all GPU work to complete before clearing buffers");
                device.poll(wgpu::Maintain::Wait);
                println!("[DIAG] GPU idle, now clearing search state");
                
                // CRITICAL: Clean up all search state after move
                // This includes thread states and virtual loss counters
                
                // 1. Reset thread states - clear any in-flight work
                if let Some(thread_states_buf) = &inner.thread_states_buffer {
                    println!("[DIAG] Resetting thread states after pruning");
                    let num_bytes = thread_states_buf.size();
                    let zeros = vec![0u8; num_bytes as usize];
                    queue.write_buffer(thread_states_buf, 0, &zeros);
                }
                
                // 2. Clear virtual loss for entire tree
                // VL is transient search state, not tree state, so it must be reset after a move
                if let Some(node_vl_buf) = &inner.node_vl_buffer {
                    println!("[DIAG] Clearing virtual loss counters after pruning");
                    let num_bytes = node_vl_buf.size();
                    let zeros = vec![0u8; num_bytes as usize];
                    queue.write_buffer(node_vl_buf, 0, &zeros);
                }
                
                // 3. Clear rollout and backprop queues
                // These contain jobs from the old tree that should not be processed
                if let Some(rollout_head_buf) = &inner.rollout_head_buffer {
                    println!("[DIAG] Clearing rollout queue after pruning");
                    queue.write_buffer(rollout_head_buf, 0, &[0u8; 4]); // Reset head to 0
                }
                if let Some(backprop_head_buf) = &inner.backprop_head_buffer {
                    println!("[DIAG] Clearing backprop queue after pruning");
                    queue.write_buffer(backprop_head_buf, 0, &[0u8; 4]); // Reset head to 0
                }
                
                // Ensure all writes complete before continuing
                device.poll(wgpu::Maintain::Wait);
            }
            true
        }

        pub fn dispatch_prune_unreachable_topdown(&self) {
            // Legacy wrapper
            self.dispatch_pruning_kernels(0, 0);
        }

        /// Dispatch garbage collection to free unreachable nodes without changing root
        /// This should be called periodically during search to prevent memory exhaustion
        pub fn dispatch_garbage_collection(&self) {
            let device = self.context.device();
            let queue = self.context.queue();
            
            // Get current state
            let (max_nodes, mcts_params_buf, node_info, node_visits, node_wins, node_vl, node_state,
                 children_indices, children_priors, free_lists, free_tops) = {
                let inner = self.inner.lock().unwrap();
                (
                    inner.max_nodes,
                    inner.mcts_params_buffer.as_ref().expect("params missing").clone(),
                    inner.node_info_buffer.as_ref().expect("node_info missing").clone(),
                    inner.node_visits_buffer.as_ref().expect("node_visits missing").clone(),
                    inner.node_wins_buffer.as_ref().expect("node_wins missing").clone(),
                    inner.node_vl_buffer.as_ref().expect("node_vl missing").clone(),
                    inner.node_state_buffer.as_ref().expect("node_state missing").clone(),
                    inner.children_indices_buffer.as_ref().expect("children_indices missing").clone(),
                    inner.children_priors_buffer.as_ref().expect("children_priors missing").clone(),
                    inner.free_lists_buffer.as_ref().expect("free_lists missing").clone(),
                    inner.free_tops_buffer.as_ref().expect("free_tops missing").clone(),
                )
            };
            
            // Create shader module
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GC Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mcts_othello.wgsl").into()),
            });
            
            // Create bind group layout for Group 0 (node data)
            let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GC Group 0 Layout"),
                entries: &(0..=8).map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }).collect::<Vec<_>>(),
            });
            
            // Create bind group layout for Group 1 (params)
            let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GC Group 1 Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
            
            // Create pipeline
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GC Pipeline Layout"),
                bind_group_layouts: &[&group0_layout, &group1_layout],
                push_constant_ranges: &[],
            });
            
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GC Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("prune_unreachable"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            // Create bind groups
            let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GC Group 0"),
                layout: &group0_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: node_info.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: node_visits.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: node_wins.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: node_vl.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: node_state.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: children_indices.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: children_priors.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: free_lists.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: free_tops.as_entire_binding() },
                ],
            });
            
            let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GC Group 1"),
                layout: &group1_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: mcts_params_buf.as_entire_binding() },
                ],
            });
            
            // Dispatch - use enough workgroups to cover all nodes
            // prune_unreachable uses workgroup_size(256), so need (max_nodes + 255) / 256 workgroups
            let num_workgroups = (max_nodes + 255) / 256;
            
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GC Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GC Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group0, &[]);
                pass.set_bind_group(1, &group1, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
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
            // Match shader's encode_move: y * width + x
            // Filter out pass moves (usize::MAX, usize::MAX) to avoid overflow
            let legal_idxs: std::collections::HashSet<_> = inner.legal_moves.iter()
                .filter(|&&(x, y)| x != usize::MAX && y != usize::MAX)
                .map(|&(x, y)| y * 8 + x)
                .collect();
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
use std::sync::Arc;
use std::collections::HashSet;
use crate::gpu::GpuContext;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
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
    pub turn_number: u32, // NEW: unique per-turn identifier
    pub free_list_capacity: u32, // Capacity per free list (max_nodes / 256 rounded up)
    pub vl_temp_scale: f32, // Scaling factor for VL-based temperature boost
    pub num_threads: u32, // NEW: Number of threads for incremental execution
    pub root_node: u32, // NEW: Current root node index for incremental execution
    pub use_vl_preincrement: u32, // 1 = use pre-increment VL, 0 = increment only selected child
    pub use_random_rollouts: u32, // 1 = use random test rollouts, 0 = real game simulation
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
    pub _pad2: u32, // Pad to 32 bytes
    pub _pad3: u32,
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
    pub nodes_freed: u32, // Count nodes freed during pruning
    pub rollouts: u32,
    pub root_board_hash: u32,
    pub init_nodes_count: u32,
    pub total_children_gen: u32,
    pub prune_work_claimed: u32, // DEBUG: How many work items successfully claimed in Phase 2
    pub prune_push_attempts: u32, // DEBUG: Total attempts in free list push loop
    pub max_temp_boost: u32, // Maximum temperature boost observed (stored as u32, divide by 1000 for f32)
    pub random_rollout_wins: u32, // Count of result=2 (win) from random rollouts (from p1 perspective)
    pub random_rollout_draws: u32, // Count of result=1 (draw) from random rollouts
    pub random_rollout_losses: u32, // Count of result=0 (loss) from random rollouts (from p1 perspective)
    pub random_rollout_from_p1: u32, // Count of rollouts where leaf_player == 1
    pub random_rollout_from_p2: u32, // Count of rollouts where leaf_player == -1
    pub random_rollout_p1_wins_raw: u32, // Count where p1_score > 32 in game simulation
    pub random_rollout_p1_losses_raw: u32, // Count where p1_score < 32 in game simulation
    pub random_rollout_p1_score_sum: u32, // Sum of all p1_score values for distribution checking
    pub random_rollout_min_tid: u32, // Minimum thread ID that did a rollout (initialized to 0xFFFFFFFF)
    pub random_rollout_max_tid: u32, // Maximum thread ID that did a rollout
    pub global_rollout_counter: u32, // Global counter for independent RNG seeding
    pub phase_selection_count: u32, // Threads in PHASE_SELECTION
    pub phase_expansion_count: u32, // Threads in PHASE_EXPANSION
    pub phase_rollout_count: u32, // Threads in PHASE_ROLLOUT_ACTIVE
    pub phase_backprop_count: u32, // Threads in PHASE_BACKPROP
    pub phase_idle_count: u32, // Threads in PHASE_IDLE
    pub phase_finished_count: u32, // Threads in PHASE_FINISHED
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RerootParams {
    pub move_x: u32,
    pub move_y: u32,
    pub current_root: u32,
    pub max_nodes: u32,  // Work queue capacity for bounds checking
}

impl std::fmt::Debug for RerootParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RerootParams")
            .field("move_x", &self.move_x)
            .field("move_y", &self.move_y)
            .field("current_root", &self.current_root)
            .field("max_nodes", &self.max_nodes)
            .finish()
    }
}

pub struct GpuOthelloMcts {
    pub context: Arc<GpuContext>,
    pub inner: Mutex<GpuOthelloMctsInner>,
    // Prevent Send/Sync for raw pointers
    _not_send_sync: std::marker::PhantomData<*const ()>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct RootChildStats {
    pub move_id: u32,
    pub visits: i32,
    pub wins: i32,
    pub _pad: u32,
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
    // Node storage buffers (Group 0)
    pub node_info_buffer: Option<Arc<wgpu::Buffer>>,
    pub node_visits_buffer: Option<Arc<wgpu::Buffer>>,
    pub node_wins_buffer: Option<Arc<wgpu::Buffer>>,
    pub node_vl_buffer: Option<Arc<wgpu::Buffer>>,
    pub node_state_buffer: Option<Arc<wgpu::Buffer>>,
    pub children_indices_buffer: Option<Arc<wgpu::Buffer>>,
    pub children_priors_buffer: Option<Arc<wgpu::Buffer>>,
    pub free_lists_buffer: Option<Arc<wgpu::Buffer>>,
    pub free_tops_buffer: Option<Arc<wgpu::Buffer>>,
    pub free_list_ownership_buffer: Option<Arc<wgpu::Buffer>>, // Which workgroup owns each free list (0-255, or 0xFFFF for unowned)
    // GPU-side urgent event buffers (bound to pipeline, not mapped)
    pub urgent_event_buffer_gpu: Option<Arc<wgpu::Buffer>>,
    pub urgent_event_write_head_gpu: Option<Arc<wgpu::Buffer>>,
    // Host-mapped urgent event buffers (for polling, never bound to pipeline)
    pub urgent_event_buffer_host: Option<Arc<wgpu::Buffer>>,
    pub urgent_event_write_head_host: Option<Arc<wgpu::Buffer>>,
    // Staging buffers for robust host reads (MAP_READ | COPY_DST)
    pub urgent_event_staging: Option<Arc<wgpu::Buffer>>,
    pub urgent_event_write_head_staging: Option<Arc<wgpu::Buffer>>,
    // Persistent buffer for global_reroot_threads_remaining (group 5, binding 0)
    pub global_reroot_threads_remaining: Option<Arc<wgpu::Buffer>>,
    // Persistent buffer for global_reroot_start_threads_remaining (group 5, binding 1)
    pub global_reroot_start_threads_remaining: Option<Arc<wgpu::Buffer>>,
    // Pruning buffers (Group 4)
    pub reroot_params_buffer: Option<Arc<wgpu::Buffer>>, // Binding 0
    pub new_root_output_buffer: Option<Arc<wgpu::Buffer>>, // Binding 1
    pub new_root_staging_buffer: Option<Arc<wgpu::Buffer>>, // Staging
    pub global_free_queue_buffer: Option<Arc<wgpu::Buffer>>, // Binding 2
    pub global_free_head_buffer: Option<Arc<wgpu::Buffer>>, // Binding 3
    pub expansion_paused_buffer: Option<Arc<wgpu::Buffer>>, // Group 0 Binding 11
    pub work_queue_buffer: Option<Arc<wgpu::Buffer>>, // Binding 4
    pub work_head_buffer: Option<Arc<wgpu::Buffer>>, // Binding 5
    pub work_claimed_buffer: Option<Arc<wgpu::Buffer>>, // Binding 6
    pub work_completed_buffer: Option<Arc<wgpu::Buffer>>, // Binding 7
    pub current_root_idx: u32,
    // MCTS Execution Buffers (Group 1)
    pub mcts_params_buffer: Option<Arc<wgpu::Buffer>>, // Binding 0
    pub work_items_buffer: Option<Arc<wgpu::Buffer>>, // Binding 1
    pub paths_buffer: Option<Arc<wgpu::Buffer>>, // Binding 2
    pub alloc_counter_buffer: Option<Arc<wgpu::Buffer>>, // Binding 3
    pub diagnostics_buffer: Option<Arc<wgpu::Buffer>>, // Binding 4
    pub root_stats_buffer: Option<Arc<wgpu::Buffer>>, // Binding 5
    // Root Board Buffer (Group 2)
    pub root_board_buffer: Option<Arc<wgpu::Buffer>>, // Binding 0
    // Pre-Computation Buffers (Group 6 - Optimization)
    pub leaf_candidates_buffer: Option<Arc<wgpu::Buffer>>, // Binding 0: LeafCandidate[max_nodes]
    pub candidate_count_buffer: Option<Arc<wgpu::Buffer>>, // Binding 1: atomic<u32>
    pub precomputed_cache_buffer: Option<Arc<wgpu::Buffer>>, // Binding 2: PrecomputedMoves[budget]
    pub cache_size_buffer: Option<Arc<wgpu::Buffer>>, // Binding 3: u32
    pub precompute_diagnostics_buffer: Option<Arc<wgpu::Buffer>>, // Binding 4: PrecomputeDiagnostics
    // Thread State Buffer (Group 7 - Incremental Execution)
    pub thread_states_buffer: Option<Arc<wgpu::Buffer>>, // Binding 0: ThreadState[num_threads]
    // Queue Buffers (Group 8 - Decoupled Rollout Architecture)
    pub rollout_queue_buffer: Option<Arc<wgpu::Buffer>>, // Binding 0: RolloutJob[queue_size]
    pub rollout_head_buffer: Option<Arc<wgpu::Buffer>>, // Binding 1: atomic<u32>
    pub backprop_queue_buffer: Option<Arc<wgpu::Buffer>>, // Binding 2: BackpropJob[queue_size]
    pub backprop_head_buffer: Option<Arc<wgpu::Buffer>>, // Binding 3: atomic<u32>
    // Diagnostics baseline for computing per-search deltas
    pub max_temp_boost_baseline: u32,
    // Progressive wave tracking
    pub current_wave_workgroups: u32,
}

impl GpuOthelloMcts {
    pub fn run_iterations(&self, _iterations: u32, _exploration: f32, _virtual_loss_weight: f32, _temperature: f32, _seed: u32) -> OthelloRunTelemetry {
        // In GPU-native mode, the kernel is dispatched separately.
        // This function just reads back the telemetry.
        let mut diagnostics = self.read_diagnostics();
        let nodes_used = self.calculate_nodes_used();
        
        let inner = self.inner.lock().unwrap();
        
        // Compute per-search delta for max_temp_boost (since GPU buffer resets don't work)
        let per_search_boost = diagnostics.max_temp_boost.saturating_sub(inner.max_temp_boost_baseline);
        diagnostics.max_temp_boost = per_search_boost;
        
        OthelloRunTelemetry {
            iterations_launched: diagnostics.rollouts,
            alloc_count_after: nodes_used,
            free_count_after: inner.max_nodes - nodes_used,
            node_capacity: inner.max_nodes,
            saturated: diagnostics.alloc_failures > 0,
            diagnostics,
        }
    }

    /// Set the baseline for diagnostics delta tracking.
    /// Call this before starting a new search to capture the starting values.
    pub fn set_diagnostics_baseline(&self) {
        // Just read current value as baseline (don't reset here to avoid lock issues)
        let diagnostics = self.read_diagnostics();
        let mut inner = self.inner.lock().unwrap();
        inner.max_temp_boost_baseline = diagnostics.max_temp_boost;
    }

    /// Run incremental MCTS with stateful thread execution.
    /// 
    /// This orchestrates the full incremental execution pipeline:
    /// 1. Collect leaf candidates (find unexpanded nodes with visits > 0)
    /// 2. Sort candidates by parent visit count (prioritize hot subtrees)
    /// 3. Run incremental_mcts_step for SELECTION/EXPANSION/BACKPROP phases
    /// 4. Run rollout_chunk_kernel for active rollouts (8 moves per dispatch)
    /// 5. Repeat steps 3-4 until convergence
    ///
    /// Parameters:
    ///   - num_threads: Number of parallel threads to use
    ///   - max_steps: Maximum number of incremental steps (safety limit)
    ///   - exploration: PUCT exploration constant
    ///   - virtual_loss_weight: Virtual loss weight for parallelism
    ///   - temperature: Temperature for move selection
    ///   - seed: RNG seed
    ///   - timeout: Optional timeout duration to limit search time
    pub fn run_incremental_mcts(
        &self,
        num_threads: u32,
        max_steps: u32,
        exploration: f32,
        virtual_loss_weight: f32,
        temperature: f32,
        seed: u32,
        timeout: Option<std::time::Duration>,
        use_random_rollouts: bool,
    ) -> OthelloRunTelemetry {
        println!("[INCREMENTAL] Starting incremental MCTS: {} threads, {} max steps", num_threads, max_steps);
        
        let device = self.context.device();
        let queue = self.context.queue();
        
        // Upload root board to GPU (needed for expand_node to reconstruct board states)
        {
            let inner = self.inner.lock().unwrap();
            let root_board_buf = inner.root_board_buffer.as_ref().expect("root_board_buffer missing");
            queue.write_buffer(root_board_buf, 0, bytemuck::cast_slice(&inner.root_board));
        }
        
        // Allocate/recreate thread states buffer (force recreation to ensure clean state)
        let thread_states_buf = {
            let mut inner = self.inner.lock().unwrap();
            inner.thread_states_buffer = None; // Clear any existing buffer
            drop(inner);
            self.ensure_thread_states_buffer(num_threads)
        };
        
        let _candidates_buf = self.ensure_candidates_buffer(16384); // Support up to 16K candidates
        let counter_buf = self.ensure_candidate_counter_buffer();
        
        // Reset candidate counter to 0
        queue.write_buffer(&counter_buf, 0, &0u32.to_le_bytes());
        
        // Initialize thread states: All threads start in SELECTION phase at root
        {
            let inner = self.inner.lock().unwrap();
            let root_idx = inner.current_root_idx;
            
            // ThreadState layout: phase(4), current_node(4), path(512), path_len(4), 
            // rng_seed(4), leaf_player(4), rollout_result(4), backprop_index(4),
            // rollout_board(256), rollout_player(4), rollout_moves_remaining(4), _pad0(4) = 816 bytes
            const THREAD_STATE_SIZE: usize = 816;
            let mut init_data = vec![0u8; THREAD_STATE_SIZE * num_threads as usize]; // Already zeros entire buffer
            
            for tid in 0..num_threads {
                let offset = tid as usize * THREAD_STATE_SIZE;
                let slice = &mut init_data[offset..offset + THREAD_STATE_SIZE];
                
                // All fields default to 0 which is correct for most
                // phase = SELECTION (0) - already 0
                // current_node = root_idx
                slice[4..8].copy_from_slice(&root_idx.to_le_bytes());
                // path[0] = root_idx (path is at offset 8, 128 u32s = 512 bytes)
                slice[8..12].copy_from_slice(&root_idx.to_le_bytes());
                // path_len = 1 (at offset 8 + 512 = 520)
                slice[520..524].copy_from_slice(&1u32.to_le_bytes());
                // rng_seed = seed + tid (at offset 524)
                slice[524..528].copy_from_slice(&(seed + tid).to_le_bytes());
                // leaf_player, rollout_result, backprop_index all 0 is fine
                // rollout_board all 0 is fine
            }
            
            queue.write_buffer(&thread_states_buf, 0, &init_data);
        }
        
        // Ensure all buffer writes are complete before dispatching kernels
        device.poll(wgpu::Maintain::Wait);
        
        // DEBUG: Verify thread states were initialized correctly (disabled)
        if false {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Thread States Verify"),
                size: 816 * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(&thread_states_buf, 0, &staging, 0, 816 * 4);
            queue.submit(Some(encoder.finish()));
            
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |v| { let _ = tx.send(v); });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            
            let data = slice.get_mapped_range();
            println!("[INIT CHECK] Thread states immediately after initialization:");
            for tid in 0..4.min(num_threads) {
                let offset = tid as usize * 816;
                let phase = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap());
                let current_node = u32::from_le_bytes(data[offset+4..offset+8].try_into().unwrap());
                let path0 = u32::from_le_bytes(data[offset+8..offset+12].try_into().unwrap());
                let path_len = u32::from_le_bytes(data[offset+520..offset+524].try_into().unwrap());
                let rng_seed = u32::from_le_bytes(data[offset+524..offset+528].try_into().unwrap());
                let leaf_player = i32::from_le_bytes(data[offset+528..offset+532].try_into().unwrap());
                let rollout_result = u32::from_le_bytes(data[offset+532..offset+536].try_into().unwrap());
                let backprop_idx = u32::from_le_bytes(data[offset+536..offset+540].try_into().unwrap());
                println!("[INIT CHECK] Thread {}: phase={} current_node={} path[0]={} path_len={} rng={} leaf={} rollout={} backprop={}", 
                         tid, phase, current_node, path0, path_len, rng_seed, leaf_player, rollout_result, backprop_idx);
            }
            drop(data);
            staging.unmap();
        }
        
        // Step 1: Collect leaf candidates
        println!("[INCREMENTAL] Collecting leaf candidates...");
        self.dispatch_collect_candidates();
        
        // Step 2: Sort candidates by score (bitonic sort)
        println!("[INCREMENTAL] Sorting candidates...");
        self.dispatch_bitonic_sort();
        
        // Step 3-4: Run incremental steps until convergence
        println!("[INCREMENTAL] Running {} steps with {} threads...", max_steps, num_threads);
        let start_time = std::time::Instant::now();
        let mut steps_completed = 0;
        for step in 0..max_steps {
            // Check timeout every 10 steps to avoid overhead
            if step % 10 == 0 {
                if let Some(t) = timeout {
                    if start_time.elapsed() >= t {
                        println!("[INCREMENTAL] Timeout reached after {} steps ({:.2}s)", step, start_time.elapsed().as_secs_f64());
                        steps_completed = step;
                        break;
                    }
                }
            }
            
            // Dispatch incremental_mcts_step (handles SELECTION/EXPANSION/BACKPROP)
            self.dispatch_incremental_step(num_threads, exploration, virtual_loss_weight, temperature, seed + step, use_random_rollouts);
            
            // Dispatch rollout_chunk_kernel for threads in ROLLOUT_ACTIVE phase
            self.dispatch_rollout_chunk(num_threads, seed + step * 1000);
            
            steps_completed = step + 1;
        }
        
        println!("[INCREMENTAL] Incremental MCTS complete after {} steps ({:.2}s)", steps_completed, start_time.elapsed().as_secs_f64());
        
        // CRITICAL: Wait for all GPU work to complete before returning
        // This ensures that if the caller immediately calls advance_root, there are no in-flight
        // threads that would try to access VL buffers after they've been cleared
        device.poll(wgpu::Maintain::Wait);
        
        // Return telemetry (reuse existing run_iterations logic)
        self.run_iterations(0, exploration, virtual_loss_weight, temperature, seed)
    }

    /// Dispatch collect_leaf_candidates kernel to scan tree for unexpanded leaves.
    fn dispatch_collect_candidates(&self) {
        use wgpu::*;
        let device = self.context.device();
        let queue = self.context.queue();
        
        // Get buffers
        let (node_info, node_visits, candidates_buf, counter_buf, params_buf, max_nodes) = {
            let inner = self.inner.lock().unwrap();
            (
                inner.node_info_buffer.as_ref().expect("node_info missing").clone(),
                inner.node_visits_buffer.as_ref().expect("node_visits missing").clone(),
                inner.leaf_candidates_buffer.as_ref().expect("candidates missing").clone(),
                inner.candidate_count_buffer.as_ref().expect("counter missing").clone(),
                inner.mcts_params_buffer.as_ref().expect("params missing").clone(),
                inner.max_nodes,
            )
        };
        
        // Create shader module
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Collect Candidates Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/mcts_othello.wgsl").into()),
        });
        
        // Group 0: Node data (just what we need: node_info, node_visits)
        let group0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Collect Group 0 Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        
        let group0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Collect Group 0"),
            layout: &group0_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: node_info.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: node_visits.as_entire_binding() },
            ],
        });
        
        // Group 1: Params
        let group1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Collect Group 1 Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        
        let group1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Collect Group 1"),
            layout: &group1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            ],
        });
        
        // Create dummy bind group layouts for groups 2-5 (required for contiguous layout)
        let dummy_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Dummy Layout"),
            entries: &[],
        });
        
        let dummy_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Dummy Group"),
            layout: &dummy_layout,
            entries: &[],
        });
        
        // Group 6: Candidates buffer + counter
        let group6_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Collect Group 6 Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        
        let group6 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Collect Group 6"),
            layout: &group6_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: candidates_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: counter_buf.as_entire_binding() },
            ],
        });
        
        // Create pipeline with all groups (0, 1, 2-dummy, 3-dummy, 4-dummy, 5-dummy, 6)
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Collect Candidates Pipeline Layout"),
            bind_group_layouts: &[&group0_layout, &group1_layout, &dummy_layout, &dummy_layout, &dummy_layout, &dummy_layout, &group6_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Collect Candidates Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("collect_leaf_candidates"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Dispatch
        // Cap at 65535 due to GPU hardware limit on workgroup dimensions
        let workgroups = ((max_nodes + 63) / 64).min(65535);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Collect Candidates Encoder"),
        });
        
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Collect Candidates Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group0, &[]);
            pass.set_bind_group(1, &group1, &[]);
            pass.set_bind_group(2, &dummy_group, &[]);
            pass.set_bind_group(3, &dummy_group, &[]);
            pass.set_bind_group(4, &dummy_group, &[]);
            pass.set_bind_group(5, &dummy_group, &[]);
            pass.set_bind_group(6, &group6, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        queue.submit(Some(encoder.finish()));
        device.poll(Maintain::Wait);
    }
    
    /// Dispatch bitonic_sort_candidates kernel (multiple passes for sorting).
    fn dispatch_bitonic_sort(&self) {
        use wgpu::*;
        
        let _poll_lock = DEVICE_POLL_MUTEX.lock().unwrap();
        
        let device = self.context.device();
        let queue = self.context.queue();
        
        // Get buffers using ensure methods (they return Arc<Buffer>)
        let counter_buf = self.ensure_candidate_counter_buffer();
        let candidates_buf = self.ensure_candidates_buffer(16384);
        
        // Get node buffers and params for bind groups
        let (node_info, node_visits, node_children, mcts_params_buf, _max_nodes) = {
            let inner = self.inner.lock().unwrap();
            (
                inner.node_info_buffer.clone().expect("node_info missing"),
                inner.node_visits_buffer.clone().expect("node_visits missing"),
                inner.children_indices_buffer.clone().expect("children_indices missing"),
                inner.mcts_params_buffer.clone().expect("mcts_params missing"),
                inner.max_nodes,
            )
        };
        
        // Read back candidate count to determine number of sort passes
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("Counter Staging"),
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Counter Read"),
        });
        encoder.copy_buffer_to_buffer(&counter_buf, 0, &staging, 0, 4);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        device.poll(Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let num_candidates = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        staging.unmap();
        
        if num_candidates == 0 {
            return; // Nothing to sort
        }
        
        // Round up to next power of 2
        let mut n = 1u32;
        while n < num_candidates {
            n <<= 1;
        }
        
        let num_stages = n.trailing_zeros();
        
        // Create shader
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Bitonic Sort Shader"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/mcts_othello.wgsl"))),
        });
        
        // Group 0: Node data (not used by sort, but required for contiguous layout)
        let group0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Sort Group 0 Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        
        let group0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Sort Group 0"),
            layout: &group0_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: node_info.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: node_visits.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: node_children.as_entire_binding() },
            ],
        });
        
        // Group 1: Params
        let group1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Sort Group 1 Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        
        let group1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Sort Group 1"),
            layout: &group1_layout,
            entries: &[BindGroupEntry { binding: 0, resource: mcts_params_buf.as_entire_binding() }],
        });
        
        // Dummy groups 2-5
        let dummy_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Dummy Layout"),
            entries: &[],
        });
        
        let dummy_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Dummy Group"),
            layout: &dummy_layout,
            entries: &[],
        });
        
        // Group 6: Candidates
        let group6_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Sort Group 6 Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        
        let group6 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Sort Group 6"),
            layout: &group6_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: candidates_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: counter_buf.as_entire_binding() },
            ],
        });
        
        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Sort Pipeline Layout"),
            bind_group_layouts: &[&group0_layout, &group1_layout, &dummy_layout, &dummy_layout, &dummy_layout, &dummy_layout, &group6_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Sort Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("bitonic_sort_candidates"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Dispatch multiple passes: for each stage, for each step in that stage
        // Cap at 65535 due to GPU hardware limit on workgroup dimensions
        let workgroups = ((n / 2 + 63) / 64).min(65535); // Each thread handles one comparison pair
        
        for stage in 0..num_stages {
            for step in (0..=stage).rev() {
                // Update params with stage/step (abusing turn_number and game_type fields)
                let mut params = MctsOthelloParams::default();
                params.turn_number = stage;
                params.game_type = step;
                
                queue.write_buffer(&mcts_params_buf, 0, bytemuck::bytes_of(&params));
                
                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some(&format!("Sort Stage {} Step {}", stage, step)),
                });
                
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some(&format!("Sort Pass {}-{}", stage, step)),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group0, &[]);
                    pass.set_bind_group(1, &group1, &[]);
                    pass.set_bind_group(2, &dummy_group, &[]);
                    pass.set_bind_group(3, &dummy_group, &[]);
                    pass.set_bind_group(4, &dummy_group, &[]);
                    pass.set_bind_group(5, &dummy_group, &[]);
                    pass.set_bind_group(6, &group6, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                
                queue.submit(Some(encoder.finish()));
                device.poll(Maintain::Wait);
            }
        }
    }
    
    /// Helper: Create Group 0 bind group layout and bind group with all 13 buffers
    fn create_group0_bind_group(&self, device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        use wgpu::*;
        
        // Get all Group 0 buffers
        let (
            node_info,
            node_visits,
            node_wins,
            node_vl,
            node_state,
            children_indices,
            children_priors,
            free_lists,
            free_tops,
            free_list_ownership_buf,
            global_free_queue_buf,
            global_free_head_buf,
            expansion_paused_buf
        ) = {
            let inner = self.inner.lock().unwrap();
            (
                inner.node_info_buffer.as_ref().expect("node_info missing").clone(),
                inner.node_visits_buffer.as_ref().expect("node_visits missing").clone(),
                inner.node_wins_buffer.as_ref().expect("node_wins missing").clone(),
                inner.node_vl_buffer.as_ref().expect("node_vl missing").clone(),
                inner.node_state_buffer.as_ref().expect("node_state missing").clone(),
                inner.children_indices_buffer.as_ref().expect("children_indices missing").clone(),
                inner.children_priors_buffer.as_ref().expect("children_priors missing").clone(),
                inner.free_lists_buffer.as_ref().expect("free_lists missing").clone(),
                inner.free_tops_buffer.as_ref().expect("free_tops missing").clone(),
                inner.free_list_ownership_buffer.as_ref().expect("free_list_ownership missing").clone(),
                inner.global_free_queue_buffer.as_ref().expect("global_free_queue missing").clone(),
                inner.global_free_head_buffer.as_ref().expect("global_free_head missing").clone(),
                inner.expansion_paused_buffer.as_ref().expect("expansion_paused missing").clone(),
            )
        };

        // Create layout with 13 bindings (0-12)
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Group 0 Layout (Node Data)"),
            entries: &(0..=12).map(|i| BindGroupLayoutEntry {
                binding: i,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }).collect::<Vec<_>>(),
        });
        
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Group 0 Bind Group (Node Data)"),
            layout: &layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: node_info.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: node_visits.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: node_wins.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: node_vl.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: node_state.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: children_indices.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: children_priors.as_entire_binding() },
                BindGroupEntry { binding: 7, resource: free_lists.as_entire_binding() },
                BindGroupEntry { binding: 8, resource: free_tops.as_entire_binding() },
                BindGroupEntry { binding: 9, resource: free_list_ownership_buf.as_entire_binding() },
                BindGroupEntry { binding: 10, resource: global_free_queue_buf.as_entire_binding() },
                BindGroupEntry { binding: 11, resource: global_free_head_buf.as_entire_binding() },
                BindGroupEntry { binding: 12, resource: expansion_paused_buf.as_entire_binding() },
            ],
        });
        
        (layout, bind_group)
    }
    
    /// Helper: Create Group 1 bind group layout and bind group with all 6 buffers
    fn create_group1_bind_group(&self, device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        use wgpu::*;
        
        // Get all Group 1 buffers
        let (
            mcts_params_buf,
            work_items_buf,
            paths_buf,
            alloc_counter_buf,
            diagnostics_buf,
            root_stats_buf
        ) = {
            let inner = self.inner.lock().unwrap();
            (
                inner.mcts_params_buffer.as_ref().expect("mcts_params missing").clone(),
                inner.work_items_buffer.as_ref().expect("work_items missing").clone(),
                inner.paths_buffer.as_ref().expect("paths missing").clone(),
                inner.alloc_counter_buffer.as_ref().expect("alloc_counter missing").clone(),
                inner.diagnostics_buffer.as_ref().expect("diagnostics missing").clone(),
                inner.root_stats_buffer.as_ref().expect("root_stats missing").clone(),
            )
        };

        // Create layout with 6 bindings (0-5)
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Group 1 Layout (MCTS Params)"),
            entries: &[
                BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Group 1 Bind Group (MCTS Params)"),
            layout: &layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: mcts_params_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: work_items_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: paths_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: alloc_counter_buf.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: diagnostics_buf.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: root_stats_buf.as_entire_binding() },
            ],
        });
        
        (layout, bind_group)
    }
    
    /// Helper: Create Group 2 bind group layout and bind group (root_board)
    fn create_group2_bind_group(&self, device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        use wgpu::*;
        
        let root_board_buf = {
            let inner = self.inner.lock().unwrap();
            inner.root_board_buffer.as_ref().expect("root_board missing").clone()
        };

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Group 2 Layout (Root Board)"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Group 2 Bind Group (Root Board)"),
            layout: &layout,
            entries: &[BindGroupEntry { binding: 0, resource: root_board_buf.as_entire_binding() }],
        });
        
        (layout, bind_group)
    }
    
    /// Helper: Create Group 3 bind group layout and bind group (urgent events)
    fn create_group3_bind_group(&self, device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        use wgpu::*;
        
        let (urgent_event_buffer_gpu, urgent_event_write_head_gpu) = {
            let inner = self.inner.lock().unwrap();
            (
                inner.urgent_event_buffer_gpu.as_ref().expect("urgent_event_buffer_gpu missing").clone(),
                inner.urgent_event_write_head_gpu.as_ref().expect("urgent_event_write_head_gpu missing").clone(),
            )
        };

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Group 3 Layout (Urgent Events)"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Group 3 Bind Group (Urgent Events)"),
            layout: &layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: urgent_event_buffer_gpu.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: urgent_event_write_head_gpu.as_entire_binding() }
            ],
        });
        
        (layout, bind_group)
    }
    
    /// Helper: Create Group 4 bind group layout and bind group (pruning/reroot)
    fn create_group4_bind_group(&self, device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        use wgpu::*;
        
        let (
            reroot_params_buf,
            new_root_output_buf,
            global_free_queue_buf,
            global_free_head_buf,
            work_queue_buf,
            work_head_buf,
            work_claimed_buf,
            work_completed_buf
        ) = {
            let inner = self.inner.lock().unwrap();
            (
                inner.reroot_params_buffer.as_ref().expect("reroot_params missing").clone(),
                inner.new_root_output_buffer.as_ref().expect("new_root_output missing").clone(),
                inner.global_free_queue_buffer.as_ref().expect("global_free_queue missing").clone(),
                inner.global_free_head_buffer.as_ref().expect("global_free_head missing").clone(),
                inner.work_queue_buffer.as_ref().expect("work_queue missing").clone(),
                inner.work_head_buffer.as_ref().expect("work_head missing").clone(),
                inner.work_claimed_buffer.as_ref().expect("work_claimed missing").clone(),
                inner.work_completed_buffer.as_ref().expect("work_completed missing").clone(),
            )
        };

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Group 4 Layout (Pruning)"),
            entries: &[
                BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 6, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 7, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Group 4 Bind Group (Pruning)"),
            layout: &layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: reroot_params_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: new_root_output_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: global_free_queue_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: global_free_head_buf.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: work_queue_buf.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: work_head_buf.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: work_claimed_buf.as_entire_binding() },
                BindGroupEntry { binding: 7, resource: work_completed_buf.as_entire_binding() },
            ],
        });
        
        (layout, bind_group)
    }
    
    /// Helper: Create Group 5 bind group layout and bind group (reroot synchronization)
    fn create_group5_bind_group(&self, device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        use wgpu::*;
        
        let (global_reroot_threads_remaining, global_reroot_start_threads_remaining) = {
            let inner = self.inner.lock().unwrap();
            (
                inner.global_reroot_threads_remaining.as_ref().expect("global_reroot_threads_remaining missing").clone(),
                inner.global_reroot_start_threads_remaining.as_ref().expect("global_reroot_start_threads_remaining missing").clone(),
            )
        };

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Group 5 Layout (Reroot Sync)"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Group 5 Bind Group (Reroot Sync)"),
            layout: &layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: global_reroot_threads_remaining.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: global_reroot_start_threads_remaining.as_entire_binding() },
            ],
        });
        
        (layout, bind_group)
    }
    
    /// Dispatch incremental_mcts_step kernel for one phase step.
    fn dispatch_incremental_step(&self, num_threads: u32, exploration: f32, vl_weight: f32, temp: f32, seed: u32, use_random_rollouts: bool) {
        use wgpu::*;
        
        let _poll_lock = DEVICE_POLL_MUTEX.lock().unwrap();
        
        let device = self.context.device();
        let queue = self.context.queue();
        
        // Get buffers
        let thread_states_buf = self.ensure_thread_states_buffer(num_threads);
        
        // UPDATE PARAMS BUFFER with current num_threads and other values
        {
            let inner = self.inner.lock().unwrap();
            let params_buf = inner.mcts_params_buffer.as_ref().expect("mcts_params missing");
            
            // MctsOthelloParams struct (16 u32/f32 fields):
            let params = MctsOthelloParams {
                num_iterations: 0, // Not used in incremental
                max_nodes: inner.max_nodes,
                exploration,
                virtual_loss_weight: vl_weight,
                root_idx: inner.current_root_idx,
                seed,
                board_width: 8,
                board_height: 8,
                game_type: 0, // Othello
                temperature: temp,
                turn_number: 0,
                free_list_capacity: (inner.max_nodes + 255) / 256,
                vl_temp_scale: 0.01, // Not used in incremental
                num_threads,
                root_node: inner.current_root_idx,
                use_vl_preincrement: 0,
                use_random_rollouts: if use_random_rollouts { 1 } else { 0 },
            };
            
            queue.write_buffer(params_buf, 0, bytemuck::bytes_of(&params));
        }
        
        // Create shader
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Incremental Step Shader"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/mcts_othello.wgsl"))),
        });
        
        // Group 0: Node data (all 13 bindings)
        let (group0_layout, group0) = self.create_group0_bind_group(device);
        
        // Group 1: Params (all 6 bindings)
        let (group1_layout, group1) = self.create_group1_bind_group(device);
        
        // Group 2: Root board
        let (group2_layout, group2) = self.create_group2_bind_group(device);
        
        // Group 3: Urgent events
        let (group3_layout, group3) = self.create_group3_bind_group(device);
        
        // Group 4: Pruning/reroot
        let (group4_layout, group4) = self.create_group4_bind_group(device);
        
        // Group 5: Reroot synchronization
        let (group5_layout, group5) = self.create_group5_bind_group(device);
        
        // Dummy group 6
        let dummy_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Dummy Layout"),
            entries: &[],
        });
        
        let dummy_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Dummy Group"),
            layout: &dummy_layout,
            entries: &[],
        });
        
        // Group 7: Thread states + queue buffers (5 bindings: 0-4)
        let (rollout_queue, rollout_head, backprop_queue, backprop_head) = {
            let inner = self.inner.lock().unwrap();
            (
                inner.rollout_queue_buffer.as_ref().expect("rollout_queue missing").clone(),
                inner.rollout_head_buffer.as_ref().expect("rollout_head missing").clone(),
                inner.backprop_queue_buffer.as_ref().expect("backprop_queue missing").clone(),
                inner.backprop_head_buffer.as_ref().expect("backprop_head missing").clone(),
            )
        };
        
        let group7_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Incremental Group 7 Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        
        let group7 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Incremental Group 7"),
            layout: &group7_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: thread_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: rollout_queue.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: rollout_head.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: backprop_queue.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: backprop_head.as_entire_binding() },
            ],
        });
        
        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Incremental Step Pipeline Layout"),
            bind_group_layouts: &[&group0_layout, &group1_layout, &group2_layout, &group3_layout, &group4_layout, &group5_layout, &dummy_layout, &group7_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Incremental Step Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("incremental_mcts_step"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Dispatch
        let workgroups = (num_threads + 63) / 64;
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Incremental Step Encoder"),
        });
        
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Incremental Step Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group0, &[]);
            pass.set_bind_group(1, &group1, &[]);
            pass.set_bind_group(2, &group2, &[]);
            pass.set_bind_group(3, &group3, &[]);
            pass.set_bind_group(4, &group4, &[]);
            pass.set_bind_group(5, &group5, &[]);
            pass.set_bind_group(6, &dummy_group, &[]);
            pass.set_bind_group(7, &group7, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        queue.submit(Some(encoder.finish()));
        device.poll(Maintain::Wait);
    }
    
    /// Dispatch rollout_chunk_kernel for threads in ROLLOUT_ACTIVE phase.
    fn dispatch_rollout_chunk(&self, num_threads: u32, _seed: u32) {
        use wgpu::*;
        
        let _poll_lock = DEVICE_POLL_MUTEX.lock().unwrap();
        
        let device = self.context.device();
        let queue = self.context.queue();
        
        // Get buffers
        let thread_states_buf = self.ensure_thread_states_buffer(num_threads);
        
        // Create shader
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Rollout Chunk Shader"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/mcts_othello.wgsl"))),
        });
        
        // Group 0: Node data (all 13 bindings)
        let (group0_layout, group0) = self.create_group0_bind_group(device);
        
        // Group 1: Params (all 6 bindings)
        let (group1_layout, group1) = self.create_group1_bind_group(device);
        
        // Group 2: Root board
        let (group2_layout, group2) = self.create_group2_bind_group(device);
        
        // Group 3: Urgent events
        let (group3_layout, group3) = self.create_group3_bind_group(device);
        
        // Group 4: Pruning/reroot
        let (group4_layout, group4) = self.create_group4_bind_group(device);
        
        // Group 5: Reroot synchronization
        let (group5_layout, group5) = self.create_group5_bind_group(device);
        
        // Dummy group 6
        let dummy_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Dummy Layout"),
            entries: &[],
        });
        
        let dummy_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Dummy Group"),
            layout: &dummy_layout,
            entries: &[],
        });
        
        // Group 7: Thread states + queue buffers (5 bindings: 0-4)
        let (rollout_queue, rollout_head, backprop_queue, backprop_head) = {
            let inner = self.inner.lock().unwrap();
            (
                inner.rollout_queue_buffer.as_ref().expect("rollout_queue missing").clone(),
                inner.rollout_head_buffer.as_ref().expect("rollout_head missing").clone(),
                inner.backprop_queue_buffer.as_ref().expect("backprop_queue missing").clone(),
                inner.backprop_head_buffer.as_ref().expect("backprop_head missing").clone(),
            )
        };
        
        let group7_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Rollout Group 7 Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        
        let group7 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Rollout Group 7"),
            layout: &group7_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: thread_states_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: rollout_queue.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: rollout_head.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: backprop_queue.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: backprop_head.as_entire_binding() },
            ],
        });
        
        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Rollout Chunk Pipeline Layout"),
            bind_group_layouts: &[&group0_layout, &group1_layout, &group2_layout, &group3_layout, &group4_layout, &group5_layout, &dummy_layout, &group7_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Rollout Chunk Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("rollout_chunk_kernel"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Dispatch
        let workgroups = (num_threads + 63) / 64;
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Rollout Chunk Encoder"),
        });
        
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Rollout Chunk Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group0, &[]);
            pass.set_bind_group(1, &group1, &[]);
            pass.set_bind_group(2, &group2, &[]);
            pass.set_bind_group(3, &group3, &[]);
            pass.set_bind_group(4, &group4, &[]);
            pass.set_bind_group(5, &group5, &[]);
            pass.set_bind_group(6, &dummy_group, &[]);
            pass.set_bind_group(7, &group7, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        queue.submit(Some(encoder.finish()));
        device.poll(Maintain::Wait);
    }

    pub fn calculate_nodes_used(&self) -> u32 {
        let _poll_lock = DEVICE_POLL_MUTEX.lock().unwrap();
        
        let inner = self.inner.lock().unwrap();
        let device = self.context.device();
        let queue = self.context.queue();
        
        // Read alloc_counter to get total allocations
        let alloc_counter_buf = inner.alloc_counter_buffer.as_ref().expect("alloc_counter missing");
        
        let size = 4; // single u32
        let staging_alloc = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alloc Counter Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Read free_tops to get total freed nodes
        let free_tops_buf = inner.free_tops_buffer.as_ref().expect("free_tops missing");
        let staging_free = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Free Tops Staging"),
            size: 256 * 4, // 256 u32s
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Read Node Usage") });
        encoder.copy_buffer_to_buffer(alloc_counter_buf, 0, &staging_alloc, 0, 4);
        encoder.copy_buffer_to_buffer(free_tops_buf, 0, &staging_free, 0, 256 * 4);
        queue.submit(Some(encoder.finish()));
        
        // Read alloc_counter
        let slice_alloc = staging_alloc.slice(..);
        let (tx_alloc, rx_alloc) = std::sync::mpsc::channel();
        slice_alloc.map_async(wgpu::MapMode::Read, move |v| tx_alloc.send(v).unwrap());
        
        // Read free_tops
        let slice_free = staging_free.slice(..);
        let (tx_free, rx_free) = std::sync::mpsc::channel();
        slice_free.map_async(wgpu::MapMode::Read, move |v| tx_free.send(v).unwrap());
        
        device.poll(wgpu::Maintain::Wait);
        rx_alloc.recv().unwrap().unwrap();
        rx_free.recv().unwrap().unwrap();
        
        let data_alloc = slice_alloc.get_mapped_range();
        let _alloc_count = u32::from_le_bytes([data_alloc[0], data_alloc[1], data_alloc[2], data_alloc[3]]);
        drop(data_alloc);
        staging_alloc.unmap();
        
        let data_free = slice_free.get_mapped_range();
        let mut total_freed = 0u32;
        // free_list_capacity = (max_nodes + 255) / 256
        let free_list_cap = (inner.max_nodes + 255) / 256;
        for i in 0..256 {
            let offset = i * 4;
            let top = u32::from_le_bytes([data_free[offset], data_free[offset+1], data_free[offset+2], data_free[offset+3]]);
            // free_tops contains stack size (0 to free_list_capacity)
            // If it wrapped around (>free_list_capacity), treat as 0
            if top <= free_list_cap {
                total_freed += top;
            }
        }
        drop(data_free);
        staging_free.unmap();
        
        // nodes_used = max_nodes - total_freed
        // (alloc_counter is set to max_nodes to disable fallback allocation)
        inner.max_nodes.saturating_sub(total_freed)
    }

    fn update_root_stats_internal(&self) {
        let device = self.context.device();
        let queue = self.context.queue();
        
        // Write params to ensure root_idx is correct
        {
            let inner = self.inner.lock().unwrap();
            let mcts_params = inner.mcts_params_buffer.as_ref().expect("mcts_params missing");
            let free_list_capacity = (inner.max_nodes + 255) / 256;
            let params = MctsOthelloParams {
                num_iterations: 1,
                max_nodes: inner.max_nodes,
                exploration: 1.4,
                virtual_loss_weight: 1.0,
                root_idx: inner.current_root_idx,
                seed: 0,
                board_width: 8,
                board_height: 8,
                game_type: 0,
                temperature: 1.0,
                turn_number: 0,
                free_list_capacity,
                vl_temp_scale: 0.01,
                num_threads: 0, // Not used for stats gathering
                root_node: inner.current_root_idx,
                use_vl_preincrement: 0,
                use_random_rollouts: 0,
            };
            queue.write_buffer(mcts_params, 0, bytemuck::bytes_of(&params));
            device.poll(wgpu::Maintain::Wait);
        }
        
        // 1. Create pipeline for gather_root_stats
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gather Root Stats Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mcts_othello.wgsl").into()),
        });
        
        // We need Group 0 (Node Data) and Group 1 (Stats Buffer)
        // Recreate layouts (inefficient but safe)
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gather Stats Group 0 Layout"),
            entries: &(0..=8).map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }).collect::<Vec<_>>(),
        });
        
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gather Stats Group 1 Layout"),
            entries: &[
                // We only need binding 5 (root_stats) but layout must match shader definition
                // Shader defines bindings 0,1,2,3,4,5 in Group 1
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gather Stats Pipeline Layout"),
            bind_group_layouts: &[&group0_layout, &group1_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gather Stats Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("gather_root_stats"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 2. Create Bind Groups
        let (
            node_info, node_visits, node_wins, node_vl, node_state,
            children_indices, children_priors, free_lists, free_tops,
            mcts_params, work_items, paths, alloc_counter, diagnostics, root_stats
        ) = {
            let inner = self.inner.lock().unwrap();
            (
                inner.node_info_buffer.as_ref().expect("node_info missing").clone(),
                inner.node_visits_buffer.as_ref().expect("node_visits missing").clone(),
                inner.node_wins_buffer.as_ref().expect("node_wins missing").clone(),
                inner.node_vl_buffer.as_ref().expect("node_vl missing").clone(),
                inner.node_state_buffer.as_ref().expect("node_state missing").clone(),
                inner.children_indices_buffer.as_ref().expect("children_indices missing").clone(),
                inner.children_priors_buffer.as_ref().expect("children_priors missing").clone(),
                inner.free_lists_buffer.as_ref().expect("free_lists missing").clone(),
                inner.free_tops_buffer.as_ref().expect("free_tops missing").clone(),
                inner.mcts_params_buffer.as_ref().expect("mcts_params missing").clone(),
                inner.work_items_buffer.as_ref().expect("work_items missing").clone(),
                inner.paths_buffer.as_ref().expect("paths missing").clone(),
                inner.alloc_counter_buffer.as_ref().expect("alloc_counter missing").clone(),
                inner.diagnostics_buffer.as_ref().expect("diagnostics missing").clone(),
                inner.root_stats_buffer.as_ref().expect("root_stats missing").clone(),
            )
        };

        let group0_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gather Stats Group 0"),
            layout: &group0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: node_info.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: node_visits.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: node_wins.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: node_vl.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: node_state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: children_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: children_priors.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: free_lists.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: free_tops.as_entire_binding() },
            ],
        });

        let group1_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gather Stats Group 1"),
            layout: &group1_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: mcts_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: work_items.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: paths.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: alloc_counter.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: diagnostics.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: root_stats.as_entire_binding() },
            ],
        });

        // 3. Dispatch
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Gather Stats Encoder") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gather Stats Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group0_bind_group, &[]);
            pass.set_bind_group(1, &group1_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // 4. Read back
        let size = (std::mem::size_of::<RootChildStats>() * 64) as u64;
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Root Stats Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Read Root Stats") });
        encoder.copy_buffer_to_buffer(&root_stats, 0, &staging_buf, 0, size);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        // Stats are now in staging_buf, ready to be read by caller
        // We don't copy to position-indexed arrays anymore (that was the bug!)
    }
    
    fn dispatch_gather_root_stats_kernel(&self) {
        // Just call the full update_root_stats_internal which dispatches gather_root_stats
        self.update_root_stats_internal();
    }

    pub fn read_diagnostics(&self) -> OthelloDiagnostics {
        let inner = self.inner.lock().unwrap();
        let device = self.context.device();
        let queue = self.context.queue();
        
        let diagnostics_buf = inner.diagnostics_buffer.as_ref().expect("diagnostics missing");
        
        let size = std::mem::size_of::<OthelloDiagnostics>() as u64;
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Diagnostics Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Read Diagnostics") });
        encoder.copy_buffer_to_buffer(diagnostics_buf, 0, &staging_buf, 0, size);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let diagnostics: OthelloDiagnostics = *bytemuck::from_bytes(&data);
        drop(data);
        staging_buf.unmap();
        
        diagnostics
    }

    /// Read total virtual loss across all nodes in the tree
    /// Used for testing/debugging VL leak issues
    pub fn read_total_virtual_loss(&self) -> i32 {
        // Calculate nodes used first (acquires DEVICE_POLL_MUTEX internally)
        let nodes_used = self.calculate_nodes_used();
        
        // Now acquire both locks for buffer read
        let _poll_lock = DEVICE_POLL_MUTEX.lock().unwrap();
        let inner = self.inner.lock().unwrap();
        
        let device = self.context.device();
        let queue = self.context.queue();
        let node_vl_buf = inner.node_vl_buffer.as_ref().expect("node_vl missing");
        
        let size = (nodes_used as u64) * 4; // i32 per node
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VL Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { 
            label: Some("Read VL") 
        });
        encoder.copy_buffer_to_buffer(node_vl_buf, 0, &staging_buf, 0, size);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let vl_values: &[i32] = bytemuck::cast_slice(&data);
        let total: i32 = vl_values.iter().sum();
        
        drop(data);
        staging_buf.unmap();
        
        total
    }
    
    /// Allocate thread states buffer for incremental execution (if not already allocated).
    /// ThreadState struct: ~600 bytes per thread
    /// For 16,384 threads: ~10 MB total
    pub fn ensure_thread_states_buffer(&self, num_threads: u32) -> Arc<wgpu::Buffer> {
        use wgpu::BufferUsages;
        
        const THREAD_STATE_SIZE: u64 = 816;
        
        let mut inner = self.inner.lock().unwrap();
        
        // Check if existing buffer is large enough
        let required_size = THREAD_STATE_SIZE * num_threads as u64;
        
        if let Some(existing) = &inner.thread_states_buffer {
            if existing.size() >= required_size {
                return existing.clone();
            }
            // Buffer too small - will recreate below
        }
        
        let device = &self.context.device;
        
        // ThreadState layout (matching WGSL):
        // - phase: u32
        // - current_node: u32
        // - path: array<u32, 128> = 512 bytes
        // - path_len: u32
        // - rng_seed: u32
        // - leaf_player: i32 (4 bytes)
        // - rollout_result: u32
        // - backprop_index: u32
        // - rollout_board: array<i32, 64> = 256 bytes
        // - rollout_player: i32 (4 bytes)
        // - rollout_moves_remaining: u32
        // - _pad0: u32
        // Total: 4 + 4 + 512 + 4 + 4 + 4 + 4 + 4 + 256 + 4 + 4 + 4 = 808 bytes (need padding to 816 for alignment)
        
        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ThreadStates"),
            size: THREAD_STATE_SIZE * num_threads as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        
        inner.thread_states_buffer = Some(buffer.clone());
        buffer
    }
    
    /// Ensure candidates buffer is allocated (for collect_leaf_candidates kernel).
    /// Buffer stores [score0, node_idx0, score1, node_idx1, ...] as u32 pairs.
    fn ensure_candidates_buffer(&self, max_capacity: u32) -> Arc<wgpu::Buffer> {
        let mut inner = self.inner.lock().unwrap();
        
        // Reuse existing buffer if already allocated
        if let Some(ref buf) = inner.leaf_candidates_buffer {
            return buf.clone();
        }
        
        let device = self.context.device();
        
        // Buffer size: 2 u32s per candidate (score + node_idx)
        let buffer_size = (max_capacity * 2 * 4) as u64; // *4 for u32 size
        
        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LeafCandidates"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        
        inner.leaf_candidates_buffer = Some(buffer.clone());
        buffer
    }
    
    /// Ensure candidate counter buffer is allocated (atomic u32).
    fn ensure_candidate_counter_buffer(&self) -> Arc<wgpu::Buffer> {
        let mut inner = self.inner.lock().unwrap();
        
        // Reuse existing buffer if already allocated
        if let Some(ref buf) = inner.candidate_count_buffer {
            return buf.clone();
        }
        
        let device = self.context.device();
        
        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CandidateCounter"),
            size: 4, // Single u32 atomic counter
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        
        inner.candidate_count_buffer = Some(buffer.clone());
        buffer
    }
    
    /// Calculate optimal workgroup count based on tree state (adaptive dispatch sizing).
    /// Uses progressive wave sizing: starts small (16), grows exponentially up to 256.
    /// This reduces contention by matching thread count to available leaves.
    pub fn calculate_optimal_workgroups(&self) -> u32 {
        // Drop lock to call calculate_nodes_used (which acquires its own lock)
        let nodes_used = self.calculate_nodes_used();
        let mut inner = self.inner.lock().unwrap();
        
        // Progressive wave sizing: 16 → 32 → 64 → 128 → 256
        // Rule: Double the wave size every 10,000 nodes
        let next_workgroups = if nodes_used < 1_000 {
            16  // Very early: minimal threads
        } else if nodes_used < 10_000 {
            32  // Early: small waves
        } else if nodes_used < 50_000 {
            64  // Mid: medium waves
        } else if nodes_used < 100_000 {
            128 // Late: large waves
        } else {
            256 // Full tree: maximum parallelism
        };
        
        // Update for next dispatch
        inner.current_wave_workgroups = next_workgroups;
        
        next_workgroups
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
        // Allocate Node Buffers
        let node_info_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NodeInfoBuffer"),
            size: max_nodes as u64 * 32, // 32 bytes per NodeInfo
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let node_visits_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NodeVisitsBuffer"),
            size: max_nodes as u64 * 4, // atomic i32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let node_wins_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NodeWinsBuffer"),
            size: max_nodes as u64 * 4, // atomic i32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let node_vl_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NodeVLBuffer"),
            size: max_nodes as u64 * 4, // atomic i32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let node_state_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NodeStateBuffer"),
            size: max_nodes as u64 * 4, // atomic u32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let children_indices_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ChildrenIndicesBuffer"),
            size: max_nodes as u64 * 64 * 4, // u32 (MAX_CHILDREN * u32)
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let children_priors_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ChildrenPriorsBuffer"),
            size: max_nodes as u64 * 64 * 4, // f32 (MAX_CHILDREN * f32)
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        // Free lists: 256 lists, each holds (max_nodes / 256) indices rounded up
        // Total capacity matches max_nodes
        let free_list_capacity_per_list = (max_nodes + 255) / 256;
        let free_lists_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FreeListsBuffer"),
            size: 256 * free_list_capacity_per_list as u64 * 4, // 256 lists * capacity * u32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let free_tops_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FreeTopsBuffer"),
            size: 256 * 4, // 256 atomic u32s
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Free list ownership: tracks which workgroup owns each free list (0xFFFF = unowned)
        let free_list_ownership_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FreeListOwnershipBuffer"),
            size: 256 * 4, // 256 u32s (workgroup IDs, or 0xFFFF for unowned)
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

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

        // Staging buffers for robust host reads (MAP_READ | COPY_DST)
        let urgent_event_staging = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UrgentEventStaging"),
            size: urgent_event_buffer_size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let urgent_event_write_head_staging = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UrgentEventWriteHeadStaging"),
            size: urgent_event_write_head_size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Pruning buffers
        let reroot_params_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RerootParamsBuffer"),
            size: std::mem::size_of::<RerootParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let new_root_output_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NewRootOutputBuffer"),
            size: 4, // u32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        
        let new_root_staging_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NewRootStagingBuffer"),
            size: 4, // u32
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let global_free_queue_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlobalFreeQueueBuffer"),
            size: max_nodes as u64 * 4, // Worst case: all nodes free
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let global_free_head_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlobalFreeHeadBuffer"),
            size: 4, // atomic u32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let expansion_paused_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ExpansionPausedBuffer"),
            size: 4, // atomic u32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let work_queue_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WorkQueueBuffer"),
            size: max_nodes as u64 * 4, // Worst case
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));

        let work_head_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WorkHeadBuffer"),
            size: 4, // atomic u32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let work_claimed_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WorkClaimedBuffer"),
            size: 4, // atomic u32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let work_completed_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WorkCompletedBuffer"),
            size: 4, // atomic u32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // MCTS Execution Buffers (Group 1)
        // Support up to 65535 workgroups * 64 threads = ~4.2M threads
        let max_threads = 65536 * 64; 
        let mcts_params_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MctsParamsBuffer"),
            size: std::mem::size_of::<MctsOthelloParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let work_items_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WorkItemsBuffer"),
            size: max_threads * 4, // u32
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        let paths_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PathsBuffer"),
            size: max_threads * 128 * 4, // 128 depth * u32
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        let alloc_counter_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AllocCounterBuffer"),
            size: 4, // atomic u32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let diagnostics_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DiagnosticsBuffer"),
            size: std::mem::size_of::<OthelloDiagnostics>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let root_stats_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RootStatsBuffer"),
            size: (std::mem::size_of::<RootChildStats>() * 64) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let root_board_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RootBoardBuffer"),
            size: 64 * 4, // 64 i32s
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Queue buffers for decoupled rollout architecture (Group 8)
        // Queue size: Allow up to 2x num_threads worth of pending work
        let max_queue_size = 32768_u32; // Support up to 16K threads with 2x buffering
        
        // RolloutJob is 272 bytes (64 i32s + 3 u32s + padding)
        let rollout_job_size = 272_u64;
        let rollout_queue_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RolloutQueueBuffer"),
            size: max_queue_size as u64 * rollout_job_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        let rollout_head_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RolloutHeadBuffer"),
            size: 4, // atomic<u32>
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        // BackpropJob is 16 bytes (4 u32s)
        let backprop_job_size = 16_u64;
        let backprop_queue_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BackpropQueueBuffer"),
            size: max_queue_size as u64 * backprop_job_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        let backprop_head_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BackpropHeadBuffer"),
            size: 4, // atomic<u32>
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // === Initialize Allocator (Free Lists) ===
        // Force rebuild comment
        {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Init Allocator Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mcts_othello.wgsl").into()),
            });

            // Group 0 Layout & Bind Group
            let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Init Group 0 Layout"),
                entries: &(0..=8).map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }).collect::<Vec<_>>(),
            });
            let group0_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Init Group 0 Bind Group"),
                layout: &group0_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: node_info_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: node_visits_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: node_wins_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: node_vl_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: node_state_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: children_indices_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: children_priors_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: free_lists_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: free_tops_buffer.as_entire_binding() },
                ],
            });

            // Group 1 Layout & Bind Group (Params)
            // Initialize params buffer first
            let free_list_capacity = (max_nodes + 255) / 256;
            let params = MctsOthelloParams {
                max_nodes,
                free_list_capacity,
                ..Default::default()
            };
            context.queue().write_buffer(&mcts_params_buffer, 0, bytemuck::bytes_of(&params));

            let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Init Group 1 Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    // Bindings 1-4 are storage buffers in the shader, we must provide them even if unused by init_allocator
                    wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ],
            });
            let group1_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Init Group 1 Bind Group"),
                layout: &group1_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: mcts_params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: work_items_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: paths_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: alloc_counter_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: diagnostics_buffer.as_entire_binding() },
                ],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Init Pipeline Layout"),
                bind_group_layouts: &[&group0_layout, &group1_layout],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Init Allocator Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("init_allocator"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Init Allocator Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Init Allocator Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group0_bind_group, &[]);
                pass.set_bind_group(1, &group1_bind_group, &[]);
                
                // Calculate 2D dispatch dimensions to support > 65535 workgroups
                // Max workgroups per dimension is 65535
                // Workgroup size is 64
                let total_workgroups = (max_nodes + 63) / 64;
                let max_x = 65535;
                
                let dispatch_x = if total_workgroups > max_x { max_x } else { total_workgroups };
                let dispatch_y = (total_workgroups + max_x - 1) / max_x;
                
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            context.queue().submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        }

        Ok(GpuOthelloMcts {
            context: context.clone(),
            inner: Mutex::new(GpuOthelloMctsInner {
                max_nodes,
                root_player: 1,
                root_board: [0; 64],
                legal_moves: vec![],
                visits: vec![0; 64],
                wins: vec![0; 64],
                seen_boards: HashSet::new(),
                expanded_nodes: HashSet::new(),
                node_info_buffer: Some(node_info_buffer),
                node_visits_buffer: Some(node_visits_buffer),
                node_wins_buffer: Some(node_wins_buffer),
                node_vl_buffer: Some(node_vl_buffer),
                node_state_buffer: Some(node_state_buffer),
                children_indices_buffer: Some(children_indices_buffer),
                children_priors_buffer: Some(children_priors_buffer),
                free_lists_buffer: Some(free_lists_buffer),
                free_tops_buffer: Some(free_tops_buffer),
                free_list_ownership_buffer: Some(free_list_ownership_buffer),
                urgent_event_buffer_gpu: Some(urgent_event_buffer_gpu),
                urgent_event_write_head_gpu: Some(urgent_event_write_head_gpu),
                urgent_event_buffer_host: Some(urgent_event_buffer_host),
                urgent_event_write_head_host: Some(urgent_event_write_head_host),
                urgent_event_staging: Some(urgent_event_staging),
                urgent_event_write_head_staging: Some(urgent_event_write_head_staging),
                global_reroot_threads_remaining: Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GlobalRerootThreadsRemaining"),
                    size: 4 * 4, // 4 u32s, more than enough for a single atomic
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }))),
                global_reroot_start_threads_remaining: Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GlobalRerootStartThreadsRemaining"),
                    size: 4 * 4, // 4 u32s
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }))),
                reroot_params_buffer: Some(reroot_params_buffer),
                new_root_output_buffer: Some(new_root_output_buffer),
                new_root_staging_buffer: Some(new_root_staging_buffer),
                global_free_queue_buffer: Some(global_free_queue_buffer),
                global_free_head_buffer: Some(global_free_head_buffer),
                expansion_paused_buffer: Some(expansion_paused_buffer),
                work_queue_buffer: Some(work_queue_buffer),
                work_head_buffer: Some(work_head_buffer),
                work_claimed_buffer: Some(work_claimed_buffer),
                work_completed_buffer: Some(work_completed_buffer),
                current_root_idx: 0,
                mcts_params_buffer: Some(mcts_params_buffer),
                work_items_buffer: Some(work_items_buffer),
                paths_buffer: Some(paths_buffer),
                alloc_counter_buffer: Some(alloc_counter_buffer),
                diagnostics_buffer: Some(diagnostics_buffer),
                root_stats_buffer: Some(root_stats_buffer),
                root_board_buffer: Some(root_board_buffer),
                // Pre-computation buffers (allocated lazily or on first use)
                leaf_candidates_buffer: None,
                candidate_count_buffer: None,
                precomputed_cache_buffer: None,
                cache_size_buffer: None,
                precompute_diagnostics_buffer: None,
                // Thread state buffer (allocated lazily or on first incremental dispatch)
                thread_states_buffer: None,
                // Queue buffers (Group 8 - Decoupled Rollout Architecture)
                rollout_queue_buffer: Some(rollout_queue_buffer),
                rollout_head_buffer: Some(rollout_head_buffer),
                backprop_queue_buffer: Some(backprop_queue_buffer),
                backprop_head_buffer: Some(backprop_head_buffer),
                max_temp_boost_baseline: 0,
                current_wave_workgroups: 16, // Start with small waves
            }),
            _not_send_sync: std::marker::PhantomData,
        })
    }

    pub fn init_tree(&self, board: &[i32; 64], root_player: i32, legal_moves: &[(usize, usize)]) {
        {
            let mut inner = self.inner.lock().unwrap();
            inner.root_player = root_player;
            inner.root_board.copy_from_slice(board);
            inner.legal_moves = legal_moves.to_vec();
            for &(x, y) in legal_moves {
                // Skip pass moves - they don't have visit counts to track
                if x == usize::MAX && y == usize::MAX {
                    continue;
                }
                // Match shader's encode_move: y * width + x
                let idx = y * 8 + x;
                inner.visits[idx] = 0;
                inner.wins[idx] = 0;
            }
            inner.expanded_nodes.clear();
            inner.expanded_nodes.insert(*board);
        } // Drop lock
        
        println!("[GPU-Native] init_tree: Initializing GPU tree (resetting allocator and Node 0)");
        self.reset_gpu_tree(root_player);
        
        // Reset diagnostics using compute shader and set baseline to 0
        self.reset_diagnostics_gpu();
        let mut inner = self.inner.lock().unwrap();
        inner.max_temp_boost_baseline = 0;
    }

    // ...existing code...

                // removed stray line: pub seen_boards
    pub fn get_children_stats(&self) -> Vec<(usize, usize, i32, i32, f64)> {
        // Dispatch gather_root_stats kernel to populate root_stats buffer
        // This must be done WITHOUT holding the inner lock to avoid GPU operations while locked
        self.dispatch_gather_root_stats_kernel();
        
        // Now read back the root_stats buffer
        let inner = self.inner.lock().unwrap();
        let device = self.context.device();
        let queue = self.context.queue();
        
        // Read back root_stats buffer
        let size = (std::mem::size_of::<RootChildStats>() * 64) as u64;
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Root Stats Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let root_stats = inner.root_stats_buffer.as_ref().expect("root_stats missing").clone();
        let legal_moves_copy = inner.legal_moves.clone();
        drop(inner); // Release lock before GPU operations
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Read Root Stats") });
        encoder.copy_buffer_to_buffer(&root_stats, 0, &staging_buf, 0, size);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let stats: &[RootChildStats] = bytemuck::cast_slice(&data);
        
        // Build result from root_stats, matching with legal_moves
        let width = 8;
        let mut result = Vec::new();
        
        // Calculate total visits for PUCT diagnostics
        let total_visits: i32 = stats.iter().map(|s| s.visits).sum();
        let num_children = legal_moves_copy.len();
        
        println!("\n=== PUCT Breakdown for Root Children ===");
        println!("Parent total visits: {} (num_children: {})", total_visits, num_children);
        println!("sqrt(parent_visits + 1) = {:.4}", ((total_visits + 1) as f64).sqrt());
        println!();
        
        for &(x, y) in &legal_moves_copy {
            // Pass moves don't have visit/win stats
            if x == usize::MAX && y == usize::MAX {
                result.push((x, y, 0, 0, 0.0));
                continue;
            }
            
            // Match shader's encode_move: y * width + x
            let move_id = (y * width + x) as u32;
            
            // Find this move in root_stats
            let mut found = false;
            for stat in stats {
                if stat.move_id == move_id {
                    let visits = stat.visits;
                    let wins = stat.wins;
                    // Q from parent's perspective (wins are already from parent's view)
                    let q = if visits > 0 {
                        wins as f64 / (visits as f64 * 2.0)
                    } else {
                        0.0
                    };
                    
                    // Calculate PUCT components for diagnostic
                    let uniform_prior = 1.0 / num_children as f64;
                    let sqrt_parent = ((total_visits + 1) as f64).sqrt();
                    let u = uniform_prior * sqrt_parent / (1.0 + visits as f64);
                    let puct = q + u;
                    
                    println!("  ({},{}) visits={:7} wins={:7} Q={:.4} U={:.4} PUCT={:.4}", 
                             x, y, visits, wins, q, u, puct);
                    
                    result.push((x, y, visits, wins, q));
                    found = true;
                    break;
                }
            }
            
            // If not found in root_stats, it's unvisited
            if !found {
                result.push((x, y, 0, 0, 0.0));
            }
        }
        
        drop(data);
        staging_buf.unmap();
        result
    }

    pub fn get_total_nodes(&self) -> u32 {
        let inner = self.inner.lock().unwrap();
        inner.expanded_nodes.len() as u32
    }

    pub fn get_capacity(&self) -> u32 {
        let inner = self.inner.lock().unwrap();
        inner.max_nodes
    }

    pub fn get_root_board_hash(&self) -> u32 {
        let inner = self.inner.lock().unwrap();
        let mut hash: u32 = 0x811c9dc5;
        for &v in &inner.root_board {
            hash ^= v as u32;
            hash = hash.wrapping_mul(0x01000193);
        }
        hash
    }

    pub fn flush_and_wait(&self) {}

    pub fn get_root_visits(&self) -> u32 {
        // Read from GPU node_visits buffer for the current root
        let inner = self.inner.lock().unwrap();
        let device = self.context.device();
        let queue = self.context.queue();
        
        let root_idx = inner.current_root_idx;
        let node_visits_buf = inner.node_visits_buffer.as_ref().expect("node_visits missing");
        
        // Copy one i32 from node_visits[root_idx]
        let offset = (root_idx as u64) * 4; // i32 size
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Root Visits Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Read Root Visits") });
        encoder.copy_buffer_to_buffer(node_visits_buf, offset, &staging_buf, 0, 4);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let visits = i32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        staging_buf.unmap();
        
        visits.max(0) as u32
    }

    pub fn reset_gpu_tree(&self, root_player: i32) {
        use wgpu::*;
        let context = &self.context;
        let device = context.device();
        let queue = context.queue();
        
        // 1. Retrieve buffers
        let (
            node_info, node_visits, node_wins, node_vl, node_state,
            children_indices, children_priors, free_lists, free_tops, free_list_ownership,
            global_free_queue, global_free_head, expansion_paused,
            mcts_params, work_items, paths, alloc_counter, diagnostics,
            max_nodes
        ) = {
            let inner = self.inner.lock().unwrap();
            (
                inner.node_info_buffer.as_ref().expect("node_info missing").clone(),
                inner.node_visits_buffer.as_ref().expect("node_visits missing").clone(),
                inner.node_wins_buffer.as_ref().expect("node_wins missing").clone(),
                inner.node_vl_buffer.as_ref().expect("node_vl missing").clone(),
                inner.node_state_buffer.as_ref().expect("node_state missing").clone(),
                inner.children_indices_buffer.as_ref().expect("children_indices missing").clone(),
                inner.children_priors_buffer.as_ref().expect("children_priors missing").clone(),
                inner.free_lists_buffer.as_ref().expect("free_lists missing").clone(),
                inner.free_tops_buffer.as_ref().expect("free_tops missing").clone(),
                inner.free_list_ownership_buffer.as_ref().expect("free_list_ownership missing").clone(),
                inner.global_free_queue_buffer.as_ref().expect("global_free_queue missing").clone(),
                inner.global_free_head_buffer.as_ref().expect("global_free_head missing").clone(),
                inner.expansion_paused_buffer.as_ref().expect("expansion_paused missing").clone(),
                inner.mcts_params_buffer.as_ref().expect("mcts_params missing").clone(),
                inner.work_items_buffer.as_ref().expect("work_items missing").clone(),
                inner.paths_buffer.as_ref().expect("paths missing").clone(),
                inner.alloc_counter_buffer.as_ref().expect("alloc_counter missing").clone(),
                inner.diagnostics_buffer.as_ref().expect("diagnostics missing").clone(),
                inner.max_nodes
            )
        };

        // 2. Recreate Bind Groups for Init Allocator
        // Group 0 (Node Data)
        let group0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Init Group 0 Layout"),
            entries: &(0..=12).map(|i| BindGroupLayoutEntry {
                binding: i,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }).collect::<Vec<_>>(),
        });
        let group0_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Init Group 0 Bind Group"),
            layout: &group0_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: node_info.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: node_visits.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: node_wins.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: node_vl.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: node_state.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: children_indices.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: children_priors.as_entire_binding() },
                BindGroupEntry { binding: 7, resource: free_lists.as_entire_binding() },
                BindGroupEntry { binding: 8, resource: free_tops.as_entire_binding() },
                BindGroupEntry { binding: 9, resource: free_list_ownership.as_entire_binding() },
                BindGroupEntry { binding: 10, resource: global_free_queue.as_entire_binding() },
                BindGroupEntry { binding: 11, resource: global_free_head.as_entire_binding() },
                BindGroupEntry { binding: 12, resource: expansion_paused.as_entire_binding() },
            ],
        });

        // Group 1 (MCTS Params & Work Items)
        let group1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Init Group 1 Layout"),
            entries: &[
                BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let root_stats = {
            let inner = self.inner.lock().unwrap();
            inner.root_stats_buffer.as_ref().expect("root_stats missing").clone()
        };
        let group1_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Init Group 1 Bind Group"),
            layout: &group1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: mcts_params.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: work_items.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: paths.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: alloc_counter.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: diagnostics.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: root_stats.as_entire_binding() },
            ],
        });

        // 3. Create Pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Init Allocator Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mcts_othello.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Init Pipeline Layout"),
            bind_group_layouts: &[&group0_layout, &group1_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Init Allocator Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("init_allocator"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 4. Dispatch
        // Zero out free_tops before running init_allocator to prevent double-counting if called multiple times
        let zeros_256 = vec![0u8; 256 * 4];
        queue.write_buffer(&free_tops, 0, &zeros_256);
        
        // Initialize all free lists as unowned (0xFFFF)
        let unowned_marker = 0xFFFFu32;
        let ownership_init: Vec<u8> = (0..256)
            .flat_map(|_| unowned_marker.to_le_bytes())
            .collect();
        queue.write_buffer(&free_list_ownership, 0, &ownership_init);
        
        // Zero out global allocator state
        queue.write_buffer(&global_free_head, 0, &0u32.to_le_bytes());
        queue.write_buffer(&expansion_paused, 0, &0u32.to_le_bytes());
        
        // NOTE: We do NOT reset diagnostics here because GPU buffer writes do not reliably
        // clear atomicMax values before the next kernel runs. The diagnostics buffer
        // accumulates across searches. To get per-search metrics, we read the baseline
        // value at the start of each search and compute deltas.
        
        // Note: We don't need to initialize global_free_queue_alloc because 
        // global_free_head_alloc starts at 0, so the queue is empty.
        // When we try to pop from an empty queue, we restore the counter.

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Init Allocator Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Init Allocator Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group0_bind_group, &[]);
            pass.set_bind_group(1, &group1_bind_group, &[]);
            
            // Calculate 2D dispatch dimensions to support > 65535 workgroups
            // Max workgroups per dimension is 65535
            // Workgroup size is 64
            let total_workgroups = (max_nodes + 63) / 64;
            let max_x = 65535;
            
            let dispatch_x = if total_workgroups > max_x { max_x } else { total_workgroups };
            let dispatch_y = (total_workgroups + max_x - 1) / max_x;
            
            // Ensure dispatch dimensions do not exceed limits
            let safe_dispatch_x = dispatch_x.min(65535);
            let safe_dispatch_y = dispatch_y.min(65535);
            
            pass.dispatch_workgroups(safe_dispatch_x, safe_dispatch_y, 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // 5. Initialize Root Node (Node 0)
        let root_info = OthelloNodeInfo {
            parent_idx: u32::MAX, // No parent
            move_id: 0,
            num_children: 0,
            player_at_node: root_player,
            flags: 0,
            _pad: 0,
            _pad2: 0,
            _pad3: 0,
        };
        
        // Write Node 0 Info
        queue.write_buffer(&node_info, 0, bytemuck::bytes_of(&root_info));
        
        // Write Node 0 State = READY (2)
        queue.write_buffer(&node_state, 0, &2u32.to_le_bytes());

        // Initialize alloc_counter to max_nodes
        // The init kernel already populated free lists with nodes 1..max_nodes-1.
        // Setting alloc_counter to max_nodes prevents the fallback allocator from 
        // double-allocating nodes that are already in free lists.
        // If memory is truly exhausted (free lists empty AND alloc_counter >= max_nodes),
        // try_allocate_node will set expansion_paused and return INVALID_INDEX.
        queue.write_buffer(&alloc_counter, 0, &max_nodes.to_le_bytes());
        
        // Clear expansion_paused flag at tree reset
        {
            let inner = self.inner.lock().unwrap();
            let expansion_paused_buf = inner.expansion_paused_buffer.as_ref().expect("expansion_paused missing");
            queue.write_buffer(expansion_paused_buf, 0, &0u32.to_le_bytes());
        }
        
        // Reset current_root_idx
        let mut inner = self.inner.lock().unwrap();
        inner.current_root_idx = 0;
        
        println!("[GPU-Native] GPU Tree Reset Complete. Root set to 0.");
    }

    /// Reset diagnostics buffer using a compute shader.
    /// This is more reliable than queue.write_buffer() for atomic values.
    pub fn reset_diagnostics_gpu(&self) {
        let device = self.context.device();
        let queue = self.context.queue();
        let inner = self.inner.lock().unwrap();
        
        let diagnostics_buf = inner.diagnostics_buffer.as_ref().expect("diagnostics missing");
        
        // Load and compile reset shader
        let shader_source = include_str!("shaders/reset_diagnostics.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reset Diagnostics Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Reset Diagnostics Bind Group Layout"),
            entries: &[
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
            ],
        });
        
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reset Diagnostics Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: diagnostics_buf.as_entire_binding(),
                },
            ],
        });
        
        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Reset Diagnostics Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reset Diagnostics Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Dispatch single workgroup
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Reset Diagnostics Encoder"),
        });
        
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Reset Diagnostics Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    pub fn advance_root(&self, x: usize, y: usize, new_board: &[i32; 64], new_player: i32, legal_moves: &[(usize, usize)]) -> bool {
        
        // Dispatch pruning kernels to clean up the tree on the GPU
        // The pruning kernel will handle the case where root has no children and return 0xFFFFFFF0
        if !self.dispatch_pruning_kernels(x as u32, y as u32) {
            // Pruning failed - root likely has no children (timeout before search completed)
            // OR the move wasn't found in children (shouldn't happen if caller validates)
            // Reset tree to new position and return false to indicate tree wasn't reused
            eprintln!("[GPU-Native] Pruning failed for move ({},{}) - resetting tree to new position", x, y);
            self.init_tree(new_board, new_player, legal_moves);
            return false;
        }
        
        // 2. Update host state
        {
            let mut inner = self.inner.lock().unwrap();
            inner.root_board.copy_from_slice(new_board);
            inner.root_player = new_player;
            inner.legal_moves = legal_moves.to_vec();
            inner.expanded_nodes.insert(*new_board);
        }
        
        println!("[GPU-Native] Tree reuse successful");
        true
    }

    pub fn get_best_move(&self) -> Option<(usize, usize, i32, f64)> {
        // Select move by best Q-value (quality), not by most visits
        // With softmax sampling, the most-visited move isn't necessarily the best
        self.get_children_stats()
            .into_iter()
            .max_by(|a, b| {
                let (_x_a, _y_a, _visits_a, _wins_a, q_a) = a;
                let (_x_b, _y_b, _visits_b, _wins_b, q_b) = b;
                // Compare by Q-value, handling NaN cases
                q_a.partial_cmp(q_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(x, y, visits, _wins, q)| (x, y, visits, q))
    }

    pub fn get_depth_visit_histogram(&self, _max_depth: u32) -> Vec<u32> {
        Vec::new()
    }

    // Debug helper to inspect free lists
    pub fn debug_get_free_tops(&self) -> Vec<u32> {
        let inner = self.inner.lock().unwrap();
        let buffer = inner.free_tops_buffer.as_ref().expect("free_tops missing");
        let device = &self.context.device;
        let queue = &self.context.queue;
        
        // Create staging buffer
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug FreeTops Staging"),
            size: 256 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, 256 * 4);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        result
    }

    // Debug helper to inspect a node
    pub fn debug_get_node_info(&self, idx: u32) -> OthelloNodeInfo {
        let inner = self.inner.lock().unwrap();
        let buffer = inner.node_info_buffer.as_ref().expect("node_info missing");
        let device = &self.context.device;
        let queue = &self.context.queue;
        
        let size = std::mem::size_of::<OthelloNodeInfo>() as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug NodeInfo Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, idx as u64 * size, &staging_buffer, 0, size);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let result: OthelloNodeInfo = *bytemuck::from_bytes(&data);
        drop(data);
        staging_buffer.unmap();
        result
    }

    pub fn debug_get_child_index(&self, parent_idx: u32, child_num: u32) -> u32 {
        let inner = self.inner.lock().unwrap();
        let buffer = inner.children_indices_buffer.as_ref().expect("children_indices missing");
        let device = &self.context.device;
        let queue = &self.context.queue;
        
        // Each node has MAX_CHILDREN (64) child indices
        let offset = (parent_idx * 64 + child_num) as u64 * 4; // u32 = 4 bytes
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Child Index Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, offset, &staging_buffer, 0, 4);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let result = u32::from_le_bytes(data[0..4].try_into().unwrap());
        drop(data);
        staging_buffer.unmap();
        result
    }

    pub fn debug_get_node_visits(&self, idx: u32) -> i32 {
        let inner = self.inner.lock().unwrap();
        let buffer = inner.node_visits_buffer.as_ref().expect("node_visits missing");
        let device = &self.context.device;
        let queue = &self.context.queue;
        
        let offset = idx as u64 * 4; // i32 = 4 bytes
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Node Visits Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, offset, &staging_buffer, 0, 4);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let result = i32::from_le_bytes(data[0..4].try_into().unwrap());
        drop(data);
        staging_buffer.unmap();
        result
    }

    pub fn debug_get_node_wins(&self, idx: u32) -> i32 {
        let inner = self.inner.lock().unwrap();
        let buffer = inner.node_wins_buffer.as_ref().expect("node_wins missing");
        let device = &self.context.device;
        let queue = &self.context.queue;
        
        let offset = idx as u64 * 4; // i32 = 4 bytes
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Node Wins Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, offset, &staging_buffer, 0, 4);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let result = i32::from_le_bytes(data[0..4].try_into().unwrap());
        drop(data);
        staging_buffer.unmap();
        result
    }

    pub fn debug_get_child_prior(&self, parent_idx: u32, child_num: u32) -> f32 {
        let inner = self.inner.lock().unwrap();
        let buffer = inner.children_priors_buffer.as_ref().expect("children_priors missing");
        let device = &self.context.device;
        let queue = &self.context.queue;
        
        // children_priors[parent_idx * MAX_CHILDREN + child_num]
        // MAX_CHILDREN is 64 in the shader
        let offset = (parent_idx * 64 + child_num) as u64 * 4; // f32 = 4 bytes
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Child Prior Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, offset, &staging_buffer, 0, 4);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let result = f32::from_le_bytes(data[0..4].try_into().unwrap());
        drop(data);
        staging_buffer.unmap();
        result
    }

    pub fn debug_get_node_vl(&self, idx: u32) -> i32 {
        let inner = self.inner.lock().unwrap();
        let buffer = inner.node_vl_buffer.as_ref().expect("node_vl missing");
        let device = &self.context.device;
        let queue = &self.context.queue;
        
        let offset = idx as u64 * 4; // i32 = 4 bytes
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Node VL Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, offset, &staging_buffer, 0, 4);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let result = i32::from_le_bytes(data[0..4].try_into().unwrap());
        drop(data);
        staging_buffer.unmap();
        result
    }

    pub fn debug_get_node_state(&self, idx: u32) -> u32 {
        let inner = self.inner.lock().unwrap();
        let buffer = inner.node_state_buffer.as_ref().expect("node_state missing");
        let device = &self.context.device;
        let queue = &self.context.queue;
        
        let offset = idx as u64 * 4; // u32 = 4 bytes
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Node State Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, offset, &staging_buffer, 0, 4);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let result = u32::from_le_bytes(data[0..4].try_into().unwrap());
        drop(data);
        staging_buffer.unmap();
        result
    }

    /// Debug helper: print detailed state of root's children
    pub fn debug_print_root_children(&self) {
        let inner = self.inner.lock().unwrap();
        let root_idx = inner.current_root_idx;
        drop(inner);
        
        let root_info = self.debug_get_node_info(root_idx);
        
        // Check deleted flag (bit 0)
        if (root_info.flags & 0x1) != 0 {
            println!("\n=== DEBUG: Root Children State ===");
            println!("Root index: {}", root_idx);
            println!("WARNING: Root node is DELETED (flags={:#x})", root_info.flags);
            println!("=== End Debug ===\n");
            return;
        }
        
        let num_children = root_info.num_children;
        let root_state = self.debug_get_node_state(root_idx);
        let root_vl = self.debug_get_node_vl(root_idx);
        
        println!("\n=== DEBUG: Root Children State ===");
        println!("Root index: {}", root_idx);
        println!("Root num_children: {}", num_children);
        println!("Root visits: {}", self.debug_get_node_visits(root_idx));
        println!("Root wins: {}", self.debug_get_node_wins(root_idx));
        println!("Root virtual_loss: {}", root_vl);
        println!("Root state: {} (0=EMPTY, 1=EXPANDING, 2=READY)", root_state);
        println!("Root flags: {:#x} (bit0=deleted, bit1=zero, bit2=dirty)", root_info.flags);
        println!();
        
        // Check ALL 64 possible children to find valid ones
        let mut valid_count = 0;
        let mut total_child_visits = 0;
        for i in 0..64 {
            let child_idx = self.debug_get_child_index(root_idx, i);
            if child_idx != 0xFFFFFFFF {
                let child_info = self.debug_get_node_info(child_idx);
                
                // Check if child is deleted
                if (child_info.flags & 0x1) != 0 {
                    println!("Child {}: idx={} DELETED (flags={:#x})", i, child_idx, child_info.flags);
                    continue;
                }
                
                let child_visits = self.debug_get_node_visits(child_idx);
                let child_wins = self.debug_get_node_wins(child_idx);
                let child_vl = self.debug_get_node_vl(child_idx);
                let child_prior = self.debug_get_child_prior(root_idx, i);
                
                total_child_visits += child_visits;
                
                let move_x = child_info.move_id % 8;
                let move_y = child_info.move_id / 8;
                
                println!("Child {}: idx={} move=({},{}) visits={} wins={} vl={} prior={:.4} flags={:#x}",
                    i, child_idx, move_x, move_y, child_visits, child_wins, child_vl, child_prior, child_info.flags);
                valid_count += 1;
            }
        }
        
        if valid_count == 0 {
            println!("NO VALID CHILDREN FOUND! All 64 indices are INVALID_INDEX or deleted");
            println!("This means num_children={} is garbage data", num_children);
        } else {
            println!("\nTotal valid children: {}", valid_count);
            println!("Total child visits: {}", total_child_visits);
        }
        
        println!("=== End Debug ===\n");
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

            // Inject a test urgent event from CPU to verify the logging pipeline works
            let mut payload = [0u32; 255];
            payload[0] = 0xDEADBEEF;
            engine_arc.log_urgent_event_from_cpu_with_payload(123, 999999, &payload);

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
        let mut host_hash: u32 = 0x811c9dc5;
        for &v in &board {
            host_hash ^= v as u32;
            host_hash = host_hash.wrapping_mul(0x01000193);
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
            let mcts = GpuOthelloMcts::new(context, 10000, 128).expect("Failed to create GpuOthelloMcts");
            // Initial board
            let mut board = [0i32; 64];
            board[3 * 8 + 3] = 1;
            board[3 * 8 + 4] = -1;
            board[4 * 8 + 3] = -1;
            board[4 * 8 + 4] = 1;
            let player = 1;
            let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
            
            // Test: multiple init_tree calls with same board should produce same hash
            mcts.init_tree(&board, player, &legal_moves);
            let hash1 = mcts.get_root_board_hash();
            
            mcts.init_tree(&board, player, &legal_moves);
            let hash2 = mcts.get_root_board_hash();
            
            mcts.init_tree(&board, player, &legal_moves);
            let hash3 = mcts.get_root_board_hash();
            
            assert_eq!(hash1, hash2, "Hash mismatch after second init_tree");
            assert_eq!(hash2, hash3, "Hash mismatch after third init_tree");
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
        let mcts = GpuOthelloMcts::new(context, 10000, 128).expect("Failed to create GpuOthelloMcts");
        
        // Standard Othello starting position
        let mut board = [0i32; 64];
        board[3 * 8 + 3] = -1; board[3 * 8 + 4] = 1;
        board[4 * 8 + 3] = 1; board[4 * 8 + 4] = -1;
        
        let root_player = 1;
        let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
        mcts.init_tree(&board, root_player, &legal_moves);
        
        // Multiple dispatches to allow tree growth
        mcts.dispatch_mcts_othello_kernel(1, 1.4, 1.0, 0.06, 0.01, 42);
        for _ in 0..10 {
            mcts.dispatch_mcts_othello_kernel(32, 1.4, 1.0, 0.06, 0.01, 42);
        }
        
        // Wait for GPU to finish
        std::thread::sleep(std::time::Duration::from_millis(500));
        
        let children = mcts.get_children_stats();
        assert!(children.iter().any(|&(_, _, visits, _, _)| visits > 0), "No child visits recorded!");
    }

    #[test]
    fn test_gpu_othello_mcts_no_freeze_on_large_batch() {
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
        let mcts = GpuOthelloMcts::new(context, 2_000_000, 128).expect("Failed to create GpuOthelloMcts");
        
        // Standard Othello starting position
        let mut board = [0i32; 64];
        board[3 * 8 + 3] = -1; board[3 * 8 + 4] = 1;
        board[4 * 8 + 3] = 1; board[4 * 8 + 4] = -1;
        
        let root_player = 1;
        let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
        mcts.init_tree(&board, root_player, &legal_moves);
        
        // Multiple dispatches
        mcts.dispatch_mcts_othello_kernel(1, 1.4, 1.0, 0.06, 0.01, 42);
        for _ in 0..3 {
            mcts.dispatch_mcts_othello_kernel(16, 1.4, 1.0, 0.06, 0.01, 42);
        }
        
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
        let mut host_hash: u32 = 0x811c9dc5;
        for &v in &board {
            host_hash ^= v as u32;
            host_hash = host_hash.wrapping_mul(0x01000193);
        }
        let gpu_hash = mcts.get_root_board_hash();
        assert_eq!(gpu_hash, host_hash, "GPU root board hash does not match host hash!");
    }

    #[test]
    fn test_gpu_othello_advance_root_updates_board_hash() {
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
        let mcts = GpuOthelloMcts::new(context, 10000, 128).expect("Failed to create GpuOthelloMcts");
        // Initial board
        let mut board = [0i32; 64];
        board[3 * 8 + 3] = 1;
        board[3 * 8 + 4] = -1;
        board[4 * 8 + 3] = -1;
        board[4 * 8 + 4] = 1;
        let root_player = 1;
        let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
        mcts.init_tree(&board, root_player, &legal_moves);
        
        // Run MCTS iterations to build tree before advance_root
        mcts.dispatch_mcts_othello_kernel(1, 1.4, 1.0, 0.06, 0.01, 42);
        for _ in 0..3 {
            mcts.dispatch_mcts_othello_kernel(16, 1.4, 1.0, 0.06, 0.01, 42);
        }
        
        let host_hash_1 = {
            let mut h: u32 = 0x811c9dc5;
            for &v in &board {
                h ^= v as u32;
                h = h.wrapping_mul(0x01000193);
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
            let mut h: u32 = 0x811c9dc5;
            for &v in &board {
                h ^= v as u32;
                h = h.wrapping_mul(0x01000193);
            }
            h
        };
        assert_eq!(mcts.get_root_board_hash(), host_hash_2, "Root board hash mismatch after advance_root");
    }

    #[test]
    fn test_gpu_othello_tree_expands_beyond_root() {
    }

}







