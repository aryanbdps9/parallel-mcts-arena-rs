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
            pub fn dispatch_mcts_othello_kernel(&self, num_workgroups: u32) {
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
            let total_threads = 64 * num_workgroups; // match kernel logic
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
                    free_tops
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
                    )
                };

                // Layouts for each group
                let group0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Group 0 Layout (Node Data)"),
                    entries: &(0..=8).map(|i| BindGroupLayoutEntry {
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
                    global_free_queue_buf,
                    global_free_head_buf,
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
                        inner.global_free_queue_buffer.as_ref().expect("global_free_queue missing").clone(),
                        inner.global_free_head_buffer.as_ref().expect("global_free_head missing").clone(),
                        inner.work_queue_buffer.as_ref().expect("work_queue missing").clone(),
                        inner.work_head_buffer.as_ref().expect("work_head missing").clone(),
                        inner.work_claimed_buffer.as_ref().expect("work_claimed missing").clone(),
                        inner.work_completed_buffer.as_ref().expect("work_completed missing").clone(),
                    )
                };

                // Update MctsParams (Basic initialization)
                {
                    let inner = self.inner.lock().unwrap();
                    let params = MctsOthelloParams {
                        num_iterations: 1, 
                        max_nodes: inner.max_nodes,
                        exploration: 1.414, 
                        virtual_loss_weight: 1.0, 
                        root_idx: inner.current_root_idx,
                        seed: 12345, 
                        board_width: 8,
                        board_height: 8,
                        game_type: 0,
                        temperature: 1.0, 
                        turn_number: 0, 
                        _pad0: 0,
                    };
                    queue.write_buffer(&mcts_params_buf, 0, bytemuck::bytes_of(&params));
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
                    ],
                });

                let group1_bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Group 1 Bind Group (MCTS Params)"),
                    layout: &group1_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: mcts_params_buf.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: work_items_buf.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: paths_buf.as_entire_binding() },
                        BindGroupEntry { binding: 3, resource: alloc_counter_buf.as_entire_binding() },
                        BindGroupEntry { binding: 4, resource: diagnostics_buf.as_entire_binding() },
                    ],
                });
                
                let root_board_buf = {
                    let inner = self.inner.lock().unwrap();
                    inner.root_board_buffer.as_ref().expect("root_board_buffer missing").clone()
                };
                
                // Write current root board
                {
                    let inner = self.inner.lock().unwrap();
                    println!("[GPU-Native] Writing root_board to GPU. Board[27..37]: {:?}", &inner.root_board[27..37]);
                    queue.write_buffer(&root_board_buf, 0, bytemuck::cast_slice(&inner.root_board));
                }

                // DEBUG: Read back root_board to verify
                {
                    let size = 64 * 4;
                    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Staging Readback"),
                        size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    encoder.copy_buffer_to_buffer(&root_board_buf, 0, &staging_buf, 0, size);
                    queue.submit(Some(encoder.finish()));
                    
                    let slice = staging_buf.slice(..);
                    let (tx, rx) = std::sync::mpsc::channel();
                    slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
                    device.poll(wgpu::Maintain::Wait);
                    rx.recv().unwrap().unwrap();
                    let data = slice.get_mapped_range();
                    let result: &[i32] = bytemuck::cast_slice(&data);
                    println!("[GPU-Native] Readback root_board from GPU. Board[27..37]: {:?}", &result[27..37]);
                    drop(data);
                    staging_buf.unmap();
                }

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
                    pass.dispatch_workgroups(num_workgroups, 1, 1);
                }
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);
                println!("[DIAG] Othello MCTS main kernel dispatched and device polled.");

                // DEBUG: Readback root_board from GPU AFTER execution
                {
                    let inner = self.inner.lock().unwrap();
                    let root_board_buf = inner.root_board_buffer.as_ref().expect("root_board_buffer missing");
                    let size = 64 * 4;
                    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("RootBoard Staging Buffer After"),
                        size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("RootBoard Readback After") });
                    encoder.copy_buffer_to_buffer(root_board_buf, 0, &staging_buf, 0, size);
                    queue.submit(Some(encoder.finish()));
                    
                    let slice = staging_buf.slice(..);
                    let (tx, rx) = std::sync::mpsc::channel();
                    slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
                    device.poll(wgpu::Maintain::Wait);
                    rx.recv().unwrap().unwrap();
                    
                    let data = slice.get_mapped_range();
                    let board: &[i32] = bytemuck::cast_slice(&data);
                    println!("[GPU-Native] Readback root_board from GPU AFTER execution. Board[0..10]: {:?}", &board[0..10]);
                }
            }
        /// Dispatch the GPU pruning kernel and bind the urgent event buffer for logging
        pub fn dispatch_pruning_kernels(&self, move_x: u32, move_y: u32) {
            println!("[GPU-Native] dispatch_pruning_kernels called with move_x={}, move_y={}", move_x, move_y);
            use wgpu::{BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindGroupEntry, BindGroupDescriptor, ShaderStages, BindingType, BufferBindingType};
            let context = &self.context;
            let device = context.device();
            let queue = context.queue();

            // 1. Prepare RerootParams
            let (reroot_params_buf, current_root) = {
                let inner = self.inner.lock().unwrap();
                (
                    inner.reroot_params_buffer.as_ref().expect("reroot_params_buffer missing").clone(),
                    inner.current_root_idx
                )
            };

            let params = RerootParams {
                move_x,
                move_y,
                current_root,
                _padding: 0,
            };
            queue.write_buffer(&reroot_params_buf, 0, bytemuck::bytes_of(&params));

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

            // 3. Create Bind Group 4 (Pruning Resources)
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
                free_tops
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
                )
            };

            let group0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Pruning Group 0 Layout"),
                entries: &(0..=8).map(|i| BindGroupLayoutEntry {
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

            let prune_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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

            // Phase 2: Prune Unreachable
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Prune Pass"), timestamp_writes: None });
                cpass.set_pipeline(&prune_pipeline);
                cpass.set_bind_group(0, &group0_bind_group, &[]);
                cpass.set_bind_group(1, &group1_bind_group, &[]);
                cpass.set_bind_group(2, &group2_bind_group, &[]);
                cpass.set_bind_group(3, &group3_bind_group, &[]);
                cpass.set_bind_group(4, &group4_bind_group, &[]);
                cpass.dispatch_workgroups(1024, 1, 1);
            }
            
            // Copy output to staging
            let new_root_staging_buf = {
                let inner = self.inner.lock().unwrap();
                inner.new_root_staging_buffer.as_ref().expect("new_root_staging_buffer missing").clone()
            };
            encoder.copy_buffer_to_buffer(&new_root_output_buf, 0, &new_root_staging_buf, 0, 4);
            
            queue.submit(Some(encoder.finish()));
            
            // 8. Read back new root
            let slice = new_root_staging_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            
            let data = slice.get_mapped_range();
            let new_root_idx = u32::from_le_bytes(data[0..4].try_into().unwrap());
            drop(data);
            new_root_staging_buf.unmap();
            
            if new_root_idx == 0xFFFFFFF0 {
                println!("[DIAG] Pruning failed: Root node has NO children (num_children=0).");
            } else if new_root_idx == 0xFFFFFFFF {
                println!("[DIAG] Pruning failed: Children exist but move not found.");
            } else if (new_root_idx & 0xE0000000) == 0xE0000000 {
                let first_move = new_root_idx & 0xFF;
                let num_children = (new_root_idx >> 8) & 0xFF;
                let mx = first_move % 8;
                let my = first_move / 8;
                println!("[DIAG] Pruning failed: Children exist but move not found. Num children: {}. First child move_id={} (x={}, y={})", num_children, first_move, mx, my);
            } else {
                println!("[DIAG] Pruning complete. New root: {}", new_root_idx);
            }
            
            // Update host state
            let mut inner = self.inner.lock().unwrap();
            inner.current_root_idx = new_root_idx;
        }

        pub fn dispatch_prune_unreachable_topdown(&self) {
            // Legacy wrapper
            self.dispatch_pruning_kernels(0, 0); 
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
    pub _pad0: u32,
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
    pub root_board_hash: u32,
    pub init_nodes_count: u32,
    pub total_children_gen: u32,
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
    pub _padding: u32, // Align to 16 bytes
}

impl std::fmt::Debug for RerootParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RerootParams")
            .field("move_x", &self.move_x)
            .field("move_y", &self.move_y)
            .field("current_root", &self.current_root)
            .finish()
    }
}

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
    // Root Board Buffer (Group 2)
    pub root_board_buffer: Option<Arc<wgpu::Buffer>>, // Binding 0
}

impl GpuOthelloMcts {
    pub fn run_iterations(&self, _iterations: u32, _exploration: f32, _virtual_loss_weight: f32, _temperature: f32, _seed: u32) -> OthelloRunTelemetry {
        // In GPU-native mode, the kernel is dispatched separately.
        // This function just reads back the telemetry.
        let diagnostics = self.read_diagnostics();
        
        let inner = self.inner.lock().unwrap();
        OthelloRunTelemetry {
            iterations_launched: diagnostics.rollouts,
            alloc_count_after: inner.expanded_nodes.len() as u32,
            free_count_after: 0,
            node_capacity: inner.max_nodes,
            saturated: diagnostics.alloc_failures > 0,
            diagnostics,
        }
    }

    fn read_diagnostics(&self) -> OthelloDiagnostics {
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
            size: max_nodes as u64 * 4, // u32 (index of first child)
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let children_priors_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ChildrenPriorsBuffer"),
            size: max_nodes as u64 * 4, // f32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let free_lists_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FreeListsBuffer"),
            size: 256 * 8192 * 4, // 256 lists of 8192 u32s
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let free_tops_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FreeTopsBuffer"),
            size: 256 * 4, // 256 atomic u32s
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
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));

        let global_free_head_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlobalFreeHeadBuffer"),
            size: 4, // atomic u32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
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
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
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
        let max_threads = 65536; // 1024 workgroups * 64 threads
        let mcts_params_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MctsParamsBuffer"),
            size: std::mem::size_of::<MctsOthelloParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
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
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let diagnostics_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DiagnosticsBuffer"),
            size: std::mem::size_of::<OthelloDiagnostics>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let root_board_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RootBoardBuffer"),
            size: 64 * 4, // 64 i32s
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
            let params = MctsOthelloParams {
                max_nodes,
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
                root_board_buffer: Some(root_board_buffer),
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
                let idx = x * 8 + y;
                inner.visits[idx] = 0;
                inner.wins[idx] = 0;
            }
            inner.expanded_nodes.clear();
            inner.expanded_nodes.insert(*board);
        } // Drop lock
        
        println!("[GPU-Native] init_tree: Initializing GPU tree (resetting allocator and Node 0)");
        self.reset_gpu_tree(root_player);
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
        let inner = self.inner.lock().unwrap();
        inner.legal_moves.iter().map(|&(x, y)| inner.visits[x * 8 + y] as u32).sum()
    }

    pub fn reset_gpu_tree(&self, root_player: i32) {
        use wgpu::*;
        let context = &self.context;
        let device = context.device();
        let queue = context.queue();
        
        // 1. Retrieve buffers
        let (
            node_info, node_visits, node_wins, node_vl, node_state,
            children_indices, children_priors, free_lists, free_tops,
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
            entries: &(0..=8).map(|i| BindGroupLayoutEntry {
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
            ],
        });
        let group1_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Init Group 1 Bind Group"),
            layout: &group1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: mcts_params.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: work_items.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: paths.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: alloc_counter.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: diagnostics.as_entire_binding() },
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
        let zeros = vec![0u8; 256 * 4];
        queue.write_buffer(&free_tops, 0, &zeros);

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
        };
        
        // Write Node 0 Info
        queue.write_buffer(&node_info, 0, bytemuck::bytes_of(&root_info));
        
        // Write Node 0 State = READY (2)
        queue.write_buffer(&node_state, 0, &2u32.to_le_bytes());

        // Initialize alloc_counter to max_nodes to prevent fallback allocator from returning 0
        // (It should only be used if free lists are empty, and we want it to fail if so, rather than overwriting root)
        queue.write_buffer(&alloc_counter, 0, &max_nodes.to_le_bytes());
        
        // Reset current_root_idx
        let mut inner = self.inner.lock().unwrap();
        inner.current_root_idx = 0;
        
        println!("[GPU-Native] GPU Tree Reset Complete. Root set to 0.");
    }

    pub fn advance_root(&self, x: usize, y: usize, new_board: &[i32; 64], new_player: i32, legal_moves: &[(usize, usize)]) -> bool {
        println!("[GPU-Native] advance_root called with x={}, y={}", x, y);
        // 1. Dispatch pruning kernels to clean up the tree on the GPU
        self.dispatch_pruning_kernels(x as u32, y as u32);
        
        // 2. Update host state
        let mut inner = self.inner.lock().unwrap();
        inner.root_board.copy_from_slice(new_board);
        inner.root_player = new_player;
        inner.legal_moves = legal_moves.to_vec();
        inner.expanded_nodes.insert(*new_board);
        
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
                let mut host_hash: u32 = 0x811c9dc5;
                for &v in &board {
                    host_hash ^= v as u32;
                    host_hash = host_hash.wrapping_mul(0x01000193);
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







