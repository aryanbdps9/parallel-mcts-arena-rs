//! GPU MCTS Accelerator
//!
//! This module provides high-level GPU-accelerated operations for MCTS,
//! including batch PUCT calculation and expansion decision making.

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, Buffer, BufferUsages,
    CommandEncoderDescriptor, util::DeviceExt,
};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

use super::context::{GpuContext, GpuError};

/// Data structure representing a node for GPU PUCT calculation
///
/// This is the GPU-compatible representation of node statistics
/// used for batch PUCT score computation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuNodeData {
    /// Number of visits to this node
    pub visits: i32,
    /// Accumulated wins (scaled: 2=win, 1=draw, 0=loss per visit)
    pub wins: i32,
    /// Virtual losses for parallel coordination
    pub virtual_losses: i32,
    /// Parent node's visit count
    pub parent_visits: i32,
    /// Prior probability (typically uniform = 1/num_children)
    pub prior_prob: f32,
    /// Exploration parameter (C_puct)
    pub exploration: f32,
    /// Padding for alignment
    pub _padding: [f32; 2],
}

impl GpuNodeData {
    /// Creates a new GpuNodeData instance
    pub fn new(
        visits: i32,
        wins: i32,
        virtual_losses: i32,
        parent_visits: i32,
        prior_prob: f32,
        exploration: f32,
    ) -> Self {
        Self {
            visits,
            wins,
            virtual_losses,
            parent_visits,
            prior_prob,
            exploration,
            _padding: [0.0; 2],
        }
    }
}

/// Result of GPU PUCT calculation
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuPuctResult {
    /// Calculated PUCT score
    pub puct_score: f32,
    /// Q value (exploitation term)
    pub q_value: f32,
    /// Exploration term
    pub exploration_term: f32,
    /// Original node index
    pub node_index: u32,
}

/// Input data for expansion decision
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuExpansionInput {
    /// Node depth in tree
    pub depth: u32,
    /// Current visit count
    pub visits: i32,
    /// 1 if node has no children, 0 otherwise
    pub is_leaf: u32,
    /// 1 if game state is terminal, 0 otherwise
    pub is_terminal: u32,
}

/// Output of expansion decision
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuExpansionOutput {
    /// 1 if should expand, 0 otherwise
    pub should_expand: u32,
    /// Priority for expansion
    pub expansion_priority: f32,
    /// Original node index
    pub node_index: u32,
    /// Padding
    pub _padding: u32,
}

/// Path node data for backpropagation
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuPathNode {
    /// Player who made the move leading to this node
    pub player_who_moved: i32,
    /// Game winner (-1 for no winner/draw)
    pub winner: i32,
    /// 1 if game ended in draw
    pub is_draw: u32,
    /// Padding
    pub _padding: u32,
}

/// Backpropagation update data
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuBackpropUpdate {
    /// How much to add to visits (always 1)
    pub visit_delta: i32,
    /// Reward to add: 2=win, 1=draw, 0=loss
    pub reward: i32,
    /// Index of the node to update
    pub node_index: u32,
    /// Padding
    pub _padding: u32,
}

/// Uniform parameters for GPU compute shaders
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuParams {
    /// Number of elements to process
    pub num_elements: u32,
    /// Reserved parameter 1 (max_nodes for expansion)
    pub param1: u32,
    /// Reserved parameter 2 (current_nodes for expansion)
    pub param2: u32,
    /// Reserved parameter 3 (random seed for expansion)
    pub param3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuSimulationParams {
    pub board_width: u32,
    pub board_height: u32,
    pub current_player: i32,
    pub use_heuristic: u32,
    pub seed: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuSimulationResult {
    pub score: f32,
}

/// Result of batch expansion decision
#[derive(Debug)]
pub struct BatchExpansionResult {
    /// Indices of nodes that should be expanded
    pub nodes_to_expand: Vec<usize>,
    /// Expansion priorities for the selected nodes
    pub priorities: Vec<f32>,
}

/// GPU MCTS Accelerator for batch operations
///
/// This struct orchestrates GPU-accelerated MCTS operations,
/// managing buffer allocations and compute dispatches.
pub struct GpuMctsAccelerator {
    /// The GPU context with device and pipelines
    context: Arc<GpuContext>,
    /// Pre-allocated staging buffer for PUCT input
    puct_input_buffer: Option<Buffer>,
    /// Pre-allocated staging buffer for PUCT output
    puct_output_buffer: Option<Buffer>,
    /// Pre-allocated staging buffer for reading results
    puct_staging_buffer: Option<Buffer>,
    /// Current capacity of pre-allocated buffers
    current_capacity: usize,
    
    // Reusable buffers for simulation
    sim_input_buffer: Option<Buffer>,
    sim_output_buffer: Option<Buffer>,
    sim_staging_buffer: Option<Buffer>,
    sim_params_buffer: Option<Buffer>,
    sim_bind_group: Option<wgpu::BindGroup>,
    sim_current_capacity: usize,

    /// Statistics: total GPU compute time in microseconds
    total_gpu_time_us: u64,
    /// Statistics: number of GPU dispatches
    dispatch_count: u64,
}

impl GpuMctsAccelerator {
    /// Creates a new GPU MCTS accelerator
    ///
    /// # Arguments
    /// * `context` - The GPU context to use for operations
    pub fn new(context: Arc<GpuContext>) -> Self {
        Self {
            context,
            puct_input_buffer: None,
            puct_output_buffer: None,
            puct_staging_buffer: None,
            current_capacity: 0,
            sim_input_buffer: None,
            sim_output_buffer: None,
            sim_staging_buffer: None,
            sim_params_buffer: None,
            sim_bind_group: None,
            sim_current_capacity: 0,
            total_gpu_time_us: 0,
            dispatch_count: 0,
        }
    }

    /// Ensures buffers are allocated with sufficient capacity
    fn ensure_puct_buffers(&mut self, num_nodes: usize) {
        if self.current_capacity >= num_nodes && self.puct_input_buffer.is_some() {
            return; // Existing buffers are sufficient
        }

        // Round up to next power of 2 for efficient reallocation
        let new_capacity = num_nodes.next_power_of_two().max(256);
        
        let input_size = (new_capacity * std::mem::size_of::<GpuNodeData>()) as u64;
        let output_size = (new_capacity * std::mem::size_of::<GpuPuctResult>()) as u64;

        self.puct_input_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("PUCT Input Buffer"),
            size: input_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.puct_output_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("PUCT Output Buffer"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        self.puct_staging_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("PUCT Staging Buffer"),
            size: output_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.current_capacity = new_capacity;
    }

    /// Computes PUCT scores for a batch of nodes on the GPU
    ///
    /// This is the main entry point for GPU-accelerated move selection.
    /// It takes node statistics and returns PUCT scores computed in parallel.
    ///
    /// # Arguments
    /// * `nodes` - Slice of node data to compute PUCT scores for
    ///
    /// # Returns
    /// * `Ok(Vec<GpuPuctResult>)` - PUCT results for each input node
    /// * `Err(GpuError)` - If computation fails
    pub fn compute_puct_batch(&mut self, nodes: &[GpuNodeData]) -> Result<Vec<GpuPuctResult>, GpuError> {
        let num_nodes = nodes.len();
        if num_nodes == 0 {
            return Ok(Vec::new());
        }

        // Check if batch is too small for GPU benefit
        if num_nodes < self.context.config().min_batch_threshold {
            return self.compute_puct_cpu_fallback(nodes);
        }

        let start_time = std::time::Instant::now();

        self.ensure_puct_buffers(num_nodes);

        let input_buffer = self.puct_input_buffer.as_ref().unwrap();
        let output_buffer = self.puct_output_buffer.as_ref().unwrap();
        let staging_buffer = self.puct_staging_buffer.as_ref().unwrap();

        // Write input data to GPU buffer
        self.context.queue().write_buffer(
            input_buffer,
            0,
            bytemuck::cast_slice(nodes),
        );

        // Create params uniform buffer
        let params = GpuParams {
            num_elements: num_nodes as u32,
            param1: 0,
            param2: 0,
            param3: 0,
        };
        let params_buffer = self.context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PUCT Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = self.context.device().create_bind_group(&BindGroupDescriptor {
            label: Some("PUCT Bind Group"),
            layout: self.context.puct_bind_group_layout(),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch compute
        let mut encoder = self.context.device().create_command_encoder(&CommandEncoderDescriptor {
            label: Some("PUCT Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PUCT Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(self.context.puct_pipeline());
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch with 256 threads per workgroup
            let workgroups = (num_nodes as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy results to staging buffer for CPU read
        let output_size = (num_nodes * std::mem::size_of::<GpuPuctResult>()) as u64;
        encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, output_size);

        // Submit and wait
        self.context.submit_and_wait(encoder.finish());

        // Read results from staging buffer
        let results = self.read_puct_results(staging_buffer, num_nodes)?;

        // Update statistics
        let elapsed = start_time.elapsed();
        self.total_gpu_time_us += elapsed.as_micros() as u64;
        self.dispatch_count += 1;

        Ok(results)
    }

    /// Reads PUCT results from the staging buffer
    fn read_puct_results(&self, staging_buffer: &Buffer, num_nodes: usize) -> Result<Vec<GpuPuctResult>, GpuError> {
        let buffer_slice = staging_buffer.slice(..);
        
        // Map the buffer for reading
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Poll the device until the buffer is mapped
        self.context.device().poll(wgpu::Maintain::Wait);
        
        // Wait for the mapping to complete
        pollster::block_on(rx)
            .map_err(|_| GpuError::BufferError("Buffer mapping cancelled".to_string()))?
            .map_err(|e| GpuError::BufferError(format!("Buffer mapping failed: {:?}", e)))?;

        // Read the data
        let data = buffer_slice.get_mapped_range();
        let results: Vec<GpuPuctResult> = bytemuck::cast_slice(&data[..num_nodes * std::mem::size_of::<GpuPuctResult>()])
            .to_vec();
        
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }

    /// CPU fallback for small batches where GPU overhead exceeds benefit
    fn compute_puct_cpu_fallback(&self, nodes: &[GpuNodeData]) -> Result<Vec<GpuPuctResult>, GpuError> {
        Ok(nodes.iter().enumerate().map(|(idx, node)| {
            let visits = node.visits;
            let virtual_losses = node.virtual_losses;
            let effective_visits = visits + virtual_losses;

            let parent_visits_sqrt = (node.parent_visits as f32).sqrt();
            
            let (q_value, exploration_term, puct_score) = if effective_visits == 0 {
                let exploration_term = node.exploration * node.prior_prob * parent_visits_sqrt;
                (0.0, exploration_term, exploration_term)
            } else {
                let effective_visits_f = effective_visits as f32;
                let q_value = if visits > 0 {
                    (node.wins as f32 / visits as f32) / 2.0
                } else {
                    0.0
                };
                let exploration_term = node.exploration * node.prior_prob * parent_visits_sqrt / (1.0 + effective_visits_f);
                (q_value, exploration_term, q_value + exploration_term)
            };

            GpuPuctResult {
                puct_score,
                q_value,
                exploration_term,
                node_index: idx as u32,
            }
        }).collect())
    }

    /// Computes expansion decisions for a batch of nodes
    ///
    /// Determines which leaf nodes should be expanded based on depth,
    /// visit count, and tree capacity constraints.
    ///
    /// # Arguments
    /// * `inputs` - Slice of expansion input data
    /// * `max_nodes` - Maximum nodes allowed in tree
    /// * `current_nodes` - Current number of nodes in tree
    /// * `random_seed` - Random seed for probabilistic expansion
    ///
    /// # Returns
    /// * `Ok(BatchExpansionResult)` - Nodes selected for expansion
    /// * `Err(GpuError)` - If computation fails
    pub fn compute_expansion_batch(
        &self,
        inputs: &[GpuExpansionInput],
        max_nodes: u32,
        current_nodes: u32,
        random_seed: u32,
    ) -> Result<BatchExpansionResult, GpuError> {
        let num_nodes = inputs.len();
        if num_nodes == 0 {
            return Ok(BatchExpansionResult {
                nodes_to_expand: Vec::new(),
                priorities: Vec::new(),
            });
        }

        // For small batches, use CPU
        if num_nodes < self.context.config().min_batch_threshold {
            return self.compute_expansion_cpu_fallback(inputs, max_nodes, current_nodes, random_seed);
        }

        // Create input buffer
        let input_buffer = self.context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Expansion Input Buffer"),
            contents: bytemuck::cast_slice(inputs),
            usage: BufferUsages::STORAGE,
        });

        // Create output buffer
        let output_size = (num_nodes * std::mem::size_of::<GpuExpansionOutput>()) as u64;
        let output_buffer = self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Expansion Output Buffer"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer
        let staging_buffer = self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Expansion Staging Buffer"),
            size: output_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params = GpuParams {
            num_elements: num_nodes as u32,
            param1: max_nodes,
            param2: current_nodes,
            param3: random_seed,
        };
        let params_buffer = self.context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Expansion Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = self.context.device().create_bind_group(&BindGroupDescriptor {
            label: Some("Expansion Bind Group"),
            layout: self.context.expansion_bind_group_layout(),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch
        let mut encoder = self.context.device().create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Expansion Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Expansion Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(self.context.expansion_pipeline());
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroups = (num_nodes as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        self.context.submit_and_wait(encoder.finish());

        // Read results
        let outputs = self.read_expansion_results(&staging_buffer, num_nodes)?;

        // Filter nodes that should be expanded
        let mut nodes_to_expand = Vec::new();
        let mut priorities = Vec::new();
        for output in outputs {
            if output.should_expand != 0 {
                nodes_to_expand.push(output.node_index as usize);
                priorities.push(output.expansion_priority);
            }
        }

        Ok(BatchExpansionResult {
            nodes_to_expand,
            priorities,
        })
    }

    /// Reads expansion results from staging buffer
    fn read_expansion_results(&self, staging_buffer: &Buffer, num_nodes: usize) -> Result<Vec<GpuExpansionOutput>, GpuError> {
        let buffer_slice = staging_buffer.slice(..);
        
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.context.device().poll(wgpu::Maintain::Wait);
        
        pollster::block_on(rx)
            .map_err(|_| GpuError::BufferError("Buffer mapping cancelled".to_string()))?
            .map_err(|e| GpuError::BufferError(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let results: Vec<GpuExpansionOutput> = bytemuck::cast_slice(&data[..num_nodes * std::mem::size_of::<GpuExpansionOutput>()])
            .to_vec();
        
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }

    /// CPU fallback for expansion decisions
    fn compute_expansion_cpu_fallback(
        &self,
        inputs: &[GpuExpansionInput],
        max_nodes: u32,
        current_nodes: u32,
        random_seed: u32,
    ) -> Result<BatchExpansionResult, GpuError> {
        let tree_capacity_available = current_nodes < max_nodes;
        
        let mut nodes_to_expand = Vec::new();
        let mut priorities = Vec::new();

        for (idx, input) in inputs.iter().enumerate() {
            let is_expandable = input.is_leaf != 0 && input.is_terminal == 0;
            
            if !tree_capacity_available || !is_expandable {
                continue;
            }

            // Root always expands
            if input.depth == 0 {
                nodes_to_expand.push(idx);
                priorities.push(1000.0);
                continue;
            }

            // Probabilistic expansion
            let depth_factor = 1.0 / (1.0 + input.depth as f32 * 0.5);
            let visit_factor = (input.visits as f32).sqrt() / 10.0;
            let expansion_probability = (depth_factor + visit_factor).min(1.0);

            // Simple pseudo-random
            let rand_val = Self::simple_rand(random_seed, idx as u32);
            
            if rand_val < expansion_probability {
                nodes_to_expand.push(idx);
                priorities.push(expansion_probability * (input.visits + 1) as f32);
            }
        }

        Ok(BatchExpansionResult {
            nodes_to_expand,
            priorities,
        })
    }

    /// Simple pseudo-random function matching the shader implementation
    fn simple_rand(seed: u32, idx: u32) -> f32 {
        let x = seed ^ (idx.wrapping_mul(1103515245).wrapping_add(12345));
        let y = x ^ (x >> 16);
        let z = y.wrapping_mul(0x85ebca6b);
        let w = z ^ (z >> 13);
        let v = w.wrapping_mul(0xc2b2ae35);
        let result = v ^ (v >> 16);
        result as f32 / u32::MAX as f32
    }

    /// Finds the node with the maximum PUCT score using GPU reduction
    ///
    /// # Arguments
    /// * `results` - PUCT results to find maximum from
    ///
    /// # Returns
    /// * `Ok(GpuPuctResult)` - The result with maximum PUCT score
    /// * `Err(GpuError)` - If reduction fails
    pub fn find_max_puct(&self, results: &[GpuPuctResult]) -> Result<GpuPuctResult, GpuError> {
        if results.is_empty() {
            return Err(GpuError::ComputeError("Empty results array".to_string()));
        }

        if results.len() == 1 {
            return Ok(results[0]);
        }

        // For small arrays, use CPU
        if results.len() < 1024 {
            return Ok(results.iter()
                .max_by(|a, b| a.puct_score.partial_cmp(&b.puct_score).unwrap())
                .copied()
                .unwrap());
        }

        // Use GPU reduction for large arrays
        let mut current_results = results.to_vec();
        
        while current_results.len() > 1 {
            let num_elements = current_results.len();
            let workgroups = (num_elements + 255) / 256;

            // Create input buffer
            let input_buffer = self.context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Max Reduction Input"),
                contents: bytemuck::cast_slice(&current_results),
                usage: BufferUsages::STORAGE,
            });

            // Create output buffer
            let output_size = (workgroups * std::mem::size_of::<GpuPuctResult>()) as u64;
            let output_buffer = self.context.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Max Reduction Output"),
                size: output_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Create staging buffer
            let staging_buffer = self.context.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Max Reduction Staging"),
                size: output_size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Create params
            let params = GpuParams {
                num_elements: num_elements as u32,
                param1: 0,
                param2: 0,
                param3: 0,
            };
            let params_buffer = self.context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Max Reduction Params"),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM,
            });

            // Create bind group
            let bind_group = self.context.device().create_bind_group(&BindGroupDescriptor {
                label: Some("Max Reduction Bind Group"),
                layout: self.context.max_reduction_bind_group_layout(),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            // Dispatch
            let mut encoder = self.context.device().create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Max Reduction Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Max Reduction Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(self.context.max_reduction_pipeline());
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
            self.context.submit_and_wait(encoder.finish());

            // Read results
            current_results = self.read_puct_results(&staging_buffer, workgroups)?;
        }

        Ok(current_results[0])
    }

    /// Computes Gomoku heuristic evaluation for a batch of boards
    pub fn simulate_batch(
        &mut self,
        board_data: &[i32],
        params: GpuSimulationParams,
    ) -> Result<Vec<f32>, GpuError> {
        let start_time = std::time::Instant::now();
        let num_boards = board_data.len() as u32 / (params.board_width * params.board_height);
        
        if num_boards == 0 {
            return Ok(Vec::new());
        }

        // Ensure buffers are large enough
        let required_capacity = num_boards as usize;
        if self.sim_current_capacity < required_capacity || self.sim_input_buffer.is_none() {
            let new_capacity = required_capacity.next_power_of_two().max(256);
            let board_size = (params.board_width * params.board_height) as usize;
            
            let input_size = (new_capacity * board_size * std::mem::size_of::<i32>()) as u64;
            let output_size = (new_capacity * std::mem::size_of::<GpuSimulationResult>()) as u64;

            self.sim_input_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gomoku Input Buffer"),
                size: input_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            self.sim_output_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gomoku Output Buffer"),
                size: output_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));

            self.sim_staging_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gomoku Staging Buffer"),
                size: output_size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            self.sim_params_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gomoku Params Buffer"),
                size: std::mem::size_of::<GpuSimulationParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            // Create bind group
            self.sim_bind_group = Some(self.context.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Gomoku Bind Group"),
                layout: &self.context.gomoku_eval_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.sim_input_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: self.sim_output_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.sim_params_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                ],
            }));
            
            self.sim_current_capacity = new_capacity;
        }

        // Upload data
        self.context.queue().write_buffer(
            self.sim_input_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(board_data),
        );
        
        self.context.queue().write_buffer(
            self.sim_params_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&params),
        );

        // Dispatch
        let mut encoder = self.context.device().create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Gomoku Eval Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gomoku Eval Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.context.gomoku_eval_pipeline);
            compute_pass.set_bind_group(0, self.sim_bind_group.as_ref().unwrap(), &[]);
            let workgroups = (num_boards + 63) / 64;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let output_size = (num_boards as usize * std::mem::size_of::<GpuSimulationResult>()) as u64;
        encoder.copy_buffer_to_buffer(
            self.sim_output_buffer.as_ref().unwrap(),
            0,
            self.sim_staging_buffer.as_ref().unwrap(),
            0,
            output_size,
        );
        
        self.context.submit_and_wait(encoder.finish());

        // Read results
        let staging_buffer = self.sim_staging_buffer.as_ref().unwrap();
        let buffer_slice = staging_buffer.slice(..output_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.context.device().poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().map_err(|e| GpuError::BufferError(e.to_string()))?;

        let data = buffer_slice.get_mapped_range();
        let results: &[GpuSimulationResult] = bytemuck::cast_slice(&data);
        let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
        drop(data);
        staging_buffer.unmap();

        self.total_gpu_time_us += start_time.elapsed().as_micros() as u64;
        self.dispatch_count += 1;

        Ok(scores)
    }

    /// Returns the GPU context
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    /// Returns GPU statistics
    pub fn stats(&self) -> (u64, u64, f64) {
        let avg_time = if self.dispatch_count > 0 {
            self.total_gpu_time_us as f64 / self.dispatch_count as f64
        } else {
            0.0
        };
        (self.total_gpu_time_us, self.dispatch_count, avg_time)
    }

    /// Returns debug information
    pub fn debug_info(&self) -> String {
        let (total_us, dispatches, avg_us) = self.stats();
        format!(
            "{}\nGPU Stats: {} dispatches, {:.2}ms total, {:.2}µs avg",
            self.context.debug_info(),
            dispatches,
            total_us as f64 / 1000.0,
            avg_us
        )
    }
}
