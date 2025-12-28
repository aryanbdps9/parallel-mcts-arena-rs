//! GPU MCTS Accelerator
//!
//! Provides GPU-accelerated batch operations for MCTS:
//! - PUCT score calculation
//! - Gomoku board evaluation

use wgpu::{BindGroupDescriptor, BindGroupEntry, Buffer, BufferUsages, CommandEncoderDescriptor, util::DeviceExt};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

use super::context::{GpuContext, GpuError};

// Game type constants matching shaders.rs
const GAME_GOMOKU: i32 = 0;
const GAME_CONNECT4: i32 = 1;
const GAME_OTHELLO: i32 = 2;
const GAME_BLOKUS: i32 = 3;
const GAME_HIVE: i32 = 4;

/// Node data for GPU PUCT calculation
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuNodeData {
    pub visits: i32,
    pub wins: i32,
    pub virtual_losses: i32,
    pub parent_visits: i32,
    pub prior_prob: f32,
    pub exploration: f32,
    pub _padding: [f32; 2],
}

impl GpuNodeData {
    pub fn new(
        visits: i32, wins: i32, virtual_losses: i32,
        parent_visits: i32, prior_prob: f32, exploration: f32,
    ) -> Self {
        Self { visits, wins, virtual_losses, parent_visits, prior_prob, exploration, _padding: [0.0; 2] }
    }
}

/// Result of GPU PUCT calculation
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuPuctResult {
    pub puct_score: f32,
    pub q_value: f32,
    pub exploration_term: f32,
    pub node_index: u32,
}

/// Shader parameters
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    num_elements: u32,
    _reserved: [u32; 3],
}

/// Parameters for game board evaluation
/// 
/// The current_player field is overloaded:
/// - Bits 0-7: current player (1 or -1, normalized to 1 in board data)
/// - Bits 8-15: game-specific parameter (e.g., line_size for Connect4)
/// 
/// Game types (determined by board dimensions and context):
/// - Gomoku: 15x15 or 19x19, win with 5-in-a-row
/// - Connect4: 7x6 typically, win with N-in-a-row (default 4), gravity-based
/// - Othello: 8x8, flip-based, count-based winner
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuSimulationParams {
    pub board_width: u32,
    pub board_height: u32,
    pub current_player: i32,  // Lower 8 bits: player, bits 8-15: line_size for Connect4
    pub use_heuristic: u32,
    pub seed: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSimulationResult {
    score: f32,
}

/// GPU MCTS Accelerator
pub struct GpuMctsAccelerator {
    context: Arc<GpuContext>,
    // PUCT buffers
    puct_input_buffer: Option<Buffer>,
    puct_output_buffer: Option<Buffer>,
    puct_staging_buffer: Option<Buffer>,
    puct_capacity: usize,
    // Simulation buffers
    sim_input_buffer: Option<Buffer>,
    sim_output_buffer: Option<Buffer>,
    sim_staging_buffer: Option<Buffer>,
    sim_params_buffer: Option<Buffer>,
    sim_bind_group: Option<wgpu::BindGroup>,
    sim_capacity: usize,
    // Statistics
    total_gpu_time_us: u64,
    dispatch_count: u64,
}

impl GpuMctsAccelerator {
    pub fn new(context: Arc<GpuContext>) -> Self {
        Self {
            context,
            puct_input_buffer: None,
            puct_output_buffer: None,
            puct_staging_buffer: None,
            puct_capacity: 0,
            sim_input_buffer: None,
            sim_output_buffer: None,
            sim_staging_buffer: None,
            sim_params_buffer: None,
            sim_bind_group: None,
            sim_capacity: 0,
            total_gpu_time_us: 0,
            dispatch_count: 0,
        }
    }

    fn ensure_puct_buffers(&mut self, num_nodes: usize) {
        if self.puct_capacity >= num_nodes && self.puct_input_buffer.is_some() {
            return;
        }

        let new_capacity = num_nodes.next_power_of_two().max(256);
        let input_size = (new_capacity * std::mem::size_of::<GpuNodeData>()) as u64;
        let output_size = (new_capacity * std::mem::size_of::<GpuPuctResult>()) as u64;

        self.puct_input_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("PUCT Input"),
            size: input_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.puct_output_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("PUCT Output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        self.puct_staging_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("PUCT Staging"),
            size: output_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.puct_capacity = new_capacity;
    }

    /// Compute PUCT scores for a batch of nodes
    pub fn compute_puct_batch(&mut self, nodes: &[GpuNodeData]) -> Result<Vec<GpuPuctResult>, GpuError> {
        let num_nodes = nodes.len();
        if num_nodes == 0 {
            return Ok(Vec::new());
        }

        // Use CPU for small batches
        if num_nodes < self.context.config().min_batch_threshold {
            return Ok(self.compute_puct_cpu(nodes));
        }

        let start = std::time::Instant::now();
        self.ensure_puct_buffers(num_nodes);

        let input_buffer = self.puct_input_buffer.as_ref().unwrap();
        let output_buffer = self.puct_output_buffer.as_ref().unwrap();
        let staging_buffer = self.puct_staging_buffer.as_ref().unwrap();

        // Upload input
        self.context.queue().write_buffer(input_buffer, 0, bytemuck::cast_slice(nodes));

        // Create params buffer
        let params = GpuParams { num_elements: num_nodes as u32, _reserved: [0; 3] };
        let params_buffer = self.context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PUCT Params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = self.context.device().create_bind_group(&BindGroupDescriptor {
            label: Some("PUCT Bind Group"),
            layout: self.context.puct_bind_group_layout(),
            entries: &[
                BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        // Dispatch
        let mut encoder = self.context.device().create_command_encoder(&CommandEncoderDescriptor {
            label: Some("PUCT Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PUCT Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.context.puct_pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((num_nodes as u32 + 255) / 256, 1, 1);
        }

        let output_size = (num_nodes * std::mem::size_of::<GpuPuctResult>()) as u64;
        encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, output_size);
        self.context.submit_and_wait(encoder.finish());

        // Read results
        let results = self.read_buffer(staging_buffer, num_nodes)?;

        self.total_gpu_time_us += start.elapsed().as_micros() as u64;
        self.dispatch_count += 1;

        Ok(results)
    }

    fn read_buffer<T: Pod>(&self, staging_buffer: &Buffer, count: usize) -> Result<Vec<T>, GpuError> {
        let buffer_slice = staging_buffer.slice(..);
        
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| { let _ = tx.send(result); });
        self.context.device().poll(wgpu::Maintain::Wait);
        
        pollster::block_on(rx)
            .map_err(|_| GpuError::BufferError("Mapping cancelled".to_string()))?
            .map_err(|e| GpuError::BufferError(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let results: Vec<T> = bytemuck::cast_slice(&data[..count * std::mem::size_of::<T>()]).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }

    fn compute_puct_cpu(&self, nodes: &[GpuNodeData]) -> Vec<GpuPuctResult> {
        nodes.iter().enumerate().map(|(idx, node)| {
            let effective_visits = node.visits + node.virtual_losses;
            let parent_sqrt = (node.parent_visits as f32).sqrt();
            
            let (q_value, exploration_term) = if effective_visits == 0 {
                (0.0, node.exploration * node.prior_prob * parent_sqrt)
            } else {
                let q = if node.visits > 0 { (node.wins as f32 / node.visits as f32) / 2.0 } else { 0.0 };
                let e = node.exploration * node.prior_prob * parent_sqrt / (1.0 + effective_visits as f32);
                (q, e)
            };

            GpuPuctResult {
                puct_score: q_value + exploration_term,
                q_value,
                exploration_term,
                node_index: idx as u32,
            }
        }).collect()
    }

    /// Evaluate Gomoku boards on GPU
    pub fn simulate_batch(&mut self, board_data: &[i32], params: GpuSimulationParams) -> Result<Vec<f32>, GpuError> {
        let start = std::time::Instant::now();
        let board_size = (params.board_width * params.board_height) as usize;
        if board_size == 0 {
            return Ok(Vec::new());
        }
        let num_boards = board_data.len() / board_size;
        
        if num_boards == 0 {
            return Ok(Vec::new());
        }

        // Debug print to verify params
        // println!("GPU Batch: {} boards, size {}x{}={}, total_len={}, params={:?}", 
        //    num_boards, params.board_width, params.board_height, board_size, board_data.len(), params);

        // Debug: Bypass GPU execution
        // return Ok(vec![0.0; num_boards]);

        // Ensure buffers
        self.ensure_sim_buffers(num_boards, board_size);

        // Upload data
        self.context.queue().write_buffer(self.sim_input_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(board_data));
        
        // Update params
        self.context.queue().write_buffer(self.sim_params_buffer.as_ref().unwrap(), 0, bytemuck::bytes_of(&params));

        // Dispatch
        let mut encoder = self.context.device().create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Sim Encoder"),
        });

        // Determine which pipeline to use based on game type
        let encoded = params.current_player;
        let explicit_game_type = (encoded >> 16) & 0xFF;
        // let line_size = (encoded >> 8) & 0xFF;
        
        let pipeline = match explicit_game_type {
            GAME_CONNECT4 => &self.context.connect4_eval_pipeline,
            GAME_OTHELLO => &self.context.othello_eval_pipeline,
            GAME_BLOKUS => &self.context.blokus_eval_pipeline,
            GAME_HIVE => &self.context.hive_eval_pipeline,
            GAME_GOMOKU | _ => &self.context.gomoku_eval_pipeline,
        };

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sim Pass"), timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, self.sim_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups((num_boards as u32 + 63) / 64, 1, 1);
        }

        let output_size = (num_boards * std::mem::size_of::<GpuSimulationResult>()) as u64;
        encoder.copy_buffer_to_buffer(
            self.sim_output_buffer.as_ref().unwrap(), 0,
            self.sim_staging_buffer.as_ref().unwrap(), 0,
            output_size,
        );
        self.context.submit_and_wait(encoder.finish());

        // Read results
        let staging = self.sim_staging_buffer.as_ref().unwrap();
        let buffer_slice = staging.slice(..output_size);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.context.device().poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| GpuError::BufferError(e.to_string()))?;

        let data = buffer_slice.get_mapped_range();
        let results: &[GpuSimulationResult] = bytemuck::cast_slice(&data);
        let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
        drop(data);
        staging.unmap();

        self.total_gpu_time_us += start.elapsed().as_micros() as u64;
        self.dispatch_count += 1;

        Ok(scores)
    }

    fn ensure_sim_buffers(&mut self, num_boards: usize, board_size: usize) {
        if self.sim_capacity < num_boards || self.sim_input_buffer.is_none() {
            let new_capacity = num_boards.next_power_of_two().max(256);
            let input_size = (new_capacity * board_size * std::mem::size_of::<i32>()) as u64;
            let output_size = (new_capacity * std::mem::size_of::<GpuSimulationResult>()) as u64;

            self.sim_input_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sim Input"), size: input_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST, mapped_at_creation: false,
            }));

            self.sim_output_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sim Output"), size: output_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, mapped_at_creation: false,
            }));

            self.sim_staging_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sim Staging"), size: output_size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST, mapped_at_creation: false,
            }));

            self.sim_params_buffer = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sim Params"), size: std::mem::size_of::<GpuSimulationParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST, mapped_at_creation: false,
            }));

            self.sim_bind_group = Some(self.context.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sim Bind Group"),
                layout: &self.context.eval_bind_group_layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: self.sim_input_buffer.as_ref().unwrap().as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: self.sim_output_buffer.as_ref().unwrap().as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: self.sim_params_buffer.as_ref().unwrap().as_entire_binding() },
                ],
            }));
            
            self.sim_capacity = new_capacity;
        }
    }

    pub fn stats(&self) -> (u64, u64, f64) {
        let avg = if self.dispatch_count > 0 { self.total_gpu_time_us as f64 / self.dispatch_count as f64 } else { 0.0 };
        (self.total_gpu_time_us, self.dispatch_count, avg)
    }

    /// Get the GPU context for creating other GPU-based structures
    pub fn get_context(&self) -> Arc<GpuContext> {
        self.context.clone()
    }

    pub fn debug_info(&self) -> String {
        let (total_us, dispatches, avg_us) = self.stats();
        format!("{}\nStats: {} dispatches, {:.2}ms total, {:.2}µs avg",
            self.context.debug_info(), dispatches, total_us as f64 / 1000.0, avg_us)
    }
}
