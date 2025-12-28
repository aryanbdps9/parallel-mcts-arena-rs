//! GPU-Native MCTS for Othello
//!
//! This module provides a specialized GPU MCTS engine for Othello that runs
//! complete MCTS iterations (selection, simulation, backprop) in a single
//! GPU kernel dispatch, eliminating CPU-GPU synchronization overhead.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages,
    CommandEncoderDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineLayoutDescriptor,
    ShaderModuleDescriptor, ShaderStages,
};

use super::context::GpuContext;
use super::shaders::MCTS_OTHELLO_SHADER;

// =============================================================================
// Constants (must match shader)
// =============================================================================

const MAX_CHILDREN: u32 = 64;
const MAX_PATH_LENGTH: u32 = 128;
const INVALID_INDEX: u32 = 0xFFFFFFFF;
const WORKGROUP_SIZE: u32 = 64;
const NODE_STATE_READY: u32 = 2;

// =============================================================================
// Data Structures (must match shader layout exactly)
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
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
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct OthelloNodeInfo {
    pub parent_idx: u32,
    pub move_id: u32,
    pub num_children: u32,
    pub player_at_node: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct OthelloChildStats {
    pub move_id: u32,
    pub visits: i32,
    pub wins: i32,
    pub q_value: f32,
}

/// Lightweight telemetry from a single GPU dispatch
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
    pub expansion_terminal: u32,
    pub alloc_failures: u32,
    pub rollouts: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Lightweight telemetry from a single GPU dispatch
#[derive(Debug, Clone, Copy, Default)]
pub struct OthelloRunTelemetry {
    pub iterations_launched: u32,
    pub alloc_count_after: u32,
    pub node_capacity: u32,
    pub saturated: bool,
    pub diagnostics: OthelloDiagnostics,
}

// =============================================================================
// GPU MCTS Engine for Othello
// =============================================================================

pub struct GpuOthelloMcts {
    context: Arc<GpuContext>,

    // Compute pipeline
    iteration_pipeline: ComputePipeline,

    // Bind group layouts
    node_pool_layout: BindGroupLayout,
    execution_layout: BindGroupLayout,
    board_layout: BindGroupLayout,

    // Node pool buffers
    node_info_buffer: Buffer,
    node_visits_buffer: Buffer,
    node_wins_buffer: Buffer,
    node_vl_buffer: Buffer,
    node_state_buffer: Buffer,
    children_indices_buffer: Buffer,
    children_priors_buffer: Buffer,

    // Execution state buffers
    params_buffer: Buffer,
    work_items_buffer: Buffer,
    paths_buffer: Buffer,
    alloc_counter_buffer: Buffer,
    diagnostics_buffer: Buffer,

    // Root board buffer
    root_board_buffer: Buffer,
    root_board_staging: Buffer,

    // Staging buffers for readback
    visits_staging: Buffer,
    wins_staging: Buffer,
    node_info_staging: Buffer,
    children_staging: Buffer,
    alloc_staging: Buffer,
    diagnostics_staging: Buffer,

    // Bind groups
    node_pool_bind_group: Option<BindGroup>,
    execution_bind_group: Option<BindGroup>,
    board_bind_group: Option<BindGroup>,

    // Configuration
    max_nodes: u32,
    _max_iterations: u32,
}

impl GpuOthelloMcts {
    /// Create a new GPU Othello MCTS engine
    pub fn new(context: Arc<GpuContext>, max_nodes: u32, max_iterations: u32) -> Self {
        let device = context.device();

        // Create shader module
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("MCTS Othello Shader"),
            source: wgpu::ShaderSource::Wgsl(MCTS_OTHELLO_SHADER.into()),
        });

        // Create bind group layouts
        let node_pool_layout = Self::create_node_pool_layout(device);
        let execution_layout = Self::create_execution_layout(device);
        let board_layout = Self::create_board_layout(device);

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("MCTS Othello Pipeline Layout"),
            bind_group_layouts: &[&node_pool_layout, &execution_layout, &board_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let iteration_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("MCTS Othello Iteration Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("mcts_othello_iteration"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create buffers
        let node_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Othello Node Info"),
            size: (max_nodes as usize * std::mem::size_of::<OthelloNodeInfo>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let node_visits_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Othello Node Visits"),
            size: (max_nodes as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let node_wins_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Othello Node Wins"),
            size: (max_nodes as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let node_vl_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Othello Node VL"),
            size: (max_nodes as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let node_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Othello Node State"),
            size: (max_nodes as usize * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let children_size = max_nodes as usize * MAX_CHILDREN as usize;
        let children_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Othello Children Indices"),
            size: (children_size * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let children_priors_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Othello Children Priors"),
            size: (children_size * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MCTS Othello Params"),
            size: std::mem::size_of::<MctsOthelloParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Work items size: 8 * u32 per item
        let work_items_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Othello Work Items"),
            size: (max_iterations as usize * 8 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let paths_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Othello Paths"),
            size: (max_iterations as usize * MAX_PATH_LENGTH as usize * std::mem::size_of::<u32>())
                as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let alloc_counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Othello Alloc Counter"),
            size: std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let diagnostics_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Othello Diagnostics"),
            size: std::mem::size_of::<OthelloDiagnostics>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Root board: 8x8 = 64 cells
        let root_board_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Othello Root Board"),
            size: (64 * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let root_board_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Othello Root Board Staging"),
            size: (64 * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Staging buffers for readback
        let visits_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visits Staging"),
            size: (MAX_CHILDREN as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let wins_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Wins Staging"),
            size: (MAX_CHILDREN as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let node_info_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Info Staging"),
            size: (MAX_CHILDREN as usize * std::mem::size_of::<OthelloNodeInfo>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let children_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Children Staging"),
            size: (MAX_CHILDREN as usize * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let alloc_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alloc Counter Staging"),
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let diagnostics_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Diagnostics Staging"),
            size: std::mem::size_of::<OthelloDiagnostics>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            context,
            iteration_pipeline,
            node_pool_layout,
            execution_layout,
            board_layout,
            node_info_buffer,
            node_visits_buffer,
            node_wins_buffer,
            node_vl_buffer,
            node_state_buffer,
            children_indices_buffer,
            children_priors_buffer,
            params_buffer,
            work_items_buffer,
            paths_buffer,
            alloc_counter_buffer,
            diagnostics_buffer,
            root_board_buffer,
            root_board_staging,
            visits_staging,
            wins_staging,
            node_info_staging,
            children_staging,
            alloc_staging,
            diagnostics_staging,
            node_pool_bind_group: None,
            execution_bind_group: None,
            board_bind_group: None,
            max_nodes,
            _max_iterations: max_iterations,
        }
    }

    fn create_node_pool_layout(device: &wgpu::Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Othello Node Pool Layout"),
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
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_execution_layout(device: &wgpu::Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Othello Execution Layout"),
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
        })
    }

    fn create_board_layout(device: &wgpu::Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Othello Board Layout"),
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
        })
    }

    fn create_bind_groups(&mut self) {
        let device = self.context.device();

        self.node_pool_bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("Othello Node Pool Bind Group"),
            layout: &self.node_pool_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.node_info_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.node_visits_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.node_wins_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.node_vl_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.node_state_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: self.children_indices_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: self.children_priors_buffer.as_entire_binding(),
                },
            ],
        }));

        self.execution_bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("Othello Execution Bind Group"),
            layout: &self.execution_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.work_items_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.paths_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.alloc_counter_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.diagnostics_buffer.as_entire_binding(),
                },
            ],
        }));

        self.board_bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("Othello Board Bind Group"),
            layout: &self.board_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: self.root_board_buffer.as_entire_binding(),
            }],
        }));
    }

    /// Initialize the MCTS tree with root position and legal moves
    ///
    /// # Arguments
    /// * `board` - 64-element array representing the Othello board
    /// * `root_player` - Player to move at root (1 or -1)
    /// * `legal_moves` - List of legal moves as (x, y) coordinates
    pub fn init_tree(&mut self, board: &[i32; 64], root_player: i32, legal_moves: &[(usize, usize)]) {
        let queue = self.context.queue();

        // Upload root board
        queue.write_buffer(&self.root_board_buffer, 0, bytemuck::cast_slice(board));

        // Initialize root node (index 0)
        let root_info = OthelloNodeInfo {
            parent_idx: INVALID_INDEX,
            move_id: INVALID_INDEX,
            num_children: legal_moves.len() as u32,
            player_at_node: root_player,
        };
        queue.write_buffer(&self.node_info_buffer, 0, bytemuck::bytes_of(&root_info));
        queue.write_buffer(&self.node_visits_buffer, 0, bytemuck::bytes_of(&0i32));
        queue.write_buffer(&self.node_wins_buffer, 0, bytemuck::bytes_of(&0i32));
        queue.write_buffer(&self.node_vl_buffer, 0, bytemuck::bytes_of(&0i32));
        queue.write_buffer(&self.node_state_buffer, 0, bytemuck::bytes_of(&NODE_STATE_READY));

        // Initialize children for root
        let opposite_player = -root_player;
        let uniform_prior = 1.0 / legal_moves.len().max(1) as f32;
        
        let mut child_indices = vec![INVALID_INDEX; MAX_CHILDREN as usize];
        let mut child_priors = vec![0.0f32; MAX_CHILDREN as usize];

        for (i, &(x, y)) in legal_moves.iter().enumerate() {
            if i >= MAX_CHILDREN as usize {
                break;
            }

            let child_idx = (i + 1) as u32;
            let move_id = (y * 8 + x) as u32;

            let child_info = OthelloNodeInfo {
                parent_idx: 0,
                move_id,
                num_children: 0, // Will be expanded on first visit
                player_at_node: opposite_player,
            };

            let offset = child_idx as usize * std::mem::size_of::<OthelloNodeInfo>();
            queue.write_buffer(&self.node_info_buffer, offset as u64, bytemuck::bytes_of(&child_info));

            let stat_offset = child_idx as usize * std::mem::size_of::<i32>();
            queue.write_buffer(&self.node_visits_buffer, stat_offset as u64, bytemuck::bytes_of(&0i32));
            queue.write_buffer(&self.node_wins_buffer, stat_offset as u64, bytemuck::bytes_of(&0i32));
            queue.write_buffer(&self.node_vl_buffer, stat_offset as u64, bytemuck::bytes_of(&0i32));
            queue.write_buffer(&self.node_state_buffer, stat_offset as u64, bytemuck::bytes_of(&NODE_STATE_READY));

            child_indices[i] = child_idx;
            child_priors[i] = uniform_prior;
        }

        queue.write_buffer(&self.children_indices_buffer, 0, bytemuck::cast_slice(&child_indices));
        queue.write_buffer(&self.children_priors_buffer, 0, bytemuck::cast_slice(&child_priors));

        // Set allocation counter
        let alloc_count = (legal_moves.len() + 1) as u32;
        queue.write_buffer(&self.alloc_counter_buffer, 0, bytemuck::bytes_of(&alloc_count));

        self.create_bind_groups();
    }

    /// Run MCTS iterations on GPU
    ///
    /// # Arguments
    /// * `num_iterations` - Number of parallel MCTS iterations
    /// * `exploration` - C_puct exploration parameter
    /// * `seed` - Random seed
    pub fn run_iterations(
        &mut self,
        num_iterations: u32,
        exploration: f32,
        virtual_loss_weight: f32,
        seed: u32,
    ) -> OthelloRunTelemetry {
        let queue = self.context.queue();
        let device = self.context.device();

        let params = MctsOthelloParams {
            num_iterations,
            max_nodes: self.max_nodes,
            exploration,
            virtual_loss_weight,
            root_idx: 0,
            seed,
            board_width: 8,
            board_height: 8,
            game_type: 2, // Othello
            _pad: [0; 3],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Reset diagnostics for this dispatch
        let diag_zero = OthelloDiagnostics::default();
        queue.write_buffer(
            &self.diagnostics_buffer,
            0,
            bytemuck::bytes_of(&diag_zero),
        );

        let workgroups = (num_iterations + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut encoder = self.context.device().create_command_encoder(&CommandEncoderDescriptor {
            label: Some("MCTS Othello Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MCTS Othello Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.iteration_pipeline);
            pass.set_bind_group(0, self.node_pool_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(1, self.execution_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(2, self.board_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Capture allocation counter after this dispatch for telemetry
        encoder.copy_buffer_to_buffer(&self.alloc_counter_buffer, 0, &self.alloc_staging, 0, 4);
        encoder.copy_buffer_to_buffer(
            &self.diagnostics_buffer,
            0,
            &self.diagnostics_staging,
            0,
            std::mem::size_of::<OthelloDiagnostics>() as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Periodically poll the device to flush submitted work and avoid command
        // buffer buildup when many batches are dispatched in a tight loop.
        device.poll(wgpu::Maintain::Poll);

        // Read back alloc counter
        let slice = self.alloc_staging.slice(..4);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let alloc_after: u32 = *bytemuck::from_bytes(&data);
        drop(data);
        self.alloc_staging.unmap();

        // Read diagnostics
        let diag_slice = self
            .diagnostics_staging
            .slice(..std::mem::size_of::<OthelloDiagnostics>() as u64);
        diag_slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let diag_data = diag_slice.get_mapped_range();
        let diagnostics: OthelloDiagnostics = *bytemuck::from_bytes(&diag_data);
        drop(diag_data);
        self.diagnostics_staging.unmap();

        OthelloRunTelemetry {
            iterations_launched: num_iterations,
            alloc_count_after: alloc_after,
            node_capacity: self.max_nodes,
            saturated: alloc_after >= self.max_nodes,
            diagnostics,
        }
    }

    /// Force GPU to complete all pending work
    /// Call this periodically to prevent command buffer and memory buildup
    pub fn flush_and_wait(&self) {
        let device = self.context.device();
        device.poll(wgpu::Maintain::Wait);
    }

    /// Get the best move based on visit counts
    ///
    /// Returns (move_x, move_y, visits, q_value) or None if no moves
    /// Uses pre-allocated staging buffers to avoid memory leaks
    pub fn get_best_move(&self) -> Option<(usize, usize, i32, f64)> {
        // Reuse get_children_stats which already uses pre-allocated buffers
        let stats = self.get_children_stats();
        
        if stats.is_empty() {
            return None;
        }
        
        // Find child with most visits
        stats.into_iter()
            .max_by_key(|(_, _, visits, _, _)| *visits)
            .map(|(x, y, visits, _wins, q)| (x, y, visits, q))
    }

    /// Get all children statistics for analysis
    pub fn get_all_children_stats(&self) -> Vec<OthelloChildStats> {
        // Similar to get_best_move but returns all children
        // Implementation omitted for brevity - follows same pattern
        Vec::new()
    }

    /// Get root visit count (useful for progress tracking)
    /// Uses pre-allocated staging buffer
    pub fn get_root_visits(&self) -> i32 {
        let device = self.context.device();
        let queue = self.context.queue();

        // Use pre-allocated visits_staging buffer (first 4 bytes)
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.node_visits_buffer, 0, &self.visits_staging, 0, 4);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.visits_staging.slice(..4);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let visits: i32 = *bytemuck::from_bytes(&data);
        drop(data);
        self.visits_staging.unmap();

        visits
    }

    /// Read a lightweight hash of the GPU root board for drift detection
    pub fn get_root_board_hash(&self) -> u64 {
        let device = self.context.device();
        let queue = self.context.queue();

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Copy Root Board For Hash"),
        });
        encoder.copy_buffer_to_buffer(
            &self.root_board_buffer,
            0,
            &self.root_board_staging,
            0,
            (64 * std::mem::size_of::<i32>()) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.root_board_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let values: &[i32] = bytemuck::cast_slice(&data);

        // FNV-1a 64-bit for cheap hashing
        let mut hash: u64 = 0xcbf29ce484222325;
        for v in values {
            hash ^= *v as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }

        drop(data);
        self.root_board_staging.unmap();

        hash
    }

    /// Read the current GPU root board into host memory using the pre-allocated staging buffer
    fn read_root_board(&self) -> [i32; 64] {
        let device = self.context.device();
        let queue = self.context.queue();

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Copy Root Board (read_root_board)"),
        });
        encoder.copy_buffer_to_buffer(
            &self.root_board_buffer,
            0,
            &self.root_board_staging,
            0,
            (64 * std::mem::size_of::<i32>()) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.root_board_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let mut out = [0i32; 64];
        out.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        self.root_board_staging.unmap();

        out
    }

    /// CPU-side Othello legal move generator for drift checks
    fn compute_othello_legal_moves_host(board: &[i32; 64], player: i32) -> Vec<(usize, usize)> {
        const DIRS: [(i32, i32); 8] = [
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ];
        let w = 8i32;
        let mut moves = Vec::new();
        for y in 0..w {
            for x in 0..w {
                if board[(y * w + x) as usize] != 0 {
                    continue;
                }
                let mut valid = false;
                for (dx, dy) in DIRS.iter().copied() {
                    let mut cx = x + dx;
                    let mut cy = y + dy;
                    let mut seen_opponent = false;
                    while cx >= 0 && cx < w && cy >= 0 && cy < w {
                        let cell = board[(cy * w + cx) as usize];
                        if cell == -player {
                            seen_opponent = true;
                            cx += dx;
                            cy += dy;
                            continue;
                        }
                        if cell == player && seen_opponent {
                            valid = true;
                        }
                        break;
                    }
                    if valid {
                        break;
                    }
                }
                if valid {
                    moves.push((x as usize, y as usize));
                }
            }
        }
        moves
    }

    fn apply_othello_move_host(board: &[i32; 64], x: usize, y: usize, player: i32) -> Option<[i32; 64]> {
        const DIRS: [(i32, i32); 8] = [
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ];

        if x >= 8 || y >= 8 {
            return None;
        }

        let mut out = *board;
        if out[y * 8 + x] != 0 {
            return None;
        }

        let mut any_flip = false;
        out[y * 8 + x] = player;

        for (dx, dy) in DIRS.iter().copied() {
            let mut cx = x as i32 + dx;
            let mut cy = y as i32 + dy;
            let mut path: Vec<(usize, usize)> = Vec::new();
            let mut seen_opponent = false;

            while cx >= 0 && cx < 8 && cy >= 0 && cy < 8 {
                let idx = (cy as usize) * 8 + cx as usize;
                let cell = out[idx];
                if cell == -player {
                    seen_opponent = true;
                    path.push((cx as usize, cy as usize));
                    cx += dx;
                    cy += dy;
                    continue;
                }
                if cell == player && seen_opponent {
                    // flip path
                    for (fx, fy) in path.iter().copied() {
                        out[fy * 8 + fx] = player;
                    }
                    any_flip = true;
                }
                break;
            }
        }

        if any_flip {
            Some(out)
        } else {
            None
        }
    }

    fn fnv_hash_board(board: &[i32; 64]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for v in board.iter() {
            hash ^= *v as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    fn board_to_ascii(board: &[i32; 64]) -> Vec<String> {
        let mut out = Vec::with_capacity(8);
        for row in 0..8 {
            let mut line = String::with_capacity(8);
            for col in 0..8 {
                let cell = board[row * 8 + col];
                let ch = match cell {
                    1 => 'X',
                    -1 => 'O',
                    _ => '.',
                };
                line.push(ch);
            }
            out.push(line);
        }
        out
    }

    /// Get the current root index
    pub fn get_root_idx(&self) -> u32 {
        // For now, root is always at index 0 in current implementation
        // In the future with tree reuse, this may change
        0
    }

    /// Mark a root child as invalid to prevent repeating an illegal proposal
    /// Returns true if a matching child slot was found and invalidated
    pub fn invalidate_root_child(&mut self, move_x: usize, move_y: usize) -> bool {
        let device = self.context.device();
        let queue = self.context.queue();
        let target_move_id = (move_y * 8 + move_x) as u32;

        // Read root info using pre-allocated staging buffer
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Read Root Info (invalidate_child)"),
        });
        encoder.copy_buffer_to_buffer(
            &self.node_info_buffer,
            0,
            &self.node_info_staging,
            0,
            std::mem::size_of::<OthelloNodeInfo>() as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self
            .node_info_staging
            .slice(..std::mem::size_of::<OthelloNodeInfo>() as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let root_info: OthelloNodeInfo = *bytemuck::from_bytes(&data);
        drop(data);
        self.node_info_staging.unmap();

        if root_info.num_children == 0 {
            return false;
        }

        let num_children = root_info.num_children.min(MAX_CHILDREN) as usize;

        // Read children indices
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Read Children Indices (invalidate_child)"),
        });
        encoder.copy_buffer_to_buffer(
            &self.children_indices_buffer,
            0,
            &self.children_staging,
            0,
            (num_children * std::mem::size_of::<u32>()) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.children_staging.slice(..(num_children * 4) as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let children: Vec<u32> = bytemuck::cast_slice(&data[..num_children * 4]).to_vec();
        drop(data);
        self.children_staging.unmap();

        // Read children node info to locate target move
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Read Children Info (invalidate_child)"),
        });
        for (i, &child_idx) in children.iter().enumerate() {
            if child_idx != INVALID_INDEX && i < MAX_CHILDREN as usize {
                let src_offset = child_idx as u64 * std::mem::size_of::<OthelloNodeInfo>() as u64;
                let dst_offset = i as u64 * std::mem::size_of::<OthelloNodeInfo>() as u64;
                encoder.copy_buffer_to_buffer(
                    &self.node_info_buffer,
                    src_offset,
                    &self.node_info_staging,
                    dst_offset,
                    std::mem::size_of::<OthelloNodeInfo>() as u64,
                );
            }
        }
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self
            .node_info_staging
            .slice(..(num_children * std::mem::size_of::<OthelloNodeInfo>()) as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let children_info: Vec<OthelloNodeInfo> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.node_info_staging.unmap();

        // Find slot for the target move
        let mut found_slot: Option<usize> = None;
        for (i, info) in children_info.iter().enumerate() {
            if info.move_id == target_move_id {
                found_slot = Some(i);
                break;
            }
        }

        let slot = match found_slot {
            Some(s) => s,
            None => return false,
        };

        // Invalidate child entry and zero its prior to remove it from consideration
        let slot_u64 = slot as u64 * 4;
        let invalid = INVALID_INDEX;
        queue.write_buffer(&self.children_indices_buffer, slot_u64, bytemuck::bytes_of(&invalid));
        let zero_prior: f32 = 0.0;
        queue.write_buffer(&self.children_priors_buffer, slot_u64, bytemuck::bytes_of(&zero_prior));

        // Reset root stats so selection restarts cleanly after invalidation
        queue.write_buffer(&self.node_visits_buffer, 0, bytemuck::bytes_of(&0i32));
        queue.write_buffer(&self.node_wins_buffer, 0, bytemuck::bytes_of(&0i32));
        queue.write_buffer(&self.node_vl_buffer, 0, bytemuck::bytes_of(&0i32));
        queue.write_buffer(&self.node_state_buffer, 0, bytemuck::bytes_of(&NODE_STATE_READY));

        true
    }

    /// Get detailed stats for all children of the root
    /// Returns Vec of (move_x, move_y, visits, wins, q_value)
    /// Uses pre-allocated staging buffers to avoid memory leaks
    pub fn get_children_stats(&self) -> Vec<(usize, usize, i32, i32, f64)> {
        let device = self.context.device();
        let queue = self.context.queue();

        // Read root node info using pre-allocated staging buffer
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Read Root Info"),
        });
        encoder.copy_buffer_to_buffer(
            &self.node_info_buffer,
            0,
            &self.node_info_staging,
            0,
            std::mem::size_of::<OthelloNodeInfo>() as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.node_info_staging.slice(..std::mem::size_of::<OthelloNodeInfo>() as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let root_info: OthelloNodeInfo = *bytemuck::from_bytes(&data);
        drop(data);
        self.node_info_staging.unmap();

        if root_info.num_children == 0 {
            return Vec::new();
        }

        let num_children = root_info.num_children.min(MAX_CHILDREN) as usize;
        
        // Read children indices using pre-allocated staging buffer
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Read Children"),
        });
        encoder.copy_buffer_to_buffer(
            &self.children_indices_buffer,
            0,
            &self.children_staging,
            0,
            (num_children * std::mem::size_of::<u32>()) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.children_staging.slice(..(num_children * std::mem::size_of::<u32>()) as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let children: Vec<u32> = bytemuck::cast_slice(&data[..num_children * 4]).to_vec();
        drop(data);
        self.children_staging.unmap();

        // Batch read all children visits and wins in one go
        // visits_staging has space for MAX_CHILDREN i32 values
        // wins_staging has space for MAX_CHILDREN i32 values
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Batch Read Children Stats"),
        });
        
        // Copy visits and wins for each valid child
        for (i, &child_idx) in children.iter().enumerate() {
            if child_idx != INVALID_INDEX && i < MAX_CHILDREN as usize {
                let src_offset = child_idx as u64 * 4;
                let dst_offset = i as u64 * 4;
                encoder.copy_buffer_to_buffer(&self.node_visits_buffer, src_offset, &self.visits_staging, dst_offset, 4);
                encoder.copy_buffer_to_buffer(&self.node_wins_buffer, src_offset, &self.wins_staging, dst_offset, 4);
            }
        }
        
        // Also batch read node info for move_id
        for (i, &child_idx) in children.iter().enumerate() {
            if child_idx != INVALID_INDEX && i < MAX_CHILDREN as usize {
                let src_offset = child_idx as u64 * std::mem::size_of::<OthelloNodeInfo>() as u64;
                let dst_offset = i as u64 * std::mem::size_of::<OthelloNodeInfo>() as u64;
                encoder.copy_buffer_to_buffer(&self.node_info_buffer, src_offset, &self.node_info_staging, dst_offset, std::mem::size_of::<OthelloNodeInfo>() as u64);
            }
        }
        queue.submit(std::iter::once(encoder.finish()));

        // Read visits
        let slice = self.visits_staging.slice(..(num_children * 4) as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let visits_data: Vec<i32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.visits_staging.unmap();

        // Read wins
        let slice = self.wins_staging.slice(..(num_children * 4) as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let wins_data: Vec<i32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.wins_staging.unmap();

        // Read node info for move_ids
        let slice = self.node_info_staging.slice(..(num_children * std::mem::size_of::<OthelloNodeInfo>()) as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let info_data: &[OthelloNodeInfo] = bytemuck::cast_slice(&data);
        
        let mut stats = Vec::with_capacity(num_children);
        for (i, &child_idx) in children.iter().enumerate() {
            if child_idx == INVALID_INDEX || i >= num_children {
                continue;
            }
            
            let visits = visits_data[i];
            let wins = wins_data[i];
            let move_id = info_data[i].move_id;
            
            if move_id == INVALID_INDEX {
                if visits > 100 {
                    println!("[GPU-Native WARNING] Child {} has INVALID_INDEX move_id but {} visits!", 
                        i, visits);
                }
                continue;
            }

            let x = (move_id % 8) as usize;
            let y = (move_id / 8) as usize;
            // Q-value from root's perspective: children store wins from their (opponent's) perspective
            // Standard MCTS Q-value: wins normalized to [0, 1]
            // Backprop handles perspective, so no inversion needed here
            let q = if visits > 0 { 
                wins as f64 / (visits as f64 * 2.0)
            } else { 
                0.5
            };
            
            stats.push((x, y, visits, wins, q));
        }
        
        drop(data);
        self.node_info_staging.unmap();

        stats
    }

    pub fn get_total_nodes(&self) -> u32 {
        let device = self.context.device();
        let queue = self.context.queue();

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Read Alloc Counter"),
        });
        encoder.copy_buffer_to_buffer(&self.alloc_counter_buffer, 0, &self.alloc_staging, 0, 4);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.alloc_staging.slice(..4);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let count: u32 = *bytemuck::from_bytes(&data);
        drop(data);
        self.alloc_staging.unmap();

        count
    }

    /// Build a visit histogram by depth to understand search spread
    /// Bins are [0, max_bins - 1]; the last bin is an overflow bucket.
    pub fn get_depth_visit_histogram(&self, max_bins: usize) -> Vec<u32> {
        if max_bins == 0 {
            return Vec::new();
        }

        let total_nodes = self.get_total_nodes().min(self.max_nodes);
        if total_nodes == 0 {
            return Vec::new();
        }

        let device = self.context.device();
        let queue = self.context.queue();

        let node_bytes = total_nodes as usize * std::mem::size_of::<OthelloNodeInfo>();
        let visit_bytes = total_nodes as usize * std::mem::size_of::<i32>();

        let node_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Depth Histogram Node Staging"),
            size: node_bytes as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visit_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Depth Histogram Visit Staging"),
            size: visit_bytes as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Copy Depth Histogram Buffers"),
        });
        encoder.copy_buffer_to_buffer(&self.node_info_buffer, 0, &node_staging, 0, node_bytes as u64);
        encoder.copy_buffer_to_buffer(&self.node_visits_buffer, 0, &visit_staging, 0, visit_bytes as u64);
        queue.submit(std::iter::once(encoder.finish()));

        let node_slice = node_staging.slice(..node_bytes as u64);
        node_slice.map_async(wgpu::MapMode::Read, |_| {});
        let visit_slice = visit_staging.slice(..visit_bytes as u64);
        visit_slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let node_data = node_slice.get_mapped_range();
        let visit_data = visit_slice.get_mapped_range();
        let nodes: &[OthelloNodeInfo] = bytemuck::cast_slice(&node_data);
        let visits: &[i32] = bytemuck::cast_slice(&visit_data);

        let mut depths = vec![0u32; total_nodes as usize];
        for idx in 1..total_nodes as usize {
            let parent = nodes[idx].parent_idx;
            if parent == INVALID_INDEX {
                depths[idx] = 0;
                continue;
            }

            let parent_idx = parent as usize;
            if parent_idx >= idx || parent_idx >= depths.len() {
                depths[idx] = 0;
            } else {
                depths[idx] = depths[parent_idx].saturating_add(1);
            }
        }

        let mut histogram = vec![0u32; max_bins];
        for (idx, depth) in depths.iter().enumerate() {
            let visit_count = visits.get(idx).copied().unwrap_or(0);
            if visit_count <= 0 {
                continue;
            }

            let bin = if (*depth as usize) < max_bins {
                *depth as usize
            } else {
                max_bins - 1
            };
            histogram[bin] = histogram[bin].saturating_add(visit_count as u32);
        }

        drop(node_data);
        drop(visit_data);
        node_staging.unmap();
        visit_staging.unmap();

        // Trim trailing empty bins for cleaner logging, but keep at least the overflow bin
        while histogram.len() > 1 {
            if let Some(&last) = histogram.last() {
                if last == 0 {
                    histogram.pop();
                    continue;
                }
            }
            break;
        }

        histogram
    }

    /// Advance the root to the child corresponding to the given move
    /// This enables tree reuse between consecutive searches
    ///
    /// # Arguments
    /// * `move_x` - X coordinate of the move (column)
    /// * `move_y` - Y coordinate of the move (row)
    /// * `new_board` - The new board state after the move
    /// * `new_legal_moves` - Legal moves from the new position
    ///
    /// # Returns
    /// true if successfully advanced to existing child, false if had to reinitialize
    /// Uses pre-allocated staging buffers to avoid memory leaks
    pub fn advance_root(
        &mut self,
        move_x: usize,
        move_y: usize,
        new_board: &[i32; 64],
        new_player: i32,
        new_legal_moves: &[(usize, usize)],
    ) -> bool {
        let device = self.context.device();
        let queue = self.context.queue();
        let target_move_id = (move_y * 8 + move_x) as u32;

        // Early capacity guard: if allocator is near exhaustion, abort reuse
        {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Read Alloc Counter (advance_root)"),
            });
            encoder.copy_buffer_to_buffer(&self.alloc_counter_buffer, 0, &self.alloc_staging, 0, 4);
            queue.submit(std::iter::once(encoder.finish()));

            let slice = self.alloc_staging.slice(..4);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range();
            let alloc_now: u32 = *bytemuck::from_bytes(&data);
            drop(data);
            self.alloc_staging.unmap();

            if alloc_now >= self.max_nodes.saturating_sub(MAX_CHILDREN) {
                panic!(
                    "GPU-Native alloc counter near capacity during advance_root ({} / {}); aborting reuse",
                    alloc_now, self.max_nodes
                );
            }
        }

        // Read root node info using pre-allocated staging buffer
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Read Root Info"),
        });
        encoder.copy_buffer_to_buffer(
            &self.node_info_buffer,
            0,
            &self.node_info_staging,
            0,
            std::mem::size_of::<OthelloNodeInfo>() as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.node_info_staging.slice(..std::mem::size_of::<OthelloNodeInfo>() as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let root_info: OthelloNodeInfo = *bytemuck::from_bytes(&data);
        drop(data);
        self.node_info_staging.unmap();

        if root_info.num_children == 0 {
            panic!("GPU-Native advance_root: root has no children; aborting reuse");
        }

        let num_children = root_info.num_children.min(MAX_CHILDREN) as usize;

        // Read children indices using pre-allocated staging buffer
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Read Children Indices"),
        });
        encoder.copy_buffer_to_buffer(
            &self.children_indices_buffer,
            0,
            &self.children_staging,
            0,
            (num_children * std::mem::size_of::<u32>()) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.children_staging.slice(..(num_children * 4) as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let children: Vec<u32> = bytemuck::cast_slice(&data[..num_children * 4]).to_vec();
        drop(data);
        self.children_staging.unmap();

        // Batch read all children's node info to find the matching move
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Batch Read Children Info"),
        });
        for (i, &child_idx) in children.iter().enumerate() {
            if child_idx != INVALID_INDEX && i < MAX_CHILDREN as usize {
                let src_offset = child_idx as u64 * std::mem::size_of::<OthelloNodeInfo>() as u64;
                let dst_offset = i as u64 * std::mem::size_of::<OthelloNodeInfo>() as u64;
                encoder.copy_buffer_to_buffer(&self.node_info_buffer, src_offset, &self.node_info_staging, dst_offset, std::mem::size_of::<OthelloNodeInfo>() as u64);
            }
        }
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.node_info_staging.slice(..(num_children * std::mem::size_of::<OthelloNodeInfo>()) as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let children_info: Vec<OthelloNodeInfo> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.node_info_staging.unmap();

        // Build a small summary for diagnostics in the mismatch case
        let mut child_summaries: Vec<String> = Vec::with_capacity(num_children);
        for (i, &child_idx) in children.iter().enumerate() {
            if child_idx == INVALID_INDEX {
                continue;
            }
            let info = children_info[i];
            let mv = info.move_id;
            let (mx, my) = if mv == INVALID_INDEX {
                (usize::MAX, usize::MAX)
            } else {
                ((mv % 8) as usize, (mv / 8) as usize)
            };
            child_summaries.push(format!(
                "slot={} idx={} move_id={} ({},{}) kids={}",
                i, child_idx, mv, mx, my, info.num_children
            ));
        }

        // Read current GPU root board to validate the tree state before re-rooting
        let root_board_vals = self.read_root_board();

        // Validate that applying the host move to the GPU root board reproduces the provided new_board
        if let Some(simulated_next) = Self::apply_othello_move_host(
            &root_board_vals,
            move_x,
            move_y,
            root_info.player_at_node,
        ) {
            if simulated_next != *new_board {
                let gpu_hash = Self::fnv_hash_board(&root_board_vals);
                let simulated_hash = Self::fnv_hash_board(&simulated_next);
                let host_hash = Self::fnv_hash_board(new_board);
                let gpu_ascii = Self::board_to_ascii(&root_board_vals);
                let simulated_ascii = Self::board_to_ascii(&simulated_next);
                let host_ascii = Self::board_to_ascii(new_board);
                eprintln!(
                    "[GPU-Native] advance_root drift detected; rebuilding tree. gpu_hash={} simulated_hash={} host_hash={} move=({}, {}) gpu_board={:?} simulated_board={:?} host_board={:?}",
                    gpu_hash,
                    simulated_hash,
                    host_hash,
                    move_x,
                    move_y,
                    gpu_ascii,
                    simulated_ascii,
                    host_ascii
                );

                // Reinitialize tree to eliminate drift
                self.init_tree(new_board, new_player, new_legal_moves);
                return false;
            }
        } else {
            // Illegal move w.r.t GPU root board; rebuild to stay safe
            let gpu_hash = Self::fnv_hash_board(&root_board_vals);
            let host_hash = Self::fnv_hash_board(new_board);
            let gpu_ascii = Self::board_to_ascii(&root_board_vals);
            let host_ascii = Self::board_to_ascii(new_board);
            eprintln!(
                "[GPU-Native] advance_root move illegal on GPU root board; rebuilding tree. gpu_hash={} host_hash={} move=({}, {}) gpu_board={:?} host_board={:?}",
                gpu_hash,
                host_hash,
                move_x,
                move_y,
                gpu_ascii,
                host_ascii
            );
            self.init_tree(new_board, new_player, new_legal_moves);
            return false;
        }

        // Build actual move set from GPU children for invariant checking (current root position)
        let mut actual_moves: Vec<(usize, usize)> = children
            .iter()
            .enumerate()
            .filter_map(|(i, &child_idx)| {
                if child_idx == INVALID_INDEX {
                    return None;
                }
                let mv = children_info[i].move_id;
                if mv == INVALID_INDEX {
                    return None;
                }
                Some(((mv % 8) as usize, (mv / 8) as usize))
            })
            .collect();

        // Expected legal moves list (host computed for the *new* position) for later diagnostics
        let mut expected_moves: Vec<String> = Vec::with_capacity(new_legal_moves.len());
        for (x, y) in new_legal_moves.iter().copied() {
            expected_moves.push(format!("({}, {})", x, y));
        }

        // Validate the current root children against the GPU root board + player to catch drift
        let mut expected_root_moves =
            Self::compute_othello_legal_moves_host(&root_board_vals, root_info.player_at_node);

        actual_moves.sort_unstable();
        expected_root_moves.sort_unstable();

        if actual_moves != expected_root_moves {
            let board_hash = Self::fnv_hash_board(&root_board_vals);
            let board_ascii = Self::board_to_ascii(&root_board_vals);

            panic!(
                "GPU-Native advance_root root children mismatch vs GPU root board/player. actual={:?} expected={:?} player_at_root={} board_hash={} board_ascii={:?}; children: {:?}",
                actual_moves,
                expected_root_moves,
                root_info.player_at_node,
                board_hash,
                board_ascii,
                child_summaries
            );
        }

        // Find the child with the matching move
        let mut found_child_idx: Option<u32> = None;
        let mut found_child_slot: Option<usize> = None;
        let mut found_child_num_children: u32 = 0;

        for (i, &child_idx) in children.iter().enumerate() {
            if child_idx == INVALID_INDEX {
                continue;
            }
            if children_info[i].move_id == target_move_id {
                found_child_idx = Some(child_idx);
                found_child_slot = Some(i);
                found_child_num_children = children_info[i].num_children;
                break;
            }
        }

        let _ = found_child_slot; // Suppress unused warning

        match found_child_idx {
            Some(child_idx) => {
                // Found the child! Copy its subtree to become the new root
                // Read child's stats using pre-allocated staging buffers
                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Read Child Stats"),
                });
                let stats_offset = child_idx as u64 * 4;
                encoder.copy_buffer_to_buffer(&self.node_visits_buffer, stats_offset, &self.visits_staging, 0, 4);
                encoder.copy_buffer_to_buffer(&self.node_wins_buffer, stats_offset, &self.wins_staging, 0, 4);
                queue.submit(std::iter::once(encoder.finish()));

                let slice = self.visits_staging.slice(..4);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::Maintain::Wait);
                let data = slice.get_mapped_range();
                let child_visits: i32 = *bytemuck::from_bytes(&data);
                drop(data);
                self.visits_staging.unmap();

                let slice = self.wins_staging.slice(..4);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::Maintain::Wait);
                let data = slice.get_mapped_range();
                let child_wins: i32 = *bytemuck::from_bytes(&data);
                drop(data);
                self.wins_staging.unmap();

                println!("[GPU-Native] Advancing root to child {} (move {},{}) with {} visits, {} wins",
                    child_idx, move_x, move_y, child_visits, child_wins);

                // Update root board
                queue.write_buffer(&self.root_board_buffer, 0, bytemuck::cast_slice(new_board));

                // Read child's children info using pre-allocated children_staging
                let child_children_offset = child_idx as u64 * MAX_CHILDREN as u64 * 4;

                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Read Child Children"),
                });
                encoder.copy_buffer_to_buffer(&self.children_indices_buffer, child_children_offset, &self.children_staging, 0, (MAX_CHILDREN as usize * 4) as u64);
                queue.submit(std::iter::once(encoder.finish()));

                let slice = self.children_staging.slice(..(MAX_CHILDREN as usize * 4) as u64);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::Maintain::Wait);

                let data = slice.get_mapped_range();
                let mut child_children: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                self.children_staging.unmap();

                // Filter out invalid children based on num_children
                // The buffer might contain garbage/zeros beyond num_children
                let valid_count = found_child_num_children.min(MAX_CHILDREN) as usize;
                
                // Ensure the vector is clean: keep valid ones, set rest to INVALID_INDEX
                for i in 0..MAX_CHILDREN as usize {
                    if i >= valid_count {
                        child_children[i] = INVALID_INDEX;
                    }
                }

                // Count valid grandchildren
                let grandchild_count = child_children.iter().filter(|&&idx| idx != INVALID_INDEX).count() as u32;

                // Create new root info (the child becomes root)
                let new_root_info = OthelloNodeInfo {
                    parent_idx: INVALID_INDEX, // Root has no parent
                    move_id: INVALID_INDEX, // Root has no associated move
                    num_children: if grandchild_count > 0 { grandchild_count } else { new_legal_moves.len() as u32 },
                    player_at_node: new_player,
                };

                // Write new root info to index 0
                queue.write_buffer(&self.node_info_buffer, 0, bytemuck::bytes_of(&new_root_info));
                queue.write_buffer(&self.node_visits_buffer, 0, bytemuck::bytes_of(&child_visits));
                queue.write_buffer(&self.node_wins_buffer, 0, bytemuck::bytes_of(&child_wins));
                queue.write_buffer(&self.node_vl_buffer, 0, bytemuck::bytes_of(&0i32));
                queue.write_buffer(&self.node_state_buffer, 0, bytemuck::bytes_of(&NODE_STATE_READY));

                // Rebuild children fresh for the new position to avoid stale/mismatched subtree
                let opposite_player = -new_player;
                let uniform_prior = 1.0 / new_legal_moves.len().max(1) as f32;

                // Reset root metadata/stats to match the new legal move set
                let refreshed_root_info = OthelloNodeInfo {
                    parent_idx: INVALID_INDEX,
                    move_id: INVALID_INDEX,
                    num_children: new_legal_moves.len().min(MAX_CHILDREN as usize) as u32,
                    player_at_node: new_player,
                };
                queue.write_buffer(&self.node_info_buffer, 0, bytemuck::bytes_of(&refreshed_root_info));
                queue.write_buffer(&self.node_visits_buffer, 0, bytemuck::bytes_of(&0i32));
                queue.write_buffer(&self.node_wins_buffer, 0, bytemuck::bytes_of(&0i32));
                queue.write_buffer(&self.node_vl_buffer, 0, bytemuck::bytes_of(&0i32));
                queue.write_buffer(&self.node_state_buffer, 0, bytemuck::bytes_of(&NODE_STATE_READY));
                
                // Read current alloc counter
                let alloc_staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Alloc Staging"),
                    size: 4,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
                encoder.copy_buffer_to_buffer(&self.alloc_counter_buffer, 0, &alloc_staging, 0, 4);
                queue.submit(std::iter::once(encoder.finish()));

                let slice = alloc_staging.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::Maintain::Wait);
                let data = slice.get_mapped_range();
                let mut alloc_count: u32 = *bytemuck::from_bytes(&data);
                drop(data);
                alloc_staging.unmap();

                let mut new_child_indices = vec![INVALID_INDEX; MAX_CHILDREN as usize];
                let mut new_child_priors = vec![0.0f32; MAX_CHILDREN as usize];

                for (i, &(x, y)) in new_legal_moves.iter().enumerate() {
                    if i >= MAX_CHILDREN as usize {
                        break;
                    }

                    if alloc_count >= self.max_nodes {
                        panic!("GPU-Native out of nodes during advance_root (alloc_count={} max_nodes={})", alloc_count, self.max_nodes);
                    }

                    let new_child_idx = alloc_count;
                    alloc_count += 1;
                    let move_id = (y * 8 + x) as u32;

                    let child_info = OthelloNodeInfo {
                        parent_idx: 0,
                        move_id,
                        num_children: 0,
                        player_at_node: opposite_player,
                    };

                    let offset = new_child_idx as usize * std::mem::size_of::<OthelloNodeInfo>();
                    if offset as u64 + std::mem::size_of::<OthelloNodeInfo>() as u64 <= self.node_info_buffer.size() {
                        queue.write_buffer(&self.node_info_buffer, offset as u64, bytemuck::bytes_of(&child_info));
                    }

                    let stat_offset = new_child_idx as usize * std::mem::size_of::<i32>();
                    if stat_offset as u64 + 4 <= self.node_visits_buffer.size() {
                        queue.write_buffer(&self.node_visits_buffer, stat_offset as u64, bytemuck::bytes_of(&0i32));
                        queue.write_buffer(&self.node_wins_buffer, stat_offset as u64, bytemuck::bytes_of(&0i32));
                        queue.write_buffer(&self.node_vl_buffer, stat_offset as u64, bytemuck::bytes_of(&0i32));
                        queue.write_buffer(&self.node_state_buffer, stat_offset as u64, bytemuck::bytes_of(&NODE_STATE_READY));
                    }

                    new_child_indices[i] = new_child_idx;
                    new_child_priors[i] = uniform_prior;
                }

                queue.write_buffer(&self.children_indices_buffer, 0, bytemuck::cast_slice(&new_child_indices));
                queue.write_buffer(&self.children_priors_buffer, 0, bytemuck::cast_slice(&new_child_priors));
                queue.write_buffer(&self.alloc_counter_buffer, 0, bytemuck::bytes_of(&alloc_count));

                self.create_bind_groups();
                true
            }
            None => {
                panic!(
                    "GPU-Native advance_root move ({},{}) not found in children (root num_children={}, slots_checked={}); children: {:?}; expected: {:?}",
                    move_x,
                    move_y,
                    root_info.num_children,
                    num_children,
                    child_summaries,
                    expected_moves
                );
            }
        }
    }
}
