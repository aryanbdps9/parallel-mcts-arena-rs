//! GPU-Native MCTS for Othello - Clean Rust Implementation
//!
//! This module provides GPU-native MCTS for Othello with complete tree reuse across turns.
//!
//! ## Architecture
//! - Root board buffer: Holds current game state
//! - Root node (index 0): Standard node with parent=INVALID, move=INVALID
//! - All nodes represent game states via path from root
//! - State reconstruction: root_board + apply moves along path
//! - No transposition: Same move from different parents = different nodes
//! 
//! ## Key Operations
//! - init_tree: Initialize tree with root position and its children
//! - run_iterations: Run GPU MCTS iterations (selection, expansion, simulation, backprop)
//! - advance_root: Move a child to root, keep its subtree, free siblings
//! - get_children_stats: Extract visit counts and values for policy

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
const INVALID_INDEX: u32 = 0xFFFFFFFF;
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
    free_list_buffer: Buffer,
    free_top_buffer: Buffer,

    // Execution state buffers
    params_buffer: Buffer,
    work_items_buffer: Buffer,
    paths_buffer: Buffer,
    alloc_counter_buffer: Buffer,
    diagnostics_buffer: Buffer,

    // Root board buffer
    root_board_buffer: Buffer,

    // Staging buffers for readback
    node_info_staging: Buffer,
    children_staging: Buffer,
    priors_staging: Buffer,
    visits_staging: Buffer,
    wins_staging: Buffer,
    alloc_staging: Buffer,
    free_top_staging: Buffer,
    diagnostics_staging: Buffer,

    // Bind groups
    node_pool_bind_group: Option<BindGroup>,
    execution_bind_group: Option<BindGroup>,
    board_bind_group: Option<BindGroup>,

    // Configuration
    max_nodes: u32,
}

impl GpuOthelloMcts {
    /// Create a new GPU Othello MCTS engine
    pub fn new(
        context: Arc<GpuContext>,
        max_nodes: u32,
        max_iterations: u32,
    ) -> Self {
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

        // Create node pool buffers
        let node_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Info Buffer"),
            size: (max_nodes as u64) * std::mem::size_of::<OthelloNodeInfo>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let node_visits_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Visits Buffer"),
            size: (max_nodes as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let node_wins_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Wins Buffer"),
            size: (max_nodes as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let node_vl_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Virtual Loss Buffer"),
            size: (max_nodes as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let node_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node State Buffer"),
            size: (max_nodes as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let children_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Children Indices Buffer"),
            size: (max_nodes as u64) * (MAX_CHILDREN as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let children_priors_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Children Priors Buffer"),
            size: (max_nodes as u64) * (MAX_CHILDREN as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let free_list_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Free List Buffer"),
            size: (max_nodes as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let free_top_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Free Top Buffer"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create execution state buffers
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<MctsOthelloParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let work_items_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Work Items Buffer"),
            size: (max_iterations as u64) * 32,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let paths_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Paths Buffer"),
            size: (max_iterations as u64) * 128 * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let alloc_counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alloc Counter Buffer"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let diagnostics_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Diagnostics Buffer"),
            size: std::mem::size_of::<OthelloDiagnostics>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Root board buffer (8x8 = 64 cells, i32 each)
        let root_board_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Root Board Buffer"),
            size: 64 * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffers
        let node_info_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Info Staging"),
            size: (MAX_CHILDREN as u64) * std::mem::size_of::<OthelloNodeInfo>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let children_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Children Staging"),
            size: (MAX_CHILDREN as u64) * 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let priors_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Priors Staging"),
            size: (MAX_CHILDREN as u64) * 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visits_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visits Staging"),
            size: (MAX_CHILDREN as u64) * 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let wins_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Wins Staging"),
            size: (MAX_CHILDREN as u64) * 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let alloc_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alloc Staging"),
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let free_top_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Free Top Staging"),
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

        let mut engine = Self {
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
            free_list_buffer,
            free_top_buffer,
            params_buffer,
            work_items_buffer,
            paths_buffer,
            alloc_counter_buffer,
            diagnostics_buffer,
            root_board_buffer,
            node_info_staging,
            children_staging,
            priors_staging,
            visits_staging,
            wins_staging,
            alloc_staging,
            free_top_staging,
            diagnostics_staging,
            node_pool_bind_group: None,
            execution_bind_group: None,
            board_bind_group: None,
            max_nodes,
        };

        engine.create_bind_groups();
        engine
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
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 8,
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
                BindGroupEntry {
                    binding: 7,
                    resource: self.free_list_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: self.free_top_buffer.as_entire_binding(),
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
                num_children: 0,
                player_at_node: opposite_player,
            };

            let offset = child_idx as u64 * std::mem::size_of::<OthelloNodeInfo>() as u64;
            queue.write_buffer(&self.node_info_buffer, offset, bytemuck::bytes_of(&child_info));

            let stat_offset = child_idx as u64 * 4;
            queue.write_buffer(&self.node_visits_buffer, stat_offset, bytemuck::bytes_of(&0i32));
            queue.write_buffer(&self.node_wins_buffer, stat_offset, bytemuck::bytes_of(&0i32));
            queue.write_buffer(&self.node_vl_buffer, stat_offset, bytemuck::bytes_of(&0i32));
            queue.write_buffer(&self.node_state_buffer, stat_offset, bytemuck::bytes_of(&NODE_STATE_READY));

            child_indices[i] = child_idx;
            child_priors[i] = uniform_prior;
        }

        queue.write_buffer(&self.children_indices_buffer, 0, bytemuck::cast_slice(&child_indices));
        queue.write_buffer(&self.children_priors_buffer, 0, bytemuck::cast_slice(&child_priors));

        // Reset free list
        queue.write_buffer(&self.free_top_buffer, 0, bytemuck::bytes_of(&0u32));

        // Set allocation counter
        let alloc_count = (legal_moves.len() + 1) as u32;
        queue.write_buffer(&self.alloc_counter_buffer, 0, bytemuck::bytes_of(&alloc_count));

        // Reset diagnostics
        let zero_diag = OthelloDiagnostics::default();
        queue.write_buffer(&self.diagnostics_buffer, 0, bytemuck::bytes_of(&zero_diag));
    }

    /// Run MCTS iterations on GPU
    pub fn run_iterations(
        &mut self,
        num_iterations: u32,
        exploration: f32,
        virtual_loss_weight: f32,
        seed: u32,
    ) -> OthelloRunTelemetry {
        let queue = self.context.queue();
        let device = self.context.device();

        // Set parameters
        let params = MctsOthelloParams {
            num_iterations,
            max_nodes: self.max_nodes,
            exploration,
            virtual_loss_weight,
            root_idx: 0,
            seed,
            board_width: 8,
            board_height: 8,
            game_type: 0,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Reset diagnostics
        let zero_diag = OthelloDiagnostics::default();
        queue.write_buffer(&self.diagnostics_buffer, 0, bytemuck::bytes_of(&zero_diag));

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("MCTS Iteration"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MCTS Compute Pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.iteration_pipeline);
            cpass.set_bind_group(0, self.node_pool_bind_group.as_ref().unwrap(), &[]);
            cpass.set_bind_group(1, self.execution_bind_group.as_ref().unwrap(), &[]);
            cpass.set_bind_group(2, self.board_bind_group.as_ref().unwrap(), &[]);

            let workgroup_size = 64;
            let num_workgroups = (num_iterations + workgroup_size - 1) / workgroup_size;
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Copy diagnostics and alloc counter for readback
        encoder.copy_buffer_to_buffer(
            &self.diagnostics_buffer,
            0,
            &self.diagnostics_staging,
            0,
            std::mem::size_of::<OthelloDiagnostics>() as u64,
        );
        encoder.copy_buffer_to_buffer(&self.alloc_counter_buffer, 0, &self.alloc_staging, 0, 4);

        queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let diagnostics = self.read_diagnostics();
        let alloc_count = self.read_u32(&self.alloc_staging);

        OthelloRunTelemetry {
            iterations_launched: num_iterations,
            alloc_count_after: alloc_count,
            node_capacity: self.max_nodes,
            saturated: alloc_count >= self.max_nodes,
            diagnostics,
        }
    }

    /// Get children statistics for root node
    pub fn get_children_stats(&self) -> Vec<(usize, usize, i32, i32, f64)> {
        let device = self.context.device();
        let queue = self.context.queue();

        // Read root node info
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

        let root_info = self.read_node_info(0);
        let num_children = root_info.num_children.min(MAX_CHILDREN) as usize;

        if num_children == 0 {
            return Vec::new();
        }

        // Read children indices, visits, and wins
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Read Children Stats"),
        });
        encoder.copy_buffer_to_buffer(
            &self.children_indices_buffer,
            0,
            &self.children_staging,
            0,
            (num_children * 4) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let children_indices = self.read_children_indices(num_children);

        // Read stats for each child
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Batch Read Child Stats"),
        });
        for i in 0..num_children {
            let child_idx = children_indices[i];
            if child_idx != INVALID_INDEX {
                let offset = child_idx as u64 * 4;
                let dst_offset = i as u64 * 4;
                encoder.copy_buffer_to_buffer(&self.node_visits_buffer, offset, &self.visits_staging, dst_offset, 4);
                encoder.copy_buffer_to_buffer(&self.node_wins_buffer, offset, &self.wins_staging, dst_offset, 4);
            }
        }
        queue.submit(std::iter::once(encoder.finish()));

        let visits = self.read_i32_array(&self.visits_staging, num_children);
        let wins = self.read_i32_array(&self.wins_staging, num_children);

        // Also need to read the node info for each child to get move_id
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Batch Read Child Info"),
        });
        for i in 0..num_children {
            let child_idx = children_indices[i];
            if child_idx != INVALID_INDEX {
                let src_offset = child_idx as u64 * std::mem::size_of::<OthelloNodeInfo>() as u64;
                let dst_offset = i as u64 * std::mem::size_of::<OthelloNodeInfo>() as u64;
                encoder.copy_buffer_to_buffer(&self.node_info_buffer, src_offset, &self.node_info_staging, dst_offset, std::mem::size_of::<OthelloNodeInfo>() as u64);
            }
        }
        queue.submit(std::iter::once(encoder.finish()));

        let children_info = self.read_node_info_array(num_children);

        // Build result
        let mut result = Vec::new();
        for i in 0..num_children {
            let child_idx = children_indices[i];
            if child_idx == INVALID_INDEX {
                continue;
            }

            let move_id = children_info[i].move_id;
            let x = (move_id % 8) as usize;
            let y = (move_id / 8) as usize;
            let v = visits[i];
            let w = wins[i];
            let q = if v > 0 { w as f64 / (2.0 * v as f64) } else { 0.0 };

            result.push((x, y, v, w, q));
        }

        result
    }

    /// Get root visits
    pub fn get_root_visits(&self) -> i32 {
        let device = self.context.device();
        let queue = self.context.queue();

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Read Root Visits"),
        });
        encoder.copy_buffer_to_buffer(&self.node_visits_buffer, 0, &self.visits_staging, 0, 4);
        queue.submit(std::iter::once(encoder.finish()));

        self.read_i32(&self.visits_staging)
    }

    /// Get total allocated nodes
    pub fn get_total_nodes(&self) -> u32 {
        self.read_u32(&self.alloc_staging)
    }

    /// Get hash of root board
    pub fn get_root_board_hash(&self) -> u64 {
        let device = self.context.device();
        let queue = self.context.queue();

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Board Staging"),
            size: 64 * 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Read Root Board"),
        });
        encoder.copy_buffer_to_buffer(&self.root_board_buffer, 0, &staging, 0, 64 * 4);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let board: &[i32; 64] = bytemuck::from_bytes(&data);

        let mut hash: u64 = 0xcbf29ce484222325;
        for &v in board.iter() {
            hash ^= v as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }

        drop(data);
        staging.unmap();

        hash
    }

    /// Flush and wait for GPU to complete
    pub fn flush_and_wait(&self) {
        self.context.device().poll(wgpu::Maintain::Wait);
    }

    // Helper functions for reading GPU data
    fn read_u32(&self, buffer: &Buffer) -> u32 {
        let device = self.context.device();
        let slice = buffer.slice(..4);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let val = *bytemuck::from_bytes(&data);
        drop(data);
        buffer.unmap();
        val
    }

    fn read_i32(&self, buffer: &Buffer) -> i32 {
        let device = self.context.device();
        let slice = buffer.slice(..4);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let val = *bytemuck::from_bytes(&data);
        drop(data);
        buffer.unmap();
        val
    }

    fn read_diagnostics(&self) -> OthelloDiagnostics {
        let device = self.context.device();
        let slice = self.diagnostics_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let val = *bytemuck::from_bytes(&data);
        drop(data);
        self.diagnostics_staging.unmap();
        val
    }

    fn read_node_info(&self, idx: usize) -> OthelloNodeInfo {
        let device = self.context.device();
        let offset = idx as u64 * std::mem::size_of::<OthelloNodeInfo>() as u64;
        let size = std::mem::size_of::<OthelloNodeInfo>() as u64;
        let slice = self.node_info_staging.slice(offset..offset + size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let val = *bytemuck::from_bytes(&data);
        drop(data);
        self.node_info_staging.unmap();
        val
    }

    fn read_node_info_array(&self, count: usize) -> Vec<OthelloNodeInfo> {
        let device = self.context.device();
        let size = count as u64 * std::mem::size_of::<OthelloNodeInfo>() as u64;
        let slice = self.node_info_staging.slice(..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let val = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.node_info_staging.unmap();
        val
    }

    fn read_children_indices(&self, count: usize) -> Vec<u32> {
        let device = self.context.device();
        let size = count as u64 * 4;
        let slice = self.children_staging.slice(..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let val = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.children_staging.unmap();
        val
    }

    fn read_i32_array(&self, buffer: &Buffer, count: usize) -> Vec<i32> {
        let device = self.context.device();
        let size = count as u64 * 4;
        let slice = buffer.slice(..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let val = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffer.unmap();
        val
    }

    /// Advance root to a child node (stub - not implemented in clean version)
    pub fn advance_root(
        &mut self,
        _move_x: usize,
        _move_y: usize,
        new_board: &[i32; 64],
        new_player: i32,
        new_legal_moves: &[(usize, usize)],
    ) -> bool {
        // For now, just rebuild the tree
        self.init_tree(new_board, new_player, new_legal_moves);
        false  // Indicates we rebuilt instead of reused
    }

    /// Get best move (for compatibility)
    pub fn get_best_move(&self) -> Option<(usize, usize, i32, f64)> {
        let stats = self.get_children_stats();
        if stats.is_empty() {
            return None;
        }

        // Find child with most visits
        let best = stats.iter().max_by_key(|(_, _, v, _, _)| *v)?;
        Some((best.0, best.1, best.2, best.4))  // (x, y, visits, q)
    }

    /// Get depth visit histogram (stub for compatibility)
    pub fn get_depth_visit_histogram(&self, _max_depth: u32) -> Vec<u32> {
        // Return empty histogram - this was a diagnostic feature
        Vec::new()
    }
}
