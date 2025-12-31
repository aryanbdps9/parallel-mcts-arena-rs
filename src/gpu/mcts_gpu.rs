/// Struct to hold urgent event buffers for fine-grained locking
pub struct UrgentEventBuffers {
    pub urgent_event_buffer: Buffer,
    pub urgent_event_staging_buffer: Buffer,
    pub urgent_event_write_head_buffer: Buffer,
    pub urgent_event_write_head_staging: Buffer,
    /// Atomic flag for synchronizing buffer access
    pub urgent_event_buffer_in_use: std::sync::atomic::AtomicBool,
}
/// GPU-Native MCTS Implementation
///
/// This module provides a fully GPU-based Monte Carlo Tree Search where all four phases
/// (selection, expansion, simulation, backpropagation) run on the GPU without CPU-GPU
/// synchronization during iterations.
///
/// ## Architecture
/// - Pre-allocated node pool on GPU (no dynamic allocation during search)
/// - Index-based tree structure (no pointers)
/// - Atomic operations for thread coordination
/// - Virtual losses prevent path convergence
///
/// ## Benefits over Hybrid Approach
/// - No stale path problem (paths are always current)
/// - No CPU-GPU sync overhead during iterations
/// - True parallel MCTS with coherent tree state

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages,
    CommandEncoderDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineLayoutDescriptor,
    ShaderModuleDescriptor, ShaderStages,
};

use super::context::GpuContext;
use super::shaders::MCTS_TREE_SHADER;

// =============================================================================
// Constants (must match shader)
// =============================================================================

const MAX_CHILDREN: u32 = 64;
const MAX_PATH_LENGTH: u32 = 128;
const INVALID_INDEX: u32 = 0xFFFFFFFF;
// Must match WGSL error codes
const SELECT_BEST_CHILD_NO_CHILDREN: u32 = 0xFFFFFFFE;
const SELECT_BEST_CHILD_NO_VALID: u32 = 0xFFFFFFFD;
const SELECT_BEST_CHILD_SOFTMAX_PANIC: u32 = 0xFFFFFFFC;
const WORKGROUP_SIZE: u32 = 64;

// Node states
const NODE_STATE_READY: u32 = 2;

// =============================================================================
// Data Structures (must match shader layout exactly)
// =============================================================================

/// MCTS parameters sent to GPU
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct MctsParams {
    pub num_iterations: u32,
    pub max_nodes: u32,
    pub exploration: f32,
    pub root_idx: u32,
    pub seed: u32,
    pub board_width: u32,
    pub board_height: u32,
    pub game_type: u32,
}

/// Non-atomic node info (read-mostly after initialization)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct NodeInfo {
    pub parent_idx: u32,
    pub move_id: u32,
    pub num_children: u32,
    pub player_at_node: i32,
}

/// Per-iteration work tracking
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct WorkItem {
    pub current_node: u32,
    pub leaf_node: u32,
    pub path_length: u32,
    pub status: u32,
    pub sim_result: i32,
    pub leaf_player: i32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Statistics from the tree
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct TreeStats {
    pub total_nodes: u32,
    pub root_visits: i32,
    pub root_wins: i32,
    pub _pad: u32,
}

/// Child node data for best move selection
#[derive(Debug, Clone)]
pub struct ChildStats {
    pub move_id: u32,
    pub visits: i32,
    pub wins: i32,
    pub q_value: f64,
}

// =============================================================================
// GPU MCTS Engine
// =============================================================================

/// GPU-native MCTS engine
///
/// Manages the GPU resources for running MCTS entirely on the GPU.
pub struct GpuMctsEngine {
    pub context: Arc<GpuContext>,

    // Compute pipelines
    select_pipeline: ComputePipeline,
    backprop_pipeline: ComputePipeline,
    stats_pipeline: ComputePipeline,

    // Bind group layouts
    node_pool_layout: BindGroupLayout,
    execution_layout: BindGroupLayout,
    game_state_layout: BindGroupLayout,
    stats_layout: BindGroupLayout,

    // Node pool buffers
    node_info_buffer: Buffer,
    node_visits_buffer: Buffer,
    node_wins_buffer: Buffer,
    node_vl_buffer: Buffer,
    node_state_buffer: Buffer,
    children_indices_buffer: Buffer,
    children_priors_buffer: Buffer,
    // Hybrid allocator buffers
    free_lists_buffer: Buffer,        // [256][8192] u32s, per-workgroup free lists
    free_tops_buffer: Buffer,         // [256] u32s, per-workgroup free list tops

    // Execution state buffers
    params_buffer: Buffer,
    work_items_buffer: Buffer,
    paths_buffer: Buffer,
    alloc_counter_buffer: Buffer,

    // Game state buffers (for simulation boards)
    sim_boards_buffer: Buffer,

    // Stats buffer
    stats_buffer: Buffer,
    stats_staging_buffer: Buffer,

    // Urgent event buffers (lock-free)
    pub urgent_event_buffers: std::sync::Arc<UrgentEventBuffers>,

    // Bind groups (created per dispatch)
    pub node_pool_bind_group: Option<BindGroup>,
    pub execution_bind_group: Option<BindGroup>,
    pub game_state_bind_group: Option<BindGroup>,
    pub stats_bind_group: Option<BindGroup>,
    pub urgent_event_bind_group: Option<BindGroup>,

    // Configuration
    max_nodes: u32,
    _max_iterations: u32,
    _board_size: u32,
}

impl GpuMctsEngine {
    /// Log a custom urgent event from the CPU with a specific payload (for testing)
    pub fn log_urgent_event_from_cpu_with_payload(&self, event_type: u32, timestamp: u64, payload: &[u32; 255]) {
        let urgent_event_buffers = &self.urgent_event_buffers;
        while urgent_event_buffers.urgent_event_buffer_in_use.swap(true, std::sync::atomic::Ordering::Acquire) {
            std::thread::yield_now();
        }
        use crate::gpu::mcts_othello::UrgentEvent;
        let event = UrgentEvent {
            timestamp: timestamp as u32,
            event_type,
            _pad: 0,
            payload: *payload,
        };
        let queue = self.context.queue();
        let event_bytes: &[u8; std::mem::size_of::<UrgentEvent>()] = unsafe { std::mem::transmute(&event) };
        let device = self.context.device();
        let urgent_event_buffers = &self.urgent_event_buffers;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("UrgentEventWriteHeadRead"),
        });
        encoder.copy_buffer_to_buffer(
            &urgent_event_buffers.urgent_event_write_head_buffer,
            0,
            &urgent_event_buffers.urgent_event_write_head_staging,
            0,
            4,
        );
        queue.submit(Some(encoder.finish()));
        let slice = urgent_event_buffers.urgent_event_write_head_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let mut write_head = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        urgent_event_buffers.urgent_event_write_head_staging.unmap();
        let slot = (write_head % 256) as u64;
        let offset = slot * 1024;
        queue.write_buffer(&urgent_event_buffers.urgent_event_buffer, offset, event_bytes);
        write_head = write_head.wrapping_add(1);
        let write_head_bytes = write_head.to_le_bytes();
        queue.write_buffer(&urgent_event_buffers.urgent_event_write_head_buffer, 0, &write_head_bytes);
        urgent_event_buffers.urgent_event_buffer_in_use.store(false, std::sync::atomic::Ordering::Release);
    }

    /// Log a string as an UrgentEvent from the CPU into the urgent_event_buffer (for test/diagnostic)
    pub fn log_urgent_event_from_cpu(&self, event_type: u32, timestamp: u64, message: &str) {
                // Acquire atomic flag for exclusive buffer access
                let urgent_event_buffers = &self.urgent_event_buffers;
                while urgent_event_buffers.urgent_event_buffer_in_use.swap(true, std::sync::atomic::Ordering::Acquire) {
                    std::thread::yield_now();
                }
        use crate::gpu::mcts_othello::UrgentEvent;
        let mut payload = [0u32; 255];
        let msg_bytes = message.as_bytes();
        // Copy message bytes into payload as u32 words
        for (i, chunk) in msg_bytes.chunks(4).enumerate().take(255) {
            let mut word = [0u8; 4];
            for (j, b) in chunk.iter().enumerate() {
                word[j] = *b;
            }
            payload[i] = u32::from_le_bytes(word);
        }
        let event = UrgentEvent {
            timestamp: timestamp as u32,
            event_type,
            _pad: 0,
            payload,
        };
        let queue = self.context.queue();
        // SAFETY: UrgentEvent is #[repr(C)] and tightly packed for FFI/buffer transfer
        let event_bytes: &[u8; std::mem::size_of::<UrgentEvent>()] = unsafe { std::mem::transmute(&event) };

        // Read the current write head (synchronously)
        let device = self.context.device();
        let urgent_event_buffers = &self.urgent_event_buffers;
        // Copy write head to staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("UrgentEventWriteHeadRead"),
        });
        encoder.copy_buffer_to_buffer(
            &urgent_event_buffers.urgent_event_write_head_buffer,
            0,
            &urgent_event_buffers.urgent_event_write_head_staging,
            0,
            4,
        );
        queue.submit(Some(encoder.finish()));
        // Map and read
        let slice = urgent_event_buffers.urgent_event_write_head_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let mut write_head = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        urgent_event_buffers.urgent_event_write_head_staging.unmap();

        // Now safe to write to the buffers
        // Write the event at the next slot in the ring buffer
        let slot = (write_head % 256) as u64;
        let offset = slot * 1024;
        queue.write_buffer(&urgent_event_buffers.urgent_event_buffer, offset, event_bytes);

        // Increment the write head and write it back
        write_head = write_head.wrapping_add(1);
        let write_head_bytes = write_head.to_le_bytes();
        queue.write_buffer(&urgent_event_buffers.urgent_event_write_head_buffer, 0, &write_head_bytes);

        // Release atomic flag
        urgent_event_buffers.urgent_event_buffer_in_use.store(false, std::sync::atomic::Ordering::Release);
    }
    /// Create a new GPU MCTS engine
    ///
    /// # Arguments
    /// * `context` - GPU context with device and queue
    /// * `max_nodes` - Maximum number of nodes in the tree
    /// * `max_iterations` - Maximum parallel iterations per dispatch
    /// * `board_width` - Game board width
    /// * `board_height` - Game board height
    pub fn new(
        context: Arc<GpuContext>,
        _max_nodes: u32,
        max_iterations: u32,
        board_width: u32,
        board_height: u32,
    ) -> Self {
        eprintln!("[DIAG] GpuMctsEngine::new: before device");
        let device = context.device();
        eprintln!("[DIAG] GpuMctsEngine::new: after device");

        eprintln!("[DIAG] GpuMctsEngine::new: before free_lists_buffer");
        let free_lists_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Free Lists"),
            size: (256 * 8192 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        eprintln!("[DIAG] GpuMctsEngine::new: after free_lists_buffer");
        let free_tops_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Free Tops"),
            size: (256 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        eprintln!("[DIAG] GpuMctsEngine::new: after free_tops_buffer");

        eprintln!("[DIAG] GpuMctsEngine::new: before create_shader_module");
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("MCTS Tree Shader"),
            source: wgpu::ShaderSource::Wgsl(MCTS_TREE_SHADER.into()),
        });
        eprintln!("[DIAG] GpuMctsEngine::new: after create_shader_module");

        eprintln!("[DIAG] GpuMctsEngine::new: before create_node_pool_layout");
        let node_pool_layout = Self::create_node_pool_layout(device);
        eprintln!("[DIAG] GpuMctsEngine::new: after create_node_pool_layout");
        let execution_layout = Self::create_execution_layout(device);
        eprintln!("[DIAG] GpuMctsEngine::new: after create_execution_layout");
        let game_state_layout = Self::create_game_state_layout(device);
        eprintln!("[DIAG] GpuMctsEngine::new: after create_game_state_layout");
        let stats_layout = Self::create_stats_layout(device);
        eprintln!("[DIAG] GpuMctsEngine::new: after create_stats_layout");

        eprintln!("[DIAG] GpuMctsEngine::new: before create_pipeline_layout");
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("MCTS Pipeline Layout"),
            bind_group_layouts: &[
                &node_pool_layout,
                &execution_layout,
                &game_state_layout,
                &stats_layout,
            ],
            push_constant_ranges: &[],
        });
        eprintln!("[DIAG] GpuMctsEngine::new: after create_pipeline_layout");

        eprintln!("[DIAG] GpuMctsEngine::new: before create_select_pipeline");
        let select_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("MCTS Select Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("mcts_select"),
            compilation_options: Default::default(),
            cache: None,
        });
        eprintln!("[DIAG] GpuMctsEngine::new: after create_select_pipeline");

        eprintln!("[DIAG] GpuMctsEngine::new: before create_backprop_pipeline");
        let backprop_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("MCTS Backprop Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("mcts_backprop"),
            compilation_options: Default::default(),
            cache: None,
        });
        eprintln!("[DIAG] GpuMctsEngine::new: after create_backprop_pipeline");

        eprintln!("[DIAG] GpuMctsEngine::new: before create_stats_pipeline");
        let stats_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("MCTS Stats Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("mcts_get_stats"),
            compilation_options: Default::default(),
            cache: None,
        });
        eprintln!("[DIAG] GpuMctsEngine::new: after create_stats_pipeline");

        // Create buffers
        eprintln!("[DIAG] GpuMctsEngine::new: before board_size and node_info_buffer");
        let board_size = board_width * board_height;
        let limits = device.limits();
        let max_nodes = 2_000_000; // Restore to normal test value
        let node_info_size = (max_nodes as usize * std::mem::size_of::<NodeInfo>()) as u64;
        eprintln!("[DIAG] Device limits: max_buffer_size={} max_storage_buffer_binding_size={}", limits.max_buffer_size, limits.max_storage_buffer_binding_size);
        eprintln!("[DIAG] node_info_buffer requested size: {} (max_nodes={} * {})", node_info_size, max_nodes, std::mem::size_of::<NodeInfo>());
        eprintln!("[DIAG] before node_info_buffer");
        let node_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Info"),
            size: node_info_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        eprintln!("[DIAG] after node_info_buffer");

        eprintln!("[DIAG] before node_visits_buffer");
        let _node_visits_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Visits"),
            size: (max_nodes as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        eprintln!("[DIAG] after node_visits_buffer");

        eprintln!("[DIAG] before node_wins_buffer");
        let _node_wins_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Wins"),
            size: (max_nodes as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        eprintln!("[DIAG] after node_wins_buffer");

        eprintln!("[DIAG] before node_vl_buffer");
        let _node_vl_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node VL"),
            size: (max_nodes as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        eprintln!("[DIAG] after node_vl_buffer");

        eprintln!("[DIAG] before node_state_buffer");
        let _node_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node State"),
            size: (max_nodes as usize * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        eprintln!("[DIAG] after node_state_buffer");

        eprintln!("[DIAG] before children_indices_buffer");
        let _children_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Children Indices"),
            size: (max_nodes as usize * std::mem::size_of::<u32>() * 8) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        eprintln!("[DIAG] after children_indices_buffer");

        eprintln!("[DIAG] before children_priors_buffer");
        let _children_priors_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Children Priors"),
            size: (max_nodes as usize * std::mem::size_of::<f32>() * 8) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        eprintln!("[DIAG] after children_priors_buffer");

        let node_visits_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Visits"),
            size: (max_nodes as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let node_wins_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Wins"),
            size: (max_nodes as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let node_vl_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node VL"),
            size: (max_nodes as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let node_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node State"),
            size: (max_nodes as usize * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let children_size = max_nodes as usize * MAX_CHILDREN as usize;
        let children_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Children Indices"),
            size: (children_size * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let children_priors_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Children Priors"),
            size: (children_size * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MCTS Params"),
            size: std::mem::size_of::<MctsParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let work_items_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Work Items"),
            size: (max_iterations as usize * std::mem::size_of::<WorkItem>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let paths_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Paths"),
            size: (max_iterations as usize * MAX_PATH_LENGTH as usize * std::mem::size_of::<u32>())
                as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let alloc_counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alloc Counter"),
            size: std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sim_boards_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sim Boards"),
            size: (max_iterations as usize * board_size as usize * std::mem::size_of::<i32>())
                as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tree Stats"),
            size: std::mem::size_of::<TreeStats>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let stats_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Stats Staging"),
            size: std::mem::size_of::<TreeStats>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Urgent event buffer (256 x 1024 bytes)
        let urgent_event_buffers = std::sync::Arc::new(UrgentEventBuffers {
            urgent_event_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Urgent Event Buffer"),
                size: 264_192,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            urgent_event_staging_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Urgent Event Staging Buffer"),
                size: 264_192,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            urgent_event_write_head_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Urgent Event Write Head Buffer"),
                size: 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            urgent_event_write_head_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Urgent Event Write Head Staging"),
                size: 4,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            urgent_event_buffer_in_use: std::sync::atomic::AtomicBool::new(false),
        });

        let mut engine = Self {
            context: context.clone(),
            select_pipeline,
            backprop_pipeline,
            stats_pipeline,
            node_pool_layout,
            execution_layout,
            game_state_layout,
            stats_layout,
            node_info_buffer,
            node_visits_buffer,
            node_wins_buffer,
            node_vl_buffer,
            node_state_buffer,
            children_indices_buffer,
            children_priors_buffer,
            free_lists_buffer,
            free_tops_buffer,
            params_buffer,
            work_items_buffer,
            paths_buffer,
            alloc_counter_buffer,
            sim_boards_buffer,
            stats_buffer,
            stats_staging_buffer,
            urgent_event_buffers: urgent_event_buffers.clone(),
            node_pool_bind_group: None,
            execution_bind_group: None,
            game_state_bind_group: None,
            stats_bind_group: None,
            urgent_event_bind_group: None,
            max_nodes,
            _max_iterations: max_iterations,
            _board_size: board_size,
        };
        engine.create_bind_groups(&device);
        engine
    }

    /// Poll urgent events from the GPU ring buffer
    pub fn poll_urgent_events(&self) -> Vec<[u8; 1024]> {
        let urgent_event_buffers = &self.urgent_event_buffers;
        // Acquire atomic flag for exclusive buffer access
        while urgent_event_buffers.urgent_event_buffer_in_use.swap(true, std::sync::atomic::Ordering::Acquire) {
            std::thread::yield_now();
        }
        let device = self.context.device();
        let queue = self.context.queue();
        let urgent_event_buffers = &self.urgent_event_buffers;
        // Diagnostics: print buffer addresses and sizes
        // Copy urgent event buffer and write head to staging
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Urgent Event Poll Encoder"),
        });
        encoder.copy_buffer_to_buffer(&urgent_event_buffers.urgent_event_buffer, 0, &urgent_event_buffers.urgent_event_staging_buffer, 0, 256 * 1024);
        encoder.copy_buffer_to_buffer(&urgent_event_buffers.urgent_event_write_head_buffer, 0, &urgent_event_buffers.urgent_event_write_head_staging, 0, 4);
        queue.submit(std::iter::once(encoder.finish()));
        // Map and read write head
        let write_head_slice = urgent_event_buffers.urgent_event_write_head_staging.slice(..);
        write_head_slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let write_head_data = write_head_slice.get_mapped_range();
        let write_head = u32::from_le_bytes([write_head_data[0], write_head_data[1], write_head_data[2], write_head_data[3]]);
        drop(write_head_data);
        urgent_event_buffers.urgent_event_write_head_staging.unmap();
        // Map and read urgent event buffer
        let event_slice = urgent_event_buffers.urgent_event_staging_buffer.slice(..);
        event_slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let event_data = event_slice.get_mapped_range();
            if event_data.len() >= 16 {
            }
        let mut events = Vec::new();
        for i in 0..(write_head.min(256)) {
            let start = (i as usize) * 1024;
            let end = start + 1024;
            let mut event = [0u8; 1024];
            event.copy_from_slice(&event_data[start..end]);
            // Print first few bytes for diagnostics
            if i < 4 {
                eprintln!("[DIAG] urgent_event[{}] bytes: {:?}", i, &event[..16]);
            }
            events.push(event);
        }
        drop(event_data);
        urgent_event_buffers.urgent_event_staging_buffer.unmap();
        // Release atomic flag
        urgent_event_buffers.urgent_event_buffer_in_use.store(false, std::sync::atomic::Ordering::Release);
        events
    }

    fn create_node_pool_layout(device: &wgpu::Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Node Pool Layout"),
            entries: &[
                // node_info
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
                // node_visits (atomic)
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
                // node_wins (atomic)
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
                // node_vl (atomic)
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
                // node_state (atomic)
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
                // children_indices
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
                // children_priors
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
                // free_lists_buffer
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
                // free_tops_buffer
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
                // (removed: node_generations_buffer, no generation-based cleanup)
                BindGroupLayoutEntry {
                    binding: 9,
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
            label: Some("Execution Layout"),
            entries: &[
                // params (uniform)
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
                // work_items
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
                // paths
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
                // alloc_counter (atomic)
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
            ],
        })
    }

    fn create_game_state_layout(device: &wgpu::Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Game State Layout"),
            entries: &[
                // sim_boards
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
            ],
        })
    }

    fn create_stats_layout(device: &wgpu::Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Stats Layout"),
            entries: &[
                // tree_stats
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
            ],
        })
    }

    /// Initialize the tree with root node and its children
    ///
    /// # Arguments
    /// * `root_player` - Player whose turn it is at root (1 or -1)
    /// * `children_moves` - List of (move_id, prior) for each legal move
    pub fn init_tree(&mut self, root_player: i32, children_moves: &[(u32, f32)]) {
        let queue = self.context.queue();

        // Initialize root node
        let root_info = NodeInfo {
            parent_idx: INVALID_INDEX,
            move_id: INVALID_INDEX,
            num_children: children_moves.len() as u32,
            player_at_node: root_player,
        };

        queue.write_buffer(&self.node_info_buffer, 0, bytemuck::bytes_of(&root_info));
        queue.write_buffer(&self.node_visits_buffer, 0, bytemuck::bytes_of(&0i32));
        queue.write_buffer(&self.node_wins_buffer, 0, bytemuck::bytes_of(&0i32));
        queue.write_buffer(&self.node_vl_buffer, 0, bytemuck::bytes_of(&0i32));
        queue.write_buffer(
            &self.node_state_buffer,
            0,
            bytemuck::bytes_of(&NODE_STATE_READY),
        );

        // Initialize children for root
        let opposite_player = -root_player;
        let mut child_indices = vec![INVALID_INDEX; MAX_CHILDREN as usize];
        let mut child_priors = vec![0.0f32; MAX_CHILDREN as usize];

        for (i, &(move_id, prior)) in children_moves.iter().enumerate() {
            if i >= MAX_CHILDREN as usize {
                break;
            }

            let child_idx = (i + 1) as u32; // Children start at index 1

            // Initialize child node
            let child_info = NodeInfo {
                parent_idx: 0, // Root is parent
                move_id,
                num_children: 0, // Will be populated on expansion
                player_at_node: opposite_player,
            };

            let offset = (child_idx as usize) * std::mem::size_of::<NodeInfo>();
            queue.write_buffer(
                &self.node_info_buffer,
                offset as u64,
                bytemuck::bytes_of(&child_info),
            );

            // Initialize child stats
            let child_offset = (child_idx as usize) * std::mem::size_of::<i32>();
            queue.write_buffer(
                &self.node_visits_buffer,
                child_offset as u64,
                bytemuck::bytes_of(&0i32),
            );
            queue.write_buffer(
                &self.node_wins_buffer,
                child_offset as u64,
                bytemuck::bytes_of(&0i32),
            );
            queue.write_buffer(
                &self.node_vl_buffer,
                child_offset as u64,
                bytemuck::bytes_of(&0i32),
            );
            queue.write_buffer(
                &self.node_state_buffer,
                child_offset as u64,
                bytemuck::bytes_of(&NODE_STATE_READY),
            );

            child_indices[i] = child_idx;
            child_priors[i] = prior;
        }

        // Write children arrays for root
        queue.write_buffer(
            &self.children_indices_buffer,
            0,
            bytemuck::cast_slice(&child_indices),
        );
        queue.write_buffer(
            &self.children_priors_buffer,
            0,
            bytemuck::cast_slice(&child_priors),
        );

        // Set alloc counter (root + children allocated)
        let alloc_count = (children_moves.len() + 1) as u32;
        queue.write_buffer(
            &self.alloc_counter_buffer,
            0,
            bytemuck::bytes_of(&alloc_count),
        );

    }

    pub fn create_bind_groups(&mut self, device: &wgpu::Device) {
                // Create urgent event bind group (@group(3) in shader)
                let urgent_event_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Urgent Event Layout"),
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
                let urgent_event_buffers = &self.urgent_event_buffers;
                self.urgent_event_bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Urgent Event Bind Group"),
                    layout: &urgent_event_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: urgent_event_buffers.urgent_event_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: urgent_event_buffers.urgent_event_write_head_buffer.as_entire_binding() },
                    ],
                }));
        let device = self.context.device();

        self.node_pool_bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("Node Pool Bind Group"),
            layout: &self.node_pool_layout,
            entries: &[ 
                BindGroupEntry { binding: 0, resource: self.node_info_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: self.node_visits_buffer.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: self.node_wins_buffer.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: self.node_vl_buffer.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: self.node_state_buffer.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: self.children_indices_buffer.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: self.children_priors_buffer.as_entire_binding() },
                BindGroupEntry { binding: 7, resource: self.free_lists_buffer.as_entire_binding() },
                BindGroupEntry { binding: 8, resource: self.free_tops_buffer.as_entire_binding() },
                BindGroupEntry { binding: 9, resource: urgent_event_buffers.urgent_event_buffer.as_entire_binding() },
            ],
        }));

        self.execution_bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("Execution Bind Group"),
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
            ],
        }));

        self.game_state_bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("Game State Bind Group"),
            layout: &self.game_state_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: self.sim_boards_buffer.as_entire_binding(),
            }],
        }));

        self.stats_bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("Stats Bind Group"),
            layout: &self.stats_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: self.stats_buffer.as_entire_binding(),
            }],
        }));
    }

    /// Run MCTS iterations on GPU
    ///
    /// # Arguments
    /// * `num_iterations` - Number of parallel iterations
    /// * `exploration` - C_puct exploration parameter
    /// * `seed` - Random seed
    /// * `board_width` - Game board width
    /// * `board_height` - Game board height
    /// * `game_type` - Game type ID (0=Gomoku, 1=Connect4, 2=Othello, etc.)
    pub fn run_iterations(
        &mut self,
        num_iterations: u32,
        exploration: f32,
        seed: u32,
        board_width: u32,
        board_height: u32,
        game_type: u32,
    ) {
        println!("[DIAG] run_iterations: ENTERED");
        let queue = self.context.queue();
        println!("[DIAG] run_iterations: got queue");
        // Upload parameters
        let params = MctsParams {
            num_iterations,
            max_nodes: self.max_nodes,
            exploration,
            root_idx: 0,
            seed,
            board_width,
            board_height,
            game_type,
        };
        println!("[DIAG] run_iterations: writing params");
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        // Calculate workgroups
        let workgroups = (num_iterations + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        println!("[DIAG] run_iterations: creating encoder");
        let mut encoder = self
            .context
            .device()
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("MCTS Iteration Encoder"),
            });
        println!("[DIAG] run_iterations: selection pass");
        // Selection pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MCTS Select Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.select_pipeline);
            pass.set_bind_group(0, self.node_pool_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(1, self.execution_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(2, self.game_state_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(3, self.urgent_event_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(4, self.stats_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        println!("[DIAG] run_iterations: backprop pass");
        // TODO: Add simulation pass here (game-specific)
        // Backpropagation pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MCTS Backprop Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backprop_pipeline);
            pass.set_bind_group(0, self.node_pool_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(1, self.execution_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(2, self.game_state_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(3, self.urgent_event_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(4, self.stats_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        println!("[DIAG] run_iterations: queue.submit");
        queue.submit(std::iter::once(encoder.finish()));
        println!("[DIAG] run_iterations: end");
    }

    /// Get statistics from the tree (requires GPU sync)
    pub fn get_stats(&self) -> TreeStats {
        let device = self.context.device();
        let queue = self.context.queue();

        // Run stats kernel
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Stats Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Stats Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.stats_pipeline);
            pass.set_bind_group(0, self.node_pool_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(1, self.execution_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(2, self.game_state_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(3, self.stats_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Copy to staging
        encoder.copy_buffer_to_buffer(
            &self.stats_buffer,
            0,
            &self.stats_staging_buffer,
            0,
            std::mem::size_of::<TreeStats>() as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = self.stats_staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let stats: TreeStats = *bytemuck::from_bytes(&data);
        drop(data);
        self.stats_staging_buffer.unmap();

        stats
    }

    /// Get best move from root based on visit counts
    ///
    /// Returns (move_id, visits, wins, q_value) for the best child
    pub fn get_best_move(&self) -> Option<ChildStats> {
        let device = self.context.device();
        let queue = self.context.queue();

        // Read root's children info
        // First, read root's node_info to get num_children
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Child Stats Staging"),
            size: (MAX_CHILDREN as usize * std::mem::size_of::<u32>() * 2
                + std::mem::size_of::<NodeInfo>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // We need to read: node_info[0], children_indices[0..MAX_CHILDREN], node_visits for each child
        // This is complex - for now, let's read the full children array

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Best Move Encoder"),
        });

        // Copy root's node info
        encoder.copy_buffer_to_buffer(
            &self.node_info_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of::<NodeInfo>() as u64,
        );

        // Copy children indices for root
        encoder.copy_buffer_to_buffer(
            &self.children_indices_buffer,
            0,
            &staging_buffer,
            std::mem::size_of::<NodeInfo>() as u64,
            (MAX_CHILDREN as usize * std::mem::size_of::<u32>()) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();

        let root_info: NodeInfo =
            *bytemuck::from_bytes(&data[..std::mem::size_of::<NodeInfo>()]);
        let children_start = std::mem::size_of::<NodeInfo>();
        let children_indices_slice: &[u32] = bytemuck::cast_slice(
            &data[children_start..children_start + MAX_CHILDREN as usize * std::mem::size_of::<u32>()],
        );
        // Copy to owned Vec before dropping data
        let children_indices: Vec<u32> = children_indices_slice.to_vec();

        drop(data);
        staging_buffer.unmap();

        if root_info.num_children == 0 {
            return None;
        }

        // Now read visits/wins for each child
        let mut best: Option<ChildStats> = None;
        let mut best_visits = -1;

        for i in 0..root_info.num_children as usize {
            let child_idx = children_indices[i];
            // Handle explicit error codes from select_best_child
            if child_idx == INVALID_INDEX ||
               child_idx == SELECT_BEST_CHILD_NO_CHILDREN ||
               child_idx == SELECT_BEST_CHILD_NO_VALID {
                continue;
            }
            if child_idx == SELECT_BEST_CHILD_SOFTMAX_PANIC {
                panic!("select_best_child: SOFTMAX_PANIC error code returned by GPU. This indicates a bug in the selection logic. Please check diagnostics and shader code.");
            }

            // Read child stats (this is inefficient - should batch)
            let child_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Child Stat Staging"),
                size: (std::mem::size_of::<NodeInfo>() + std::mem::size_of::<i32>() * 2) as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Child Stat Encoder"),
            });

            let info_offset = child_idx as u64 * std::mem::size_of::<NodeInfo>() as u64;
            let visits_offset = child_idx as u64 * std::mem::size_of::<i32>() as u64;
            let wins_offset = child_idx as u64 * std::mem::size_of::<i32>() as u64;

            encoder.copy_buffer_to_buffer(
                &self.node_info_buffer,
                info_offset,
                &child_staging,
                0,
                std::mem::size_of::<NodeInfo>() as u64,
            );
            encoder.copy_buffer_to_buffer(
                &self.node_visits_buffer,
                visits_offset,
                &child_staging,
                std::mem::size_of::<NodeInfo>() as u64,
                std::mem::size_of::<i32>() as u64,
            );
            encoder.copy_buffer_to_buffer(
                &self.node_wins_buffer,
                wins_offset,
                &child_staging,
                (std::mem::size_of::<NodeInfo>() + std::mem::size_of::<i32>()) as u64,
                std::mem::size_of::<i32>() as u64,
            );

            queue.submit(std::iter::once(encoder.finish()));

            let slice = child_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let child_info: NodeInfo =
                *bytemuck::from_bytes(&data[..std::mem::size_of::<NodeInfo>()]);
            let visits: i32 = *bytemuck::from_bytes(
                &data[std::mem::size_of::<NodeInfo>()
                    ..std::mem::size_of::<NodeInfo>() + std::mem::size_of::<i32>()],
            );
            let wins: i32 = *bytemuck::from_bytes(
                &data[std::mem::size_of::<NodeInfo>() + std::mem::size_of::<i32>()..],
            );

            drop(data);
            child_staging.unmap();

            if visits > best_visits {
                best_visits = visits;
                let q_value = if visits > 0 {
                    (wins as f64) / (visits as f64 * 2.0)
                } else {
                    0.0
                };
                best = Some(ChildStats {
                    move_id: child_info.move_id,
                    visits,
                    wins,
                    q_value,
                });
            }
        }

        best
    }
}
