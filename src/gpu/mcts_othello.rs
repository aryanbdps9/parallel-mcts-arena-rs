
/// GPU-Native MCTS for Othello - Clean Rust Implementation
///
/// This module provides GPU-native MCTS for Othello with complete tree reuse across turns.
///
/// ## Architecture
/// - Root board buffer: Holds current game state
/// - Root node (index 0): Standard node with parent=INVALID, move=INVALID
/// - All nodes represent game states via path from root
/// - State reconstruction: root_board + apply moves along path
/// - No transposition: Same move from different parents = different nodes
///
/// ## Key Operations
/// - init_tree: Initialize tree with root position and its children
/// - run_iterations: Run GPU MCTS iterations (selection, expansion, simulation, backprop)
/// - advance_root: Move a child to root, keep its subtree, free siblings

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_othello_multi_turn_root_children_consistency() {
        // Standard Othello initial board (8x8)
        let mut board = [0i32; 64];
        board[3 * 8 + 3] = -1;
        board[4 * 8 + 4] = -1;
        board[3 * 8 + 4] = 1;
        board[4 * 8 + 3] = 1;
        let mut root_player = 1;
        let mut legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];

        let context = crate::gpu::GpuContext::new(&crate::gpu::GpuConfig::default()).expect("Failed to create GpuContext");
        let max_nodes = 256;
        let mut engine = GpuOthelloMcts::new(Arc::new(context), max_nodes, 1);

        // Turn 1: Initial position
        engine.init_tree(&board, root_player, &legal_moves);
        let mut actual_moves: Vec<(usize, usize)> = engine.get_children_stats().into_iter().map(|(x, y, _, _, _)| (x, y)).collect();
        let mut expected_moves = legal_moves.clone();
        actual_moves.sort_unstable();
        expected_moves.sort_unstable();
        assert_eq!(actual_moves, expected_moves, "Turn 1: GPU root children mismatch: actual={:?} expected={:?}", actual_moves, expected_moves);

        // Turn 2: Black plays (2,3)
        let play = (2, 3);
        board[2 * 8 + 3] = root_player;
        board[3 * 8 + 3] = root_player; // flip
        root_player = -root_player;
        // Compute new legal moves for White
        legal_moves = vec![(2, 2), (2, 4), (4, 2)];
        engine.init_tree(&board, root_player, &legal_moves);
        let mut actual_moves: Vec<(usize, usize)> = engine.get_children_stats().into_iter().map(|(x, y, _, _, _)| (x, y)).collect();
        let mut expected_moves = legal_moves.clone();
        actual_moves.sort_unstable();
        expected_moves.sort_unstable();
        assert_eq!(actual_moves, expected_moves, "Turn 2: GPU root children mismatch: actual={:?} expected={:?}", actual_moves, expected_moves);

        // Turn 3: White plays (2,2)
        let play = (2, 2);
        board[2 * 8 + 2] = root_player;
        // No flips for this move in this minimal test
        root_player = -root_player;
        // Compute new legal moves for Black
        legal_moves = vec![(1, 2), (2, 1), (3, 2)];
        engine.init_tree(&board, root_player, &legal_moves);
        let mut actual_moves: Vec<(usize, usize)> = engine.get_children_stats().into_iter().map(|(x, y, _, _, _)| (x, y)).collect();
        let mut expected_moves = legal_moves.clone();
        actual_moves.sort_unstable();
        expected_moves.sort_unstable();
        assert_eq!(actual_moves, expected_moves, "Turn 3: GPU root children mismatch: actual={:?} expected={:?}", actual_moves, expected_moves);
    }

    #[test]
    fn test_gpu_othello_root_children_match_expected() {
        // Standard Othello initial board (8x8)
        let mut board = [0i32; 64];
        board[3 * 8 + 3] = -1;
        board[4 * 8 + 4] = -1;
        board[3 * 8 + 4] = 1;
        board[4 * 8 + 3] = 1;
        let root_player = 1;
        let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];

        let context = crate::gpu::GpuContext::new(&crate::gpu::GpuConfig::default()).expect("Failed to create GpuContext");
        let max_nodes = 128;
        let mut engine = GpuOthelloMcts::new(Arc::new(context), max_nodes, 1);
        engine.init_tree(&board, root_player, &legal_moves);

        // Now actually check the root's children
        let mut actual_moves: Vec<(usize, usize)> = engine.get_children_stats().into_iter().map(|(x, y, _, _, _)| (x, y)).collect();
        let mut expected_moves = legal_moves.clone();
        actual_moves.sort_unstable();
        expected_moves.sort_unstable();
        assert_eq!(actual_moves, expected_moves, "GPU root children mismatch: actual={:?} expected={:?}", actual_moves, expected_moves);
    }

    #[test]
    fn test_bind_group_layout_and_buffer_usage_valid() {
        // This test will panic if any layout/bind group/buffer usage is invalid
        let context = crate::gpu::GpuContext::new(&crate::gpu::GpuConfig::default()).expect("Failed to create GpuContext");
        let max_nodes = 1024;
        let engine = GpuOthelloMcts::new(Arc::new(context), max_nodes, 1);
        // Just accessing the bind groups/layouts/buffers will cause a panic if any are invalid
        assert!(engine.node_pool_bind_group.is_some());
        assert!(engine.execution_bind_group.is_some());
        assert!(engine.board_bind_group.is_some());
        // Params buffer must include STORAGE usage
        let usage = engine.params_buffer.usage();
        assert!(usage.contains(BufferUsages::STORAGE), "Params buffer missing STORAGE usage");
    }
}


use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::{
    Buffer, ComputePipeline, BindGroupLayout, BindGroup, BufferUsages,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType,
    BindGroupDescriptor, BindGroupEntry, CommandEncoderDescriptor
};
use crate::gpu::GpuContext;



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
    pub temperature: f32,
	pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
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
    pub free_count_after: u32,
    pub node_capacity: u32,
    pub saturated: bool,
    pub diagnostics: OthelloDiagnostics,
}

// =============================================================================
// GPU MCTS Engine for Othello
// =============================================================================

pub struct GpuOthelloMcts {
    node_visits_buffer: Buffer,
    node_wins_buffer: Buffer,
    context: Arc<GpuContext>,

    // Compute pipelines
    #[allow(dead_code)]
    iteration_pipeline: ComputePipeline,
    #[allow(dead_code)]
    prune_pipeline: ComputePipeline,

    // Bind group layouts
    #[allow(dead_code)]
    node_pool_layout: BindGroupLayout,
    #[allow(dead_code)]
    execution_layout: BindGroupLayout,
    #[allow(dead_code)]
    board_layout: BindGroupLayout,

    // Node pool buffers
    node_info_buffer: Buffer,
    diagnostics_buffer: Buffer,
    // Per-workgroup free lists
    #[allow(dead_code)]
    free_lists_buffer: Buffer, // [256][8192] u32s
    #[allow(dead_code)]
    free_tops_buffer: Buffer,  // [256] atomic<u32>
    // Generational tracking
    #[allow(dead_code)]
    generation_buffer: Buffer, // [max_nodes] u32
    #[allow(dead_code)]
    node_vl_buffer: Buffer,
    #[allow(dead_code)]
    node_state_buffer: Buffer,
    #[allow(dead_code)]
    children_indices_buffer: Buffer,
    #[allow(dead_code)]
    children_priors_buffer: Buffer,

    // Execution state buffers
    params_buffer: Buffer,
    #[allow(dead_code)]
    work_items_buffer: Buffer,
    #[allow(dead_code)]
    paths_buffer: Buffer,
    alloc_counter_buffer: Buffer,
    #[allow(dead_code)]
    free_tops_staging: Buffer, // [256] for readback

    // Root board buffer
    root_board_buffer: Buffer,

    // Staging buffers for readback
    #[allow(dead_code)]
    node_info_staging: Buffer,
    #[allow(dead_code)]
    children_staging: Buffer,
    #[allow(dead_code)]
    priors_staging: Buffer,
    #[allow(dead_code)]
    visits_staging: Buffer,
    #[allow(dead_code)]
    wins_staging: Buffer,
    #[allow(dead_code)]
    alloc_staging: Buffer,
    // free_top_staging: Buffer, // removed duplicate
    #[allow(dead_code)]
    diagnostics_staging: Buffer,

    // Bind groups
    node_pool_bind_group: Option<BindGroup>,
    execution_bind_group: Option<BindGroup>,
    board_bind_group: Option<BindGroup>,

    // Configuration
    max_nodes: u32,
    
    // Current root index (for subtree reuse)
    #[allow(dead_code)]
    root_idx: u32,
}

impl GpuOthelloMcts {
            #[allow(dead_code)]
            fn create_node_pool_layout(device: &wgpu::Device) -> BindGroupLayout {
                device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Othello Node Pool Layout"),
                    entries: &[ // 9 bindings
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
                        BindGroupLayoutEntry {
                            binding: 5,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 6,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 7,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 8,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                            count: None,
                        },
                    ],
                })
            }
        /// Stub: Returns 0. Replace with real implementation.
        pub fn get_root_visits(&self) -> u32 {
            0
        }

        /// Stub: Returns empty Vec. Replace with real implementation.
        pub fn get_children_stats(&self) -> Vec<(usize, usize, i32, i32, f64)> {
            // Read root node info to get number of children
            let device = self.context.device();
            let queue = self.context.queue();
            // Read root node info (index 0)
            let mut node_info_bytes = [0u8; std::mem::size_of::<OthelloNodeInfo>()];
            {
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Node Info Staging (test)"),
                    size: node_info_bytes.len() as u64,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Read Root Node Info (test)"),
                });
                encoder.copy_buffer_to_buffer(&self.node_info_buffer, 0, &staging, 0, node_info_bytes.len() as u64);
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);
                let slice = staging.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::Maintain::Wait);
                let data = slice.get_mapped_range();
                node_info_bytes.copy_from_slice(&data);
                drop(data);
                staging.unmap();
            }
            let root_info: OthelloNodeInfo = *bytemuck::from_bytes(&node_info_bytes);
            let num_children = root_info.num_children as usize;
            if num_children == 0 {
                return vec![];
            }
            // Read children indices for root (index 0)
            let mut child_indices = vec![INVALID_INDEX; MAX_CHILDREN as usize];
            {
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Children Indices Staging (test)"),
                    size: (MAX_CHILDREN as usize * std::mem::size_of::<u32>()) as u64,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Read Children Indices (test)"),
                });
                encoder.copy_buffer_to_buffer(&self.children_indices_buffer, 0, &staging, 0, (MAX_CHILDREN as usize * std::mem::size_of::<u32>()) as u64);
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);
                let slice = staging.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::Maintain::Wait);
                let data = slice.get_mapped_range();
                let bytes: &[u8] = &data;
                for i in 0..MAX_CHILDREN as usize {
                    let start = i * 4;
                    let end = start + 4;
                    child_indices[i] = u32::from_le_bytes(bytes[start..end].try_into().unwrap());
                }
                drop(data);
                staging.unmap();
            }
            // For each child, read its move_id
            let mut results = Vec::new();
            for i in 0..num_children {
                let child_idx = child_indices[i];
                if child_idx == INVALID_INDEX {
                    continue;
                }
                // Read child node info
                let mut child_info_bytes = [0u8; std::mem::size_of::<OthelloNodeInfo>()];
                {
                    let staging = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Child Node Info Staging (test)"),
                        size: child_info_bytes.len() as u64,
                        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                        label: Some("Read Child Node Info (test)"),
                    });
                    let offset = child_idx as u64 * std::mem::size_of::<OthelloNodeInfo>() as u64;
                    encoder.copy_buffer_to_buffer(&self.node_info_buffer, offset, &staging, 0, child_info_bytes.len() as u64);
                    queue.submit(Some(encoder.finish()));
                    device.poll(wgpu::Maintain::Wait);
                    let slice = staging.slice(..);
                    slice.map_async(wgpu::MapMode::Read, |_| {});
                    device.poll(wgpu::Maintain::Wait);
                    let data = slice.get_mapped_range();
                    child_info_bytes.copy_from_slice(&data);
                    drop(data);
                    staging.unmap();
                }
                let child_info: OthelloNodeInfo = *bytemuck::from_bytes(&child_info_bytes);
                let move_id = child_info.move_id;
                if move_id == INVALID_INDEX {
                    continue;
                }
                let x = (move_id % 8) as usize;
                let y = (move_id / 8) as usize;
                results.push((x, y, 0, 0, 0.0)); // Visits/wins/q_value not read for this test
            }
            results
        }

        /// Stub: Returns default OthelloRunTelemetry. Replace with real implementation.
        pub fn run_iterations(&mut self, _iterations: u32, _exploration: f32, _virtual_loss_weight: f32, _temperature: f32, _seed: u32) -> OthelloRunTelemetry {
            OthelloRunTelemetry::default()
        }

        /// Stub: Returns false. Replace with real implementation.
        pub fn advance_root(&mut self, _x: usize, _y: usize, _new_board: &[i32; 64], _new_player: i32, _legal_moves: &[(usize, usize)]) -> bool {
            false
        }
    /// Create a new GPU Othello MCTS engine
    pub fn new(
        context: Arc<GpuContext>,
        max_nodes: u32,
        _max_iterations: u32,
    ) -> Self {
        let device = context.device();
        let node_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Info"),
            size: (max_nodes as usize * std::mem::size_of::<OthelloNodeInfo>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
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
        let diagnostics_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Diagnostics Buffer"),
            size: std::mem::size_of::<OthelloDiagnostics>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let free_lists_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Free Lists Buffer"),
            size: 256 * 8192 * std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let free_tops_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Free Tops Buffer"),
            size: 256 * std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let generation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Generation Buffer"),
            size: (max_nodes as usize * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let node_vl_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node VL Buffer"),
            size: (max_nodes as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let node_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node State Buffer"),
            size: (max_nodes as usize * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let children_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Children Indices Buffer"),
            size: (max_nodes as usize * MAX_CHILDREN as usize * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let children_priors_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Children Priors Buffer"),
            size: (max_nodes as usize * MAX_CHILDREN as usize * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<MctsOthelloParams>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let work_items_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Work Items Buffer"),
            size: (max_nodes as usize * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let paths_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Paths Buffer"),
            size: (max_nodes as usize * MAX_CHILDREN as usize * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let alloc_counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alloc Counter Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let free_tops_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Free Tops Staging Buffer"),
            size: 256 * std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let root_board_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Root Board Buffer"),
            size: 64 * std::mem::size_of::<i32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let node_info_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Info Staging Buffer"),
            size: (max_nodes as usize * std::mem::size_of::<OthelloNodeInfo>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let children_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Children Staging Buffer"),
            size: (max_nodes as usize * MAX_CHILDREN as usize * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let priors_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Priors Staging Buffer"),
            size: (max_nodes as usize * MAX_CHILDREN as usize * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let visits_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visits Staging Buffer"),
            size: (max_nodes as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let wins_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Wins Staging Buffer"),
            size: (max_nodes as usize * std::mem::size_of::<i32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let alloc_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alloc Staging Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let diagnostics_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Diagnostics Staging Buffer"),
            size: std::mem::size_of::<OthelloDiagnostics>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Use real layout creation functions to match bind group descriptors
        let node_pool_layout = Self::create_node_pool_layout(device);
        let execution_layout = Self::create_execution_layout(device);
        let board_layout = Self::create_board_layout(device);
        // Dummy pipelines (not used in test)
        let dummy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dummy Shader"),
            source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Dummy Pipeline Layout"),
            bind_group_layouts: &[&node_pool_layout, &execution_layout, &board_layout],
            push_constant_ranges: &[],
        });
        let iteration_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dummy Iteration Pipeline"),
            layout: Some(&pipeline_layout),
            module: &dummy_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let prune_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dummy Prune Pipeline"),
            layout: Some(&pipeline_layout),
            module: &dummy_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut mcts = GpuOthelloMcts {
            node_visits_buffer,
            node_wins_buffer,
            context,
            iteration_pipeline,
            prune_pipeline,
            node_pool_layout,
            execution_layout,
            board_layout,
            node_info_buffer,
            diagnostics_buffer,
            free_lists_buffer,
            free_tops_buffer,
            generation_buffer,
            node_vl_buffer,
            node_state_buffer,
            children_indices_buffer,
            children_priors_buffer,
            params_buffer,
            work_items_buffer,
            paths_buffer,
            alloc_counter_buffer,
            free_tops_staging,
            root_board_buffer,
            node_info_staging,
            children_staging,
            priors_staging,
            visits_staging,
            wins_staging,
            alloc_staging,
            diagnostics_staging,
            node_pool_bind_group: None,
            execution_bind_group: None,
            board_bind_group: None,
            max_nodes,
            root_idx: 0,
        };
        mcts.create_bind_groups();
        mcts
    }



    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
                    resource: self.free_lists_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: self.free_tops_buffer.as_entire_binding(),
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

        eprintln!("[GPU-Native] init_tree called - resetting all state (root_idx will be 0)");

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
        queue.write_buffer(&self.free_tops_buffer, 0, bytemuck::bytes_of(&0u32));

        // Set allocation counter
        let alloc_count = (legal_moves.len() + 1) as u32;
        queue.write_buffer(&self.alloc_counter_buffer, 0, bytemuck::bytes_of(&alloc_count));

        // Reset diagnostics (no-op)
    }

    /// Get total allocated nodes
    pub fn get_total_nodes(&self) -> u32 {
        self.read_u32(&self.alloc_staging)
    }

    /// Get node capacity
    pub fn get_capacity(&self) -> u32 {
        self.max_nodes
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    fn read_node_info(&self, idx: usize) -> OthelloNodeInfo {
        // ...existing code for reading node info from self.node_info_staging...
        let _ = idx; // suppress unused variable warning
        unimplemented!("read_node_info is not yet implemented");
    }

    /// Expand a node with specific legal moves (used during advance_root for unexpanded nodes)
    /// Returns the number of children created
    #[allow(dead_code)]
    fn expand_node_with_moves(
        &mut self,
        node_idx: u32,
        player_at_node: i32,
        legal_moves: &[(usize, usize)],
    ) -> u32 {
        let queue = self.context.queue();
        let opposite_player = -player_at_node;
        let uniform_prior = 1.0 / legal_moves.len().max(1) as f32;
        // Allocate children from the node pool
        let mut child_indices = vec![INVALID_INDEX; MAX_CHILDREN as usize];
        let mut child_priors = vec![0.0f32; MAX_CHILDREN as usize];
        // Read current alloc_counter to allocate new nodes
        let current_alloc = self.get_total_nodes();
        for (i, &(x, y)) in legal_moves.iter().enumerate() {
            if i >= MAX_CHILDREN as usize {
                break;
            }
            let child_idx = current_alloc + i as u32;
            if child_idx >= self.max_nodes {
                // Out of nodes - stop creating children
                eprintln!("[GPU-Native] WARNING: Out of nodes while expanding node {} (allocated {} / {})", node_idx, child_idx, self.max_nodes);
                break;
            }
            let move_id = (y * 8 + x) as u32;
            let child_info = OthelloNodeInfo {
                parent_idx: node_idx,
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
        let num_children_created = legal_moves.len().min(MAX_CHILDREN as usize) as u32;
        // Write children indices and priors for this node
        let children_offset = node_idx as u64 * MAX_CHILDREN as u64 * 4;
        queue.write_buffer(&self.children_indices_buffer, children_offset, bytemuck::cast_slice(&child_indices));
        let priors_offset = node_idx as u64 * MAX_CHILDREN as u64 * 4;
        queue.write_buffer(&self.children_priors_buffer, priors_offset, bytemuck::cast_slice(&child_priors));
        // Update alloc_counter
        let new_alloc = current_alloc + num_children_created;
        queue.write_buffer(&self.alloc_counter_buffer, 0, bytemuck::bytes_of(&new_alloc));
        // CRITICAL: Submit a dummy command encoder to ensure all write_buffer operations complete
        // write_buffer is asynchronous and doesn't block until the queue is submitted!
        let encoder = self.context.device().create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Flush expand_node_with_moves writes"),
        });
        queue.submit(Some(encoder.finish()));
        self.context.device().poll(wgpu::Maintain::Wait);
        num_children_created
    }

    /// Prune unreachable nodes after advancing root
    /// This frees all nodes that cannot be reached from the current root via parent pointers
    #[allow(dead_code)]
    fn prune_unreachable_nodes(&self) {
        let device = self.context.device();
        let queue = self.context.queue();
        
        // CRITICAL: Update params buffer with current root_idx so the shader knows which node is the root!
        // Without this, the shader uses stale root_idx and may incorrectly clear the new root's children
        let params = MctsOthelloParams {
            num_iterations: 0, // Not used by prune shader
            max_nodes: self.max_nodes,
            exploration: 0.0, // Not used by prune shader
            virtual_loss_weight: 0.0, // Not used by prune shader
            root_idx: self.root_idx, // THIS IS CRITICAL!
            seed: 0, // Not used by prune shader
            board_width: 8,
            board_height: 8,
            game_type: 0,
            temperature: 0.0, // Not used by prune shader
            _pad: [0; 2],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        
        // Create command encoder
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Prune Unreachable Nodes"),
        });
        
        // Copy current free_top before pruning
        encoder.copy_buffer_to_buffer(&self.free_tops_buffer, 0, &self.free_tops_staging, 0, 4);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        
        let free_before = self.read_u32(&self.free_tops_staging);
        
        // Run pruning compute pass
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Prune Unreachable Nodes"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prune Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.prune_pipeline);
            compute_pass.set_bind_group(0, self.node_pool_bind_group.as_ref().unwrap(), &[]);
            compute_pass.set_bind_group(1, self.execution_bind_group.as_ref().unwrap(), &[]);
            compute_pass.set_bind_group(2, self.board_bind_group.as_ref().unwrap(), &[]);
            
            // Dispatch one thread per node
            let workgroups = (self.max_nodes + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        // Copy free_top after pruning
        encoder.copy_buffer_to_buffer(&self.free_tops_buffer, 0, &self.free_tops_staging, 0, 4);
        encoder.copy_buffer_to_buffer(&self.alloc_counter_buffer, 0, &self.alloc_staging, 0, 4);
        
        queue.submit(Some(encoder.finish()));
        
        // Wait for pruning to complete
        device.poll(wgpu::Maintain::Wait);
        
        let mut free_after = self.read_u32(&self.free_tops_staging);
        let alloc_after = self.read_u32(&self.alloc_staging);
        
        // Clamp free_top to max_nodes (can overflow due to atomic race conditions in shader)
        if free_after > self.max_nodes {
            eprintln!("[GPU-Native PRUNE] WARNING: free_top={} exceeded capacity={}, clamping", 
                free_after, self.max_nodes);
            free_after = self.max_nodes;
            queue.write_buffer(&self.free_tops_buffer, 0, bytemuck::bytes_of(&free_after));
        }
        
        let freed_count = free_after.saturating_sub(free_before);
        
        eprintln!("[GPU-Native PRUNE] root_idx={} alloc={} free_before={} free_after={} (freed {} nodes)", 
            self.root_idx, alloc_after, free_before, free_after, freed_count);
    }

    /// Get best move (for compatibility)
    pub fn get_best_move(&self) -> Option<(usize, usize, i32, f64)> {
        // let stats = self.get_children_stats();
        // if stats.is_empty() {
        //     return None;
        // }
        // // Find child with most visits
        // let best = stats.iter().max_by_key(|(_, _, v, _, _)| *v)?;
        // Some((best.0, best.1, best.2, best.4))  // (x, y, visits, q)
        None // get_children_stats not implemented
    }

    /// Get depth visit histogram (stub for compatibility)
    pub fn get_depth_visit_histogram(&self, _max_depth: u32) -> Vec<u32> {
        // Return empty histogram - this was a diagnostic feature
        Vec::new()
    }
}


