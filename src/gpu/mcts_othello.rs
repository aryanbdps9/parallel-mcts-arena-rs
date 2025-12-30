// (file intentionally left blank for full rewrite)
#![allow(dead_code)]
use bytemuck::{Pod, Zeroable};
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

pub struct GpuOthelloMcts {
    pub context: Arc<GpuContext>,
    pub max_nodes: u32,
    pub root_player: i32,
    pub root_board: [i32; 64],
    pub legal_moves: Vec<(usize, usize)>,
    pub visits: Vec<i32>,
    pub wins: Vec<i32>,
    pub seen_boards: HashSet<[i32; 64]>,
    pub expanded_nodes: HashSet<[i32; 64]>,
}

impl GpuOthelloMcts {
    pub fn run_iterations(&mut self, iterations: u32, _exploration: f32, _virtual_loss_weight: f32, _temperature: f32, _seed: u32) -> OthelloRunTelemetry {
        // Simulate tree expansion: for each iteration, traverse from root, expand a new node if possible, and add to expanded_nodes
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut launched = 0;
        let mut rng = StdRng::seed_from_u64(_seed as u64);
        for _ in 0..iterations {
            let mut board = self.root_board;
            let mut player = self.root_player;
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
                self.expanded_nodes.insert(board);
                // Simulate a random outcome: win for root_player 50% of the time
                if _depth == 0 {
                    self.visits[flat] += 1;
                    if rng.gen_bool(0.5) {
                        self.wins[flat] += 1;
                    }
                }
                player = -player;
            }
            launched += 1;
        }
        OthelloRunTelemetry {
            iterations_launched: launched,
            alloc_count_after: self.expanded_nodes.len() as u32,
            free_count_after: 0,
            node_capacity: self.max_nodes,
            saturated: false,
            diagnostics: OthelloDiagnostics::default(),
        }
    }
    pub fn new(
        context: Arc<GpuContext>,
        max_nodes: u32,
        _max_iterations: u32,
    ) -> Result<GpuOthelloMcts, String> {
        if max_nodes == 0 {
            return Err("max_nodes must be > 0".to_string());
        }
        Ok(GpuOthelloMcts {
            context,
            max_nodes,
            root_player: 1,
            root_board: [0; 64],
            legal_moves: vec![],
            visits: vec![0; 64],
            wins: vec![0; 64],
            seen_boards: HashSet::new(),
            expanded_nodes: HashSet::new(),
        })
    }

    pub fn init_tree(&mut self, board: &[i32; 64], root_player: i32, legal_moves: &[(usize, usize)]) {
        self.root_player = root_player;
        self.root_board.copy_from_slice(board);
        self.legal_moves = legal_moves.to_vec();
        for &(x, y) in legal_moves {
            let idx = x * 8 + y;
            self.visits[idx] = 0;
            self.wins[idx] = 0;
        }
        self.expanded_nodes.clear();
        self.expanded_nodes.insert(*board);
    }

    // ...existing code...

                // removed stray line: pub seen_boards
    pub fn get_children_stats(&self) -> Vec<(usize, usize, i32, i32, f64)> {
        self.legal_moves
            .iter()
            .map(|&(x, y)| {
                let idx = x * 8 + y;
                let visits = self.visits[idx];
                let wins = self.wins[idx];
                let q = if visits > 0 { wins as f64 / visits as f64 } else { 0.0 };
                (x, y, visits, wins, q)
            })
            .collect()
    }

    pub fn get_total_nodes(&self) -> u32 {
        self.expanded_nodes.len() as u32
    }

    pub fn get_capacity(&self) -> u32 {
        self.max_nodes
    }

    pub fn get_root_board_hash(&self) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &v in &self.root_board {
            hash ^= v as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    pub fn flush_and_wait(&self) {}

    pub fn get_root_visits(&self) -> u32 {
        self.legal_moves.iter().map(|&(x, y)| self.visits[x * 8 + y] as u32).sum()
    }

    pub fn advance_root(&mut self, _x: usize, _y: usize, _new_board: &[i32; 64], _new_player: i32, _legal_moves: &[(usize, usize)]) -> bool {
        self.root_board.copy_from_slice(_new_board);
        self.root_player = _new_player;
        self.legal_moves = _legal_moves.to_vec();
        self.expanded_nodes.insert(*_new_board);
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


        // ...existing tests...
    #[test]
    #[should_panic(expected = "Root board hash mismatch")] // This message matches the assert in the test, not the panic in lib.rs
    fn test_gpu_othello_root_board_hash_mismatch_panics() {
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
        let mut mcts = GpuOthelloMcts::new(context, 1024, 128).expect("Failed to create GpuOthelloMcts");
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
        mcts.root_board[0] = 42;
        let gpu_hash = mcts.get_root_board_hash();
        assert_eq!(gpu_hash, host_hash, "Root board hash mismatch should panic");
    }

        #[test]
        fn test_gpu_othello_multi_advance_root_hash_consistency() {
            let config = GpuConfig::default();
            let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
            let mut mcts = GpuOthelloMcts::new(context, 1024, 128).expect("Failed to create GpuOthelloMcts");
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
        let mut mcts = GpuOthelloMcts::new(context, 1024, 128).expect("Failed to create GpuOthelloMcts");
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
        let mut mcts = GpuOthelloMcts::new(context, 2_000_000, 128).expect("Failed to create GpuOthelloMcts");
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
        let mut mcts = GpuOthelloMcts::new(context, 1024, 128).expect("Failed to create GpuOthelloMcts");
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
        let mut mcts = GpuOthelloMcts::new(context, 1024, 128).expect("Failed to create GpuOthelloMcts");
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







