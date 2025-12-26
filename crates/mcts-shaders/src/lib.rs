#![no_std]
#![feature(asm_experimental_arch)]

use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct SimulationParams {
    pub board_width: u32,
    pub board_height: u32,
    pub current_player: i32,
    pub use_heuristic: u32,
    pub seed: u32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct SimulationResult {
    pub score: f32,
}

// Game type constants
const GAME_CONNECT4: u32 = 1;

// PCG Random Number Generator
struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn pcg_hash(&mut self) -> u32 {
        let state = self.state.wrapping_mul(747796405).wrapping_add(2891336453);
        self.state = state;
        let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
        (word >> 22) ^ word
    }

    fn rand(&mut self) -> f32 {
        let hash = self.pcg_hash();
        hash as f32 / 4294967296.0
    }

    fn rand_range(&mut self, min: u32, max: u32) -> u32 {
        min + (self.rand() * (max - min) as f32) as u32
    }
}

fn get_line_size(params: &SimulationParams) -> i32 {
    let encoded = params.current_player;
    let line_size = (encoded >> 8) & 0xFF;
    if line_size > 0 {
        line_size
    } else {
        5 // Default for Gomoku
    }
}

#[spirv(compute(threads(64)))]
pub fn evaluate_connect4(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] boards: &[i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] results: &mut [SimulationResult],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] params: &SimulationParams,
) {
    let idx = global_id.x as usize;
    if idx >= results.len() {
        return;
    }

    let board_size = (params.board_width * params.board_height) as usize;
    let board_start = idx * board_size;
    
    // Simple validation to prevent out of bounds
    if board_start + board_size > boards.len() {
        results[idx].score = 0.0;
        return;
    }

    let current_player = 1;
    let line_size = get_line_size(params);
    let mut rng = Rng::new(params.seed.wrapping_add((idx as u32).wrapping_mul(719393)));

    // Simple heuristic scoring instead of full rollout to avoid complex memory access
    let mut score = 0.0f32;
    
    // Count pieces for each player
    let mut player_count = 0;
    let mut opponent_count = 0;
    
    for i in 0..board_size {
        if board_start + i < boards.len() {
            let cell = boards[board_start + i];
            if cell == current_player {
                player_count += 1;
            } else if cell == -current_player {
                opponent_count += 1;
            }
        }
    }
    
    // Simple heuristic: more pieces = better position
    score = (player_count as f32 - opponent_count as f32) * 0.1;
    
    // Add some randomness
    score += (rng.rand() - 0.5) * 0.2;
    
    results[idx].score = score;
}
