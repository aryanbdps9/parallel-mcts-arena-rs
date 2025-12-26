//! WGSL Compute Shaders for MCTS GPU Acceleration
//!
//! This module contains compute shaders for:
//! - PUCT score calculation (selection phase)
//! - Multi-game board evaluation (simulation phase)
//!   - Gomoku: 5-in-a-row on square boards
//!   - Connect4: N-in-a-row with gravity
//!   - Othello: Flip-based capture game
//!   - Blokus: Polyomino placement game
//!   - Hive: Hexagonal tile placement game

use super::embedded_wgsl;

/// Connect4 evaluation shader (generated at build time from rust-gpu SPIR-V)
pub const CONNECT4_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/connect4.wgsl"));

/// PUCT calculation shader (embedded WGSL)
pub fn puct_wgsl() -> &'static str {
	embedded_wgsl::puct_wgsl()
}

pub fn gomoku_wgsl() -> &'static str {
	embedded_wgsl::gomoku_wgsl()
}

pub fn othello_wgsl() -> &'static str {
	embedded_wgsl::othello_wgsl()
}

pub fn blokus_wgsl() -> &'static str {
	embedded_wgsl::blokus_wgsl()
}

pub fn hive_wgsl() -> &'static str {
	embedded_wgsl::hive_wgsl()
}
