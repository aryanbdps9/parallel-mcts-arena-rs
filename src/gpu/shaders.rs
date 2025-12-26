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

/// PUCT calculation shader
pub const PUCT_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/puct.wgsl"));

pub const GOMOKU_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/gomoku.wgsl"));

// Connect4 now uses Rust-GPU SPIR-V shader instead of WGSL
#[allow(dead_code)]
pub const CONNECT4_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/connect4.wgsl"));

pub const OTHELLO_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/othello.wgsl"));

pub const BLOKUS_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/blokus.wgsl"));

pub const HIVE_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/hive.wgsl"));

/// Compiled SPIR-V shader module containing all kernels
/// Generated from crates/mcts-shaders
pub const SHADER_MODULE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mcts_shaders.spv"));
