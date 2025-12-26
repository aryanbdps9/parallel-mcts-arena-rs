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

/// Generated WGSL module (rust-gpu SPIR-V translated to WGSL at build time)
///
/// This module contains all compute entry points used by the runtime.
pub const MCTS_SHADERS_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/mcts_shaders.wgsl"));

/// Generated SPIR-V module (rust-gpu output, validated at build time)
pub const MCTS_SHADERS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mcts_shaders.spv"));
