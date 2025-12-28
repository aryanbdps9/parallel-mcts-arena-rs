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
//! - GPU-native MCTS tree (all 4 phases on GPU)

/// PUCT calculation shader
pub const PUCT_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/puct.wgsl"));

pub const GOMOKU_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/gomoku.wgsl"));

pub const CONNECT4_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/connect4.wgsl"));

pub const OTHELLO_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/othello.wgsl"));

pub const BLOKUS_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/blokus.wgsl"));

pub const HIVE_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/hive.wgsl"));

/// GPU-native MCTS tree shader (selection, expansion, simulation, backprop all on GPU)
pub const MCTS_TREE_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/mcts_tree.wgsl"));

/// GPU-native MCTS for Othello (complete iteration in one kernel)
pub const MCTS_OTHELLO_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/mcts_othello.wgsl"));
