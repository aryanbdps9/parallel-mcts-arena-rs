//! # GPU Acceleration Module for MCTS
//!
//! This module provides GPU-accelerated operations for the Monte Carlo Tree Search algorithm.
//! It leverages WebGPU (wgpu) for cross-platform GPU compute capabilities.
//!
//! ## Key Features
//! - **Batch PUCT Calculation**: Compute PUCT scores for many nodes in parallel on the GPU
//! - **Automatic Fallback**: Falls back to CPU if GPU is unavailable
//!
//! ## Architecture
//! The GPU module consists of:
//! - `GpuContext`: Manages GPU device, queues, and resources
//! - `GpuMctsAccelerator`: Orchestrates GPU-accelerated MCTS operations
//! - Compute shaders for PUCT calculation and state evaluation

mod context;
mod accelerator;
mod shaders;

pub use context::GpuContext;
pub use accelerator::{GpuMctsAccelerator, GpuNodeData, GpuPuctResult, BatchExpansionResult, GpuExpansionInput, GpuSimulationParams};

/// Configuration for GPU acceleration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Maximum number of nodes to process in a single GPU batch
    pub max_batch_size: usize,
    /// Whether to prefer high-performance GPU over low-power GPU
    pub prefer_high_performance: bool,
    /// Minimum number of nodes before using GPU (below this, CPU is faster due to transfer overhead)
    pub min_batch_threshold: usize,
    /// Enable debug output for GPU operations
    pub debug_mode: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 65536, // 64K nodes per batch
            prefer_high_performance: true,
            min_batch_threshold: 256, // Don't use GPU for less than 256 nodes
            debug_mode: false,
        }
    }
}

/// Result of GPU initialization
#[derive(Debug)]
pub enum GpuInitResult {
    /// GPU initialized successfully
    Success(GpuContext),
    /// GPU not available, falling back to CPU
    Unavailable(String),
    /// GPU initialization failed with error
    Error(String),
}

/// Attempts to initialize the GPU for MCTS acceleration
///
/// This function will try to create a GPU context with the specified configuration.
/// If the GPU is not available or initialization fails, it returns an appropriate error.
///
/// # Arguments
/// * `config` - GPU configuration options
///
/// # Returns
/// * `GpuInitResult` indicating success or failure with details
pub fn try_init_gpu(config: &GpuConfig) -> GpuInitResult {
    match GpuContext::new(config) {
        Ok(ctx) => GpuInitResult::Success(ctx),
        Err(e) => {
            let error_str = format!("{}", e);
            if error_str.contains("No suitable GPU adapter found") {
                GpuInitResult::Unavailable(error_str)
            } else {
                GpuInitResult::Error(error_str)
            }
        }
    }
}
