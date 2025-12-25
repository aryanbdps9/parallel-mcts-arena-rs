//! GPU Context Management
//!
//! This module handles GPU device initialization, resource management,
//! and provides the foundational infrastructure for GPU-accelerated MCTS operations.

use wgpu::{
    Device, Queue, Instance, InstanceDescriptor, RequestAdapterOptions,
    PowerPreference, DeviceDescriptor, Features, Limits, ShaderModule,
    ComputePipeline, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType,
    PipelineLayoutDescriptor, ComputePipelineDescriptor, Buffer, BufferDescriptor,
    BufferUsages, Maintain,
};
use std::sync::Arc;

use super::GpuConfig;
use super::shaders;

/// Represents an error that occurred during GPU operations
#[derive(Debug)]
pub enum GpuError {
    /// No suitable GPU adapter was found
    NoAdapter(String),
    /// Failed to request a GPU device
    DeviceRequest(String),
    /// Shader compilation failed
    ShaderCompilation(String),
    /// Buffer operation failed
    BufferError(String),
    /// Pipeline creation failed
    PipelineError(String),
    /// Compute dispatch failed
    ComputeError(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter(msg) => write!(f, "No suitable GPU adapter found: {}", msg),
            GpuError::DeviceRequest(msg) => write!(f, "Failed to request GPU device: {}", msg),
            GpuError::ShaderCompilation(msg) => write!(f, "Shader compilation failed: {}", msg),
            GpuError::BufferError(msg) => write!(f, "Buffer operation failed: {}", msg),
            GpuError::PipelineError(msg) => write!(f, "Pipeline creation failed: {}", msg),
            GpuError::ComputeError(msg) => write!(f, "Compute dispatch failed: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

/// GPU Context holding device, queue, and compiled pipelines
///
/// This is the main entry point for GPU operations. It manages the GPU device
/// and provides access to pre-compiled compute pipelines for MCTS operations.
pub struct GpuContext {
    /// The GPU device handle
    device: Arc<Device>,
    /// The command queue for submitting work
    queue: Arc<Queue>,
    /// Information about the GPU adapter
    adapter_info: wgpu::AdapterInfo,
    /// Pre-compiled PUCT calculation pipeline
    puct_pipeline: ComputePipeline,
    /// Pre-compiled expansion decision pipeline
    expansion_pipeline: ComputePipeline,
    /// Pre-compiled backpropagation pipeline
    backprop_pipeline: ComputePipeline,
    /// Pre-compiled max reduction pipeline
    max_reduction_pipeline: ComputePipeline,
    /// Bind group layouts for each pipeline
    puct_bind_group_layout: BindGroupLayout,
    expansion_bind_group_layout: BindGroupLayout,
    backprop_bind_group_layout: BindGroupLayout,
    max_reduction_bind_group_layout: BindGroupLayout,
    /// Configuration
    config: GpuConfig,
    /// Maximum buffer size supported
    max_buffer_size: u64,
}

impl GpuContext {
    /// Creates a new GPU context with the specified configuration
    ///
    /// This function initializes the GPU device, compiles shaders, and creates
    /// compute pipelines. It will return an error if no suitable GPU is found.
    ///
    /// # Arguments
    /// * `config` - GPU configuration options
    ///
    /// # Returns
    /// * `Ok(GpuContext)` if initialization succeeds
    /// * `Err(GpuError)` if initialization fails
    pub fn new(config: &GpuConfig) -> Result<Self, GpuError> {
        // Create instance with all backends enabled
        let instance = Instance::new(InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter with specified power preference
        let power_preference = if config.prefer_high_performance {
            PowerPreference::HighPerformance
        } else {
            PowerPreference::LowPower
        };

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| GpuError::NoAdapter("No suitable GPU adapter found".to_string()))?;

        let adapter_info = adapter.get_info();
        
        if config.debug_mode {
            eprintln!("GPU Adapter: {} ({:?})", adapter_info.name, adapter_info.backend);
            eprintln!("Driver: {}", adapter_info.driver);
        }

        // Request device with reasonable limits
        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("MCTS GPU Device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Get maximum buffer size
        let limits = device.limits();
        let max_buffer_size = limits.max_buffer_size;

        // Compile shaders
        let puct_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PUCT Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::PUCT_SHADER.into()),
        });

        let expansion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Expansion Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::EXPANSION_SHADER.into()),
        });

        let backprop_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Backprop Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::BACKPROP_SHADER.into()),
        });

        let max_reduction_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Max Reduction Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::MAX_REDUCTION_SHADER.into()),
        });

        // Create bind group layouts
        let puct_bind_group_layout = Self::create_puct_bind_group_layout(&device);
        let expansion_bind_group_layout = Self::create_expansion_bind_group_layout(&device);
        let backprop_bind_group_layout = Self::create_backprop_bind_group_layout(&device);
        let max_reduction_bind_group_layout = Self::create_max_reduction_bind_group_layout(&device);

        // Create pipelines
        let puct_pipeline = Self::create_compute_pipeline(
            &device,
            &puct_shader,
            "compute_puct",
            &puct_bind_group_layout,
            "PUCT Pipeline",
        );

        let expansion_pipeline = Self::create_compute_pipeline(
            &device,
            &expansion_shader,
            "compute_expansion",
            &expansion_bind_group_layout,
            "Expansion Pipeline",
        );

        let backprop_pipeline = Self::create_compute_pipeline(
            &device,
            &backprop_shader,
            "compute_backprop",
            &backprop_bind_group_layout,
            "Backprop Pipeline",
        );

        let max_reduction_pipeline = Self::create_compute_pipeline(
            &device,
            &max_reduction_shader,
            "reduce_max",
            &max_reduction_bind_group_layout,
            "Max Reduction Pipeline",
        );

        Ok(Self {
            device,
            queue,
            adapter_info,
            puct_pipeline,
            expansion_pipeline,
            backprop_pipeline,
            max_reduction_pipeline,
            puct_bind_group_layout,
            expansion_bind_group_layout,
            backprop_bind_group_layout,
            max_reduction_bind_group_layout,
            config: config.clone(),
            max_buffer_size,
        })
    }

    /// Creates the bind group layout for PUCT computation
    fn create_puct_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("PUCT Bind Group Layout"),
            entries: &[
                // Input nodes buffer
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output results buffer
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
                // Params uniform buffer
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Creates the bind group layout for expansion computation
    fn create_expansion_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Expansion Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
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
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Creates the bind group layout for backpropagation computation
    fn create_backprop_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Backprop Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
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
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Creates the bind group layout for max reduction
    fn create_max_reduction_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Max Reduction Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
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
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Creates a compute pipeline with the specified shader and entry point
    fn create_compute_pipeline(
        device: &Device,
        shader: &ShaderModule,
        entry_point: &str,
        bind_group_layout: &BindGroupLayout,
        label: &str,
    ) -> ComputePipeline {
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{} Layout", label)),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Returns the GPU adapter information
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Returns the GPU device handle
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the command queue
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Returns the configuration
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    /// Returns the PUCT compute pipeline
    pub fn puct_pipeline(&self) -> &ComputePipeline {
        &self.puct_pipeline
    }

    /// Returns the expansion compute pipeline
    pub fn expansion_pipeline(&self) -> &ComputePipeline {
        &self.expansion_pipeline
    }

    /// Returns the backpropagation compute pipeline
    pub fn backprop_pipeline(&self) -> &ComputePipeline {
        &self.backprop_pipeline
    }

    /// Returns the max reduction compute pipeline
    pub fn max_reduction_pipeline(&self) -> &ComputePipeline {
        &self.max_reduction_pipeline
    }

    /// Returns the PUCT bind group layout
    pub fn puct_bind_group_layout(&self) -> &BindGroupLayout {
        &self.puct_bind_group_layout
    }

    /// Returns the expansion bind group layout
    pub fn expansion_bind_group_layout(&self) -> &BindGroupLayout {
        &self.expansion_bind_group_layout
    }

    /// Returns the backpropagation bind group layout
    pub fn backprop_bind_group_layout(&self) -> &BindGroupLayout {
        &self.backprop_bind_group_layout
    }

    /// Returns the max reduction bind group layout
    pub fn max_reduction_bind_group_layout(&self) -> &BindGroupLayout {
        &self.max_reduction_bind_group_layout
    }

    /// Returns the maximum buffer size supported by this device
    pub fn max_buffer_size(&self) -> u64 {
        self.max_buffer_size
    }

    /// Creates a GPU buffer with the specified size and usage flags
    pub fn create_buffer(&self, label: &str, size: u64, usage: BufferUsages) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Submits a command buffer and waits for completion
    pub fn submit_and_wait(&self, command_buffer: wgpu::CommandBuffer) {
        self.queue.submit(std::iter::once(command_buffer));
        self.device.poll(Maintain::Wait);
    }

    /// Returns a debug string with GPU information
    pub fn debug_info(&self) -> String {
        format!(
            "GPU: {} ({:?})\nDriver: {}\nMax Buffer: {} MB\nMax Batch: {} nodes",
            self.adapter_info.name,
            self.adapter_info.backend,
            self.adapter_info.driver,
            self.max_buffer_size / 1024 / 1024,
            self.config.max_batch_size
        )
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("adapter_name", &self.adapter_info.name)
            .field("backend", &self.adapter_info.backend)
            .field("max_buffer_size", &self.max_buffer_size)
            .field("config", &self.config)
            .finish()
    }
}
