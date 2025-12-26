//! GPU Context Management
//!
//! Handles GPU device initialization and compute pipeline management.

use wgpu::{
    Device, Queue, Instance, InstanceDescriptor, RequestAdapterOptions,
    PowerPreference, DeviceDescriptor, Features, Limits, ShaderModule,
    ComputePipeline, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType,
    PipelineLayoutDescriptor, ComputePipelineDescriptor, Maintain,
};
use std::sync::Arc;

use super::GpuConfig;
use super::shaders;

/// GPU operation errors
#[derive(Debug)]
pub enum GpuError {
    NoAdapter(String),
    DeviceRequest(String),
    BufferError(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter(msg) => write!(f, "No GPU adapter: {}", msg),
            GpuError::DeviceRequest(msg) => write!(f, "GPU device error: {}", msg),
            GpuError::BufferError(msg) => write!(f, "Buffer error: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

/// GPU Context with device, queue, and compute pipelines
pub struct GpuContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    adapter_info: wgpu::AdapterInfo,
    puct_pipeline: ComputePipeline,
    pub gomoku_eval_pipeline: ComputePipeline,
    pub connect4_eval_pipeline: ComputePipeline,
    pub othello_eval_pipeline: ComputePipeline,
    pub blokus_eval_pipeline: ComputePipeline,
    pub hive_eval_pipeline: ComputePipeline,
    puct_bind_group_layout: BindGroupLayout,
    pub eval_bind_group_layout: BindGroupLayout,
    config: GpuConfig,
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

        // Decide on required features (SPIR-V passthrough is optional and checked for support)
        let adapter_features = adapter.features();
        let use_spirv = adapter_features.contains(Features::SPIRV_SHADER_PASSTHROUGH);
        let requested_features = if use_spirv {
            Features::SPIRV_SHADER_PASSTHROUGH
        } else {
            Features::empty()
        };

        // Request device with reasonable limits
        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("MCTS GPU Device"),
                features: requested_features,
                limits: Limits::default(),
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

        // Load SPIR-V shader module for Rust-based kernels (if supported).
        // NOTE: `include_bytes!` returns a byte slice with no alignment guarantee; `bytemuck::cast_slice`
        // can panic here. Convert bytes -> u32 words explicitly to avoid alignment issues.
        let spirv_shader = if use_spirv {
            let bytes = shaders::SHADER_MODULE_SPV;
            if bytes.len() % 4 != 0 {
                eprintln!("SPIR-V module length is not a multiple of 4 ({} bytes); disabling SPIR-V pipeline", bytes.len());
                None
            } else {
                let words: Vec<u32> = bytes
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Some(unsafe {
                    device.create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
                        label: Some("Rust-GPU SPIR-V Shader"),
                        source: std::borrow::Cow::Owned(words),
                    })
                })
            }
        } else {
            None
        };

        // Create bind group layouts
        let puct_bind_group_layout = Self::create_puct_bind_group_layout(&device);
        let eval_bind_group_layout = Self::create_eval_bind_group_layout(&device);

        // Create pipelines
        let puct_pipeline = Self::create_compute_pipeline(
            &device,
            &puct_shader,
            "compute_puct",
            &puct_bind_group_layout,
            "PUCT Pipeline",
        );

        // Helper closure to create game pipelines
        let create_game_pipeline = |source: String, entry: &str, name: &str| -> ComputePipeline {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{} Shader", name)),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            Self::create_compute_pipeline(
                &device,
                &shader,
                entry,
                &eval_bind_group_layout,
                &format!("{} Pipeline", name),
            )
        };

        let gomoku_eval_pipeline = create_game_pipeline(
            shaders::GOMOKU_SHADER.to_string(),
            "evaluate_gomoku",
            "Gomoku Eval"
        );

        // Use SPIR-V pipeline for Connect4 when available, otherwise WGSL fallback
        let connect4_eval_pipeline = if let Some(spirv) = spirv_shader.as_ref() {
            Self::create_compute_pipeline(
                &device,
                spirv,
                "evaluate_connect4",
                &eval_bind_group_layout,
                "Connect4 Eval (Rust-GPU)",
            )
        } else {
            let connect4_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Connect4 WGSL Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::CONNECT4_SHADER.into()),
            });
            Self::create_compute_pipeline(
                &device,
                &connect4_shader,
                "evaluate_connect4",
                &eval_bind_group_layout,
                "Connect4 Eval (WGSL fallback)",
            )
        };

        let othello_eval_pipeline = create_game_pipeline(
            shaders::OTHELLO_SHADER.to_string(),
            "evaluate_othello",
            "Othello Eval"
        );

        let blokus_eval_pipeline = create_game_pipeline(
            shaders::BLOKUS_SHADER.to_string(),
            "evaluate_blokus",
            "Blokus Eval"
        );

        let hive_eval_pipeline = create_game_pipeline(
            shaders::HIVE_SHADER.to_string(),
            "evaluate_hive",
            "Hive Eval"
        );

        Ok(Self {
            device,
            queue,
            adapter_info,
            puct_pipeline,
            gomoku_eval_pipeline,
            connect4_eval_pipeline,
            othello_eval_pipeline,
            blokus_eval_pipeline,
            hive_eval_pipeline,
            puct_bind_group_layout,
            eval_bind_group_layout,
            config: config.clone(),
            max_buffer_size,
        })
    }

    /// Creates the bind group layout for game evaluation
    fn create_eval_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Eval Bind Group Layout"),
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

    /// Creates the bind group layout for PUCT computation
    fn create_puct_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("PUCT Bind Group Layout"),
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
            entry_point: entry_point,
        })
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    pub fn puct_pipeline(&self) -> &ComputePipeline {
        &self.puct_pipeline
    }

    pub fn puct_bind_group_layout(&self) -> &BindGroupLayout {
        &self.puct_bind_group_layout
    }

    /// Submits a command buffer and waits for completion
    pub fn submit_and_wait(&self, command_buffer: wgpu::CommandBuffer) {
        self.queue.submit(std::iter::once(command_buffer));
        self.device.poll(Maintain::Wait);
    }

    /// Returns a debug string with GPU information
    pub fn debug_info(&self) -> String {
        format!(
            "GPU: {} ({:?}), Driver: {}",
            self.adapter_info.name,
            self.adapter_info.backend,
            self.adapter_info.driver,
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
