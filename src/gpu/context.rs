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
use std::env;
use std::borrow::Cow;

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
        // Choose backend(s). On Windows, prefer DX12 first to avoid hitting
        // Naga's SPIR-V backend at runtime (WGSL->SPIR-V), which can panic for
        // some generated shaders.
        //
        // Override with `MCTS_WGPU_BACKEND=dx12|vulkan|all` if needed.
        let override_backend = env::var("MCTS_WGPU_BACKEND").ok();

        // Request adapter with specified power preference
        let power_preference = if config.prefer_high_performance {
            PowerPreference::HighPerformance
        } else {
            PowerPreference::LowPower
        };

        let request_options = RequestAdapterOptions {
            power_preference,
            compatible_surface: None,
            force_fallback_adapter: false,
        };

        let mut try_backends: Vec<wgpu::Backends> = Vec::new();
        match override_backend.as_deref() {
            Some("dx12") => try_backends.push(wgpu::Backends::DX12),
            Some("vulkan") => try_backends.push(wgpu::Backends::VULKAN),
            Some("all") => try_backends.push(wgpu::Backends::all()),
            Some(other) => {
                eprintln!("Unknown MCTS_WGPU_BACKEND='{other}', falling back to defaults");
            }
            None => {}
        }

        if try_backends.is_empty() {
            if cfg!(windows) {
                // Vulkan first (we can load SPIR-V directly), then fall back.
                try_backends.push(wgpu::Backends::VULKAN);
                try_backends.push(wgpu::Backends::all());
            } else {
                try_backends.push(wgpu::Backends::all());
            }
        }

        let mut adapter: Option<wgpu::Adapter> = None;
        let mut instance: Option<Instance> = None;
        for backends in try_backends {
            let inst = Instance::new(InstanceDescriptor {
                backends,
                ..Default::default()
            });

            if let Some(a) = pollster::block_on(inst.request_adapter(&request_options)) {
                adapter = Some(a);
                instance = Some(inst);
                break;
            }
        }

        let _instance = instance.ok_or_else(|| GpuError::NoAdapter("No suitable GPU adapter found".to_string()))?;
        let adapter = adapter.unwrap();

        let adapter_info = adapter.get_info();
        
        if config.debug_mode {
            eprintln!("GPU Adapter: {} ({:?})", adapter_info.name, adapter_info.backend);
            eprintln!("Driver: {}", adapter_info.driver);
        }

        // Request device
        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("MCTS GPU Device"),
                features: Features::empty(),
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

        // Compile shaders.
        // On Vulkan, prefer feeding validated rust-gpu SPIR-V directly to avoid
        // runtime WGSL->SPIR-V codegen (which can panic inside Naga).
        let mcts_shader = match adapter_info.backend {
            wgpu::Backend::Vulkan => {
                let bytes = shaders::MCTS_SHADERS_SPV;
                if bytes.len() % 4 != 0 {
                    return Err(GpuError::DeviceRequest("Invalid SPIR-V byte length".to_string()));
                }

                let mut words = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }

                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("MCTS Shaders (rust-gpu SPIR-V)"),
                    source: wgpu::ShaderSource::SpirV(Cow::Owned(words)),
                })
            }
            _ => device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MCTS Shaders (generated WGSL)"),
                source: wgpu::ShaderSource::Wgsl(shaders::MCTS_SHADERS_WGSL.into()),
            }),
        };

        // Create bind group layouts
        let puct_bind_group_layout = Self::create_puct_bind_group_layout(&device);
        let eval_bind_group_layout = Self::create_eval_bind_group_layout(&device);

        // Create pipelines
        // NOTE: Each pipeline uses its own bind group 0 layout.
        let puct_pipeline = Self::create_compute_pipeline(
            &device,
            &mcts_shader,
            "compute_puct",
            &[&puct_bind_group_layout],
            "PUCT Pipeline",
        );

        let gomoku_eval_pipeline = Self::create_compute_pipeline(
            &device,
            &mcts_shader,
            "evaluate_gomoku",
            &[&eval_bind_group_layout],
            "Gomoku Eval Pipeline",
        );

        let connect4_eval_pipeline = Self::create_compute_pipeline(
            &device,
            &mcts_shader,
            "evaluate_connect4",
            &[&eval_bind_group_layout],
            "Connect4 Eval Pipeline",
        );

        let othello_eval_pipeline = Self::create_compute_pipeline(
            &device,
            &mcts_shader,
            "evaluate_othello",
            &[&eval_bind_group_layout],
            "Othello Eval Pipeline",
        );

        let blokus_eval_pipeline = Self::create_compute_pipeline(
            &device,
            &mcts_shader,
            "evaluate_blokus",
            &[&eval_bind_group_layout],
            "Blokus Eval Pipeline",
        );

        let hive_eval_pipeline = Self::create_compute_pipeline(
            &device,
            &mcts_shader,
            "evaluate_hive",
            &[&eval_bind_group_layout],
            "Hive Eval Pipeline",
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
        bind_group_layouts: &[&BindGroupLayout],
        label: &str,
    ) -> ComputePipeline {
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{} Layout", label)),
            bind_group_layouts,
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
