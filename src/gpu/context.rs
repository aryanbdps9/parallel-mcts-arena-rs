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
use std::time::{Duration, Instant};
use std::path::PathBuf;
use std::sync::Mutex;

#[cfg(feature = "gpu")]
use futures::FutureExt;
#[cfg(feature = "gpu")]
use std::pin::pin;

use super::GpuConfig;
use super::shaders;

#[cfg(windows)]
fn find_dxc_binaries() -> Option<(PathBuf, PathBuf)> {
    // Allow explicit override first.
    if let (Ok(dxc), Ok(dxil)) = (env::var("MCTS_DXC_PATH"), env::var("MCTS_DXIL_PATH")) {
        let dxc = PathBuf::from(dxc);
        let dxil = PathBuf::from(dxil);
        if dxc.is_file() && dxil.is_file() {
            return Some((dxc, dxil));
        }
    }

    // Try common Windows SDK locations.
    let program_files_x86 = env::var_os("ProgramFiles(x86)").map(PathBuf::from);
    let Some(pfx86) = program_files_x86 else {
        return None;
    };

    let kits_root = pfx86.join("Windows Kits").join("10");
    let dxil_candidate = kits_root.join("Redist").join("D3D").join("x64").join("dxil.dll");
    if !dxil_candidate.is_file() {
        return None;
    }

    let bin_root = kits_root.join("bin");

    // Prefer versioned bin directories (e.g. bin/10.0.22621.0/x64/dxcompiler.dll).
    let mut best: Option<PathBuf> = None;
    if let Ok(entries) = std::fs::read_dir(&bin_root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let candidate = path.join("x64").join("dxcompiler.dll");
            if candidate.is_file() {
                // Pick the lexicographically greatest version dir name, which tends to match latest.
                match &best {
                    None => best = Some(candidate),
                    Some(prev) => {
                        if candidate.parent().and_then(|p| p.parent()).map(|p| p.file_name()).flatten()
                            > prev.parent().and_then(|p| p.parent()).map(|p| p.file_name()).flatten()
                        {
                            best = Some(candidate);
                        }
                    }
                }
            }
        }
    }

    // Fallback: non-versioned bin/x64/dxcompiler.dll.
    let fallback = bin_root.join("x64").join("dxcompiler.dll");
    if best.is_none() && fallback.is_file() {
        best = Some(fallback);
    }

    best.map(|dxc| (dxc, dxil_candidate))
}

/// GPU operation errors
#[derive(Debug)]
pub enum GpuError {
    NoAdapter(String),
    DeviceRequest(String),
    BufferError(String),
    Cancelled,
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter(msg) => write!(f, "No GPU adapter: {}", msg),
            GpuError::DeviceRequest(msg) => write!(f, "GPU device error: {}", msg),
            GpuError::BufferError(msg) => write!(f, "Buffer error: {}", msg),
            GpuError::Cancelled => write!(f, "GPU operation cancelled"),
        }
    }
}

impl std::error::Error for GpuError {}

/// GPU Context with device, queue, and compute pipelines
pub struct GpuContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    adapter_info: wgpu::AdapterInfo,
    mcts_shader: ShaderModule,
    puct_pipeline: ComputePipeline,
    gomoku_eval_pipeline: Mutex<Option<Arc<ComputePipeline>>>,
    connect4_eval_pipeline: Mutex<Option<Arc<ComputePipeline>>>,
    othello_eval_pipeline: Mutex<Option<Arc<ComputePipeline>>>,
    blokus_eval_pipeline: Mutex<Option<Arc<ComputePipeline>>>,
    hive_eval_pipeline: Mutex<Option<Arc<ComputePipeline>>>,
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
        // Backend override precedence:
        // 1) `GpuConfig.backend_override` (typically driven via CLI args)
        // 2) `MCTS_WGPU_BACKEND=dx12|vulkan|all` env var (legacy)
        let override_backend: Option<Cow<'_, str>> = config
            .backend_override
            .as_deref()
            .map(Cow::Borrowed)
            .or_else(|| env::var("MCTS_WGPU_BACKEND").ok().map(Cow::Owned));

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
            let mut desc = InstanceDescriptor {
                backends,
                ..Default::default()
            };
            // Prefer DXC on DX12: FXC is significantly less compatible with
            // modern shader constructs and tends to choke on some Naga output.
            if cfg!(windows) {
                #[cfg(windows)]
                {
                    let dxc = find_dxc_binaries();
                    if config.debug_mode {
                        match &dxc {
                            Some((dxc_path, dxil_path)) => {
                                eprintln!("[dx12] Using DXC: {} + {}", dxc_path.display(), dxil_path.display());
                            }
                            None => {
                                eprintln!("[dx12] DXC not found; wgpu may fall back to FXC. Set MCTS_DXC_PATH + MCTS_DXIL_PATH to override.");
                            }
                        }
                    }

                    desc.dx12_shader_compiler = match dxc {
                        Some((dxc_path, dxil_path)) => wgpu::Dx12Compiler::Dxc {
                            dxil_path: Some(dxil_path),
                            dxc_path: Some(dxc_path),
                        },
                        None => wgpu::Dx12Compiler::Dxc {
                            dxil_path: None,
                            dxc_path: None,
                        },
                    };
                }
                #[cfg(not(windows))]
                {
                    desc.dx12_shader_compiler = wgpu::Dx12Compiler::Dxc {
                        dxil_path: None,
                        dxc_path: None,
                    };
                }
            }
            let inst = Instance::new(desc);

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

        // Prevent background threads from panicking the whole AI worker when wgpu
        // encounters an internal/validation error (e.g. shader compilation on DX12).
        // Errors are still captured via push/pop_error_scope and propagated as Err.
        device.on_uncaptured_error(Box::new(|err| {
            eprintln!("[wgpu] uncaptured error: {err}");
        }));

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Get maximum buffer size
        let limits = device.limits();
        let max_buffer_size = limits.max_buffer_size;

        // Create shaders + pipelines.
        // DX12 can overflow the default Rust thread stack during shader translation for large modules.
        // Run pipeline creation on a larger-stack thread to avoid crashing/hanging the app.
        let (mcts_shader, puct_bind_group_layout, eval_bind_group_layout, puct_pipeline) = {
            let device = device.clone();
            let backend = adapter_info.backend;
            let debug_mode = config.debug_mode;

            let init = move || -> Result<(_, _, _, _), GpuError> {
                if backend == wgpu::Backend::Dx12 && debug_mode {
                    eprintln!("[dx12-init] creating shader module");
                }
                // Compile shaders.
                // Prefer feeding validated rust-gpu SPIR-V directly when possible.
                // - Vulkan: avoids runtime WGSL->SPIR-V codegen (which can panic inside Naga).
                // On DX12, prefer WGSL to avoid SPIR-V->HLSL translation stalls on some drivers.
                let mcts_shader = match backend {
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

                // Create PUCT pipeline eagerly (always required).
                // NOTE: Each pipeline uses its own bind group 0 layout.
                if backend == wgpu::Backend::Dx12 && debug_mode {
                    eprintln!("[dx12-init] creating pipeline: PUCT Pipeline");
                }
                let puct_pipeline = Self::create_compute_pipeline_checked(
                    &device,
                    &mcts_shader,
                    "compute_puct",
                    &[&puct_bind_group_layout],
                    "PUCT Pipeline",
                )?;

                Ok((mcts_shader, puct_bind_group_layout, eval_bind_group_layout, puct_pipeline))
            };

            if backend == wgpu::Backend::Dx12 {
                let join = std::thread::Builder::new()
                    .name("dx12-pipeline-init".to_string())
                    .stack_size(64 * 1024 * 1024)
                    .spawn(init)
                    .map_err(|e| GpuError::DeviceRequest(format!("Failed to spawn DX12 init thread: {e}")))?;
                join.join().map_err(|_| GpuError::DeviceRequest("DX12 init thread panicked".to_string()))??
            } else {
                init()?
            }
        };

        Ok(Self {
            device,
            queue,
            adapter_info,
            mcts_shader,
            puct_pipeline,
            gomoku_eval_pipeline: Mutex::new(None),
            connect4_eval_pipeline: Mutex::new(None),
            othello_eval_pipeline: Mutex::new(None),
            blokus_eval_pipeline: Mutex::new(None),
            hive_eval_pipeline: Mutex::new(None),
            puct_bind_group_layout,
            eval_bind_group_layout,
            config: config.clone(),
            max_buffer_size,
        })
    }

    pub fn eval_pipeline_for_game(&self, game_type: u32) -> Result<Arc<ComputePipeline>, GpuError> {
        // Keep this in sync with the shader crate's GAME_* constants.
        const GAME_GOMOKU: u32 = 0;
        const GAME_CONNECT4: u32 = 1;
        const GAME_OTHELLO: u32 = 2;
        const GAME_BLOKUS: u32 = 3;
        const GAME_HIVE: u32 = 4;

        let (slot, entry_point, label) = match game_type {
            GAME_GOMOKU => (&self.gomoku_eval_pipeline, "evaluate_gomoku", "Gomoku Eval Pipeline"),
            GAME_CONNECT4 => (&self.connect4_eval_pipeline, "evaluate_connect4", "Connect4 Eval Pipeline"),
            GAME_OTHELLO => (&self.othello_eval_pipeline, "evaluate_othello", "Othello Eval Pipeline"),
            GAME_BLOKUS => (&self.blokus_eval_pipeline, "evaluate_blokus", "Blokus Eval Pipeline"),
            GAME_HIVE => (&self.hive_eval_pipeline, "evaluate_hive", "Hive Eval Pipeline"),
            other => {
                return Err(GpuError::DeviceRequest(format!(
                    "Unknown game type for eval pipeline: {other}"
                )))
            }
        };

        // Fast path: already created.
        if let Some(existing) = slot.lock().unwrap().as_ref() {
            return Ok(existing.clone());
        }

        // Slow path: create under lock (simple and safe; creation is expensive anyway).
        let mut guard = slot.lock().unwrap();
        if let Some(existing) = guard.as_ref() {
            return Ok(existing.clone());
        }

        if self.adapter_info.backend == wgpu::Backend::Dx12 && self.config.debug_mode {
            eprintln!("[dx12-init] creating pipeline (lazy): {label}");
        }
        let pipeline = Self::create_compute_pipeline_checked(
            &self.device,
            &self.mcts_shader,
            entry_point,
            &[&self.eval_bind_group_layout],
            label,
        )?;
        let arc = Arc::new(pipeline);
        *guard = Some(arc.clone());
        Ok(arc)
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
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
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

    /// Creates a compute pipeline but captures wgpu validation errors instead of panicking.
    fn create_compute_pipeline_checked(
        device: &Device,
        shader: &ShaderModule,
        entry_point: &str,
        bind_group_layouts: &[&BindGroupLayout],
        label: &str,
    ) -> Result<ComputePipeline, GpuError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = Self::create_compute_pipeline(device, shader, entry_point, bind_group_layouts, label);

        // Drive wgpu so the error scope can resolve.
        // On some backends (notably DX12 with large shaders), shader translation can be very slow
        // or even get stuck; we hard-timeout so callers don't hang indefinitely.
        let mut fut = pin!(device.pop_error_scope());
        let start = Instant::now();
        let timeout = Duration::from_secs(10);
        loop {
            device.poll(Maintain::Poll);

            if let Some(err) = fut.as_mut().now_or_never() {
                if let Some(err) = err {
                    return Err(GpuError::DeviceRequest(format!(
                        "Failed to create compute pipeline '{label}': {err}"
                    )));
                }
                break;
            }

            if start.elapsed() > timeout {
                return Err(GpuError::DeviceRequest(format!(
                    "Timed out creating compute pipeline '{label}' after {timeout:?}"
                )));
            }

            std::thread::yield_now();
        }

        Ok(pipeline)
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
