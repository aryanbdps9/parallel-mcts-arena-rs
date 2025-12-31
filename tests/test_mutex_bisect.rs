use std::sync::Mutex;
use mcts::UrgentEventBuffers;
use std::sync::Arc;
use mcts::gpu::{GpuContext, GpuConfig};

pub struct BisectEngine3 {
    pub context: Arc<GpuContext>,
    pub urgent_event_buffers: Arc<UrgentEventBuffers>,
}

#[test]
fn test_mutex_bisect_engine3() {
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    let device = context.device();
    let urgent_event_buffers = Arc::new(UrgentEventBuffers {
        urgent_event_buffer: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Urgent Event Buffer"),
            size: 256 * 1024,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        urgent_event_staging_buffer: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Urgent Event Staging Buffer"),
            size: 256 * 1024,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        urgent_event_write_head_buffer: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Urgent Event Write Head Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        urgent_event_write_head_staging: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Urgent Event Write Head Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        urgent_event_buffer_in_use: std::sync::atomic::AtomicBool::new(false)
    });
    let s = Arc::new(BisectEngine3 { context, urgent_event_buffers });
    println!("[DIAG] constructed Arc<BisectEngine3> with Arc<GpuContext> and urgent_event_buffers");
    no_op(s);
    println!("[DIAG] returned from no_op");
}
pub struct BisectEngine2 {
    pub context: Arc<GpuContext>,
    pub node_info_buffer: wgpu::Buffer,
    pub node_visits_buffer: wgpu::Buffer,
    pub node_wins_buffer: wgpu::Buffer,
    pub node_vl_buffer: wgpu::Buffer,
    pub node_state_buffer: wgpu::Buffer,
    pub children_indices_buffer: wgpu::Buffer,
    pub children_priors_buffer: wgpu::Buffer,
    pub free_lists_buffer: wgpu::Buffer,
    pub free_tops_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,
    pub work_items_buffer: wgpu::Buffer,
    pub paths_buffer: wgpu::Buffer,
    pub alloc_counter_buffer: wgpu::Buffer,
    pub sim_boards_buffer: wgpu::Buffer,
    pub stats_buffer: wgpu::Buffer,
    pub stats_staging_buffer: wgpu::Buffer,
    pub node_pool_layout: wgpu::BindGroupLayout,
    pub execution_layout: wgpu::BindGroupLayout,
    pub game_state_layout: wgpu::BindGroupLayout,
    pub stats_layout: wgpu::BindGroupLayout,
    pub select_pipeline: wgpu::ComputePipeline,
    pub backprop_pipeline: wgpu::ComputePipeline,
    pub stats_pipeline: wgpu::ComputePipeline,
}

#[test]
fn test_mutex_bisect_engine2() {
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    let device = context.device();
    // Buffers
    let node_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Node Info"),
        size: (1024 * std::mem::size_of::<mcts::gpu::NodeInfo>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let node_visits_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Node Visits"),
        size: (1024 * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let node_wins_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Node Wins"),
        size: (1024 * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let node_vl_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Node VL"),
        size: (1024 * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let node_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Node State"),
        size: (1024 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let children_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Children Indices"),
        size: (1024 * 8 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let children_priors_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Children Priors"),
        size: (1024 * 8 * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let free_lists_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Free Lists"),
        size: (256 * 8192 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let free_tops_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Free Tops"),
        size: (256 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MCTS Params"),
        size: std::mem::size_of::<mcts::gpu::MctsParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let work_items_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Work Items"),
        size: (64 * std::mem::size_of::<mcts::gpu::WorkItem>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let paths_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Paths"),
        size: (64 * 128 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let alloc_counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Alloc Counter"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let sim_boards_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sim Boards"),
        size: (64 * 42 * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Tree Stats"),
        size: std::mem::size_of::<mcts::gpu::TreeStats>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let stats_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Stats Staging"),
        size: std::mem::size_of::<mcts::gpu::TreeStats>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Layouts
    let node_pool_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Node Pool Layout"),
        entries: &[],
    });
    let execution_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Execution Layout"),
        entries: &[],
    });
    let game_state_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Game State Layout"),
        entries: &[],
    });
    let stats_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Stats Layout"),
        entries: &[],
    });
    // Pipelines
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Dummy Shader"),
        source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&node_pool_layout, &execution_layout, &game_state_layout, &stats_layout],
        push_constant_ranges: &[],
    });
    let select_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Select Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let backprop_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Backprop Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let stats_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Stats Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let s = Arc::new(Mutex::new(BisectEngine2 {
        context, node_info_buffer, node_visits_buffer, node_wins_buffer, node_vl_buffer,
        node_state_buffer, children_indices_buffer, children_priors_buffer, free_lists_buffer, free_tops_buffer,
        params_buffer, work_items_buffer, paths_buffer, alloc_counter_buffer, sim_boards_buffer, stats_buffer, stats_staging_buffer,
        node_pool_layout, execution_layout, game_state_layout, stats_layout,
        select_pipeline, backprop_pipeline, stats_pipeline
    }));
    println!("[DIAG] constructed Arc<Mutex<BisectEngine2>> with all wgpu::Buffer, BindGroupLayout, and ComputePipeline fields");
    no_op(s);
    println!("[DIAG] returned from no_op");
}
pub struct BisectEngine1 {
    pub context: Arc<GpuContext>,
    pub node_info_buffer: wgpu::Buffer,
    pub node_visits_buffer: wgpu::Buffer,
    pub node_wins_buffer: wgpu::Buffer,
    pub node_vl_buffer: wgpu::Buffer,
    pub node_state_buffer: wgpu::Buffer,
    pub children_indices_buffer: wgpu::Buffer,
    pub children_priors_buffer: wgpu::Buffer,
    pub free_lists_buffer: wgpu::Buffer,
    pub free_tops_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,
    pub work_items_buffer: wgpu::Buffer,
    pub paths_buffer: wgpu::Buffer,
    pub alloc_counter_buffer: wgpu::Buffer,
    pub sim_boards_buffer: wgpu::Buffer,
    pub stats_buffer: wgpu::Buffer,
    pub stats_staging_buffer: wgpu::Buffer,
}

#[test]
fn test_mutex_bisect_engine1() {
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    let device = context.device();
    let node_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Node Info"),
        size: (1024 * std::mem::size_of::<mcts::gpu::NodeInfo>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let node_visits_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Node Visits"),
        size: (1024 * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let node_wins_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Node Wins"),
        size: (1024 * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let node_vl_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Node VL"),
        size: (1024 * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let node_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Node State"),
        size: (1024 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let children_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Children Indices"),
        size: (1024 * 8 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let children_priors_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Children Priors"),
        size: (1024 * 8 * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let free_lists_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Free Lists"),
        size: (256 * 8192 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let free_tops_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Free Tops"),
        size: (256 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MCTS Params"),
        size: std::mem::size_of::<mcts::gpu::MctsParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let work_items_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Work Items"),
        size: (64 * std::mem::size_of::<mcts::gpu::WorkItem>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let paths_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Paths"),
        size: (64 * 128 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let alloc_counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Alloc Counter"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let sim_boards_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sim Boards"),
        size: (64 * 42 * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Tree Stats"),
        size: std::mem::size_of::<mcts::gpu::TreeStats>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let stats_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Stats Staging"),
        size: std::mem::size_of::<mcts::gpu::TreeStats>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let s = Arc::new(Mutex::new(BisectEngine1 {
        context, node_info_buffer, node_visits_buffer, node_wins_buffer, node_vl_buffer,
        node_state_buffer, children_indices_buffer, children_priors_buffer, free_lists_buffer, free_tops_buffer,
        params_buffer, work_items_buffer, paths_buffer, alloc_counter_buffer, sim_boards_buffer, stats_buffer, stats_staging_buffer
    }));
    println!("[DIAG] constructed Arc<Mutex<BisectEngine1>> with all wgpu::Buffer fields");
    no_op(s);
    println!("[DIAG] returned from no_op");
}

pub struct TestStruct1 {
    pub context: Arc<GpuContext>,
    pub accel: Arc<mcts::gpu::GpuMctsAccelerator>,
    pub engine: Arc<mcts::gpu::GpuMctsEngine>,
}


#[test]
fn test_mutex_arc_context_field() {
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    let accel = Arc::new(mcts::gpu::GpuMctsAccelerator::new(context.clone()));
    let engine = Arc::new(mcts::gpu::GpuMctsEngine::new(context.clone(), 1024, 64, 7, 6));
    let s = Arc::new(TestStruct1 { context, accel, engine });
    println!("[DIAG] constructed Arc<TestStruct1> with Arc<GpuContext>, Arc<GpuMctsAccelerator>, Arc<GpuMctsEngine>");
    no_op(s);
    println!("[DIAG] returned from no_op");
}

fn no_op<T>(_x: Arc<T>) {}

pub struct BisectEngine4 {
    pub context: Arc<GpuContext>,
    pub node_pool_bind_group: Option<wgpu::BindGroup>,
    pub execution_bind_group: Option<wgpu::BindGroup>,
    pub game_state_bind_group: Option<wgpu::BindGroup>,
    pub stats_bind_group: Option<wgpu::BindGroup>,
    pub urgent_event_bind_group: Option<wgpu::BindGroup>,
}

#[test]
fn test_mutex_bisect_engine4() {
    let config = GpuConfig::default();
    let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
    // Create dummy bind groups (None for all)
    let s = Arc::new(BisectEngine4 {
        context,
        node_pool_bind_group: None,
        execution_bind_group: None,
        game_state_bind_group: None,
        stats_bind_group: None,
        urgent_event_bind_group: None,
    });
    println!("[DIAG] constructed Arc<BisectEngine4> with Arc<GpuContext> and Option<BindGroup> fields");
    no_op(s);
    println!("[DIAG] returned from no_op");
}
