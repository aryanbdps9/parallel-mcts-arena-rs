use clap::Parser;
use mcts::{MCTS, SearchStatistics};
use std::time::{Duration, Instant};

// Re-export games so game_wrapper can find them via crate::games
pub use mcts::games;

// Include game wrapper
#[path = "../game_wrapper.rs"]
mod game_wrapper;
use game_wrapper::GameWrapper;
use games::gomoku::GomokuState;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Board size for Gomoku (default: 15)
    #[arg(long, default_value_t = 15)]
    board_size: usize,

    /// Search duration in seconds (default: 5)
    #[arg(long, default_value_t = 5)]
    duration: u64,

    /// Number of threads for CPU search (default: 32)
    #[arg(long, default_value_t = 32)]
    threads: usize,

    /// Number of threads for CPU-only benchmark (default: 16)
    #[arg(long, default_value_t = 16)]
    cpu_bench_threads: usize,

    /// Number of threads for GPU search (default: 4096)
    #[arg(long, default_value_t = 4096)]
    gpu_threads: usize,

    /// Use heuristic evaluation instead of random rollouts for GPU simulations.
    ///
    /// Heuristic evaluation is faster and gives stronger play but is game-specific.
    /// Random rollouts are slower but work for any game.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    gpu_use_heuristic: bool,

    /// Max nodes (default: 10,000,000)
    #[arg(long, default_value_t = 10_000_000)]
    max_nodes: usize,

    /// Override wgpu backend selection ("dx12" | "vulkan" | "all").
    #[arg(long, value_parser = ["dx12", "vulkan", "all"])]
    wgpu_backend: Option<String>,

    /// Safety timeout for GPU buffer readback mapping (milliseconds).
    #[arg(long, default_value_t = 10_000)]
    gpu_readback_timeout_ms: u64,

    /// Sleep between GPU readback poll iterations (milliseconds). Use 0 to busy-yield.
    #[arg(long, default_value_t = 1)]
    gpu_readback_poll_sleep_ms: u64,

    /// Minimum number of nodes before using GPU.
    #[arg(long, default_value_t = 256)]
    gpu_min_batch_threshold: usize,

    /// Prefer high performance GPU adapter.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    gpu_prefer_high_performance: bool,

    /// Run only the GPU benchmark (skip CPU benchmark).
    /// Useful on iGPUs to avoid CPU benchmarking warming/throttling the GPU.
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    gpu_only: bool,

    /// How long to wait (max) at the end of the GPU search to drain pending GPU evaluations
    /// before reporting stats / selecting the final move.
    /// Set to 0 to disable draining.
    #[arg(long, default_value_t = 500)]
    gpu_drain_timeout_ms: u64,
}

fn main() {
    let args = Args::parse();

    println!("Parallel MCTS Arena - Benchmark Tool");
    println!("====================================");
    println!("Game: Gomoku ({}x{})", args.board_size, args.board_size);
    println!("Duration: {} seconds", args.duration);
    println!("CPU Bench Threads: {}", args.cpu_bench_threads);
    println!("GPU Worker Threads: {}", args.threads);
    println!("GPU Batch Size: {}", args.gpu_threads);
    println!("GPU Use Heuristic: {}", args.gpu_use_heuristic);
    println!("Max Nodes: {}", args.max_nodes);
    println!("------------------------------------");

    #[cfg(debug_assertions)]
    println!("WARNING: Running in debug mode. Performance will be significantly lower.\nUse --release for accurate benchmarks.\n");

    // Setup game
    let game = GameWrapper::Gomoku(GomokuState::new(args.board_size, 5));
    let exploration_constant = 1.414;

    // Use a large number of iterations so timeout controls the duration
    let iterations = 100_000_000;

    // CPU Benchmark
    if !args.gpu_only {
        println!("\nRunning CPU Benchmark...");
        let mut mcts_cpu = MCTS::new(exploration_constant, args.cpu_bench_threads, args.max_nodes);

        let start = Instant::now();
        let (_move, stats) = mcts_cpu.search(&game, iterations, 0, args.duration);
        let duration = start.elapsed();

        print_stats("CPU", &stats, duration);
    }

    // GPU Benchmark
    #[cfg(feature = "gpu")]
    {
        println!("\nRunning GPU Benchmark...");
        
        let gpu_config = mcts::gpu::GpuConfig {
            max_batch_size: args.gpu_threads,
            prefer_high_performance: args.gpu_prefer_high_performance,
            min_batch_threshold: args.gpu_min_batch_threshold,
            backend_override: args.wgpu_backend.clone(),
            readback_timeout_ms: args.gpu_readback_timeout_ms,
            readback_poll_sleep_ms: args.gpu_readback_poll_sleep_ms,
            ..Default::default()
        };
        
        let (mut mcts_gpu, gpu_msg) = MCTS::with_gpu_config(
            exploration_constant, 
            args.threads, // Use CPU threads for the thread pool
            args.max_nodes,
            gpu_config,
            args.gpu_use_heuristic
        );
        
        if let Some(msg) = gpu_msg {
            println!("GPU Init: {}", msg);
        }

        mcts_gpu.set_gpu_drain_timeout_ms(args.gpu_drain_timeout_ms);

        let start = Instant::now();
        let (_move, stats) = mcts_gpu.search(&game, iterations, 0, args.duration);
        let duration = start.elapsed();
        
        print_stats("GPU", &stats, duration);
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        println!("\nGPU Benchmark skipped (feature 'gpu' not enabled)");
    }
}

fn print_stats(name: &str, stats: &SearchStatistics, duration: Duration) {
    let secs = duration.as_secs_f64();
    let nps = stats.total_nodes as f64 / secs;
    let sps = stats.root_visits as f64 / secs;
    
    println!("{} Results:", name);
    println!("  Total Nodes: {}", stats.total_nodes);
    println!("  Time: {:.3}s", secs);
    println!("  NPS: {:.0} nodes/sec", nps);
    println!("  SPS: {:.0} sims/sec", sps);
    println!("  Root Visits: {}", stats.root_visits);
}
