#![cfg(feature = "gpu")]

use mcts::gpu::{GpuConfig, GpuContext};

fn main() {
    let config = GpuConfig {
        backend_override: Some("dx12".to_string()),
        debug_mode: true,
        min_batch_threshold: 0,
        ..Default::default()
    };

    let stack_size = 64 * 1024 * 1024;
    let join = std::thread::Builder::new()
        .name("dx12-init".to_string())
        .stack_size(stack_size)
        .spawn(move || GpuContext::new(&config))
        .expect("failed to spawn dx12-init thread");

    match join.join() {
        Ok(Ok(_ctx)) => {
            println!("DX12 init OK");
            std::process::exit(0);
        }
        Ok(Err(err)) => {
            eprintln!("DX12 init ERR: {err}");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("DX12 init panicked");
            std::process::exit(1);
        }
    }
}
