//! Test real kernel emission guard with a nontrivial board state and root_player -1
#[cfg(test)]
mod tests {
    use mcts::{GpuOthelloMcts, GpuConfig, GpuContext};
    use std::sync::Arc;
    #[test]
    fn test_real_kernel_emission_guard_nontrivial_board() {
        let config = GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
        let engine = GpuOthelloMcts::new(context.clone(), 128, 1).expect("Failed to create engine");
        let device = context.device();
        let queue = context.queue();
        let num_workgroups = 2u32;
        let total_threads = 64 * num_workgroups;
        let mut board = [0; 64];
        board[27] = 1; board[28] = -1; board[35] = -1; board[36] = 1;
        let root_player = -1;
        let legal_moves = &[(2, 3), (3, 2), (4, 5), (5, 4)];
        engine.init_tree(&board, root_player, legal_moves);
        {
            let inner = engine.inner.lock().unwrap();
            if let Some(buf) = &inner.global_reroot_threads_remaining {
                let temp = (total_threads as u32).to_le_bytes();
                queue.write_buffer(buf, 0, &temp);
                device.poll(wgpu::Maintain::Wait);
            }
        }
        engine.dispatch_mcts_othello_kernel(num_workgroups);
        device.poll(wgpu::Maintain::Wait);
        let mut atomic_val = 0u32;
        {
            let inner = engine.inner.lock().unwrap();
            if let Some(buf) = &inner.global_reroot_threads_remaining {
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Test Atomic Staging"),
                    size: 4,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Test Copy Encoder"),
                });
                encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, 4);
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);
                let slice = staging.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::Maintain::Wait);
                let data = slice.get_mapped_range();
                atomic_val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            }
        }
        assert_eq!(atomic_val, 0, "Real kernel emission guard (nontrivial board): atomic not decremented to zero!");
    }
}
