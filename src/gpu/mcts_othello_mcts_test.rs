#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::gpu::GpuContext;

    #[test]
    fn test_gpu_othello_mcts_node_allocation() {
        // Create a real GpuContext with default config
        let config = crate::gpu::GpuConfig::default();
        let context = Arc::new(GpuContext::new(&config).expect("Failed to create GpuContext"));
        let mut mcts = GpuOthelloMcts::new(context, 1024, 128).expect("Failed to create GpuOthelloMcts");
        let board = [0i32; 64];
        let root_player = 1;
        let legal_moves = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
        mcts.init_tree(&board, root_player, &legal_moves);
        let telemetry = mcts.run_iterations(2048, 0.1, 1.0, 0.06, 42);
        // Check that at least one node was allocated/visited
        assert!(mcts.get_total_nodes() > 0, "No nodes were allocated!");
        let children = mcts.get_children_stats();
        assert!(children.iter().any(|&(_, _, visits, _, _)| visits > 0), "No child visits recorded!");
    }
}
