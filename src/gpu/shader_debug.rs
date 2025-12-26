#[cfg(test)]
mod tests {
    use crate::gpu::{GpuContext, GpuConfig, GpuMctsAccelerator, GpuNodeData};
    use std::sync::Arc;

    #[test]
    fn debug_puct_shader() {
        // 1. Setup GPU Context with debug mode
        let config = GpuConfig {
            max_batch_size: 1024,
            prefer_high_performance: true,
            min_batch_threshold: 0, // Force GPU usage even for small batches
            debug_mode: true,
        };

        println!("Initializing GPU Context...");
        let context = match GpuContext::new(&config) {
            Ok(ctx) => Arc::new(ctx),
            Err(e) => {
                eprintln!("Failed to initialize GPU: {}", e);
                return; // Skip test if no GPU available
            }
        };

        let mut accelerator = GpuMctsAccelerator::new(context.clone());

        // 2. Create Test Data
        // Create a few nodes with known values to test different branches of the shader
        let nodes = vec![
            // Node 0: Unvisited (visits = 0)
            // Expected: q_value = 0.0, exploration_term = high
            GpuNodeData::new(0, 0, 0, 100, 0.1, 1.414),

            // Node 1: Visited, some wins
            // visits=10, wins=10 (5 real wins * 2), parent=100
            // q_value = (10/10)/2 = 0.5
            GpuNodeData::new(10, 10, 0, 100, 0.1, 1.414),

            // Node 2: Visited, virtual losses
            // visits=10, wins=10, virtual_losses=2
            // effective_visits = 12
            GpuNodeData::new(10, 10, 2, 100, 0.1, 1.414),
        ];

        println!("Input Nodes:");
        for (i, node) in nodes.iter().enumerate() {
            println!("  Node {}: {:?}", i, node);
        }

        // 3. Run Shader
        println!("\nRunning PUCT Shader...");
        let results = accelerator.compute_puct_batch(&nodes).expect("Failed to compute PUCT batch");

        // 4. Inspect Results
        println!("\nShader Results:");
        for (i, result) in results.iter().enumerate() {
            println!("  Node {}:", i);
            println!("    PUCT Score: {:.6}", result.puct_score);
            println!("    Q Value:    {:.6}", result.q_value);
            println!("    Expl Term:  {:.6}", result.exploration_term);
            println!("    Original Idx: {}", result.node_index);
            
            // Add assertions here if you want to verify correctness automatically
            if i == 0 {
                assert_eq!(result.q_value, 0.0, "Unvisited node should have Q=0");
            }
        }
    }

    #[test]
    fn debug_simulation_shader() {
        use crate::gpu::GpuSimulationParams;

        // 1. Setup GPU Context
        let config = GpuConfig {
            max_batch_size: 1024,
            prefer_high_performance: true,
            min_batch_threshold: 0,
            debug_mode: true,
        };

        println!("Initializing GPU Context for Simulation...");
        let context = match GpuContext::new(&config) {
            Ok(ctx) => Arc::new(ctx),
            Err(e) => {
                eprintln!("Failed to initialize GPU: {}", e);
                return;
            }
        };

        let mut accelerator = GpuMctsAccelerator::new(context.clone());

        // 2. Create Test Data (Gomoku Board)
        let width = 15;
        let height = 15;
        let board_size = width * height;
        let mut board = vec![0; board_size];

        // Place a few pieces
        // Center piece for player 1
        board[7 * 15 + 7] = 1;
        // Adjacent piece for player -1
        board[7 * 15 + 8] = -1;

        let params = GpuSimulationParams {
            board_width: width as u32,
            board_height: height as u32,
            current_player: 1 | (5 << 8), // Player 1, line_size 5 (though Gomoku shader might ignore line_size if hardcoded)
            use_heuristic: 0,
            seed: 12345,
        };

        // 3. Run Simulation
        println!("\nRunning Simulation Shader...");
        let results = accelerator.simulate_batch(&board, params).expect("Failed to simulate batch");

        // 4. Inspect Results
        println!("\nSimulation Results:");
        for (i, score) in results.iter().enumerate() {
            println!("  Board {}: Score = {:.6}", i, score);
            // Score should be non-zero or at least valid
        }
    }
}
