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

    #[test]
    fn debug_blokus_shader() {
        use crate::gpu::GpuSimulationParams;

        let config = GpuConfig {
            max_batch_size: 1024,
            prefer_high_performance: true,
            min_batch_threshold: 0,
            debug_mode: true,
        };

        println!("Initializing GPU Context for Blokus...");
        let context = match GpuContext::new(&config) {
            Ok(ctx) => Arc::new(ctx),
            Err(e) => {
                eprintln!("Failed to initialize GPU: {}", e);
                return;
            }
        };

        let mut accelerator = GpuMctsAccelerator::new(context.clone());

        // Create Blokus Board (20x20 + extra row)
        let width = 20;
        let height = 21; // 20 board + 1 state
        let board_size = width * height;
        let mut board = vec![0; board_size];

        // Set some pieces
        // Player 1 at (0,0)
        board[0] = 1;
        
        // Set state row (index 400+)
        // P1 pieces (all available)
        board[400] = -1; // All bits set
        board[401] = -1; // P2
        board[402] = -1; // P3
        board[403] = -1; // P4
        
        // First move flags (all true)
        board[404] = 15; // 1111 binary

        let params = GpuSimulationParams {
            board_width: width as u32,
            board_height: height as u32,
            current_player: 1 | (3 << 16), // Player 1, Game Type 3 (Blokus)
            use_heuristic: 0,
            seed: 12345,
        };

        println!("\nRunning Blokus Shader...");
        let results = accelerator.simulate_batch(&board, params).expect("Failed to simulate Blokus");

        println!("\nBlokus Results:");
        for (i, score) in results.iter().enumerate() {
            println!("  Board {}: Score = {:.6}", i, score);
        }
    }

    #[test]
    fn debug_hive_shader() {
        use crate::gpu::GpuSimulationParams;

        let config = GpuConfig {
            max_batch_size: 1024,
            prefer_high_performance: true,
            min_batch_threshold: 0,
            debug_mode: true,
        };

        println!("Initializing GPU Context for Hive...");
        let context = match GpuContext::new(&config) {
            Ok(ctx) => Arc::new(ctx),
            Err(e) => {
                eprintln!("Failed to initialize GPU: {}", e);
                return;
            }
        };

        let mut accelerator = GpuMctsAccelerator::new(context.clone());

        // Create Hive Board (32x32 + extra row)
        let width = 32;
        let height = 33; // 32 board + 1 state
        let board_size = width * height;
        let mut board = vec![0; board_size];

        // Place some pieces
        // Center is (16, 16). Index = 16*32 + 16 = 528
        // Encode: count=1, player=1, type=Queen(0) -> (1<<16) | (1<<8) | 0 = 65536 + 256 = 65792
        board[528] = 65792;

        // Set state row (index 1024+)
        // P1 Queen placed
        board[1024 + 13] = 1; // P1 Queen placed
        board[1024 + 15] = 16; // P1 Queen Q
        board[1024 + 16] = 16; // P1 Queen R
        
        board[1024 + 11] = 1; // P1 pieces placed count

        let params = GpuSimulationParams {
            board_width: width as u32,
            board_height: height as u32,
            current_player: 2 | (4 << 16), // Player 2, Game Type 4 (Hive)
            use_heuristic: 0,
            seed: 12345,
        };

        println!("\nRunning Hive Shader...");
        let results = accelerator.simulate_batch(&board, params).expect("Failed to simulate Hive");

        println!("\nHive Results:");
        for (i, score) in results.iter().enumerate() {
            println!("  Board {}: Score = {:.6}", i, score);
        }
    }
}
