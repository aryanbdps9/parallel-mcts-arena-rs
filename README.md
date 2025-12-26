# Parallel Multi-Game MCTS Arena

A multi-threaded Monte Carlo Tree Search engine that plays four classic board games with configurable AI opponents.

## Supported Games**Performance Tips**

**Optimal Settings**
- Use `--release` build for maximum performance (essential for competitive AI play)
- Set `--num-threads` to match your CPU core count (default: 8)
- Increase `--timeout-secs` for stronger play at the cost of speed
- Enable `--shared-tree` for consistent game analysis (enabled by default)

**Gomoku (Five in a Row)**
- Board: 15×15 grid (configurable)
- Goal: Get 5 pieces in a row (configurable)
- Players: 2

**Connect 4**
- Board: 7×6 grid (configurable)
- Goal: Get 4 pieces in a row (configurable)
- Players: 2

**Othello (Reversi)**
- Board: 8×8 grid (configurable)
- Goal: Have the most pieces when the board is full
- Players: 2

**Blokus**
- Board: 20×20 grid
- Goal: Place as many polyomino pieces as possible
- Players: 2-4

## Features

**AI Engine**
- Parallel Monte Carlo Tree Search (MCTS) algorithm
- Multi-threaded search with configurable thread count
- **GPU acceleration** for batch PUCT calculations (optional)
- Virtual losses to prevent thread collisions
- Memory-efficient node recycling
- Tree reuse between moves for improved performance
- Real-time search statistics

**Interface**
- Windows GUI with Direct2D rendering
- Mouse and keyboard support
- Live AI analysis and move history
- Game-specific controls and optimizations
- Debug mode with detailed search statistics

**Configuration**
- Adjustable board sizes and win conditions
- Configurable AI parameters (exploration factor, search time, thread count)
- Human vs AI or AI vs AI gameplay modes
- Command-line interface for automation

## Installation

**Prerequisites**
- Rust toolchain (install from [rustup.rs](https://rustup.rs/))
- Windows (for GUI support)

**Build and Run**
```bash
git clone https://github.com/aryanbdps9/parallel-mcts-arena-rs.git
cd parallel-mcts-arena-rs
cargo build --release --features gui
cargo run --release --features gui
```

**Build with GPU Acceleration**
```bash
# Enable GPU acceleration for faster PUCT calculations
cargo build --release --features "gui,gpu"
cargo run --release --features "gui,gpu"
```

The GPU feature uses WebGPU (wgpu) for cross-platform GPU compute and provides:
- Batch PUCT score calculation on the GPU
- Automatic fallback to CPU when GPU is unavailable
- Support for DirectX 12, Vulkan, Metal backends

**Alternative execution methods:**
```bash
# Run the binary directly (after building)
./target/release/play

# Explicitly specify the binary name
cargo run --release --features gui --bin play
```

Use `--release` for optimal performance.

## Usage

**Interactive Mode**
Launch the application and use the menu system:
1. Select a game from the main menu
2. Configure players (Human or AI)
3. Adjust settings (optional)
4. Start playing

**Command Line Mode**
```bash
# Start specific game with AI vs AI
cargo run --release --features gui -- --game Gomoku --ai-only

# Custom board size and AI settings
cargo run --release --features gui -- --game Gomoku --board-size 19 --exploration-factor 1.4 --num-threads 16

# Fast AI games for analysis
cargo run --release --features gui -- --ai-only --timeout-secs 10

# Using the binary directly (after building with cargo build --release --features gui)
./target/release/play --game Connect4 --ai-only --timeout-secs 5
```

## Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--exploration-factor` | `-e` | 4.0 | MCTS exploration vs exploitation balance |
| `--search-iterations` | `-s` | 1000000 | Maximum MCTS iterations per move |
| `--max-nodes` | `-m` | 1000000 | Maximum nodes in search tree |
| `--num-threads` | `-n` | 8 | Parallel search threads |
| `--timeout-secs` | | 60 | Maximum AI thinking time per move |
| `--game` | `-g` | None | Start with specific game |
| `--board-size` | `-b` | 15 | Board size (game-dependent) |
| `--line-size` | `-l` | 5 | Pieces needed to win (game-dependent) |
| `--ai-only` | | false | Skip player setup, run AI vs AI |
| `--shared-tree` | | true | Reuse search tree between moves |
| `--stats-interval-secs` | | 20 | Statistics update frequency |

## Controls

**Game Navigation**
- Arrow keys: Move cursor
- Enter/Space: Place piece
- Mouse click: Direct placement
- R: Restart game
- Esc: Return to menu
- Q: Quit application

**Blokus-Specific**
- R: Rotate selected piece
- P: Pass turn
- Number keys (1-9): Quick piece selection
- E: Expand all piece lists
- C: Collapse all piece lists

**Menu Navigation**
- Up/Down arrows: Navigate options
- Left/Right arrows: Adjust values
- Enter: Confirm selection
- Esc: Go back

## Technical Details

**Monte Carlo Tree Search (MCTS)**
- Builds a tree of possible game moves through simulation
- Uses PUCT (Predictor + Upper Confidence bounds applied to Trees) for node selection
- Balances exploration of new moves vs exploitation of good moves
- Parallel implementation with virtual losses to prevent thread conflicts

**Performance Optimizations**
- Multi-threaded search using Rayon thread pool
- Node recycling to reduce memory allocations
- Tree reuse between moves for consistent game analysis
- Lock-free statistics gathering where possible
- Thread-local buffers for move generation

**Architecture**
- GameState trait provides unified interface for all games
- Wrapper types allow generic MCTS engine to work with any game
- Windows GUI built with Direct2D
- Async communication between UI and AI threads

**Dependencies**
- Rust: Systems programming language with memory safety
- Windows API: Native GUI rendering with Direct2D
- Rayon: Data parallelism library
- Parking Lot: High-performance synchronization primitives
- Clap: Command line argument parsing
- Tokio: Asynchronous runtime
- Num CPUs: CPU information and control
- wgpu (optional): GPU compute for accelerated MCTS calculations

## Troubleshooting

**Performance Issues**
- Ensure you're using `--release` builds for production use
- Reduce `--num-threads` if experiencing system slowdowns
- Lower `--max-nodes` if running out of memory
- Disable debug features in release builds

**UI Issues**
- Ensure Windows is up to date for proper Direct2D support
- Try resizing window if layout appears broken
- Use Alt+F4 to force quit if application becomes unresponsive

**Installation Issues**
- Update Rust toolchain: `rustup update`
- Clear cargo cache: `cargo clean` then rebuild
- Ensure all dependencies are available for your platform

## Performance Tips

**Optimal Settings**
- Use `--release` build for maximum performance
- Set `--num-threads` to match your CPU core count
- Increase `--timeout-secs` for stronger play at the cost of speed
- Enable `--shared-tree` for consistent game analysis

**AI Tuning**
- Higher `--exploration-factor`: More exploration of new moves
- Lower `--exploration-factor`: More exploitation of proven moves
- More `--search-iterations`: Deeper analysis but slower moves
- More `--timeout-secs`: Longer thinking time (searches may end early)
- More `--max-nodes`: Better memory utilization for complex positions

## License

This project is open source. See the LICENSE file for details.
