# Parallel Multi-Game MCTS Arena

A multi-threaded Monte Carlo Tree Search engine that plays four classic board games with configurable AI opponents.

## Supported Games

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
- Virtual losses to prevent thread collisions
- Memory-efficient node recycling
- Tree reuse between moves for improved performance
- Real-time search statistics

**Interface**
- Terminal-based UI with mouse and keyboard support
- Resizable panels with drag-and-drop boundaries
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
- Modern terminal (Windows Terminal recommended)

**Build and Run**
```bash
git clone https://github.com/aryanbdps9/parallel-mcts-arena-rs.git
cd parallel-mcts-arena-rs
cargo build --release
cargo run --release
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
cargo run --release -- --game Gomoku --ai-only

# Custom board size and AI settings
cargo run --release -- --game Gomoku --board-size 19 --exploration-factor 1.4 --num-threads 16

# Fast AI games for analysis
cargo run --release -- --ai-only --timeout-secs 10
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

**Information Panels**
- Page Up/Down: Scroll statistics and move history
- Home/End: Jump to top/bottom of panels
- Mouse drag: Resize panel boundaries

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
- Terminal UI built with Ratatui and Crossterm
- Async communication between UI and AI threads

**Dependencies**
- Rust: Systems programming language with memory safety
- Ratatui: Terminal user interface framework
- Crossterm: Cross-platform terminal manipulation
- Rayon: Data parallelism library
- Parking Lot: High-performance synchronization primitives
- Clap: Command line argument parsing

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
- More `--max-nodes`: Better memory utilization for complex positions

## License

This project is open source. See the LICENSE file for details.
