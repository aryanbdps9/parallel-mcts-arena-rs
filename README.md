# Parallel Multi-Game MCTS Engine in Rust

A game engine that includes four classic board games: Gomoku (Five in a Row), Connect 4, Othello (Reversi), and Blokus. The AI uses Monte Carlo Tree Search (MCTS) that runs on multiple threads at the same time for better performance.

**Note**: This project was built with heavy use of GitHub Copilot for code help and generation.

## Supported Games

### Gomoku (Five in a Row)
- **Goal**: First player to get 5 pieces in a row wins
- **Default Board**: 19×19 (you can change this)
- **Players**: 2 (Human vs AI or AI vs AI)
- **Features**: Change board size and how many pieces you need to win

### Connect 4
- **Goal**: First player to get 4 pieces in a row wins
- **Board**: 7×6 (you can change width/height)
- **Players**: 2 (Human vs AI or AI vs AI)
- **How it works**: Pieces fall down due to gravity

### Othello (Reversi)
- **Goal**: Have the most pieces when the board is full
- **Board**: 8×8 (you can change this)
- **Players**: 2 (Human vs AI or AI vs AI)
- **How it works**: Flip opponent pieces by surrounding them

### Blokus
- **Goal**: Place as many pieces as possible on the board
- **Board**: 20×20 grid
- **Players**: Up to 4 (Human vs AI combinations)
- **Features**: 21 different shaped pieces with rotations and flips

## Key Features

### Games
- **Four different games**: All use the same AI engine
- **Terminal interface**: Play games in your terminal with mouse and keyboard
- **Human vs AI**: Choose which players are human or AI
- **Live stats**: See what the AI is thinking in real time
- **Move history**: See all moves played with timestamps

### AI Engine
- **Multi-threaded MCTS**: Uses multiple CPU cores for faster thinking
- **Thread-safe**: Multiple threads can work on the same search tree safely
- **Smart exploration**: AI balances trying new moves vs using good known moves
- **Memory management**: Reuses memory to avoid slowdowns
- **Configurable**: Change how the AI thinks (iterations, time limits, etc.)

### User Interface
- **Resizable panels**: Drag borders with your mouse to resize parts of the screen
- **Live AI info**: See AI confidence, search stats, and move values in real time
- **Mouse and keyboard**: Click on the board or use arrow keys
- **Debug view**: See detailed AI analysis and search tree info
- **Different UI for each game**: Each game has its own optimized interface

## How to Install and Run

### What You Need
- **Rust**: Download from [rustup.rs](https://rustup.rs/)
- **Any OS**: Works on Windows, macOS, or Linux
- **Good terminal**: A modern terminal that shows Unicode characters properly

### Steps to Build

1. **Get the code:**
   ```bash
   git clone <repository-url>
   cd parallel-gomoku-rs
   ```

2. **Build it:**
   ```bash
   cargo build --release
   ```

3. **Run it:**
   ```bash
   cargo run --release
   ```

## How to Use

### Menu System
The game has easy menus:
1. **Pick a game**: Choose Gomoku, Connect 4, Othello, or Blokus
2. **Set players**: Make each player human or AI
3. **Change settings**: Adjust game rules and AI behavior
4. **Play**: Use mouse or keyboard to play

### Command Line Options

| Option | Short | Default | What it does |
|--------|-------|---------|-------------|
| `--game` | `-g` | Interactive | Pick game type (gomoku, connect4, othello, blokus) |
| `--board-size` | `-b` | 9 | Board size (NxN for most games) |
| `--line-size` | `-l` | 4 | How many pieces in a row to win |
| `--num-threads` | `-n` | 8 | Number of CPU threads for AI |
| `--exploration-parameter` | `-e` | 4.0 | How much AI explores vs exploits |
| `--iterations` | `-i` | 1,000,000 | How many simulations AI runs |
| `--max-nodes` | `-m` | 1,000,000 | Memory limit for AI search tree |
| `--stats-interval-secs` | | 20 | How often to update stats |
| `--timeout-secs` | | 60 | Max time AI can think per move |
| `--ai-only` | | false | AI vs AI mode |
| `--shared-tree` | | true | Reuse search tree between moves |

### Examples

```bash
# Play Gomoku on 15x15 board, need 5 in a row, use 12 CPU threads
cargo run --release -- --game gomoku --board-size 15 --line-size 5 --num-threads 12

# Play Connect 4 with strong AI (2 million simulations, 2 minutes thinking time)
cargo run --release -- --game connect4 --iterations 2000000 --timeout-secs 120

# Watch AI vs AI play Othello
cargo run --release -- --ai-only --game othello --iterations 500000
```

## How the AI Works

The AI uses Monte Carlo Tree Search (MCTS) which works like this:
1. **Build a tree**: The AI builds a tree of possible game positions
2. **Multiple threads**: Several CPU threads work on the tree at the same time
3. **Smart selection**: Picks moves to explore based on how promising they look
4. **Random playouts**: Plays random games from each position to see who wins
5. **Learn and improve**: Updates the tree with what it learned

### Key AI Features
- **Parallel search**: Multiple threads explore different parts of the tree
- **Virtual losses**: Prevents threads from all exploring the same moves
- **Memory reuse**: Keeps useful parts of the tree between moves
- **Time limits**: Stops thinking when time runs out

## Controls

### Mouse
- **Left click**: Select moves, use menus, interact with the game
- **Right click**: Context actions (depends on the game)
- **Scroll wheel**: Scroll through move history and debug info
- **Drag borders**: Resize the different panels (board, stats, history)

### Keyboard
- **Arrow keys**: Move around menus and move cursor on board
- **Enter**: Make a move or select menu option
- **Escape**: Go back to previous menu or cancel
- **Space**: Toggle between Human/AI in setup
- **Tab**: Switch between different parts of the interface
- **1-4**: Quick game selection (1=Gomoku, 2=Connect4, 3=Othello, 4=Blokus)

### Blokus Special Controls
- **Piece selection**: Click on pieces or use number keys
- **R key**: Rotate piece
- **F key**: Flip piece
- **Hover over board**: See where piece will be placed

## Troubleshooting

### If the AI is too slow
- Use fewer `iterations` (try 100,000 instead of 1,000,000)
- Use fewer `max-nodes` (try 100,000 instead of 1,000,000)
- Increase `timeout-secs` to give the AI more time to think
- Always use `cargo run --release` for fast builds
- Try fewer threads if you have an older computer

### If the game uses too much memory
- Lower the `max-nodes` setting (try 50,000 or 100,000)
- Keep `shared-tree` enabled (this actually saves memory)
- Close other programs to free up memory

### If the display looks wrong
- Make sure your terminal supports Unicode characters
- Try a bigger terminal window (at least 80x24)
- Make sure your terminal supports colors
- Tested on Windows Terminal - try a different terminal if you have problems

### Debug Features
- Live AI statistics showing what the AI is thinking
- Move confidence indicators
- Search tree info
- Memory usage info

## Technical Architecture

### Core Components

#### **Game Engine** (`src/game_wrapper.rs`)
- Unified interface for all game types
- Move validation and game state management
- Cross-game compatibility layer

#### **MCTS Library** (`src/lib.rs`)
- Generic implementation supporting any game implementing `GameState` trait
- Thread-safe parallel search with configurable parameters
- Memory-efficient node management with recycling

#### **Terminal UI** (`src/tui.rs`)
- Event-driven interface using Ratatui framework
- Responsive layout with draggable panes
- Real-time statistics and analysis display

## Contributing

Want to help? Here's how:
1. Install Rust with `rustup`
2. Get the code and run `cargo build`
3. Run tests with `cargo test`
4. Format code with `cargo fmt`
5. Check for issues with `cargo clippy`

### Code Layout
- Game logic in `src/games/`
- User interface in `src/tui.rs`
- AI engine in `src/lib.rs`
- Game integration in `src/game_wrapper.rs`
