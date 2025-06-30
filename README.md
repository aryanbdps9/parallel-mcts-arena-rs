# 🎮 Parallel Multi-Game MCTS Arena

**A smart AI that can play 4 different board games!**

This program uses a powerful AI to play board games. The AI can think very fast because it uses many CPU cores at the same time.

## 🎯 Games You Can Play

**Four classic games are available:**
- 🔵 **Gomoku** (Five in a Row)
- 🔴 **Connect 4** 
- ⚫ **Othello** (Reversi)
- 🟦 **Blokus**

The AI uses Monte Carlo Tree Search (MCTS) algorithm to find the best moves. It runs on multiple threads to think faster and play better.

---

*This project was created with help from GitHub Copilot.*

## 📋 Game Rules

### 🔵 Gomoku (Five in a Row)
- **Goal**: Get 5 pieces in a row to win
- **Board**: 15×15 grid
- **Players**: 2 players (you vs AI, or AI vs AI)

### 🔴 Connect 4
- **Goal**: Get 4 pieces in a row to win  
- **Board**: 7×6 grid (pieces drop down due to gravity)
- **Players**: 2 players (you vs AI, or AI vs AI)

### ⚫ Othello (Reversi)
- **Goal**: Have the most pieces when the board is full
- **Board**: 8×8 grid
- **Players**: 2 players (you vs AI, or AI vs AI)

### 🟦 Blokus
- **Goal**: Place as many of your pieces as possible
- **Board**: 20×20 grid
- **Players**: 2, 3, or 4 players (humans or AI)
- **Special**: Uses colorful puzzle pieces (polyominoes)

## ✨ What Makes This Special

### 🎮 Game Features
- **4 Different Games**: All games use the same smart AI
- **Easy to Change**: Adjust board size, winning conditions, and more
- **Human vs AI**: Play against the computer or watch AI vs AI
- **Live Information**: See what the AI is thinking in real-time
- **Move History**: Review all moves made during the game

### 🧠 AI Features
- **Very Fast**: Uses multiple CPU cores to think quickly
- **Smart Decisions**: Balances trying new moves vs. using good known moves
- **Memory Efficient**: Reuses memory to run faster
- **Adjustable**: Change how long the AI thinks and how it behaves
- **AI-Only Mode**: Watch AI play against itself for learning
- **Shared Memory**: AI remembers between moves for better play
- **Live Stats**: See AI progress and statistics while it thinks

### 💻 Interface Features
- **Works Everywhere**: Adapts to any terminal size
- **Mouse + Keyboard**: Use whichever you prefer
- **Real-time Info**: See AI statistics and move analysis
- **Debug Mode**: Detailed view of AI decision-making process
- **Scrollable**: Navigate through long information lists
- **Smart Cursors**: Each game has optimized controls (Connect4 drops to bottom automatically)
- **Easy Navigation**: Use PageUp/PageDown to scroll through information

## 🚀 How to Install and Run

### What You Need
- **Rust Programming Language**: Get it from [rustup.rs](https://rustup.rs/)
- **Operating System**: Windows, macOS, or Linux (tested on Windows)
- **Terminal**: Any modern terminal (Windows Terminal recommended)

### 📦 Installation Steps

1. **Download the code:**
   ```bash
   git clone https://github.com/aryanbdps9/parallel-mcts-arena-rs.git
   cd parallel-mcts-arena-rs
   ```

2. **Build the program:**
   ```bash
   cargo build --release
   ```

3. **Run the program:**
   ```bash
   cargo run --release
   ```

*Note: Use `--release` for the best performance!*

## 🎮 How to Play

### Getting Started
When you start the program, you will see a simple menu:

1. **Choose a Game**: Pick from Gomoku, Connect 4, Othello, or Blokus
2. **Set Up Players**: Choose if each player is a human or AI
3. **Adjust Settings**: Change game rules or AI behavior (optional)
4. **Play**: Use mouse clicks or keyboard to make your moves

*Note: If you use `--ai-only` mode, you skip step 2 and watch AI vs AI immediately.*

### 🔧 Command Line Options

You can customize the AI and game settings:

| Option | Short | Default | What it does |
|--------|--------|---------|--------------|
| `--exploration-factor` | `-e` | 4.0 | How much AI explores new moves |
| `--search-iterations` | `-s` | 1000000 | How many moves AI considers |
| `--max-nodes` | `-m` | 1000000 | Maximum AI memory usage |
| `--num-threads` | `-n` | 8 | CPU cores AI can use |
| `--game` | `-g` | None | Start with specific game |
| `--board-size` | `-b` | 15 | Board size (15 for Gomoku, 8 for Othello) |
| `--line-size` | `-l` | 5 | Pieces needed to win (5 for Gomoku, 4 for Connect4) |
| `--timeout-secs` | | 60 | Max seconds AI can think per move |
| `--stats-interval-secs` | | 20 | How often to show AI progress |
| `--ai-only` | | false | Skip human setup, watch AI vs AI |
| `--shared-tree` | | true | AI remembers between moves |

**Example:** Start Gomoku AI vs AI with faster thinking:
```bash
cargo run --release -- --game Gomoku --ai-only --num-threads 16 --timeout-secs 30
```

## ⌨️ Controls

### 🕹️ While Playing Games

**Movement:**
- **Arrow Keys** ← ↑ → ↓: Move cursor around the board
- **Enter** or **Space**: Place your piece at cursor position
- **Mouse Click**: Click anywhere on board to place piece

**Game Controls:**
- **R**: Restart current game
- **Esc**: Return to main menu
- **Q**: Quit the program

**Information:**
- **PageUp/PageDown**: Scroll through AI statistics and debug info
- **Home/End**: Go to top/bottom of information panels

### 🟦 Special Blokus Controls
*When playing Blokus, you get extra controls for pieces:*

- **R**: Rotate selected piece
- **P**: Pass your turn (when you can't place any pieces)
- **Number Keys 1-9**: Select different pieces quickly
- **E**: Expand all player piece lists
- **C**: Collapse all player piece lists

### 📋 Menu Navigation

- **Up/Down Arrow Keys**: Move through menu options
- **Left/Right Arrow Keys**: Change setting values (in settings menu)
- **Enter**: Select/confirm your choice
- **Esc**: Go back to previous menu
- **Q**: Quit program

### 🖱️ Mouse Controls

- **Left Click**: Place move on game board
- **Right Click**: Special actions (varies by game)
- **Scroll Wheel**: Scroll through information panels
- **Drag**: Resize panels (grab the borders between sections)

## 💡 Tips for Better Play

### Getting the Best AI Performance
- Use `--release` build for maximum speed
- Increase `--num-threads` to match your CPU cores
- Adjust `--timeout-secs` based on how long you want to wait
- Use `--ai-only` mode to study AI strategies

### Understanding the AI
- **Higher exploration factor** = AI tries more new moves
- **More search iterations** = AI thinks deeper but slower
- **Shared tree mode** = AI learns from previous moves in the game

## 🔧 Technical Details

### What is MCTS?
Monte Carlo Tree Search (MCTS) is a smart algorithm that:
1. **Builds a tree** of possible game moves
2. **Simulates** thousands of random games
3. **Learns** which moves lead to wins
4. **Chooses** the move with the highest win rate

### Why is it Fast?
- **Parallel Processing**: Uses multiple CPU cores simultaneously
- **Smart Memory**: Reuses calculations and memory efficiently  
- **Virtual Losses**: Prevents multiple threads from exploring the same moves
- **Tree Pruning**: Removes unnecessary branches to save memory

### Built With
- **Rust**: Fast, safe systems programming language
- **Ratatui**: Terminal user interface framework
- **Rayon**: Data parallelism library
- **Crossterm**: Cross-platform terminal manipulation

---

## 📄 License

This project is open source. Check the LICENSE file for details.

## 🤝 Contributing

Found a bug or want to add a feature? Feel free to open an issue or submit a pull request!

---

**Happy Learning! 🎮✨**
