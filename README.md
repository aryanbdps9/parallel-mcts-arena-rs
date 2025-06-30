# Parallel Multi-Game MCTS Engine and Arena written in Rust

A Rust program that plays board games using an AI.

It includes four games:
- Gomoku (or Five in a Row)
- Connect 4
- Othello (or Reversi)
- Blokus

The AI uses a massively parallel implementation of Monte Carlo Tree Search (MCTS) algorithm. It can use multiple CPU threads to think faster.

**Note**: This project was made with help from GitHub Copilot.

## Games You Can Play

### Gomoku
- **Goal**: Get 5 pieces in a row to win.
- **Board**: 15×15.
- **Players**: Human vs AI, or AI vs AI.

### Connect 4
- **Goal**: Get 4 pieces in a row to win.
- **Board**: 7×6.
- **Players**: Human vs AI, or AI vs AI.

### Othello
- **Goal**: Have the most pieces on the board when the game ends.
- **Board**: 8×8.
- **Players**: Human vs AI, or AI vs AI.

### Blokus
- **Goal**: Place as many of your pieces on the board as you can.
- **Board**: 20×20.
- **Players**: Up to 4 players (human or AI).

## Features

### Game Features
- **Four Games**: Play any of the four games with the same AI.
- **Highly Configurable**: Change game rules like board size and line size.
- **Human vs AI**: You can set any player to be a human or an AI.
- **Live Stats**: See what the AI is thinking in real-time.
- **Move History**: A list of all moves made in the game.

### AI Features
- **Multi-threaded**: Uses multiple CPU cores to make the AI think faster.
- **Smart Search**: The AI knows how to balance exploring new moves and using moves it knows are good.
- **Memory Pool**: Reuses memory to run faster.
- **AI Settings**: You can change how the AI works, like how long it can think.
- **Timeout Control**: Set maximum thinking time per move to ensure responsive gameplay.
- **AI-Only Mode**: Run AI vs AI games for analysis and testing. When enabled, skips player configuration and goes straight to the game.
- **Shared Tree**: Option to share search tree between moves for improved performance.
- **Real-time Statistics**: View AI thinking progress and statistics during gameplay.

### UI Features
- **Resizable Window**: The UI adapts to the terminal size.
- **Live AI Info**: See real-time data from the AI including move statistics and search depth.
- **Mouse and Keyboard**: Use your mouse or keyboard to play.
- **Debug View**: A special view to see details about the AI's search including top moves and win rates.
- **Move History**: Track all moves made in the game with scrollable history.
- **Game-Specific Cursors**: Each game has optimized cursor behavior (e.g., Connect4 cursor automatically drops to the lowest available position).
- **Scrollable Panels**: Use PageUp/PageDown to scroll through debug information and move history.

## How to Install and Run

### Requirements
- **Rust**: You can get it from [rustup.rs](https://rustup.rs/).
- **OS**: Should work on Windows, macOS, and Linux. Tested only on Windows.
- **Terminal**: A modern terminal that can show special characters. Tested on Windows Terminal, but should work on other terminals too.

### Steps

1.  **Get the code:**
    ```bash
    git clone https://github.com/aryanbdps9/parallel-mcts-arena-rs.git
    cd parallel-mcts-arena-rs
    ```

2.  **Build the program:**
    ```bash
    cargo build --release
    ```

3.  **Run the program:**
    ```bash
    cargo run --release
    ```

## How to Use

When you run the program, you will see a menu where you can:
1.  **Pick a game**: Choose from Gomoku, Connect 4, Othello, or Blokus.
2.  **Set players**: Choose if each player is a human or an AI (skipped in AI-only mode).
3.  **Change settings**: Change game rules or how the AI behaves.
4.  **Play**: Use your mouse or keyboard to make moves.

**Note**: When AI-only mode is enabled (via `--ai-only` flag or settings), the player configuration step is skipped and you'll go directly to watching the AI play.

### Command Line Options

You can also run the program with options from the command line.

| Option | Short | Default | Description |
|---|---|---|---|
| `--exploration-factor` | `-e` | 4.0 | The exploration factor for the MCTS algorithm. |
| `--search-iterations` | `-s` | 1000000 | The number of search iterations for the MCTS algorithm. |
| `--max-nodes` | `-m` | 1000000 | The maximum number of nodes in the MCTS search tree. |
| `--num-threads` | `-n` | 8 | Number of CPU threads for the AI to use. |
| `--board-size` | `-b` | 15 | Board size (15 for Gomoku, 7 for Connect4, 8 for Othello). |
| `--line-size` | `-l` | 5 | Number of pieces in a row to win (5 for Gomoku, 4 for Connect4). |
| `--timeout-secs` | | 60 | Maximum time AI can think per move (in seconds). |
| `--stats-interval-secs` | | 20 | How often to send statistics updates (in seconds). |
| `--ai-only` | | false | Whether this is an AI vs AI only game. |
| `--shared-tree` | | true | Whether to share the search tree between moves. |

### Keyboard Controls

#### In Game:
- **Arrow Keys**: Move the cursor around the board
- **Enter/Space**: Make a move at the current cursor position
- **PageUp/PageDown**: Scroll through debug statistics
- **Home/End**: Reset debug and history scroll positions
- **R**: Restart the current game
- **Esc**: Return to the main menu
- **Q**: Quit the application

#### In Menus:
- **Up/Down**: Navigate menu options
- **Left/Right**: Adjust settings values (in settings menu)
- **Enter**: Select/confirm choice
- **Esc**: Go back to previous menu
- **Q**: Quit the application

### Mouse Controls:
- **Click**: Place a move on the board (in-game)
- **Scroll**: Scroll through debug information (in-game)
