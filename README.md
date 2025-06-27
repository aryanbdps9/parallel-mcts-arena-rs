# Parallel Gomoku with MCTS in Rust

This project is a simple implementation of the game Gomoku (also known as Five in a Row) with an AI opponent powered by a parallelized Monte Carlo Tree Search (MCTS) algorithm. The parallelization strategy is inspired by the techniques used in Leela Chess Zero, allowing for efficient exploration of the game tree on multi-core processors.

## Features

- A command-line interface to play Gomoku against an AI.
- A configurable board size and line length to win.
- A parallelized MCTS engine for the AI, using `rayon` for data parallelism.
- Thread-safe tree nodes using `Arc` and `parking_lot::Mutex` for fine-grained locking.
- Enhanced debugging output with colored highlighting for top moves based on value, wins, and visits.
- Display of the MCTS root node value to gauge the AI's confidence in the current position.
- Row and column labels on the game board for easier move identification.

## How to Build and Run

1.  **Install Rust:** If you don't have Rust installed, get it from [rustup.rs](https://rustup.rs/).

2.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd parallel-gomoku-rs
    ```

3.  **Build the project:**
    ```sh
    cargo build --release
    ```

4.  **Run the game:**
    ```sh
    ./target/release/gomoku
    ```

    You can also run it in development mode:
    ```sh
    cargo run
    ```
    The output will now include colored grids highlighting the top moves, providing more insight into the AI's decision-making process.

### Command-line Options

-   `--board-size <SIZE>`: Sets the size of the board (default: 19).
-   `--line-size <SIZE>`: Sets the number of pieces in a row needed to win (default: 5).
-   `--num-threads <THREADS>`: Sets the number of threads for the MCTS search (default: 0, which lets `rayon` decide).

Example:
```sh
cargo run -- --board-size 15 --line-size 4 --num-threads 8
```

## Parallel MCTS Implementation

The MCTS algorithm is parallelized to speed up the search for the best move. Here's an overview of the approach:

-   **Parallel Simulations:** The main search function launches multiple MCTS simulations in parallel using the `rayon` crate. Each simulation consists of the selection, expansion, simulation, and backpropagation phases.

-   **Thread-Safe Tree:** The search tree is shared across all threads. To ensure thread safety, the nodes of the tree (`Node`) are wrapped in `Arc` (Atomic Reference Counting) for shared ownership. The internal data of each node (wins, visits, children) is protected by `parking_lot::Mutex` or atomic types for safe concurrent access.

-   **Fine-Grained Locking:** Instead of locking the entire tree, only the necessary parts of a node are locked during an update. This reduces contention and improves parallelism. For example, the `wins` and `children` of a node are in separate mutexes, and `visits` is an atomic integer.

-   **UCB1 Formula:** The selection phase uses the UCB1 formula to balance exploration (visiting less-explored nodes) and exploitation (focusing on promising nodes).

This parallel implementation allows the AI to perform a much deeper search in the same amount of time, leading to stronger gameplay.
