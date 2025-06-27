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
    ./target/release/parallel-gomoku-rs
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
-   `--exploration-parameter <PARAM>`: Sets the exploration parameter for MCTS (default: 4.0).
-   `--iterations <COUNT>`: Sets the number of MCTS iterations per move (default: 1000000).
-   `--max-nodes <NODES>`: Sets the maximum number of nodes allowed in the search tree (default: 100000).

Example:
```sh
cargo run -- --board-size 15 --line-size 4 --num-threads 8 --exploration-parameter 2.0 --iterations 500000 --max-nodes 50000
```

## Parallel MCTS Implementation

The MCTS algorithm is parallelized to speed up the search for the best move. Here's an overview of the approach:

-   **Parallel Simulations:** The main search function launches multiple MCTS simulations in parallel using the `rayon` crate. Each simulation consists of the selection, expansion, simulation, and backpropagation phases.

-   **Thread-Safe Tree with `RwLock`:** The search tree is shared across all threads. To ensure thread safety, the nodes of the tree (`Node`) are wrapped in `Arc` (Atomic Reference Counting) for shared ownership. The `children` map of each node is protected by a `parking_lot::RwLock`, which allows multiple threads to simultaneously read the children (during the selection phase) while ensuring exclusive access for modifications (during the expansion phase). This significantly reduces lock contention and improves the throughput of parallel simulations, leading to much higher CPU utilization and a stronger AI.

-   **Virtual Loss System:** To reduce thread contention and prevent multiple threads from exploring the same path simultaneously, the implementation uses virtual losses. When a thread selects a node for exploration, it applies a virtual loss to make that path less attractive to other threads until the simulation completes.

-   **PUCT Formula:** The selection phase uses the PUCT (Predictor + UCB applied to Trees) formula, which is an enhanced version of UCB1 that includes prior probabilities and handles virtual losses. This provides better balance between exploration and exploitation.

-   **Memory Management:** The implementation features sophisticated memory management with node recycling and automatic tree pruning to maintain efficiency during long games. Weak children are periodically pruned based on visit statistics to prevent memory bloat.

-   **Adaptive Expansion:** Node expansion is controlled by tree size limits, depth-based probability, and visit counts to ensure efficient memory usage while maintaining search quality.

This parallel implementation allows the AI to perform a much deeper search in the same amount of time, leading to stronger gameplay and the ability to anticipate threats several moves in advance.
