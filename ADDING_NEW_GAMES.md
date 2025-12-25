# Adding a New Game

This guide outlines the steps to add a new game to the Parallel MCTS Arena.

## 1. Core Implementation

1.  Create a new file in `src/games/` (e.g., `src/games/mygame.rs`).
2.  Define your game state struct and move struct.
3.  Implement the `GameState` trait from the `mcts` crate for your state struct.
4.  Implement `fmt::Display` for your state struct.
5.  Implement `Clone`, `Debug`, `PartialEq`, `Eq`, `Hash` for your move struct.
6.  Implement `get_line_size`, `get_last_move`, and `is_legal` methods for your state struct (as required by `GameWrapper`).

## 2. Registering the Game

1.  Add your module to `src/games/mod.rs`:
    ```rust
    pub mod mygame;
    ```

2.  Open `src/game_wrapper.rs`:
    *   Import your game state and move types.
    *   Add a variant to the `GameWrapper` enum:
        ```rust
        pub enum GameWrapper {
            // ...
            MyGame(MyGameState),
        }
        ```
    *   Add a variant to the `MoveWrapper` enum:
        ```rust
        pub enum MoveWrapper {
            // ...
            MyGame(MyGameMove),
        }
        ```
    *   Update `impl fmt::Display for MoveWrapper` to handle your new move type.
    *   Update the `impl_game_dispatch!` macro call at the bottom of the file to include your new game variant:
        ```rust
        impl_game_dispatch!(Gomoku, Connect4, Blokus, Othello, MyGame);
        ```

## 3. GUI Implementation

The GUI rendering logic is in `src/gui/game_renderers/`.

1.  Create a new file `src/gui/game_renderers/mygame.rs`.
2.  Implement the rendering functions for your game using Direct2D.
3.  Register your module in `src/gui/game_renderers/mod.rs`:
    ```rust
    pub mod mygame;
    ```
4.  Update `src/gui/renderer.rs` to handle rendering your game.

## 4. Input Handling

1.  Open `src/gui/window.rs`.
2.  Update the mouse click handling to translate click coordinates into your game's move type.
3.  Wrap the move in `MoveWrapper::MyGame(move)`.

## 5. How To Play Documentation

When adding a new game, you should also add a "How to Play" help file.

1.  Create a new file `docs/how_to_play/mygame.txt` with the following sections:
    *   **Header**: Game title in a box
    *   **OBJECTIVE**: Brief description of how to win
    *   **GAMEPLAY**: How the game is played turn by turn
    *   **WINNING**: Win/lose/draw conditions
    *   **CONTROLS**: List of keyboard controls with descriptions
    *   **STRATEGY TIPS**: Optional helpful hints for players

See the existing help files (`gomoku.txt`, `connect4.txt`, `othello.txt`, `blokus.txt`) for examples of the formatting style.
