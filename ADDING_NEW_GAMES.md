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

## 3. UI Implementation

1.  Open `src/tui/widgets.rs`.
2.  Locate `draw_board`.
3.  If your game fits the standard grid layout:
    *   Update `draw_standard_board` to handle your game's specific symbols and colors.
    *   You can customize the `GenericGridConfig` (cell width, labels, etc.) in the match block.
    *   Add your symbol rendering logic to the closure passed to `GenericGrid::new`.
4.  If your game requires a custom UI (like Blokus):
    *   Create a new drawing function (e.g., `draw_mygame_board`).
    *   Add a match arm in `draw_board` to call your function.

## 4. Input Handling

1.  Open `src/tui/mouse.rs`.
2.  Update `handle_mouse_event` to translate click coordinates into your game's move type.
3.  Wrap the move in `MoveWrapper::MyGame(move)`.

## 5. Application Entry Point

1.  Open `src/main.rs` or `src/app.rs` to ensure your game can be selected and initialized.
