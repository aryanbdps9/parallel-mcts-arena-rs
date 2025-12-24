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

The UI logic is separated into game-specific modules in `src/tui/games/`.

1.  Create a new file `src/tui/games/mygame.rs`.
2.  Implement the symbol and style logic. For standard grid games, implement `get_cell_style`:
    ```rust
    use ratatui::style::{Color, Style};

    pub fn get_cell_style(cell: i32, is_cursor: bool) -> (&'static str, Style) {
        // Return symbol and style based on cell value
    }
    ```
3.  Register your module in `src/tui/games/mod.rs`:
    ```rust
    pub mod mygame;
    ```
4.  Open `src/tui/widgets.rs`:
    *   Import your module: `use crate::tui::games::mygame;`
    *   Update `draw_standard_board` to call your `get_cell_style` function in the match block.
    *   If your game needs custom grid configuration (e.g., different cell width), update the configuration match block in `draw_standard_board`.

If your game requires a completely custom UI (like Blokus):
1.  Implement your drawing functions in `src/tui/games/mygame.rs`.
2.  Update `draw_game_view` in `src/tui/widgets.rs` to dispatch to your custom view.

## 4. Input Handling

1.  Open `src/tui/mouse.rs`.
2.  Update `handle_mouse_event` to translate click coordinates into your game's move type.
3.  Wrap the move in `MoveWrapper::MyGame(move)`.

## 5. Application Entry Point

1.  Open `src/app.rs` to ensure your game is in the `games` list in `App::new` if it's not automatically picked up (currently hardcoded in `main.rs` or `app.rs`).

## 6. How To Play Documentation

When adding a new game, you should also add a "How to Play" help file that players can view by pressing `H` during gameplay.

1.  Create a new file `docs/how_to_play/mygame.txt` with the following sections:
    *   **Header**: Game title in a box
    *   **OBJECTIVE**: Brief description of how to win
    *   **GAMEPLAY**: How the game is played turn by turn
    *   **WINNING**: Win/lose/draw conditions
    *   **CONTROLS**: List of keyboard controls with descriptions
    *   **STRATEGY TIPS**: Optional helpful hints for players

2.  Open `src/components/ui/how_to_play.rs`:
    *   Add an include for your help file:
        ```rust
        const MYGAME_HELP: &str = include_str!("../../../docs/how_to_play/mygame.txt");
        ```
    *   Update `get_help_content()` to return your help text:
        ```rust
        GameWrapper::MyGame(_) => MYGAME_HELP,
        ```
    *   Update `get_game_name()` to return the display name:
        ```rust
        GameWrapper::MyGame(_) => "My Game",
        ```

See the existing help files (`gomoku.txt`, `connect4.txt`, `othello.txt`, `blokus.txt`) for examples of the formatting style.
