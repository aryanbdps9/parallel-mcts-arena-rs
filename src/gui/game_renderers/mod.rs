//! # Game Renderers Module
//!
//! This module contains the `GameRenderer` trait and implementations for each game.
//! Adding a new game to the GUI requires implementing this trait.
//!
//! ## Adding a New Game
//! 1. Create a new file in this directory (e.g., `my_game.rs`)
//! 2. Implement the `GameRenderer` trait
//! 3. Add the module to this file
//! 4. Register the renderer in `create_renderer_for_game()`

mod connect4;
mod gomoku;
mod hive;
mod othello;
mod blokus;
mod rotatable_board;

use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::gui::renderer::{Rect, Renderer};

pub use connect4::Connect4Renderer;
pub use gomoku::GomokuRenderer;
pub use hive::HiveRenderer;
pub use othello::OthelloRenderer;
pub use blokus::BlokusRenderer;
pub use rotatable_board::RotatableBoard;

/// Input event for game interaction
#[derive(Debug, Clone)]
pub enum GameInput {
    /// Mouse click at window coordinates
    Click { x: f32, y: f32 },
    /// Mouse move for hover effects
    Hover { x: f32, y: f32 },
    /// Keyboard input
    Key { code: u32, pressed: bool },
    /// Right mouse button down (start drag)
    RightDown { x: f32, y: f32 },
    /// Right mouse button up (end drag)
    RightUp { x: f32, y: f32 },
    /// Drag delta (for camera/tilt adjustment)
    Drag { dx: f32, dy: f32, shift: bool, ctrl: bool },
    /// Mouse wheel scroll
    Wheel { delta: f32, x: f32, y: f32, ctrl: bool },
}

/// Result of processing game input
#[derive(Debug, Clone)]
pub enum InputResult {
    /// No action needed
    None,
    /// A move was made
    Move(MoveWrapper),
    /// UI needs to be redrawn (e.g., hover changed)
    Redraw,
}

/// Trait for rendering game-specific UI
///
/// Implement this trait to add GUI support for a new game.
/// The trait provides methods for rendering and input handling.
///
/// # Example
/// ```rust,ignore
/// struct MyGameRenderer {
///     hover_cell: Option<(usize, usize)>,
/// }
///
/// impl GameRenderer for MyGameRenderer {
///     fn render(&self, renderer: &Renderer, game: &GameWrapper, area: Rect) {
///         // Draw the game board
///     }
///
///     fn handle_input(&mut self, input: GameInput, game: &GameWrapper, area: Rect) -> InputResult {
///         // Process mouse/keyboard input
///     }
///
///     fn game_name(&self) -> &'static str {
///         "My Game"
///     }
/// }
/// ```
pub trait GameRenderer: Send {
    /// Render the game board and pieces
    ///
    /// # Arguments
    /// * `renderer` - The Direct2D renderer
    /// * `game` - Current game state
    /// * `area` - Rectangle where the game should be rendered
    fn render(&self, renderer: &Renderer, game: &GameWrapper, area: Rect);

    /// Handle user input and return any resulting action
    ///
    /// # Arguments
    /// * `input` - The input event to process
    /// * `game` - Current game state (for validation)
    /// * `area` - Rectangle where the game is rendered (for hit testing)
    ///
    /// # Returns
    /// `InputResult` indicating what action should be taken
    fn handle_input(
        &mut self,
        input: GameInput,
        game: &GameWrapper,
        area: Rect,
    ) -> InputResult;

    /// Get the display name of the game
    fn game_name(&self) -> &'static str;

    /// Get a short description of the game
    fn game_description(&self) -> &'static str {
        ""
    }

    /// Get player name by ID
    fn player_name(&self, player_id: i32) -> String {
        match player_id {
            1 => "Player 1".to_string(),
            -1 | 2 => "Player 2".to_string(),
            3 => "Player 3".to_string(),
            4 => "Player 4".to_string(),
            _ => format!("Player {}", player_id),
        }
    }

    /// Reset renderer state (e.g., clear hover state)
    fn reset(&mut self) {}
}

/// Create the appropriate renderer for a game
pub fn create_renderer_for_game(game: &GameWrapper) -> Box<dyn GameRenderer> {
    match game {
        GameWrapper::Gomoku(_) => Box::new(GomokuRenderer::new()),
        GameWrapper::Connect4(_) => Box::new(Connect4Renderer::new()),
        GameWrapper::Othello(_) => Box::new(OthelloRenderer::new()),
        GameWrapper::Blokus(_) => Box::new(BlokusRenderer::new()),
        GameWrapper::Hive(_) => Box::new(HiveRenderer::new()),
    }
}

/// Common grid-based board rendering utilities
pub mod grid {
    use super::*;
    use crate::gui::colors::{Colors, player_color};

    /// Calculate the cell size and offset for a square grid board
    pub fn calculate_grid_layout(area: Rect, board_size: usize, padding: f32) -> GridLayout {
        // Guard against tiny areas (e.g. window resized very small). If
        // `min(width,height) < 2*padding`, the old math produced negative
        // `cell_size` and inflated offsets.
        let padded = area.inset(padding.max(0.0));
        let available_size = padded.width.min(padded.height).max(0.0);
        let cell_size = if board_size > 0 {
            available_size / board_size as f32
        } else {
            0.0
        };

        let board_width = cell_size * board_size as f32;
        let board_height = cell_size * board_size as f32;

        let offset_x = padded.x + (padded.width - board_width) / 2.0;
        let offset_y = padded.y + (padded.height - board_height) / 2.0;

        GridLayout {
            cell_size,
            offset_x,
            offset_y,
            board_size,
        }
    }

    /// Grid layout information
    #[derive(Debug, Clone, Copy)]
    pub struct GridLayout {
        pub cell_size: f32,
        pub offset_x: f32,
        pub offset_y: f32,
        pub board_size: usize,
    }

    impl GridLayout {
        /// Get the rectangle for a specific cell
        pub fn cell_rect(&self, row: usize, col: usize) -> Rect {
            Rect::new(
                self.offset_x + col as f32 * self.cell_size,
                self.offset_y + row as f32 * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
        }

        /// Get cell coordinates from screen position
        pub fn screen_to_cell(&self, x: f32, y: f32) -> Option<(usize, usize)> {
            if self.cell_size <= 0.0 {
                return None;
            }
            let col = ((x - self.offset_x) / self.cell_size).floor() as i32;
            let row = ((y - self.offset_y) / self.cell_size).floor() as i32;

            if row >= 0 && row < self.board_size as i32 && col >= 0 && col < self.board_size as i32 {
                Some((row as usize, col as usize))
            } else {
                None
            }
        }

        /// Get the board area as a Rect
        pub fn board_rect(&self) -> Rect {
            Rect::new(
                self.offset_x,
                self.offset_y,
                self.cell_size * self.board_size as f32,
                self.cell_size * self.board_size as f32,
            )
        }
    }

    /// Draw a grid of lines
    pub fn draw_grid(renderer: &Renderer, layout: &GridLayout, color: windows::Win32::Graphics::Direct2D::Common::D2D1_COLOR_F) {
        let board_rect = layout.board_rect();
        
        // Draw vertical lines
        for i in 0..=layout.board_size {
            let x = layout.offset_x + i as f32 * layout.cell_size;
            renderer.draw_line(x, board_rect.y, x, board_rect.y + board_rect.height, color, 1.0);
        }
        
        // Draw horizontal lines
        for i in 0..=layout.board_size {
            let y = layout.offset_y + i as f32 * layout.cell_size;
            renderer.draw_line(board_rect.x, y, board_rect.x + board_rect.width, y, color, 1.0);
        }
    }

    /// Draw a stone/piece at the specified cell
    pub fn draw_stone(
        renderer: &Renderer,
        layout: &GridLayout,
        row: usize,
        col: usize,
        player: i32,
        is_last_move: bool,
    ) {
        let cell = layout.cell_rect(row, col);
        let (cx, cy) = cell.center();
        let radius = layout.cell_size * 0.4;

        // Draw the stone
        renderer.fill_ellipse(cx, cy, radius, radius, player_color(player));
        
        // Draw outline for visibility
        let outline_color = if player == 1 { Colors::PLAYER_2 } else { Colors::PLAYER_1 };
        renderer.draw_ellipse(cx, cy, radius, radius, outline_color, 1.0);

        // Highlight last move
        if is_last_move {
            renderer.draw_ellipse(cx, cy, radius * 0.5, radius * 0.5, Colors::LAST_MOVE, 2.0);
        }
    }

    /// Draw hover indicator at a cell
    pub fn draw_hover(
        renderer: &Renderer,
        layout: &GridLayout,
        row: usize,
        col: usize,
    ) {
        let cell = layout.cell_rect(row, col);
        let (cx, cy) = cell.center();
        let radius = layout.cell_size * 0.4;
        
        renderer.fill_ellipse(cx, cy, radius, radius, Colors::HIGHLIGHT);
    }
}
