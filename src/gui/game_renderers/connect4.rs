//! # Connect4 Game Renderer
//!
//! Renders the Connect4 game board with pieces falling under gravity.

use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::games::connect4::Connect4Move;
use crate::gui::colors::{Colors, player_color};
use crate::gui::renderer::{Rect, Renderer};
use mcts::GameState;
use super::{GameInput, GameRenderer, InputResult, RotatableBoard};

/// Renderer for Connect4 game
pub struct Connect4Renderer {
    hover_column: Option<usize>,
    board_view: RotatableBoard,
}

impl Connect4Renderer {
    pub fn new() -> Self {
        Self {
            hover_column: None,
            board_view: RotatableBoard::new(),
        }
    }

    fn calculate_layout(&self, area: Rect, rows: usize, cols: usize) -> Connect4Layout {
        let padding = 20.0;
        let available_width = area.width - padding * 2.0;
        let available_height = area.height - padding * 2.0;

        // Calculate cell size to fit both dimensions
        let cell_width = available_width / cols as f32;
        let cell_height = available_height / rows as f32;
        let cell_size = cell_width.min(cell_height);

        let board_width = cell_size * cols as f32;
        let board_height = cell_size * rows as f32;

        Connect4Layout {
            cell_size,
            offset_x: area.x + (area.width - board_width) / 2.0,
            offset_y: area.y + (area.height - board_height) / 2.0,
            rows,
            cols,
        }
    }
}

struct Connect4Layout {
    cell_size: f32,
    offset_x: f32,
    offset_y: f32,
    rows: usize,
    cols: usize,
}

impl Connect4Layout {
    fn cell_rect(&self, row: usize, col: usize) -> Rect {
        Rect::new(
            self.offset_x + col as f32 * self.cell_size,
            self.offset_y + row as f32 * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
    }

    fn board_rect(&self) -> Rect {
        Rect::new(
            self.offset_x,
            self.offset_y,
            self.cell_size * self.cols as f32,
            self.cell_size * self.rows as f32,
        )
    }

    fn screen_to_column(&self, x: f32, _y: f32) -> Option<usize> {
        let col = ((x - self.offset_x) / self.cell_size).floor() as i32;
        if col >= 0 && col < self.cols as i32 {
            Some(col as usize)
        } else {
            None
        }
    }
}

impl GameRenderer for Connect4Renderer {
    fn render(&self, renderer: &Renderer, game: &GameWrapper, area: Rect) {
        let GameWrapper::Connect4(state) = game else { return };
        let board = state.get_board();
        let rows = board.len();
        let cols = if rows > 0 { board[0].len() } else { 7 };

        let layout = self.calculate_layout(area, rows, cols);
        let board_rect = layout.board_rect();
        let (center_x, center_y) = board_rect.center();

        // Apply board rotation/tilt transforms
        self.board_view.begin_draw(renderer, center_x, center_y);

        // Draw board background (blue frame)
        renderer.fill_rounded_rect(board_rect.inset(-10.0), 10.0, Colors::BUTTON_SELECTED);

        // Draw cells with holes
        for row in 0..rows {
            for col in 0..cols {
                let cell = layout.cell_rect(row, col);
                let (cx, cy) = cell.center();
                let radius = layout.cell_size * 0.4;

                let cell_value = board[row][col];
                if cell_value != 0 {
                    // Draw piece
                    renderer.fill_ellipse(cx, cy, radius, radius, player_color(cell_value));
                    // Piece outline
                    let outline = if cell_value == 1 { Colors::PLAYER_2 } else { Colors::PLAYER_1 };
                    renderer.draw_ellipse(cx, cy, radius, radius, outline, 2.0);
                } else {
                    // Draw empty hole
                    renderer.fill_ellipse(cx, cy, radius, radius, Colors::BACKGROUND);
                }

                // Draw coordinates
                let font_size = layout.cell_size / 8.0;
                let coord_text = format!("{},{}", row, col);
                let text_color = if cell_value == 2 { Colors::PLAYER_1 } else { Colors::TEXT_PRIMARY };
                
                renderer.draw_text_with_size(
                    &coord_text,
                    cell,
                    text_color,
                    font_size,
                    true
                );
            }
        }

        // Get last move for highlighting
        if let Some(last_moves) = state.get_last_move() {
            for (r, c) in last_moves {
                let cell = layout.cell_rect(r, c);
                let (cx, cy) = cell.center();
                let radius = layout.cell_size * 0.4;
                renderer.draw_ellipse(cx, cy, radius * 0.5, radius * 0.5, Colors::LAST_MOVE, 3.0);
            }
        }

        // Draw hover preview at top
        if let Some(col) = self.hover_column {
            // Find the row where the piece would land
            let mut landing_row = None;
            for row in (0..rows).rev() {
                if board[row][col] == 0 {
                    landing_row = Some(row);
                    break;
                }
            }

            if let Some(row) = landing_row {
                let cell = layout.cell_rect(row, col);
                let (cx, cy) = cell.center();
                let radius = layout.cell_size * 0.4;
                renderer.fill_ellipse(cx, cy, radius, radius, Colors::HIGHLIGHT);
            }
        }

        // End board transform
        self.board_view.end_draw(renderer);

        // Draw Reset Zoom button if zoomed
        if (self.board_view.scale() - 1.0).abs() > 0.01 {
            let reset_rect = Rect::new(area.x + area.width - 110.0, area.y + 10.0, 100.0, 30.0);
            renderer.fill_rounded_rect(reset_rect, 4.0, Colors::BUTTON_BG);
            renderer.draw_text("Reset Zoom", reset_rect, Colors::TEXT_PRIMARY, true);
        }
    }

    fn handle_input(
        &mut self,
        input: GameInput,
        game: &GameWrapper,
        area: Rect,
    ) -> InputResult {
        let GameWrapper::Connect4(state) = game else { return InputResult::None };
        let board = state.get_board();
        let rows = board.len();
        let cols = if rows > 0 { board[0].len() } else { 7 };
        let layout = self.calculate_layout(area, rows, cols);
        let board_rect = layout.board_rect();
        let (center_x, center_y) = board_rect.center();

        // Handle board rotation/tilt drag
        if let Some(result) = self.board_view.handle_input(&input, center_x, center_y) {
            return result;
        }

        match input {
            GameInput::Click { x, y } => {
                // Check Reset Zoom button
                if (self.board_view.scale() - 1.0).abs() > 0.01 {
                    let reset_rect = Rect::new(area.x + area.width - 110.0, area.y + 10.0, 100.0, 30.0);
                    if x >= reset_rect.x && x <= reset_rect.x + reset_rect.width &&
                       y >= reset_rect.y && y <= reset_rect.y + reset_rect.height {
                        self.board_view.reset_zoom();
                        return InputResult::Redraw;
                    }
                }

                // Transform screen coordinates to local board coordinates
                let (lx, ly) = self.board_view.screen_to_local(x, y, center_x, center_y);
                // Add center back to get board-space coordinates
                let board_x = center_x + lx;
                let board_y = center_y + ly;
                if let Some(col) = layout.screen_to_column(board_x, board_y) {
                    // Check if column has space
                    if board[0][col] == 0 {
                        let mv = MoveWrapper::Connect4(Connect4Move(col));
                        if game.is_legal(&mv) {
                            return InputResult::Move(mv);
                        }
                    }
                }
                InputResult::None
            }
            GameInput::Hover { x, y } => {
                // Transform screen coordinates to local board coordinates
                let (lx, ly) = self.board_view.screen_to_local(x, y, center_x, center_y);
                let board_x = center_x + lx;
                let board_y = center_y + ly;
                let new_hover = layout.screen_to_column(board_x, board_y).filter(|&col| board[0][col] == 0);
                if new_hover != self.hover_column {
                    self.hover_column = new_hover;
                    return InputResult::Redraw;
                }
                InputResult::None
            }
            GameInput::Key { .. } => InputResult::None,
            GameInput::Drag { .. } | GameInput::RightDown { .. } | GameInput::RightUp { .. } | GameInput::Wheel { .. } => InputResult::None,
        }
    }

    fn game_name(&self) -> &'static str {
        "Connect 4"
    }

    fn game_description(&self) -> &'static str {
        "Drop pieces to connect four in a row!"
    }

    fn player_name(&self, player_id: i32) -> String {
        match player_id {
            1 => "Red".to_string(),
            -1 => "Yellow".to_string(),
            _ => format!("Player {}", player_id),
        }
    }

    fn reset(&mut self) {
        self.hover_column = None;
        self.board_view.reset_view();
    }
}
