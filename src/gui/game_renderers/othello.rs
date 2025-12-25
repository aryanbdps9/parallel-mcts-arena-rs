//! # Othello Game Renderer
//!
//! Renders the Othello (Reversi) game board with pieces and valid moves.

use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::games::othello::OthelloMove;
use crate::gui::colors::{Colors, player_color};
use crate::gui::renderer::{Rect, Renderer};
use mcts::GameState;
use super::{GameInput, GameRenderer, InputResult, grid};

/// Renderer for Othello game
pub struct OthelloRenderer {
    hover_cell: Option<(usize, usize)>,
}

impl OthelloRenderer {
    pub fn new() -> Self {
        Self { hover_cell: None }
    }
}

impl GameRenderer for OthelloRenderer {
    fn render(&self, renderer: &Renderer, game: &GameWrapper, area: Rect) {
        let GameWrapper::Othello(state) = game else { return };
        let board = state.get_board();
        let board_size = board.len();

        let layout = grid::calculate_grid_layout(area, board_size, 20.0);

        // Draw board background (green felt)
        let board_rect = layout.board_rect();
        let green_bg = windows::Win32::Graphics::Direct2D::Common::D2D1_COLOR_F {
            r: 0.76, g: 0.60, b: 0.42, a: 1.0,
        };
        renderer.fill_rect(board_rect.inset(-5.0), green_bg);

        // Draw grid
        let dark_green = windows::Win32::Graphics::Direct2D::Common::D2D1_COLOR_F {
            r: 0.0, g: 0.35, b: 0.15, a: 1.0,
        };
        grid::draw_grid(renderer, &layout, dark_green);

        // Get valid moves for current player
        let valid_moves: Vec<(usize, usize)> = game.get_possible_moves()
            .iter()
            .filter_map(|m| {
                if let MoveWrapper::Othello(OthelloMove(r, c)) = m {
                    Some((*r, *c))
                } else {
                    None
                }
            })
            .collect();

        // Get last move for highlighting
        let last_move = state.get_last_move();
        let last_move_coords: Vec<(usize, usize)> = last_move.unwrap_or_default();

        // Draw cells
        for row in 0..board_size {
            for col in 0..board_size {
                let cell = board[row][col];
                let cell_rect = layout.cell_rect(row, col);
                let (cx, cy) = cell_rect.center();
                let radius = layout.cell_size * 0.4;

                if cell != 0 {
                    // Draw piece
                    renderer.fill_ellipse(cx, cy, radius, radius, player_color(cell));
                    // Piece outline
                    let outline = if cell == 1 { Colors::PLAYER_2 } else { Colors::PLAYER_1 };
                    renderer.draw_ellipse(cx, cy, radius, radius, outline, 1.5);

                    // Highlight last move
                    if last_move_coords.contains(&(row, col)) {
                        renderer.draw_ellipse(cx, cy, radius * 0.5, radius * 0.5, Colors::LAST_MOVE, 2.0);
                    }
                } else if valid_moves.contains(&(row, col)) {
                    // Show valid move indicator
                    let indicator_color = windows::Win32::Graphics::Direct2D::Common::D2D1_COLOR_F {
                        r: 0.3, g: 0.3, b: 0.3, a: 0.3,
                    };
                    renderer.fill_ellipse(cx, cy, radius * 0.3, radius * 0.3, indicator_color);
                }
            }
        }

        // Draw hover indicator
        if let Some((row, col)) = self.hover_cell {
            if valid_moves.contains(&(row, col)) {
                let cell_rect = layout.cell_rect(row, col);
                let (cx, cy) = cell_rect.center();
                let radius = layout.cell_size * 0.4;
                renderer.fill_ellipse(cx, cy, radius, radius, Colors::HIGHLIGHT);
            }
        }
    }

    fn handle_input(
        &mut self,
        input: GameInput,
        game: &GameWrapper,
        area: Rect,
    ) -> InputResult {
        let GameWrapper::Othello(_state) = game else { return InputResult::None };
        let board = game.get_board();
        let board_size = board.len();
        let layout = grid::calculate_grid_layout(area, board_size, 20.0);

        match input {
            GameInput::Click { x, y } => {
                if let Some((row, col)) = layout.screen_to_cell(x, y) {
                    let mv = MoveWrapper::Othello(OthelloMove(row, col));
                    if game.is_legal(&mv) {
                        return InputResult::Move(mv);
                    }
                }
                InputResult::None
            }
            GameInput::Hover { x, y } => {
                let new_hover = layout.screen_to_cell(x, y);
                if new_hover != self.hover_cell {
                    self.hover_cell = new_hover;
                    return InputResult::Redraw;
                }
                InputResult::None
            }
            GameInput::Key { .. } => InputResult::None,
        }
    }

    fn game_name(&self) -> &'static str {
        "Othello"
    }

    fn game_description(&self) -> &'static str {
        "Flip opponent pieces by trapping them!"
    }

    fn player_name(&self, player_id: i32) -> String {
        match player_id {
            1 => "Black".to_string(),
            -1 => "White".to_string(),
            _ => format!("Player {}", player_id),
        }
    }

    fn reset(&mut self) {
        self.hover_cell = None;
    }
}
