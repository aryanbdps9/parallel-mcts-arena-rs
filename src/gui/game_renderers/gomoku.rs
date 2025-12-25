//! # Gomoku Game Renderer
//!
//! Renders the Gomoku (Five in a Row) game board with stones and grid.

use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::games::gomoku::GomokuMove;
use crate::gui::colors::Colors;
use crate::gui::renderer::{Rect, Renderer};
use mcts::GameState;
use super::{GameInput, GameRenderer, InputResult, grid, RotatableBoard};

/// Renderer for Gomoku game
pub struct GomokuRenderer {
    hover_cell: Option<(usize, usize)>,
    board_view: RotatableBoard,
}

impl GomokuRenderer {
    pub fn new() -> Self {
        Self {
            hover_cell: None,
            board_view: RotatableBoard::new(),
        }
    }
}

impl GameRenderer for GomokuRenderer {
    fn render(&self, renderer: &Renderer, game: &GameWrapper, area: Rect) {
        let GameWrapper::Gomoku(state) = game else { return };
        let board = state.get_board();
        let board_size = board.len();
        
        // Calculate grid layout
        let layout = grid::calculate_grid_layout(area, board_size, 20.0);
        let board_rect = layout.board_rect();
        let (center_x, center_y) = board_rect.center();
        
        // Apply board rotation/tilt transforms
        self.board_view.begin_draw(renderer, center_x, center_y);
        
        // Draw board background
        renderer.fill_rect(board_rect.inset(-5.0), Colors::BOARD_BG);
        
        // Draw grid lines
        grid::draw_grid(renderer, &layout, Colors::BOARD_GRID);
        
        // Draw star points (standard Gomoku board markers)
        if board_size >= 13 {
            let star_points = if board_size == 15 {
                vec![(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
            } else if board_size == 19 {
                vec![(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
            } else {
                vec![]
            };
            
            for (r, c) in star_points {
                let cell = layout.cell_rect(r, c);
                let (cx, cy) = cell.center();
                renderer.fill_ellipse(cx, cy, 4.0, 4.0, Colors::BOARD_GRID);
            }
        }
        
        // Get last move for highlighting
        let last_move = state.get_last_move();
        let last_move_coords: Vec<(usize, usize)> = last_move.unwrap_or_default();
        
        // Draw all stones
        for (row, board_row) in board.iter().enumerate() {
            for (col, &cell) in board_row.iter().enumerate() {
                if cell != 0 {
                    let is_last = last_move_coords.contains(&(row, col));
                    grid::draw_stone(renderer, &layout, row, col, cell, is_last);
                }
            }
        }
        
        // Draw hover indicator
        if let Some((row, col)) = self.hover_cell {
            if row < board_size && col < board_size && board[row][col] == 0 {
                grid::draw_hover(renderer, &layout, row, col);
            }
        }
        
        // End board transform
        self.board_view.end_draw(renderer);
    }

    fn handle_input(
        &mut self,
        input: GameInput,
        game: &GameWrapper,
        area: Rect,
    ) -> InputResult {
        let GameWrapper::Gomoku(state) = game else { return InputResult::None };
        let board = state.get_board();
        let board_size = board.len();
        let layout = grid::calculate_grid_layout(area, board_size, 20.0);
        let board_rect = layout.board_rect();
        let (center_x, center_y) = board_rect.center();

        // Handle board rotation/tilt drag
        if let Some(result) = self.board_view.handle_input(&input) {
            return result;
        }

        match input {
            GameInput::Click { x, y } => {
                // Transform screen coordinates to local board coordinates
                let (lx, ly) = self.board_view.screen_to_local(x, y, center_x, center_y);
                let board_x = center_x + lx;
                let board_y = center_y + ly;
                if let Some((row, col)) = layout.screen_to_cell(board_x, board_y) {
                    let mv = MoveWrapper::Gomoku(GomokuMove(row, col));
                    if game.is_legal(&mv) {
                        return InputResult::Move(mv);
                    }
                }
                InputResult::None
            }
            GameInput::Hover { x, y } => {
                // Transform screen coordinates to local board coordinates
                let (lx, ly) = self.board_view.screen_to_local(x, y, center_x, center_y);
                let board_x = center_x + lx;
                let board_y = center_y + ly;
                let new_hover = layout.screen_to_cell(board_x, board_y);
                if new_hover != self.hover_cell {
                    self.hover_cell = new_hover;
                    return InputResult::Redraw;
                }
                InputResult::None
            }
            GameInput::Key { .. } => InputResult::None,
            GameInput::Drag { .. } | GameInput::RightDown { .. } | GameInput::RightUp { .. } => InputResult::None,
        }
    }

    fn game_name(&self) -> &'static str {
        "Gomoku"
    }

    fn game_description(&self) -> &'static str {
        "Get five in a row to win!"
    }

    fn reset(&mut self) {
        self.hover_cell = None;
        self.board_view.reset_view();
    }
}
