//! # Blokus Game Renderer
//!
//! Renders the Blokus game board with polyomino pieces for 4 players.

use crate::game_wrapper::GameWrapper;
use crate::gui::colors::Colors;
use crate::gui::renderer::{Rect, Renderer};
use mcts::GameState;
use super::{GameInput, GameRenderer, InputResult, grid};
use windows::Win32::Graphics::Direct2D::Common::D2D1_COLOR_F;

/// Colors for the 4 Blokus players
const BLOKUS_COLORS: [D2D1_COLOR_F; 4] = [
    D2D1_COLOR_F { r: 0.2, g: 0.4, b: 0.9, a: 1.0 },  // Blue (Player 1)
    D2D1_COLOR_F { r: 0.9, g: 0.8, b: 0.2, a: 1.0 },  // Yellow (Player 2)
    D2D1_COLOR_F { r: 0.9, g: 0.2, b: 0.2, a: 1.0 },  // Red (Player 3)
    D2D1_COLOR_F { r: 0.2, g: 0.8, b: 0.3, a: 1.0 },  // Green (Player 4)
];

fn player_blokus_color(player: i32) -> D2D1_COLOR_F {
    if player >= 1 && player <= 4 {
        BLOKUS_COLORS[(player - 1) as usize]
    } else {
        Colors::TEXT_SECONDARY
    }
}

/// Renderer for Blokus game
pub struct BlokusRenderer {
    hover_cell: Option<(usize, usize)>,
    // Blokus has complex piece selection, simplified for initial implementation
    selected_piece: Option<usize>,
    selected_transform: usize,
}

impl BlokusRenderer {
    pub fn new() -> Self {
        Self {
            hover_cell: None,
            selected_piece: None,
            selected_transform: 0,
        }
    }
}

impl GameRenderer for BlokusRenderer {
    fn render(&self, renderer: &Renderer, game: &GameWrapper, area: Rect) {
        let GameWrapper::Blokus(state) = game else { return };
        let board = state.get_board();
        let board_size = board.len();

        // Layout: board on left, piece selector on right
        let board_area = Rect::new(area.x, area.y, area.height, area.height);
        
        let layout = grid::calculate_grid_layout(board_area, board_size, 10.0);

        // Draw board background
        let board_rect = layout.board_rect();
        renderer.fill_rect(board_rect.inset(-3.0), Colors::PANEL_BG);

        // Draw grid with thin lines
        let light_grid = D2D1_COLOR_F { r: 0.4, g: 0.4, b: 0.45, a: 0.5 };
        
        // Draw vertical lines
        for i in 0..=board_size {
            let x = layout.offset_x + i as f32 * layout.cell_size;
            renderer.draw_line(x, board_rect.y, x, board_rect.y + board_rect.height, light_grid, 0.5);
        }
        // Draw horizontal lines
        for i in 0..=board_size {
            let y = layout.offset_y + i as f32 * layout.cell_size;
            renderer.draw_line(board_rect.x, y, board_rect.x + board_rect.width, y, light_grid, 0.5);
        }

        // Draw starting corners with their player colors
        let corner_positions = [
            (0, 0),                     // Player 1 (Blue) - top-left
            (0, board_size - 1),        // Player 2 (Yellow) - top-right
            (board_size - 1, board_size - 1), // Player 3 (Red) - bottom-right
            (board_size - 1, 0),        // Player 4 (Green) - bottom-left
        ];

        for (player_idx, &(r, c)) in corner_positions.iter().enumerate() {
            if board[r][c] == 0 {
                let cell = layout.cell_rect(r, c);
                let color = BLOKUS_COLORS[player_idx];
                let faded = D2D1_COLOR_F { a: 0.3, ..color };
                renderer.fill_rect(cell, faded);
            }
        }

        // Draw placed pieces
        for row in 0..board_size {
            for col in 0..board_size {
                let cell_value = board[row][col];
                if cell_value != 0 {
                    let cell = layout.cell_rect(row, col);
                    let color = player_blokus_color(cell_value);
                    renderer.fill_rect(cell.inset(0.5), color);
                    
                    // Add subtle border
                    let border_color = D2D1_COLOR_F { 
                        r: color.r * 0.6, 
                        g: color.g * 0.6, 
                        b: color.b * 0.6, 
                        a: 1.0 
                    };
                    renderer.draw_rect(cell.inset(0.5), border_color, 1.0);
                }
            }
        }

        // Highlight last move
        if let Some(last_moves) = state.get_last_move() {
            for (r, c) in last_moves {
                let cell = layout.cell_rect(r, c);
                renderer.draw_rect(cell, Colors::LAST_MOVE, 2.0);
            }
        }

        // Draw hover indicator
        if let Some((row, col)) = self.hover_cell {
            if row < board_size && col < board_size && board[row][col] == 0 {
                let cell = layout.cell_rect(row, col);
                renderer.fill_rect(cell, Colors::HIGHLIGHT);
            }
        }

        // Draw current player indicator
        let current_player = game.get_current_player();
        let status_area = Rect::new(board_area.x + board_area.width + 10.0, area.y + 10.0, 200.0, 40.0);
        renderer.fill_rounded_rect(status_area, 5.0, Colors::PANEL_BG);
        
        let player_name = match current_player {
            1 => "Blue's Turn",
            2 => "Yellow's Turn",
            3 => "Red's Turn",
            4 => "Green's Turn",
            _ => "Unknown",
        };
        renderer.draw_text(player_name, status_area, player_blokus_color(current_player), true);

        // Note: Full piece selection panel would be complex
        // For now, show instructions
        let help_area = Rect::new(board_area.x + board_area.width + 10.0, area.y + 60.0, 200.0, 100.0);
        renderer.fill_rounded_rect(help_area, 5.0, Colors::PANEL_BG);
        renderer.draw_small_text(
            "Blokus requires complex\npiece selection UI.\nUse TUI for full experience.",
            help_area.with_padding(10.0),
            Colors::TEXT_SECONDARY,
            true,
        );
    }

    fn handle_input(
        &mut self,
        input: GameInput,
        game: &GameWrapper,
        area: Rect,
    ) -> InputResult {
        let GameWrapper::Blokus(_state) = game else { return InputResult::None };
        let board = game.get_board();
        let board_size = board.len();
        
        let board_area = Rect::new(area.x, area.y, area.height, area.height);
        let layout = grid::calculate_grid_layout(board_area, board_size, 10.0);

        match input {
            GameInput::Click { x, y } => {
                // Blokus requires piece selection before placement
                // For now, just update hover position
                if let Some((_row, _col)) = layout.screen_to_cell(x, y) {
                    // TODO: Full piece selection implementation
                    // For now, Blokus moves should use TUI
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
            GameInput::Key { code: _, pressed } => {
                // Could use keyboard to select pieces
                if pressed {
                    // R to rotate, F to flip, 0-9 for piece selection, etc.
                }
                InputResult::None
            }
        }
    }

    fn game_name(&self) -> &'static str {
        "Blokus"
    }

    fn game_description(&self) -> &'static str {
        "Place polyomino pieces corner-to-corner!"
    }

    fn player_name(&self, player_id: i32) -> String {
        match player_id {
            1 => "Blue".to_string(),
            2 => "Yellow".to_string(),
            3 => "Red".to_string(),
            4 => "Green".to_string(),
            _ => format!("Player {}", player_id),
        }
    }

    fn reset(&mut self) {
        self.hover_cell = None;
        self.selected_piece = None;
        self.selected_transform = 0;
    }
}
