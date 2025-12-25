//! # Blokus Game Renderer
//!
//! Renders the Blokus game board with polyomino pieces for 4 players.
//! Features complete piece selection, transformation (rotate/flip), and ghost preview.

use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::games::blokus::{BlokusMove, BlokusState, get_blokus_pieces, Piece};
use crate::gui::colors::Colors;
use crate::gui::renderer::{Rect, Renderer};
use mcts::GameState;
use super::{GameInput, GameRenderer, InputResult, grid};
use std::collections::HashSet;
use windows::Win32::Graphics::Direct2D::Common::D2D1_COLOR_F;

/// Colors for the 4 Blokus players
const BLOKUS_COLORS: [D2D1_COLOR_F; 4] = [
    D2D1_COLOR_F { r: 0.2, g: 0.4, b: 0.9, a: 1.0 },  // Blue (Player 1)
    D2D1_COLOR_F { r: 0.9, g: 0.8, b: 0.2, a: 1.0 },  // Yellow (Player 2)
    D2D1_COLOR_F { r: 0.9, g: 0.2, b: 0.2, a: 1.0 },  // Red (Player 3)
    D2D1_COLOR_F { r: 0.2, g: 0.8, b: 0.3, a: 1.0 },  // Green (Player 4)
];

/// Faded versions of player colors for ghost pieces and indicators
const BLOKUS_COLORS_FADED: [D2D1_COLOR_F; 4] = [
    D2D1_COLOR_F { r: 0.2, g: 0.4, b: 0.9, a: 0.4 },
    D2D1_COLOR_F { r: 0.9, g: 0.8, b: 0.2, a: 0.4 },
    D2D1_COLOR_F { r: 0.9, g: 0.2, b: 0.2, a: 0.4 },
    D2D1_COLOR_F { r: 0.2, g: 0.8, b: 0.3, a: 0.4 },
];

fn player_blokus_color(player: i32) -> D2D1_COLOR_F {
    if player >= 1 && player <= 4 {
        BLOKUS_COLORS[(player - 1) as usize]
    } else {
        Colors::TEXT_SECONDARY
    }
}



/// Virtual key codes for keyboard input
mod vk {
    pub const VK_0: u32 = 0x30;
    pub const VK_1: u32 = 0x31;
    pub const VK_9: u32 = 0x39;
    pub const VK_A: u32 = 0x41;
    pub const VK_K: u32 = 0x4B;
    pub const VK_P: u32 = 0x50;
    pub const VK_R: u32 = 0x52;
    pub const VK_X: u32 = 0x58;
    pub const VK_ESCAPE: u32 = 0x1B;
    pub const VK_RETURN: u32 = 0x0D;
    pub const VK_LEFT: u32 = 0x25;
    pub const VK_UP: u32 = 0x26;
    pub const VK_RIGHT: u32 = 0x27;
    pub const VK_DOWN: u32 = 0x28;
}

/// Cached piece data for efficient rendering
struct PieceCache {
    pieces: Vec<Piece>,
}

impl PieceCache {
    fn new() -> Self {
        Self {
            pieces: get_blokus_pieces(),
        }
    }

    fn get_piece(&self, id: usize) -> Option<&Piece> {
        self.pieces.iter().find(|p| p.id == id)
    }

    fn get_shape(&self, piece_id: usize, transformation: usize) -> Option<&[(i32, i32)]> {
        self.get_piece(piece_id)
            .and_then(|p| p.transformations.get(transformation))
            .map(|v| v.as_slice())
    }

    fn get_transformation_count(&self, piece_id: usize) -> usize {
        self.get_piece(piece_id)
            .map(|p| p.transformations.len())
            .unwrap_or(0)
    }
}

/// Renderer for Blokus game
pub struct BlokusRenderer {
    /// Cached pieces for efficient lookup
    piece_cache: PieceCache,
    /// Current cursor position on board (row, col)
    cursor_pos: (usize, usize),
    /// Mouse hover position on board
    hover_cell: Option<(usize, usize)>,
    /// Currently selected piece index (0-20)
    selected_piece: Option<usize>,
    /// Current transformation index for selected piece
    selected_transform: usize,
    /// Scroll offset in the piece panel
    piece_panel_scroll: usize,
    /// Whether keyboard is controlling cursor (vs mouse)
    keyboard_mode: bool,
    /// Panel areas for hit testing (recalculated each render)
    piece_button_rects: Vec<(usize, Rect)>,  // (piece_id, rect)
}

impl BlokusRenderer {
    pub fn new() -> Self {
        Self {
            piece_cache: PieceCache::new(),
            cursor_pos: (0, 0),
            hover_cell: None,
            selected_piece: None,
            selected_transform: 0,
            piece_panel_scroll: 0,
            keyboard_mode: false,
            piece_button_rects: Vec::new(),
        }
    }

    /// Get ghost piece positions for the current selection at cursor
    fn get_ghost_positions(&self, state: &BlokusState) -> HashSet<(usize, usize)> {
        let mut positions = HashSet::new();
        
        let Some(piece_id) = self.selected_piece else {
            return positions;
        };

        let current_player = state.get_current_player();
        let available = state.get_available_pieces(current_player);
        
        if !available.contains(&piece_id) {
            return positions;
        }

        let Some(shape) = self.piece_cache.get_shape(piece_id, self.selected_transform) else {
            return positions;
        };

        let (cursor_row, cursor_col) = if self.keyboard_mode {
            self.cursor_pos
        } else if let Some(hover) = self.hover_cell {
            hover
        } else {
            return positions;
        };

        let board = state.get_board();
        let board_size = board.len();

        // Check if all positions are valid
        let mut all_valid = true;
        let mut temp_positions = Vec::new();

        for &(dr, dc) in shape {
            let r = cursor_row as i32 + dr;
            let c = cursor_col as i32 + dc;

            if r < 0 || r >= board_size as i32 || c < 0 || c >= board_size as i32 {
                all_valid = false;
                break;
            }

            temp_positions.push((r as usize, c as usize));
        }

        if all_valid {
            for (r, c) in temp_positions {
                if board[r][c] == 0 {
                    positions.insert((r, c));
                }
            }
        }

        positions
    }

    /// Check if current ghost placement would be legal
    fn is_current_placement_legal(&self, state: &BlokusState) -> bool {
        let Some(piece_id) = self.selected_piece else {
            return false;
        };

        let (row, col) = if self.keyboard_mode {
            self.cursor_pos
        } else if let Some(hover) = self.hover_cell {
            hover
        } else {
            return false;
        };

        let mv = BlokusMove(piece_id, self.selected_transform, row, col);
        state.is_legal(&mv)
    }

    /// Select piece by keyboard key
    fn select_piece_by_key(&mut self, key: u32, state: &BlokusState) -> bool {
        let piece_id = if key >= vk::VK_1 && key <= vk::VK_9 {
            (key - vk::VK_1) as usize
        } else if key == vk::VK_0 {
            9
        } else if key >= vk::VK_A && key <= vk::VK_K {
            10 + (key - vk::VK_A) as usize
        } else {
            return false;
        };

        if piece_id > 20 {
            return false;
        }

        // Check if piece is available
        let current_player = state.get_current_player();
        let available = state.get_available_pieces(current_player);

        if available.contains(&piece_id) {
            self.selected_piece = Some(piece_id);
            self.selected_transform = 0;
            self.keyboard_mode = true;
            true
        } else {
            false
        }
    }

    /// Rotate selected piece
    fn rotate_piece(&mut self) {
        if let Some(piece_id) = self.selected_piece {
            let count = self.piece_cache.get_transformation_count(piece_id);
            if count > 0 {
                self.selected_transform = (self.selected_transform + 1) % count;
            }
        }
    }

    /// Flip selected piece (reverse rotation direction)
    fn flip_piece(&mut self) {
        if let Some(piece_id) = self.selected_piece {
            let count = self.piece_cache.get_transformation_count(piece_id);
            if count > 0 {
                // Skip ahead by half the transformations to get a "flip"
                let flip_amount = count / 2;
                if flip_amount > 0 {
                    self.selected_transform = (self.selected_transform + flip_amount) % count;
                }
            }
        }
    }

    /// Render a small piece preview in the given rect
    fn render_piece_preview(
        &self,
        renderer: &Renderer,
        shape: &[(i32, i32)],
        area: Rect,
        player: i32,
        is_available: bool,
    ) {
        if shape.is_empty() {
            return;
        }

        // Calculate bounds
        let min_r = shape.iter().map(|p| p.0).min().unwrap_or(0);
        let max_r = shape.iter().map(|p| p.0).max().unwrap_or(0);
        let min_c = shape.iter().map(|p| p.1).min().unwrap_or(0);
        let max_c = shape.iter().map(|p| p.1).max().unwrap_or(0);

        let piece_height = (max_r - min_r + 1) as f32;
        let piece_width = (max_c - min_c + 1) as f32;

        // Scale to fit in preview area with some padding
        let preview_padding = 8.0;
        let available_size = (area.width - preview_padding * 2.0).min(area.height - preview_padding * 2.0);
        let cell_size = available_size / piece_height.max(piece_width);

        // Center the piece
        let piece_actual_width = piece_width * cell_size;
        let piece_actual_height = piece_height * cell_size;
        let offset_x = area.x + (area.width - piece_actual_width) / 2.0;
        let offset_y = area.y + (area.height - piece_actual_height) / 2.0;

        let color = if is_available {
            player_blokus_color(player)
        } else {
            D2D1_COLOR_F { r: 0.3, g: 0.3, b: 0.3, a: 0.5 }
        };

        for &(dr, dc) in shape {
            let x = offset_x + (dc - min_c) as f32 * cell_size;
            let y = offset_y + (dr - min_r) as f32 * cell_size;
            let cell_rect = Rect::new(x, y, cell_size - 1.0, cell_size - 1.0);
            renderer.fill_rect(cell_rect, color);
        }
    }

    /// Render player status panel
    fn render_player_status(
        &self,
        renderer: &Renderer,
        state: &BlokusState,
        area: Rect,
    ) {
        renderer.fill_rounded_rect(area, 5.0, Colors::PANEL_BG);

        let current_player = state.get_current_player();
        let line_height = 28.0;

        for player in 1..=4 {
            let y = area.y + 10.0 + (player as f32 - 1.0) * line_height;
            let row_rect = Rect::new(area.x + 10.0, y, area.width - 20.0, line_height);

            let available = state.get_available_pieces(player);
            let pieces_left = available.len();

            let player_name = match player {
                1 => "Blue",
                2 => "Yellow",
                3 => "Red",
                4 => "Green",
                _ => "?",
            };

            let indicator = if player == current_player { "► " } else { "  " };
            let text = format!("{}{}: {} pieces", indicator, player_name, pieces_left);

            let color = player_blokus_color(player);
            renderer.draw_text(&text, row_rect, color, false);

            // Highlight current player row
            if player == current_player {
                let highlight_rect = Rect::new(area.x + 5.0, y - 2.0, area.width - 10.0, line_height);
                renderer.draw_rounded_rect(highlight_rect, 3.0, color, 1.0);
            }
        }
    }
}

impl GameRenderer for BlokusRenderer {
    fn render(&self, renderer: &Renderer, game: &GameWrapper, area: Rect) {
        let GameWrapper::Blokus(state) = game else { return };
        let board = state.get_board();
        let board_size = board.len();

        // Layout: board on left (square), side panel on right
        let side_panel_width = 220.0;
        let board_available = (area.height).min(area.width - side_panel_width - 20.0);
        let board_area = Rect::new(area.x + 10.0, area.y + 10.0, board_available, board_available);
        
        let layout = grid::calculate_grid_layout(board_area, board_size, 5.0);

        // Draw board background
        let board_rect = layout.board_rect();
        renderer.fill_rect(board_rect.inset(-3.0), Colors::PANEL_BG);

        // Draw checkerboard pattern for empty cells
        for row in 0..board_size {
            for col in 0..board_size {
                let cell = layout.cell_rect(row, col);
                let is_light = (row + col) % 2 == 0;
                let empty_color = if is_light {
                    D2D1_COLOR_F { r: 0.25, g: 0.25, b: 0.28, a: 1.0 }
                } else {
                    D2D1_COLOR_F { r: 0.20, g: 0.20, b: 0.23, a: 1.0 }
                };
                renderer.fill_rect(cell, empty_color);
            }
        }

        // Draw grid with thin lines
        let grid_color = D2D1_COLOR_F { r: 0.35, g: 0.35, b: 0.40, a: 0.6 };
        for i in 0..=board_size {
            let x = layout.offset_x + i as f32 * layout.cell_size;
            renderer.draw_line(x, board_rect.y, x, board_rect.y + board_rect.height, grid_color, 0.5);
        }
        for i in 0..=board_size {
            let y = layout.offset_y + i as f32 * layout.cell_size;
            renderer.draw_line(board_rect.x, y, board_rect.x + board_rect.width, y, grid_color, 0.5);
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
                let color = BLOKUS_COLORS_FADED[player_idx];
                renderer.fill_rect(cell.inset(1.0), color);
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
                        r: (color.r * 0.6).min(1.0), 
                        g: (color.g * 0.6).min(1.0), 
                        b: (color.b * 0.6).min(1.0), 
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
                renderer.draw_rect(cell, Colors::LAST_MOVE, 2.5);
            }
        }

        // Draw ghost piece preview
        let ghost_positions = self.get_ghost_positions(state);
        if !ghost_positions.is_empty() {
            let is_legal = self.is_current_placement_legal(state);
            let current_player = state.get_current_player();

            let ghost_color = if is_legal {
                // Legal placement - use player color with transparency
                let base = player_blokus_color(current_player);
                D2D1_COLOR_F { r: base.r, g: base.g, b: base.b, a: 0.6 }
            } else {
                // Illegal placement - red tint
                D2D1_COLOR_F { r: 0.9, g: 0.2, b: 0.2, a: 0.5 }
            };

            for (row, col) in &ghost_positions {
                let cell = layout.cell_rect(*row, *col);
                renderer.fill_rect(cell.inset(1.0), ghost_color);
            }

            // Draw border around ghost piece
            let border_color = if is_legal {
                Colors::TEXT_ACCENT
            } else {
                Colors::STATUS_ERROR
            };

            for (row, col) in &ghost_positions {
                let cell = layout.cell_rect(*row, *col);
                renderer.draw_rect(cell.inset(1.0), border_color, 1.5);
            }
        }

        // Draw cursor (keyboard mode) or hover indicator
        let cursor = if self.keyboard_mode {
            Some(self.cursor_pos)
        } else {
            self.hover_cell
        };

        if let Some((row, col)) = cursor {
            if row < board_size && col < board_size {
                let cell = layout.cell_rect(row, col);
                let cursor_color = if self.keyboard_mode {
                    D2D1_COLOR_F { r: 1.0, g: 1.0, b: 0.0, a: 0.8 }
                } else {
                    Colors::HIGHLIGHT
                };
                renderer.draw_rect(cell, cursor_color, 2.0);
            }
        }

        // Side panel layout
        let panel_x = board_area.x + board_available + 15.0;
        let panel_width = side_panel_width;

        // Player status panel (top)
        let status_height = 130.0;
        let status_area = Rect::new(panel_x, area.y + 10.0, panel_width, status_height);
        self.render_player_status(renderer, state, status_area);

        // Piece selection panel (below status)
        let piece_panel_y = area.y + 10.0 + status_height + 10.0;
        let piece_panel_height = area.height - status_height - 30.0;
        let piece_panel_area = Rect::new(panel_x, piece_panel_y, panel_width, piece_panel_height);

        // Need mutable self for piece_button_rects
        // Use a slightly different approach - we'll make piece_button_rects interior mutable
        // For now, we call render_piece_panel through a mutable method
        // Actually, since render takes &self, we need to handle this differently
        // We'll store the layout info and handle clicks separately
        
        // For now, just render the panel info
        renderer.fill_rounded_rect(piece_panel_area, 5.0, Colors::PANEL_BG);

        let current_player = state.get_current_player();
        let available_pieces = state.get_available_pieces(current_player);
        let available_set: HashSet<usize> = available_pieces.iter().copied().collect();

        // Title
        let title_area = Rect::new(piece_panel_area.x, piece_panel_area.y + 5.0, piece_panel_area.width, 25.0);
        renderer.draw_text("Select Piece", title_area, Colors::TEXT_PRIMARY, true);

        // Piece grid layout - use smaller pieces to fit all 21
        let grid_start_y = piece_panel_area.y + 30.0;
        let piece_size = 38.0;
        let padding = 3.0;
        let cols = ((piece_panel_area.width - padding * 2.0) / (piece_size + padding)).floor() as usize;
        let cols = cols.max(1);

        for (idx, piece_id) in (0..21usize).enumerate() {
            let row = idx / cols;
            let col = idx % cols;

            let x = piece_panel_area.x + padding + col as f32 * (piece_size + padding);
            let y = grid_start_y + row as f32 * (piece_size + padding);

            // Skip if outside visible area (reduced margin for help text)
            if y > piece_panel_area.y + piece_panel_area.height - piece_size - 45.0 {
                continue;
            }

            let btn_rect = Rect::new(x, y, piece_size, piece_size);

            let is_available = available_set.contains(&piece_id);
            let is_selected = self.selected_piece == Some(piece_id);

            // Background color
            let bg_color = if is_selected {
                Colors::BUTTON_SELECTED
            } else if is_available {
                Colors::BUTTON_BG
            } else {
                D2D1_COLOR_F { r: 0.15, g: 0.15, b: 0.15, a: 0.5 }
            };

            renderer.fill_rounded_rect(btn_rect, 3.0, bg_color);

            // Draw piece shape preview
            if let Some(piece) = self.piece_cache.get_piece(piece_id) {
                let transform_idx = if is_selected { self.selected_transform } else { 0 };
                if let Some(shape) = piece.transformations.get(transform_idx) {
                    self.render_piece_preview(renderer, shape, btn_rect, current_player, is_available);
                }
            }

            // Selection border
            if is_selected {
                renderer.draw_rounded_rect(btn_rect, 3.0, Colors::TEXT_ACCENT, 2.0);
            }

            // Key label
            let key_label = if piece_id < 9 {
                format!("{}", piece_id + 1)
            } else if piece_id == 9 {
                "0".to_string()
            } else {
                ((b'a' + (piece_id - 10) as u8) as char).to_string()
            };

            let label_rect = Rect::new(x + 2.0, y + 2.0, 15.0, 15.0);
            let label_color = if is_available { Colors::TEXT_PRIMARY } else { Colors::TEXT_SECONDARY };
            renderer.draw_small_text(&key_label, label_rect, label_color, false);
        }

        // Controls help at bottom (compact)
        let help_y = piece_panel_area.y + piece_panel_area.height - 40.0;
        let help_rect = Rect::new(piece_panel_area.x + 5.0, help_y, piece_panel_area.width - 10.0, 38.0);
        renderer.draw_small_text(
            "R: Rotate  X: Flip  P: Pass\nArrows: Move  Enter/Click: Place",
            help_rect,
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
        let GameWrapper::Blokus(state) = game else { return InputResult::None };
        let board = game.get_board();
        let board_size = board.len();
        
        // Recalculate layout for hit testing
        let side_panel_width = 220.0;
        let board_available = (area.height).min(area.width - side_panel_width - 20.0);
        let board_area = Rect::new(area.x + 10.0, area.y + 10.0, board_available, board_available);
        let layout = grid::calculate_grid_layout(board_area, board_size, 5.0);

        // Piece panel area
        let panel_x = board_area.x + board_available + 15.0;
        let piece_panel_y = area.y + 10.0 + 140.0;
        let piece_panel_area = Rect::new(panel_x, piece_panel_y, side_panel_width, area.height - 160.0);

        match input {
            GameInput::Click { x, y } => {
                // Check if click is on board
                if let Some((row, col)) = layout.screen_to_cell(x, y) {
                    self.keyboard_mode = false;
                    self.hover_cell = Some((row, col));

                    // If we have a piece selected, try to place it
                    if let Some(piece_id) = self.selected_piece {
                        let mv = BlokusMove(piece_id, self.selected_transform, row, col);
                        if state.is_legal(&mv) {
                            return InputResult::Move(MoveWrapper::Blokus(mv));
                        }
                    }
                    return InputResult::Redraw;
                }

                // Check if click is in piece panel
                if piece_panel_area.contains(x, y) {
                    // Calculate which piece was clicked - must match render layout!
                    let grid_start_y = piece_panel_area.y + 30.0;
                    let piece_size = 38.0;
                    let padding = 3.0;
                    let cols = ((piece_panel_area.width - padding * 2.0) / (piece_size + padding)).floor() as usize;
                    let cols = cols.max(1);

                    let available_pieces = state.get_available_pieces(state.get_current_player());
                    let available_set: HashSet<usize> = available_pieces.iter().copied().collect();

                    for (idx, piece_id) in (0..21usize).enumerate() {
                        let row = idx / cols;
                        let col = idx % cols;
                        let px = piece_panel_area.x + padding + col as f32 * (piece_size + padding);
                        let py = grid_start_y + row as f32 * (piece_size + padding);
                        let btn_rect = Rect::new(px, py, piece_size, piece_size);

                        if btn_rect.contains(x, y) {
                            if available_set.contains(&piece_id) {
                                if self.selected_piece == Some(piece_id) {
                                    // Already selected - rotate on click
                                    self.rotate_piece();
                                } else {
                                    self.selected_piece = Some(piece_id);
                                    self.selected_transform = 0;
                                }
                                return InputResult::Redraw;
                            }
                            break;
                        }
                    }
                }

                InputResult::None
            }

            GameInput::Hover { x, y } => {
                if !self.keyboard_mode {
                    let new_hover = layout.screen_to_cell(x, y);
                    if new_hover != self.hover_cell {
                        self.hover_cell = new_hover;
                        return InputResult::Redraw;
                    }
                }
                InputResult::None
            }

            GameInput::Key { code, pressed } => {
                if !pressed {
                    return InputResult::None;
                }

                match code {
                    // Piece selection: 1-9, 0, a-k
                    c if (c >= vk::VK_1 && c <= vk::VK_9) || c == vk::VK_0 || (c >= vk::VK_A && c <= vk::VK_K) => {
                        if self.select_piece_by_key(c, state) {
                            return InputResult::Redraw;
                        }
                        InputResult::None
                    }

                    // Rotate piece
                    vk::VK_R => {
                        self.rotate_piece();
                        InputResult::Redraw
                    }

                    // Flip piece
                    vk::VK_X => {
                        self.flip_piece();
                        InputResult::Redraw
                    }

                    // Pass turn
                    vk::VK_P => {
                        // Check if pass is a valid move
                        let pass_move = BlokusMove(usize::MAX, 0, 0, 0);
                        if state.is_legal(&pass_move) {
                            return InputResult::Move(MoveWrapper::Blokus(pass_move));
                        }
                        InputResult::None
                    }

                    // Deselect piece
                    vk::VK_ESCAPE => {
                        self.selected_piece = None;
                        self.selected_transform = 0;
                        InputResult::Redraw
                    }

                    // Place piece
                    vk::VK_RETURN => {
                        if let Some(piece_id) = self.selected_piece {
                            let (row, col) = self.cursor_pos;
                            let mv = BlokusMove(piece_id, self.selected_transform, row, col);
                            if state.is_legal(&mv) {
                                return InputResult::Move(MoveWrapper::Blokus(mv));
                            }
                        }
                        InputResult::None
                    }

                    // Arrow keys - move cursor
                    vk::VK_LEFT => {
                        self.keyboard_mode = true;
                        if self.cursor_pos.1 > 0 {
                            self.cursor_pos.1 -= 1;
                        }
                        InputResult::Redraw
                    }
                    vk::VK_RIGHT => {
                        self.keyboard_mode = true;
                        if self.cursor_pos.1 < board_size - 1 {
                            self.cursor_pos.1 += 1;
                        }
                        InputResult::Redraw
                    }
                    vk::VK_UP => {
                        self.keyboard_mode = true;
                        if self.cursor_pos.0 > 0 {
                            self.cursor_pos.0 -= 1;
                        }
                        InputResult::Redraw
                    }
                    vk::VK_DOWN => {
                        self.keyboard_mode = true;
                        if self.cursor_pos.0 < board_size - 1 {
                            self.cursor_pos.0 += 1;
                        }
                        InputResult::Redraw
                    }

                    _ => InputResult::None,
                }
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
        self.cursor_pos = (0, 0);
        self.hover_cell = None;
        self.selected_piece = None;
        self.selected_transform = 0;
        self.piece_panel_scroll = 0;
        self.keyboard_mode = false;
        self.piece_button_rects.clear();
    }
}
