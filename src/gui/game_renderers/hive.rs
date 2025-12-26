//! # Hive Game Renderer
//!
//! Renders the Hive game with hexagonal tiles and piece icons.
//! Features piece selection panel, movement highlighting, and ghost preview.

use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::games::hive::{HexCoord, HiveMove, HiveState, Piece, PieceType};
use crate::gui::colors::{Colors, with_alpha};
use crate::gui::renderer::{Rect, Renderer};
use mcts::GameState;
use super::{GameInput, GameRenderer, InputResult, RotatableBoard};
use std::collections::HashSet;
use windows::Win32::Graphics::Direct2D::Common::D2D1_COLOR_F;

/// Colors for Hive pieces
const PLAYER_1_COLOR: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }; // Black
const PLAYER_2_COLOR: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.95, g: 0.95, b: 0.95, a: 1.0 }; // White
const PLAYER_1_TEXT: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.9, g: 0.9, b: 0.9, a: 1.0 };
const PLAYER_2_TEXT: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.1, g: 0.1, b: 0.1, a: 1.0 };

const HEX_BG_COLOR: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.76, g: 0.60, b: 0.42, a: 1.0 }; // Wood table
const HEX_GRID_COLOR: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.60, g: 0.45, b: 0.30, a: 0.4 }; // Grid lines
const HEX_EMPTY_COLOR: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.85, g: 0.72, b: 0.55, a: 0.3 }; // Empty hex fill

/// Vertical offset per stack level (in pixels, scaled by hex_size)
/// This is applied in local coordinates before the tilt transform
const STACK_OFFSET_Y: f32 = 0.6;
/// Forward offset per stack level (towards camera when tilted)
const STACK_OFFSET_Z: f32 = 0.3;

/// Virtual key codes for keyboard input
mod vk {
    pub const VK_Q: u32 = 0x51;
    pub const VK_B: u32 = 0x42;
    pub const VK_S: u32 = 0x53;
    pub const VK_G: u32 = 0x47;
    pub const VK_A: u32 = 0x41;
    pub const VK_ESCAPE: u32 = 0x1B;
}

/// Layout information for hexagonal grid
#[derive(Debug, Clone, Copy)]
pub struct HexLayout {
    /// Size of each hexagon (distance from center to corner)
    pub hex_size: f32,
    /// Center of the display area
    pub center_x: f32,
    pub center_y: f32,
    /// View offset for panning (in hex coordinates)
    pub offset_q: f32,
    pub offset_r: f32,
}

impl HexLayout {
    /// Calculate hex layout based on available area and board bounds
    fn calculate(area: Rect, state: &HiveState) -> Self {
        // Find bounds of placed pieces
        let mut min_q: i32 = 0;
        let mut max_q: i32 = 0;
        let mut min_r: i32 = 0;
        let mut max_r: i32 = 0;
        
        for coord in state.get_hex_board().keys() {
            min_q = min_q.min(coord.q);
            max_q = max_q.max(coord.q);
            min_r = min_r.min(coord.r);
            max_r = max_r.max(coord.r);
        }
        
        // Add padding
        let padding = 3;
        min_q -= padding;
        max_q += padding;
        min_r -= padding;
        max_r += padding;
        
        let q_range = (max_q - min_q + 1) as f32;
        let r_range = (max_r - min_r + 1) as f32;
        
        // Calculate hex size to fit in area
        // For pointy-top hexagons: width = sqrt(3) * size, height = 2 * size
        let sqrt3 = 3.0_f32.sqrt();
        let available_width = area.width * 0.9;
        let available_height = area.height * 0.9;
        
        // Account for hex staggering
        let hex_size_for_width = available_width / (q_range * sqrt3 + 0.5 * sqrt3);
        let hex_size_for_height = available_height / (r_range * 1.5 + 0.5);
        
        let hex_size = hex_size_for_width.min(hex_size_for_height).max(20.0).min(60.0);
        
        // Calculate center offset
        let center_q = (min_q + max_q) as f32 / 2.0;
        let center_r = (min_r + max_r) as f32 / 2.0;
        
        Self {
            hex_size,
            center_x: area.x + area.width / 2.0,
            center_y: area.y + area.height / 2.0,
            offset_q: center_q,
            offset_r: center_r,
        }
    }
    
    /// Convert hex coordinates to screen coordinates (with isometric tilt)
    /// Note: Rotation is applied via D2D transform, not here
    fn hex_to_screen(&self, q: i32, r: i32) -> (f32, f32) {
        let sqrt3 = 3.0_f32.sqrt();
        let q_adj = q as f32 - self.offset_q;
        let r_adj = r as f32 - self.offset_r;
        
        // Calculate position with tilt (rotation handled by D2D transform)
        let x_local = self.hex_size * sqrt3 * (q_adj + r_adj / 2.0);
        let y_local = self.hex_size * 1.5 * r_adj;
        
        (self.center_x + x_local, self.center_y + y_local)
    }
    
    /// Convert hex coordinates to screen coordinates with stack height offset
    fn hex_to_screen_stacked(&self, q: i32, r: i32, stack_level: usize) -> (f32, f32) {
        let (x, y) = self.hex_to_screen(q, r);
        // Move pieces up (negative Y) and forward (negative Y in local space = towards camera when tilted)
        // The Y offset lifts pieces visually, the "forward" offset separates them in the tilt direction
        let level = stack_level as f32;
        let y_offset = level * self.hex_size * STACK_OFFSET_Y;
        let z_offset = level * self.hex_size * STACK_OFFSET_Z;
        (x, y - y_offset - z_offset)
    }
    

    /// Convert local (flat, unscaled) coordinates to hex coordinates
    fn local_to_hex(&self, x: f32, y: f32) -> HexCoord {
        let sqrt3 = 3.0_f32.sqrt();
        
        let r = y / (self.hex_size * 1.5);
        let q = x / (self.hex_size * sqrt3) - r / 2.0;
        
        // Round to nearest hex
        let q_adj = q + self.offset_q;
        let r_adj = r + self.offset_r;
        
        HexCoord::new(q_adj.round() as i32, r_adj.round() as i32)
    }
}

/// Input mode for Hive
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputMode {
    /// Selecting a piece type to place
    SelectPiece,
    /// Selecting a position to place a piece
    PlacePiece(PieceType),
    /// Selecting a piece to move
    SelectMove,
    /// Selecting destination for a move
    MovePiece(HexCoord),
}

/// Renderer for Hive game
pub struct HiveRenderer {
    /// Current input mode
    mode: InputMode,
    /// Mouse hover hex position
    hover_hex: Option<HexCoord>,
    /// Valid placement/move positions (cached)
    valid_positions: HashSet<HexCoord>,
    /// Layout cache
    last_layout: Option<HexLayout>,
    /// 3D rotatable board component for tilt/rotation
    board_view: RotatableBoard,
}

impl HiveRenderer {
    pub fn new() -> Self {
        Self {
            mode: InputMode::SelectPiece,
            hover_hex: None,
            valid_positions: HashSet::new(),
            last_layout: None,
            board_view: RotatableBoard::new(),
        }
    }

    /// Draw a hexagonal tile at the given position
    #[allow(dead_code)]
    fn draw_hex(&self, renderer: &Renderer, layout: &HexLayout, q: i32, r: i32, fill_color: D2D1_COLOR_F, outline_color: Option<D2D1_COLOR_F>) {
        let (cx, cy) = layout.hex_to_screen(q, r);
        let size = layout.hex_size * 0.95; // Slightly smaller to show gaps
        
        // Draw hexagon (tilt+rotation handled by D2D transform)
        renderer.fill_iso_hexagon(cx, cy, size, fill_color);
        
        if let Some(outline) = outline_color {
            renderer.draw_iso_hexagon(cx, cy, size, outline, 2.0);
        }
    }

    /// Draw a piece on a hex at a specific stack level
    fn draw_piece(&self, renderer: &Renderer, layout: &HexLayout, q: i32, r: i32, stack_level: usize, piece: &Piece, _is_top: bool, is_selected: bool, is_last_move: bool) {
        let (cx, cy) = layout.hex_to_screen_stacked(q, r, stack_level);
        let size = layout.hex_size * 0.85;
        
        // Choose colors based on player
        let (fill_color, text_color) = if piece.player == 1 {
            (PLAYER_1_COLOR, PLAYER_1_TEXT)
        } else {
            (PLAYER_2_COLOR, PLAYER_2_TEXT)
        };
        
        // Draw hex-shaped piece (tilt+rotation handled by D2D transform)
        renderer.fill_iso_hexagon(cx, cy, size, fill_color);
        
        // Draw outline
        let outline_color = if piece.player == 1 { PLAYER_2_COLOR } else { PLAYER_1_COLOR };
        renderer.draw_iso_hexagon(cx, cy, size, outline_color, 2.0);
        
        // Draw piece icon (always draw, even for pieces under a beetle)
        let icon_size = size * 0.55;
        self.draw_piece_icon(renderer, cx, cy, icon_size, piece.piece_type, text_color);
        
        // Selection highlight
        if is_selected {
            renderer.draw_iso_hexagon(cx, cy, size + 4.0, Colors::BUTTON_SELECTED, 3.0);
        }
        
        // Last move indicator
        if is_last_move {
            renderer.fill_iso_hexagon(cx, cy, size * 0.25, Colors::LAST_MOVE);
        }

        // Draw coordinates
        let font_size = layout.hex_size / 8.0;
        let coord_text = format!("{},{}", q, r);
        
        // Calculate position on the bottom-right edge
        // Edge midpoint is at angle 60 degrees (PI/3)
        // Distance is hex_size * sqrt(3)/2 (apothem)
        // We move slightly inwards (0.85 factor) to be inside
        let angle_pos = std::f32::consts::PI / 3.0;
        let dist = layout.hex_size * 3.0_f32.sqrt() / 2.0 * 0.85;
        
        let text_cx = cx + dist * angle_pos.cos();
        let text_cy = cy + dist * angle_pos.sin();
        
        // Rotation angle: -30 degrees (-PI/6) to align with edge
        let rotation_angle = -std::f32::consts::PI / 6.0;
        
        renderer.draw_rotated_text(
            &coord_text,
            text_cx,
            text_cy,
            rotation_angle,
            Colors::BOARD_GRID,
            font_size
        );
    }

    /// Draw the icon for a specific piece type using native D2D SVG
    /// Note: Tilt is applied via D2D transform
    fn draw_piece_icon(&self, renderer: &Renderer, cx: f32, cy: f32, size: f32, piece_type: PieceType, _text_color: D2D1_COLOR_F) {
        let svg_name = match piece_type {
            PieceType::Queen => "hive_queen",
            PieceType::Beetle => "hive_beetle",
            PieceType::Spider => "hive_spider",
            PieceType::Grasshopper => "hive_grasshopper",
            PieceType::Ant => "hive_ant",
        };
        
        // Draw SVG icon centered at (cx, cy) - tilt handled by D2D transform
        let icon_size = size * 2.0;
        renderer.draw_svg(svg_name, cx, cy, icon_size, icon_size);
    }

    /// Draw valid position indicator
    fn draw_valid_position(&self, renderer: &Renderer, layout: &HexLayout, q: i32, r: i32, is_hovered: bool) {
        let (cx, cy) = layout.hex_to_screen(q, r);
        let size = layout.hex_size * 0.5;
        
        let color = if is_hovered {
            Colors::BUTTON_SELECTED
        } else {
            with_alpha(Colors::HIGHLIGHT, 0.5)
        };
        
        renderer.fill_iso_hexagon(cx, cy, size, color);
    }

    /// Draw hexagonal grid background
    fn draw_hex_grid(&self, renderer: &Renderer, layout: &HexLayout, state: &HiveState, area: Rect) {
        // Find bounds of the grid to draw
        let mut min_q: i32 = 0;
        let mut max_q: i32 = 0;
        let mut min_r: i32 = 0;
        let mut max_r: i32 = 0;
        
        // Include placed pieces in bounds
        for coord in state.get_hex_board().keys() {
            min_q = min_q.min(coord.q);
            max_q = max_q.max(coord.q);
            min_r = min_r.min(coord.r);
            max_r = max_r.max(coord.r);
        }
        
        // Include valid positions in bounds
        for coord in &self.valid_positions {
            min_q = min_q.min(coord.q);
            max_q = max_q.max(coord.q);
            min_r = min_r.min(coord.r);
            max_r = max_r.max(coord.r);
        }
        
        // Add padding for the grid
        let padding = 2;
        min_q -= padding;
        max_q += padding;
        min_r -= padding;
        max_r += padding;
        
        // Draw empty hexagon grid cells
        for q in min_q..=max_q {
            for r in min_r..=max_r {
                let (cx, cy) = layout.hex_to_screen(q, r);
                
                // Only draw if within visible area
                if cx >= area.x - layout.hex_size && cx <= area.x + area.width + layout.hex_size
                   && cy >= area.y - layout.hex_size && cy <= area.y + area.height + layout.hex_size {
                    // Don't draw grid cells where pieces are placed
                    let coord = HexCoord::new(q, r);
                    if !state.get_hex_board().contains_key(&coord) {
                        // Only draw empty cell if it's move 1 (turn 1) or adjacent to a filled cell
                        let is_turn_1 = state.get_turn() <= 1;
                        let is_adjacent = coord.neighbors().iter().any(|n| state.get_hex_board().contains_key(n));
                        
                        if is_turn_1 || is_adjacent {
                            // Draw empty hex cell (tilt+rotation handled by D2D transform)
                            renderer.fill_iso_hexagon(cx, cy, layout.hex_size * 0.92, HEX_EMPTY_COLOR);
                            renderer.draw_iso_hexagon(cx, cy, layout.hex_size * 0.92, HEX_GRID_COLOR, 1.0);

                            // Draw coordinates
                            let font_size = layout.hex_size / 8.0;
                            let coord_text = format!("{},{}", q, r);
                            
                            // Calculate position on the bottom-right edge
                            let angle_pos = std::f32::consts::PI / 3.0;
                            let dist = layout.hex_size * 3.0_f32.sqrt() / 2.0 * 0.85;
                            
                            let text_cx = cx + dist * angle_pos.cos();
                            let text_cy = cy + dist * angle_pos.sin();
                            
                            // Rotation angle: -30 degrees (-PI/6) to align with edge
                            let rotation_angle = -std::f32::consts::PI / 6.0;
                            
                            renderer.draw_rotated_text(
                                &coord_text,
                                text_cx,
                                text_cy,
                                rotation_angle,
                                Colors::BOARD_GRID,
                                font_size
                            );
                        }
                    }
                }
            }
        }
    }

    /// Draw the piece selection panel
    fn draw_piece_panel(&self, renderer: &Renderer, state: &HiveState, area: Rect) {
        let current_player = state.get_current_player();
        
        // Background
        renderer.fill_rect(area, Colors::PANEL_BG);
        
        // Title
        let title_rect = Rect::new(area.x, area.y + 5.0, area.width, 25.0);
        renderer.draw_text("Select Piece", title_rect, Colors::TEXT_PRIMARY, true);
        
        let mut y = area.y + 35.0;
        let button_height = 50.0;
        let spacing = 5.0;
        
        for piece_type in PieceType::all() {
            let count = state.pieces_in_hand(current_player, *piece_type);
            
            let button_rect = Rect::new(area.x + 10.0, y, area.width - 20.0, button_height);
            
            // Button background
            let bg_color = if count == 0 {
                Colors::BUTTON_PRESSED
            } else if matches!(self.mode, InputMode::PlacePiece(t) if t == *piece_type) {
                Colors::BUTTON_SELECTED
            } else {
                Colors::BUTTON_BG
            };
            
            renderer.fill_rounded_rect(button_rect, 5.0, bg_color);
            
            // Draw piece icon on the left side
            let icon_size = button_height * 0.35;
            let icon_cx = button_rect.x + 30.0;
            let icon_cy = button_rect.y + button_height / 2.0;
            self.draw_piece_icon(renderer, icon_cx, icon_cy, icon_size, *piece_type, Colors::TEXT_PRIMARY);
            
            // Draw count on the right side
            let count_text = format!("x{}", count);
            let text_color = if count == 0 { Colors::TEXT_SECONDARY } else { Colors::TEXT_PRIMARY };
            let count_rect = Rect::new(button_rect.x + 55.0, button_rect.y, button_rect.width - 70.0, button_height);
            renderer.draw_text(&count_text, count_rect, text_color, true);
            
            // Key hint
            let key_hint = match piece_type {
                PieceType::Queen => "Q",
                PieceType::Beetle => "B",
                PieceType::Spider => "S",
                PieceType::Grasshopper => "G",
                PieceType::Ant => "A",
            };
            let hint_rect = Rect::new(area.x + area.width - 35.0, y + (button_height - 20.0) / 2.0, 25.0, 20.0);
            renderer.draw_small_text(key_hint, hint_rect, Colors::TEXT_SECONDARY, true);
            
            y += button_height + spacing;
        }
        
        // Mode hint
        let hint_y = area.y + area.height - 60.0;
        let hint_rect = Rect::new(area.x + 5.0, hint_y, area.width - 10.0, 55.0);
        let hint_text = match self.mode {
            InputMode::SelectPiece => "Click piece or\npress Q/B/S/G/A",
            InputMode::PlacePiece(_) => "Click to place\nESC to cancel",
            InputMode::SelectMove => "Click your piece\nto move it",
            InputMode::MovePiece(_) => "Click destination\nESC to cancel",
        };
        renderer.draw_small_text(hint_text, hint_rect, Colors::TEXT_SECONDARY, true);
    }

    /// Get piece button rect for hit testing
    fn get_piece_button_rect(&self, area: Rect, piece_index: usize) -> Rect {
        let y = area.y + 35.0 + (piece_index as f32) * 55.0; // button_height (50) + spacing (5)
        Rect::new(area.x + 10.0, y, area.width - 20.0, 50.0)
    }

    /// Check if a point is in the piece panel and return the piece type if clicking a button
    fn get_piece_at_panel_click(&self, x: f32, y: f32, panel_area: Rect) -> Option<PieceType> {
        for (index, piece_type) in PieceType::all().iter().enumerate() {
            let button_rect = self.get_piece_button_rect(panel_area, index);
            if x >= button_rect.x && x <= button_rect.x + button_rect.width
                && y >= button_rect.y && y <= button_rect.y + button_rect.height
            {
                return Some(*piece_type);
            }
        }
        None
    }

    /// Get valid positions based on current mode
    fn update_valid_positions(&mut self, state: &HiveState) {
        self.valid_positions.clear();
        
        match self.mode {
            InputMode::PlacePiece(piece_type) => {
                // Find valid placement positions for this piece type
                for mv in state.get_possible_moves() {
                    if let HiveMove::Place { piece_type: pt, to } = mv {
                        if pt == piece_type {
                            self.valid_positions.insert(to);
                        }
                    }
                }
            }
            InputMode::MovePiece(from) => {
                // Find valid destinations for the selected piece
                for mv in state.get_possible_moves() {
                    if let HiveMove::Move { from: f, to } = mv {
                        if f == from {
                            self.valid_positions.insert(to);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Get piece type from key code
    fn get_piece_from_key(&self, key: u32) -> Option<PieceType> {
        match key {
            vk::VK_Q => Some(PieceType::Queen),
            vk::VK_B => Some(PieceType::Beetle),
            vk::VK_S => Some(PieceType::Spider),
            vk::VK_G => Some(PieceType::Grasshopper),
            vk::VK_A => Some(PieceType::Ant),
            _ => None,
        }
    }

    /// Check if the player must place queen (on their 4th piece)
    fn must_place_queen(&self, state: &HiveState) -> bool {
        let player = state.get_current_player();
        
        // Check if queen is placed
        !state.is_queen_placed(player)
            && state.get_hex_board()
                .values()
                .flatten()
                .filter(|p| p.player == player)
                .count() >= 3
    }
}

impl GameRenderer for HiveRenderer {
    fn render(&self, renderer: &Renderer, game: &GameWrapper, area: Rect) {
        let GameWrapper::Hive(state) = game else { return };
        
        // Split area: 80% board, 20% piece panel
        let panel_width = (area.width * 0.2).max(150.0).min(200.0);
        let board_area = Rect::new(area.x, area.y, area.width - panel_width - 10.0, area.height);
        let panel_area = Rect::new(area.x + area.width - panel_width, area.y, panel_width, area.height);
        
        // Draw board background
        renderer.fill_rect(board_area, HEX_BG_COLOR);
        
        // Calculate layout using board_view tilt/rotation
        let layout = HexLayout::calculate(board_area, state);
        
        // Set board transform (tilt + rotation around center)
        self.board_view.begin_draw(renderer, layout.center_x, layout.center_y);
        
        // Draw hexagonal grid background
        self.draw_hex_grid(renderer, &layout, state, board_area);
        
        // Get last move for highlighting
        let last_move_coords: Option<HexCoord> = match state.get_last_move() {
            Some(coords) if !coords.is_empty() => {
                // Convert back from grid coords
                let (r, c) = coords[0];
                Some(HexCoord::new(c as i32 - 10, r as i32 - 10))
            }
            _ => None,
        };
        
        // Draw valid positions
        for coord in &self.valid_positions {
            let is_hovered = self.hover_hex.as_ref() == Some(coord);
            self.draw_valid_position(renderer, &layout, coord.q, coord.r, is_hovered);
        }
        
        // Collect all pieces with their positions for depth-sorted rendering
        // Tuple: (q, r, level, piece, is_top, is_selected, is_last_move)
        let mut pieces_to_draw: Vec<(i32, i32, usize, &Piece, bool, bool, bool)> = Vec::new();
        
        for (coord, stack) in state.get_hex_board() {
            let is_selected_stack = matches!(self.mode, InputMode::MovePiece(from) if from == *coord);
            let is_last = last_move_coords.as_ref() == Some(coord);
            
            // Draw ALL pieces in the stack, not just the top one
            for (level, piece) in stack.iter().enumerate() {
                let is_top = level == stack.len() - 1;
                let is_selected = is_selected_stack && is_top;
                let is_last_move = is_last && is_top;
                pieces_to_draw.push((coord.q, coord.r, level, piece, is_top, is_selected, is_last_move));
            }
        }
        
        // Sort by r (depth), then by stack level, so pieces in front are drawn last
        pieces_to_draw.sort_by(|a, b| {
            // First sort by r (lower r = further back, drawn first)
            match a.1.cmp(&b.1) {
                std::cmp::Ordering::Equal => {
                    // Then sort by stack level (lower level = drawn first)
                    a.2.cmp(&b.2)
                }
                other => other,
            }
        });
        
        // Draw all pieces in depth order
        for (q, r, level, piece, is_top, is_selected, is_last_move) in pieces_to_draw {
            self.draw_piece(renderer, &layout, q, r, level, piece, is_top, is_selected, is_last_move);
        }
        
        // Draw hover indicator
        if let Some(hover) = &self.hover_hex {
            if !self.valid_positions.contains(hover) && !state.get_hex_board().contains_key(hover) {
                // Just show a subtle hover on empty space
                let (cx, cy) = layout.hex_to_screen(hover.q, hover.r);
                renderer.draw_iso_hexagon(cx, cy, layout.hex_size * 0.3, with_alpha(Colors::TEXT_SECONDARY, 0.3), 1.0);
            }
        }
        
        // Reset transform before drawing UI elements
        self.board_view.end_draw(renderer);
        
        // Draw piece selection panel
        self.draw_piece_panel(renderer, state, panel_area);
        
        // Draw turn indicator
        let turn_rect = Rect::new(board_area.x + 10.0, board_area.y + 10.0, 150.0, 25.0);
        let turn_text = format!("Turn {}", state.get_turn());
        renderer.draw_text(&turn_text, turn_rect, Colors::TEXT_PRIMARY, false);
        
        // Draw queen warning if needed
        if self.must_place_queen(state) {
            let warning_rect = Rect::new(board_area.x + 10.0, board_area.y + 35.0, 200.0, 25.0);
            renderer.draw_text("Must place Queen!", warning_rect, Colors::STATUS_ERROR, false);
        }

        // Draw Reset Zoom button if zoomed
        if (self.board_view.scale() - 1.0).abs() > 0.01 {
            let reset_rect = Rect::new(board_area.x + board_area.width - 110.0, board_area.y + 10.0, 100.0, 30.0);
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
        let GameWrapper::Hive(state) = game else { return InputResult::None };
        
        // Calculate layout for coordinate conversion
        let panel_width = (area.width * 0.2).max(150.0).min(200.0);
        let board_area = Rect::new(area.x, area.y, area.width - panel_width - 10.0, area.height);
        let layout = HexLayout::calculate(board_area, state);

        // Let board_view handle drag inputs for tilt/rotation
        if let Some(result) = self.board_view.handle_input(&input, layout.center_x, layout.center_y) {
            return result;
        }

        match input {
            GameInput::Click { x, y } => {
                // Check Reset Zoom button
                if (self.board_view.scale() - 1.0).abs() > 0.01 {
                    let reset_rect = Rect::new(board_area.x + board_area.width - 110.0, board_area.y + 10.0, 100.0, 30.0);
                    if x >= reset_rect.x && x <= reset_rect.x + reset_rect.width &&
                       y >= reset_rect.y && y <= reset_rect.y + reset_rect.height {
                        self.board_view.reset_zoom();
                        return InputResult::Redraw;
                    }
                }

                // Calculate panel area for hit testing
                let panel_area = Rect::new(area.x + area.width - panel_width, area.y, panel_width, area.height);
                
                // Check if click is in panel area
                if x >= panel_area.x {
                    if let Some(piece_type) = self.get_piece_at_panel_click(x, y, panel_area) {
                        let count = state.pieces_in_hand(state.get_current_player(), piece_type);
                        if count > 0 {
                            // Check if must place queen
                            if self.must_place_queen(state) && piece_type != PieceType::Queen {
                                return InputResult::None;
                            }
                            
                            self.mode = InputMode::PlacePiece(piece_type);
                            self.update_valid_positions(state);
                            return InputResult::Redraw;
                        }
                    }
                    return InputResult::None;
                }
                
                // Check if click is in board area
                if x < board_area.x + board_area.width {
                    let (lx, ly) = self.board_view.screen_to_local(x, y, layout.center_x, layout.center_y);
                    let hex = layout.local_to_hex(lx, ly);
                    
                    match self.mode {
                        InputMode::SelectPiece | InputMode::SelectMove => {
                            // Check if clicking on own piece to move it
                            if let Some(stack) = state.get_hex_board().get(&hex) {
                                if let Some(top) = stack.last() {
                                    if top.player == state.get_current_player() && state.is_queen_placed(state.get_current_player()) {
                                        self.mode = InputMode::MovePiece(hex);
                                        self.update_valid_positions(state);
                                        return InputResult::Redraw;
                                    }
                                }
                            }
                        }
                        InputMode::PlacePiece(piece_type) => {
                            if self.valid_positions.contains(&hex) {
                                let mv = HiveMove::Place { piece_type, to: hex };
                                self.mode = InputMode::SelectPiece;
                                self.valid_positions.clear();
                                return InputResult::Move(MoveWrapper::Hive(mv));
                            }
                        }
                        InputMode::MovePiece(from) => {
                            if self.valid_positions.contains(&hex) {
                                let mv = HiveMove::Move { from, to: hex };
                                self.mode = InputMode::SelectPiece;
                                self.valid_positions.clear();
                                return InputResult::Move(MoveWrapper::Hive(mv));
                            } else if hex == from {
                                // Clicked same piece - deselect
                                self.mode = InputMode::SelectMove;
                                self.valid_positions.clear();
                                return InputResult::Redraw;
                            }
                        }
                    }
                }
                InputResult::None
            }
            
            GameInput::Hover { x, y } => {
                let old_hover = self.hover_hex;
                
                if x < board_area.x + board_area.width {
                    let (lx, ly) = self.board_view.screen_to_local(x, y, layout.center_x, layout.center_y);
                    self.hover_hex = Some(layout.local_to_hex(lx, ly));
                } else {
                    self.hover_hex = None;
                }
                
                if old_hover != self.hover_hex {
                    return InputResult::Redraw;
                }
                InputResult::None
            }
            
            GameInput::Key { code, pressed } => {
                if !pressed {
                    return InputResult::None;
                }
                
                match code {
                    vk::VK_ESCAPE => {
                        // Cancel current selection
                        self.mode = InputMode::SelectPiece;
                        self.valid_positions.clear();
                        return InputResult::Redraw;
                    }
                    _ => {
                        // Check for piece selection keys
                        if let Some(piece_type) = self.get_piece_from_key(code) {
                            let count = state.pieces_in_hand(state.get_current_player(), piece_type);
                            if count > 0 {
                                // Check if must place queen
                                if self.must_place_queen(state) && piece_type != PieceType::Queen {
                                    return InputResult::None;
                                }
                                
                                self.mode = InputMode::PlacePiece(piece_type);
                                self.update_valid_positions(state);
                                return InputResult::Redraw;
                            }
                        }
                    }
                }
                InputResult::None
            }
            
            GameInput::Drag { .. } | GameInput::RightDown { .. } | GameInput::RightUp { .. } | GameInput::Wheel { .. } => {
                // Handled by board_view.handle_input above
                InputResult::None
            }
        }
    }

    fn game_name(&self) -> &'static str {
        "Hive"
    }

    fn game_description(&self) -> &'static str {
        "Surround the opponent's Queen Bee!"
    }

    fn player_name(&self, player_id: i32) -> String {
        match player_id {
            1 => "Black".to_string(),
            -1 => "White".to_string(),
            _ => format!("Player {}", player_id),
        }
    }

    fn reset(&mut self) {
        self.mode = InputMode::SelectPiece;
        self.hover_hex = None;
        self.valid_positions.clear();
        self.last_layout = None;
        self.board_view.reset_view();
    }
}
