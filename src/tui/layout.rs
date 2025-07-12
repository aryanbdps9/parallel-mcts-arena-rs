//! # Layout Module
//!
//! This module handles dynamic layout calculations and panel resizing logic
//! for the terminal user interface. It provides a flexible system for creating
//! resizable UI panels that users can adjust by dragging boundaries.
//!
//! ## Key Features
//! - **Percentage-based Layouts**: Configurable panel sizes as percentages
//! - **Drag-and-Drop Resizing**: Interactive boundary dragging for panel adjustment
//! - **Game-Specific Layouts**: Specialized layouts for different game types
//! - **Responsive Design**: Automatic adjustment to terminal size changes
//!
//! ## Layout Types
//! - **Standard Layout**: 2-panel vertical split for 2-player games
//! - **Blokus Layout**: 3-panel horizontal split for 4-player Blokus game
//! - **Settings Layout**: Menu-based layouts for configuration screens

use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// Defines which boundary can be dragged for resizing panels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DragBoundary {
    /// Boundary between game board and bottom panel
    BoardBottom,
    /// Left boundary of Blokus piece selection panel
    BlokusPieceSelectionLeft,
    /// Right boundary of Blokus piece selection panel
    BlokusPieceSelectionRight,
}

/// Configuration for resizable layout areas
///
/// Stores the current panel sizes as percentages and absolute values,
/// allowing for persistent layout customization across game sessions.
pub struct LayoutConfig {
    /// Percentage of height for the board area (0-100)
    pub board_height_percent: u8,
    /// Width of Blokus piece selection panel
    pub blokus_piece_selection_width: u16,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            board_height_percent: 30,  // Lower default since dynamic sizing handles actual requirements
            blokus_piece_selection_width: 50,
        }
    }
}

impl LayoutConfig {
    /// Calculates the main vertical layout areas for standard games
    ///
    /// Uses dynamic sizing based on actual board requirements rather than fixed percentages.
    /// Ensures the board always has adequate space while maximizing debug panel space.
    ///
    /// # Arguments
    /// * `area` - Total screen area to divide
    /// * `board_height` - Number of rows in the game board
    /// * `board_width` - Number of columns in the game board  
    /// * `game_type` - Type of game (affects space requirements)
    ///
    /// # Returns
    /// Tuple of (board_area, bottom_area) rectangles
    pub fn get_main_layout_dynamic(&self, area: Rect, board_height: usize, board_width: usize, game_type: &str) -> (Rect, Rect) {
        let required_board_height = Self::calculate_required_board_height(board_height, board_width, game_type);
        
        // Use the larger of: required height or user's dragged preference (but capped)
        let board_height_from_percent = (area.height as f32 * self.board_height_percent as f32 / 100.0) as u16;
        let actual_board_height = required_board_height.max(board_height_from_percent).min(area.height - 10); // Leave at least 10 rows for bottom
        
        let bottom_height = area.height.saturating_sub(actual_board_height);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(actual_board_height),
                Constraint::Length(bottom_height),
            ])
            .split(area);

        (chunks[0], chunks[1])
    }

    /// Calculates Blokus-specific horizontal layout
    ///
    /// Creates a three-panel horizontal layout: game board on the left,
    /// piece selection panel in the center, and player status on the right.
    /// Optimized for the unique requirements of 4-player Blokus gameplay.
    ///
    /// # Arguments
    /// * `area` - Total area to divide horizontally
    ///
    /// # Returns
    /// Tuple of (board_area, piece_selection_area, player_status_area) rectangles
    pub fn get_blokus_layout(&self, area: Rect) -> (Rect, Rect, Rect) {
        let player_status_width = 20;
        let board_width = area.width
            .saturating_sub(self.blokus_piece_selection_width + player_status_width)
            .max(40);

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(board_width),
                Constraint::Length(self.blokus_piece_selection_width),
                Constraint::Length(player_status_width),
            ])
            .split(area);

        (chunks[0], chunks[1], chunks[2])
    }

    /// Get boundary positions for drag detection
    pub fn get_drag_boundaries(&self, terminal_size: Rect) -> Vec<(DragBoundary, u16, u16)> {
        let mut boundaries = Vec::new();

        let board_height = (terminal_size.height as f32 * self.board_height_percent as f32 / 100.0) as u16;

        // Vertical boundaries
        boundaries.push((DragBoundary::BoardBottom, 0, board_height));

        boundaries
    }

    /// Detect which boundary is being clicked
    pub fn detect_boundary_click(&self, col: u16, row: u16, terminal_size: Rect, is_blokus: bool) -> Option<DragBoundary> {
        if is_blokus {
            // Blokus-specific boundaries
            let player_status_width = 20;
            let board_width = terminal_size.width
                .saturating_sub(self.blokus_piece_selection_width + player_status_width)
                .max(40);
            let left_boundary = board_width;
            let right_boundary = board_width + self.blokus_piece_selection_width;

            if col.abs_diff(left_boundary) <= 2 {
                return Some(DragBoundary::BlokusPieceSelectionLeft);
            }
            if col.abs_diff(right_boundary) <= 2 {
                return Some(DragBoundary::BlokusPieceSelectionRight);
            }
        }

        let boundaries = self.get_drag_boundaries(terminal_size);
        for (boundary_type, _boundary_col, boundary_row) in boundaries {
            match boundary_type {
                DragBoundary::BoardBottom => {
                    if row.abs_diff(boundary_row) <= 1 {
                        return Some(boundary_type);
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Handle drag events to resize panels
    pub fn handle_drag(&mut self, boundary: DragBoundary, delta: i16, _terminal_size: Rect) {
        match boundary {
            DragBoundary::BoardBottom => {
                let new_percent = ((self.board_height_percent as i16 + delta).max(20).min(80)) as u8;
                self.board_height_percent = new_percent;
            }
            DragBoundary::BlokusPieceSelectionLeft => {
                let new_width = ((self.blokus_piece_selection_width as i16 + delta).max(30).min(80)) as u16;
                self.blokus_piece_selection_width = new_width;
            }
            DragBoundary::BlokusPieceSelectionRight => {
                let new_width = ((self.blokus_piece_selection_width as i16 - delta).max(30).min(80)) as u16;
                self.blokus_piece_selection_width = new_width;
            }
        }
    }

    /// Calculate the minimum height needed for the game board based on game type and board size
    ///
    /// Takes into account the actual board dimensions and adds space for borders and labels.
    /// This ensures the board always has enough space regardless of terminal size.
    ///
    /// # Arguments
    /// * `board_height` - Number of rows in the game board
    /// * `board_width` - Number of columns in the game board  
    /// * `game_type` - Type of game (affects label requirements)
    ///
    /// # Returns
    /// Minimum height in terminal rows needed for the board area
    pub fn calculate_required_board_height(board_height: usize, board_width: usize, game_type: &str) -> u16 {
        if board_height == 0 || board_width == 0 {
            return 6; // Minimum fallback height
        }

        let mut required_height = 2; // Top and bottom borders
        
        // Add space for column labels (1 row)
        required_height += 1;
        
        // Add space for each board row (1 row per board row)
        required_height += board_height as u16;
        
        // Add small padding for visual spacing
        required_height += 1;
        
        // Game-specific adjustments
        match game_type {
            "blokus" => required_height += 2, // Blokus needs extra space
            "connect4" => {}, // Connect4 is more compact
            _ => {}, // Standard games
        }
        
        // Ensure minimum usable height
        required_height.max(8)
    }

    /// Calculate the required board width (for future horizontal layout optimizations)
    pub fn calculate_required_board_width(board_height: usize, board_width: usize, game_type: &str) -> u16 {
        if board_height == 0 || board_width == 0 {
            return 20; // Minimum fallback width
        }

        let col_width = match game_type {
            "connect4" | "othello" => 2,
            _ => 2, // Standard width
        };
        
        let needs_row_labels = game_type != "connect4";
        let row_label_width = if needs_row_labels { 2 } else { 0 };
        
        let required_width = 2 + // Left and right borders
                           row_label_width + // Row labels if needed
                           (board_width as u16 * col_width); // Board columns
        
        required_width.max(20)
    }

    /// Legacy method for backward compatibility - uses dynamic sizing now
    pub fn get_main_layout(&self, area: Rect) -> (Rect, Rect) {
        // Use reasonable defaults when we don't have game info
        self.get_main_layout_dynamic(area, 15, 15, "gomoku")
    }
}
