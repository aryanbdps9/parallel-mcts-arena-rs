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
//! - **Standard Layout**: 3-panel vertical split for 2-player games
//! - **Blokus Layout**: 3-panel horizontal split for 4-player Blokus game
//! - **Settings Layout**: Menu-based layouts for configuration screens

use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// Defines which boundary can be dragged for resizing panels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DragBoundary {
    /// Boundary between game board and instructions area
    BoardInstructions,
    /// Boundary between instructions and stats area
    InstructionsStats,
    /// Horizontal boundary between debug stats and move history
    StatsHistory,
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
    /// Percentage of height for the instructions area (0-100)
    pub instructions_height_percent: u8,
    /// Percentage of width for stats vs history split (0-100)
    pub stats_width_percent: u8,
    /// Width of Blokus piece selection panel
    pub blokus_piece_selection_width: u16,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            board_height_percent: 65,
            instructions_height_percent: 15,
            stats_width_percent: 50,
            blokus_piece_selection_width: 50,
        }
    }
}

impl LayoutConfig {
    /// Calculates the main vertical layout areas for standard games
    ///
    /// Divides the screen into three vertical sections: board area at top,
    /// game info/instructions in middle, and stats/history at bottom.
    ///
    /// # Arguments
    /// * `area` - Total screen area to divide
    ///
    /// # Returns
    /// Tuple of (board_area, instructions_area, stats_area) rectangles
    pub fn get_main_layout(&self, area: Rect) -> (Rect, Rect, Rect) {
        let board_height = (area.height as f32 * self.board_height_percent as f32 / 100.0) as u16;
        let instructions_height = (area.height as f32 * self.instructions_height_percent as f32 / 100.0) as u16;
        let stats_height = area.height.saturating_sub(board_height + instructions_height);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(board_height),
                Constraint::Length(instructions_height),
                Constraint::Length(stats_height),
            ])
            .split(area);

        (chunks[0], chunks[1], chunks[2])
    }

    /// Calculates the horizontal split for the bottom stats area
    ///
    /// Divides the stats area horizontally between debug statistics on the left
    /// and move history on the right, based on the configured width percentage.
    ///
    /// # Arguments
    /// * `area` - Stats area rectangle to divide
    ///
    /// # Returns
    /// Tuple of (debug_stats_area, move_history_area) rectangles
    pub fn get_stats_layout(&self, area: Rect) -> (Rect, Rect) {
        let stats_width = (area.width as f32 * self.stats_width_percent as f32 / 100.0) as u16;
        let history_width = area.width.saturating_sub(stats_width);

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(stats_width),
                Constraint::Length(history_width),
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
        let instructions_height = (terminal_size.height as f32 * self.instructions_height_percent as f32 / 100.0) as u16;
        let stats_start = board_height + instructions_height;

        // Vertical boundaries
        boundaries.push((DragBoundary::BoardInstructions, 0, board_height));
        boundaries.push((DragBoundary::InstructionsStats, 0, board_height + instructions_height));

        // Horizontal boundary in stats area
        let stats_width_boundary = (terminal_size.width as f32 * self.stats_width_percent as f32 / 100.0) as u16;
        boundaries.push((DragBoundary::StatsHistory, stats_width_boundary, stats_start));

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
        for (boundary_type, boundary_col, boundary_row) in boundaries {
            match boundary_type {
                DragBoundary::BoardInstructions | DragBoundary::InstructionsStats => {
                    if row.abs_diff(boundary_row) <= 1 {
                        return Some(boundary_type);
                    }
                }
                DragBoundary::StatsHistory => {
                    if row > boundary_row && col.abs_diff(boundary_col) <= 2 {
                        return Some(boundary_type);
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Handle drag events to resize panels
    pub fn handle_drag(&mut self, boundary: DragBoundary, delta: i16, terminal_size: Rect) {
        match boundary {
            DragBoundary::BoardInstructions => {
                let new_percent = ((self.board_height_percent as i16 + delta).max(20).min(80)) as u8;
                self.board_height_percent = new_percent;
            }
            DragBoundary::InstructionsStats => {
                let new_percent = ((self.instructions_height_percent as i16 + delta).max(5).min(30)) as u8;
                self.instructions_height_percent = new_percent;
            }
            DragBoundary::StatsHistory => {
                let delta_percent = (delta as f32 / terminal_size.width as f32 * 100.0) as i16;
                let new_percent = ((self.stats_width_percent as i16 + delta_percent).max(20).min(80)) as u8;
                self.stats_width_percent = new_percent;
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
}
