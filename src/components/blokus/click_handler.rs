//! Click handling utilities for piece grid components.
//!
//! This module handles the complex coordinate calculations needed to map mouse clicks
//! to specific pieces in the Blokus piece grid. The complexity comes from:
//! 1. Variable piece widths and heights
//! 2. Grid borders and separators between cells
//! 3. Uniform cell heights for consistent clicking
//! 4. Multiple layout configurations (responsive grid sizing)

use crate::components::blokus::ResponsivePieceGridConfig;

/// Handles click coordinate calculations for piece grids
///
/// The click handler needs to account for:
/// - Border offsets (if borders are enabled)
/// - Row and column separators
/// - Uniform cell sizing to prevent click detection issues
/// - Dynamic grid layouts (pieces_per_row can change based on screen size)
pub struct ClickHandler {
    /// Configuration containing piece dimensions, border settings, etc.
    config: ResponsivePieceGridConfig,
    /// Current number of pieces displayed per row (can change dynamically)
    pieces_per_row: usize,
    /// Current total number of rows in the grid
    total_rows: usize,
}

impl ClickHandler {
    /// Creates a new click handler with initial layout parameters
    ///
    /// Args:
    /// - config: Contains piece dimensions, border settings, uniform cell height
    /// - pieces_per_row: Initial number of pieces per row (will change with responsive layout)
    /// - total_rows: Initial number of rows (will change as pieces are added/removed)
    pub fn new(
        config: ResponsivePieceGridConfig,
        pieces_per_row: usize,
        total_rows: usize,
    ) -> Self {
        Self {
            config,
            pieces_per_row,
            total_rows,
        }
    }

    /// Update layout parameters when the grid is resized or reconfigured
    ///
    /// This is called whenever:
    /// - Screen size changes (responsive layout)
    /// - Number of available pieces changes
    /// - Grid configuration is modified
    pub fn update_layout(&mut self, pieces_per_row: usize, total_rows: usize) {
        self.pieces_per_row = pieces_per_row;
        self.total_rows = total_rows;
    }

    /// Calculate piece index from click coordinates
    ///
    /// This is the core method that converts raw mouse click coordinates into
    /// a grid position (row, col) that can be used to identify which piece was clicked.
    ///
    /// The coordinate system works as follows:
    /// ```
    /// ┌─────────┬─────────┬─────────┐  <- Top border (if show_borders=true)
    /// │ Piece 0 │ Piece 1 │ Piece 2 │  <- Row 0
    /// ├─────────┼─────────┼─────────┤  <- Row separator
    /// │ Piece 3 │ Piece 4 │ Piece 5 │  <- Row 1
    /// └─────────┴─────────┴─────────┘  <- Bottom border
    /// ```
    ///
    /// Complexity sources:
    /// 1. **Border offset**: If borders are enabled, we need to subtract 1 from x,y
    /// 2. **Internal grid borders**: There's a top border line inside the main border
    /// 3. **Uniform cell height**: Each cell has exactly `uniform_cell_height` lines
    /// 4. **Row separators**: Each row (except last) has a separator line below it
    /// 5. **Column separators**: Each column (except last) has a separator column
    ///
    /// Args:
    /// - local_x, local_y: Mouse coordinates relative to the grid component area
    ///
    /// Returns:
    /// - Some((row, col)): Grid position if click is within valid bounds
    /// - None: If click is outside grid or on a separator
    pub fn calculate_piece_index(&self, local_x: u16, local_y: u16) -> Option<(usize, usize)> {
        // Step 1: Account for outer border offset (the main component border)
        // If show_borders=true, the grid content starts 1 pixel in from the edge
        let click_x = local_x.saturating_sub(if self.config.show_borders { 1 } else { 0 });
        let click_y = local_y.saturating_sub(if self.config.show_borders { 1 } else { 0 });

        // Step 2: Account for the internal grid's top border
        // The grid itself has its own border system independent of the component border
        // This subtracts 1 for the top border line of the internal grid
        let click_y = click_y.saturating_sub(1); // Top border line

        // Step 3: Calculate which row was clicked
        // Each row occupies exactly (uniform_cell_height + 1) lines:
        // - uniform_cell_height lines for the actual content
        // - 1 line for the row separator (except for the last row)
        //
        // Example with uniform_cell_height=3:
        // Lines 0-2: Row 0 content
        // Line 3: Row separator
        // Lines 4-6: Row 1 content
        // Line 7: Row separator
        // Lines 8-10: Row 2 content
        let total_row_height = self.config.uniform_cell_height + 1; // Include row separator
        let row = (click_y as usize) / total_row_height;

        // Step 4: Account for the internal grid's left border
        // Similar to the top border, subtract 1 for the left border column
        let click_x = click_x.saturating_sub(1); // Left border column

        // Step 5: Calculate which column was clicked
        // Each column occupies exactly (piece_width + 1) characters:
        // - piece_width characters for the actual piece content
        // - 1 character for the column separator (except for the last column)
        //
        // Example with piece_width=5:
        // Chars 0-4: Col 0 content
        // Char 5: Column separator
        // Chars 6-10: Col 1 content
        // Char 11: Column separator
        // Chars 12-16: Col 2 content
        let separator_width = 1;
        let total_cell_width = self.config.piece_width + separator_width;
        let col = (click_x as usize) / total_cell_width;

        // Step 6: Validate that the calculated position is within grid bounds
        // This prevents clicking on separators or outside the grid from registering
        if row < self.total_rows && col < self.pieces_per_row {
            Some((row, col))
        } else {
            // Click was outside valid grid area (separator, border, or beyond grid)
            None
        }
    }

    /// Calculate piece index from row and column
    pub fn get_piece_index(&self, row: usize, col: usize) -> usize {
        row * self.pieces_per_row + col
    }
}
