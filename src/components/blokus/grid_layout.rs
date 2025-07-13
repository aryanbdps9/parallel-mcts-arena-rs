//! Grid layout calculation utilities.
//!
//! This module handles the complex logic of arranging Blokus pieces in an optimal grid layout.
//! The main challenge is creating a "near-square" arrangement that looks good and fits the screen.
//!
//! Key concepts:
//! - **Responsive design**: Grid layout changes based on available screen width
//! - **Aspect ratio optimization**: Try to keep the grid close to a 1:1 (square) ratio  
//! - **Constraint satisfaction**: Must respect min/max pieces per row limits
//! - **Dynamic recalculation**: Layout changes when pieces are added/removed or screen resizes

use crate::components::blokus::ResponsivePieceGridConfig;

/// Handles grid layout calculations for optimal piece arrangement
///
/// This component is responsible for answering the question:
/// "Given N pieces and screen constraints, how should I arrange them in a grid?"
///
/// The algorithm tries to find a layout that:
/// 1. Fits within screen width constraints (min_pieces_per_row to max_pieces_per_row)
/// 2. Produces the most "square-like" grid (closest to 1:1 aspect ratio)
/// 3. Efficiently uses available space
pub struct GridLayoutCalculator {
    /// Contains all the constraints and settings that affect layout decisions
    config: ResponsivePieceGridConfig,
}

impl GridLayoutCalculator {
    /// Creates a new layout calculator with the given configuration
    pub fn new(config: ResponsivePieceGridConfig) -> Self {
        Self { config }
    }

    /// Calculate optimal grid layout for near-square arrangement
    ///
    /// This is the core algorithm that determines how many pieces to show per row
    /// and how many total rows are needed for a given number of pieces.
    ///
    /// **Algorithm explanation:**
    /// 1. If no pieces, return minimum configuration
    /// 2. Try each possible number of columns (within min/max constraints)
    /// 3. For each column count, calculate how many rows would be needed
    /// 4. Calculate the aspect ratio (how "square-like" the resulting grid is)
    /// 5. Pick the configuration with the best (closest to 1:1) aspect ratio
    ///
    /// **Why this matters:**
    /// - A 21x1 grid (all pieces in one row) looks terrible and doesn't fit screens
    /// - A 1x21 grid (one piece per row) wastes horizontal space
    /// - A ~5x4 or ~4x5 grid looks balanced and uses space efficiently
    ///
    /// Args:
    /// - piece_count: Total number of pieces to arrange
    ///
    /// Returns:
    /// - (pieces_per_row, total_rows): The optimal grid dimensions
    pub fn calculate_optimal_layout(&self, piece_count: usize) -> (usize, usize) {
        // Handle edge case: no pieces to display
        if piece_count == 0 {
            return (self.config.min_pieces_per_row, 1);
        }

        // Find the layout that produces the most square-like grid
        let mut best_layout = (self.config.min_pieces_per_row, 1);
        let mut best_ratio = f64::INFINITY; // Start with worst possible ratio

        // Try each possible number of columns within our constraints
        for cols in self.config.min_pieces_per_row..=self.config.max_pieces_per_row {
            // Calculate how many rows we'd need for this column count
            // Using ceiling division: (a + b - 1) / b = ceil(a / b)
            let rows = (piece_count + cols - 1) / cols;

            // Calculate how "square-like" this layout is
            // Perfect square has ratio of 1.0 (cols/rows = 1)
            // Very wide grid has high ratio (e.g., 10/2 = 5.0)
            // Very tall grid has low ratio (e.g., 2/10 = 0.2)
            // We want the ratio closest to 1.0
            let ratio = if rows > 0 {
                (cols as f64 / rows as f64 - 1.0).abs() // Distance from perfect 1:1 ratio
            } else {
                f64::INFINITY // Invalid layout
            };

            // If this layout is more square-like than our current best, use it
            if ratio < best_ratio {
                best_ratio = ratio;
                best_layout = (cols, rows);
            }
        }

        best_layout
    }

    /// Calculate max pieces that can fit in available width
    pub fn calculate_max_pieces_for_width(&self, available_width: u16) -> usize {
        let separator_width = 1;
        let border_width = if self.config.show_borders { 2 } else { 0 };
        let usable_width = available_width.saturating_sub(border_width);

        if usable_width > 0 {
            ((usable_width as usize + separator_width)
                / (self.config.piece_width + separator_width))
                .max(1)
        } else {
            1
        }
    }

    /// Calculate the height needed for the grid including separators and internal borders
    pub fn calculate_total_height(&self, total_rows: usize) -> u16 {
        let content_height = total_rows as u16 * self.config.uniform_cell_height as u16;
        // Add height for row separators (one less than total rows)
        let separator_height = if total_rows > 1 {
            total_rows as u16 - 1
        } else {
            0
        };
        // Add height for top and bottom internal grid borders
        let internal_border_height = 2;
        let border_height = if self.config.show_borders { 2 } else { 0 };
        content_height + separator_height + internal_border_height + border_height
    }

    /// Update config with responsive layout constraints
    pub fn update_config_for_width(&mut self, available_width: u16) {
        let max_pieces_that_fit = self.calculate_max_pieces_for_width(available_width);
        let old_max = self.config.max_pieces_per_row;
        self.config.max_pieces_per_row = max_pieces_that_fit
            .min(old_max)
            .max(self.config.min_pieces_per_row);
    }

    /// Update maximum pieces per row based on available width
    ///
    /// This method implements responsive design by adjusting the grid constraints
    /// when the screen size changes. It's called when:
    /// - Terminal window is resized
    /// - UI layout changes (sidebars appear/disappear)
    /// - Component area allocation changes
    ///
    /// **The responsive algorithm:**
    /// 1. Calculate how many pieces can physically fit in the available width
    /// 2. Respect the original max_pieces_per_row limit (don't exceed it)
    /// 3. Ensure we don't go below min_pieces_per_row (maintain minimum usability)
    ///
    /// **Why this is needed:**
    /// - On narrow screens, we might only fit 3 pieces per row
    /// - On wide screens, we might fit 8+ pieces per row  
    /// - Without this, pieces would be cut off or render incorrectly
    ///
    /// Args:
    /// - max_pieces_that_fit: Maximum pieces that can physically fit in current width
    pub fn update_width_constraints(&mut self, max_pieces_that_fit: usize) {
        let old_max = self.config.max_pieces_per_row;

        // Apply constraints in order of priority:
        // 1. Don't exceed what physically fits on screen
        // 2. Don't exceed the original configured maximum
        // 3. Don't go below the configured minimum
        self.config.max_pieces_per_row = max_pieces_that_fit
            .min(old_max) // Respect original limit
            .max(self.config.min_pieces_per_row); // Maintain minimum usability
    }

    /// Get current config
    pub fn get_config(&self) -> &ResponsivePieceGridConfig {
        &self.config
    }

    /// Get mutable config
    pub fn get_config_mut(&mut self) -> &mut ResponsivePieceGridConfig {
        &mut self.config
    }
}
