//! Piece visualization utilities for Blokus pieces.
//!
//! This module handles the complex task of converting Blokus piece shapes into
//! visual text representations that can be displayed in a terminal UI.
//!
//! **Key challenges:**
//! 1. **Shape normalization**: Piece coordinates can be negative or scattered
//! 2. **Size constraints**: Must fit within fixed cell width/height limits  
//! 3. **Visual clarity**: Pieces must be recognizable and distinguishable
//! 4. **Consistent sizing**: All pieces should use the same amount of screen space
//!
//! **Design decisions:**
//! - Use "██" (solid blocks) to represent piece cells for high visibility
//! - Normalize all piece shapes to start from (0,0) for consistent positioning
//! - Apply padding to center pieces within their allocated space
//! - Handle edge cases like oversized pieces gracefully

/// Handles visualization of Blokus piece shapes
///
/// Each Blokus piece is defined as a list of (row, col) coordinates representing
/// the cells it occupies. This visualizer converts those coordinates into a
/// text-based representation suitable for terminal display.
///
/// Example piece shape conversion:
/// ```
/// Input: [(0,0), (0,1), (1,1)]  // L-shaped piece
/// Output: ["██  ", "  ██"]       // Visual representation  
/// ```
pub struct PieceVisualizer {
    /// Maximum width (in characters) that each piece visualization can use
    /// This ensures consistent grid alignment regardless of piece complexity
    piece_width: usize,
}

impl PieceVisualizer {
    /// Creates a new piece visualizer with the specified width constraint
    pub fn new(piece_width: usize) -> Self {
        Self { piece_width }
    }

    /// Create visual representation of a piece shape (simplified)
    ///
    /// This method converts a list of (row, col) coordinates into a visual
    /// text representation that can be displayed in the terminal.
    ///
    /// **Algorithm steps:**
    /// 1. Handle edge case of empty piece
    /// 2. Find the bounding box of the piece (min/max coordinates)
    /// 3. Create a 2D grid large enough to hold the piece
    /// 4. Fill in the piece cells with "██" (solid blocks)
    /// 5. Convert the 2D grid to a list of strings
    ///
    /// **Coordinate system:**
    /// - Input coordinates can be negative or start from any value
    /// - We normalize them to start from (0,0) for consistent display
    /// - Example: piece at [(5,3), (5,4), (6,4)] becomes [(0,0), (0,1), (1,1)]
    ///
    /// Args:
    /// - piece_shape: List of (row, col) coordinates defining the piece
    ///
    /// Returns:
    /// - Vector of strings, each representing one row of the piece visualization
    pub fn create_visual_piece_shape(&self, piece_shape: &[(i32, i32)]) -> Vec<String> {
        // Handle degenerate case: empty piece shape
        if piece_shape.is_empty() {
            return vec!["██".to_string()]; // Show a single block as fallback
        }

        // Step 1: Find the bounding box of the piece
        // This determines how much space we need for the visualization
        let (min_r, max_r, min_c, max_c) = self.get_piece_bounds(piece_shape);

        // Step 2: Calculate the dimensions of our visualization grid
        let height = (max_r - min_r + 1) as usize; // +1 because coordinates are inclusive
        let width = (max_c - min_c + 1) as usize;

        // Step 3: Create a 2D grid filled with empty space
        // Each cell is "  " (two spaces) to match the width of "██"
        let mut grid = vec![vec!["  "; width]; height];

        // Step 4: Fill in the piece cells with solid blocks
        // Normalize coordinates by subtracting the minimum values
        for &(r, c) in piece_shape {
            let gr = (r - min_r) as usize; // Grid row (normalized)
            let gc = (c - min_c) as usize; // Grid column (normalized)

            // Bounds check to prevent panics
            if gr < height && gc < width {
                grid[gr][gc] = "██"; // Place solid block
            }
        }

        // Step 5: Convert the 2D grid to a vector of strings
        // Each row becomes one string in the result
        grid.iter()
            .map(|row| {
                row.iter()
                    .map(|cell| {
                        if *cell == "██" {
                            "██".to_string() // Solid block for piece cells
                        } else {
                            "  ".to_string() // Empty space for non-piece cells
                        }
                    })
                    .collect::<String>()
            })
            .collect()
    }

    /// Get the piece label for display (1-9, 0, a-k)
    ///
    /// **Labeling System Explanation:**
    /// Blokus has 21 different piece shapes (pentominoes and smaller pieces).
    /// To display them compactly in terminal UI, we use a specific labeling:
    /// - Pieces 0-8: Display as "1"-"9" (human-friendly 1-based numbering)
    /// - Piece 9: Display as "0" (wraps around for single digit)
    /// - Pieces 10-20: Display as "a"-"k" (letters for remaining pieces)
    ///
    /// This system ensures:
    /// - All labels are single characters (consistent width)
    /// - Easy visual distinction between pieces
    /// - Intuitive numbering for most common pieces
    ///
    /// Args:
    /// - piece_idx: Zero-based piece index (0-20)
    ///
    /// Returns:
    /// - Single character string label for the piece
    pub fn get_piece_label(&self, piece_idx: usize) -> String {
        if piece_idx < 9 {
            (piece_idx + 1).to_string()
        } else if piece_idx == 9 {
            "0".to_string()
        } else {
            ((b'a' + (piece_idx - 10) as u8) as char).to_string()
        }
    }

    /// Pad content to exact piece width
    ///
    /// **Purpose:** Ensure all piece content has consistent width for grid alignment
    ///
    /// **Why this is necessary:**
    /// - Terminal grids require fixed-width columns for proper alignment
    /// - Piece shapes have varying natural widths (1-5 characters)
    /// - Visual consistency demands uniform cell sizes
    /// - Labels and empty spaces need same width as piece content
    ///
    /// **Padding Algorithm:**
    /// 1. **Too narrow:** Add spaces to both sides, centering the content
    ///    - Split padding evenly between left and right
    ///    - Give extra space to right side if padding is odd
    /// 2. **Too wide:** Truncate content to fit within piece_width
    /// 3. **Exact fit:** Return content unchanged
    ///
    /// **Example scenarios:**
    /// - piece_width=6, content="██": "  ██  " (2 spaces each side)
    /// - piece_width=5, content="██": " ██  " (1 left, 2 right)
    /// - piece_width=4, content="██████": "████" (truncated)
    ///
    /// Args:
    /// - content: Text content to pad (piece shape, label, or empty space)
    ///
    /// Returns:
    /// - String padded/truncated to exactly piece_width characters
    pub fn pad_content_to_width(&self, content: &str) -> String {
        let current_width = content.chars().count();
        if current_width < self.piece_width {
            let total_padding = self.piece_width - current_width;
            let left_padding = total_padding / 2;
            let right_padding = total_padding - left_padding;
            format!(
                "{}{}{}",
                " ".repeat(left_padding),
                content,
                " ".repeat(right_padding)
            )
        } else if current_width > self.piece_width {
            content.chars().take(self.piece_width).collect()
        } else {
            content.to_string()
        }
    }

    /// Create empty padded content
    ///
    /// **Purpose:** Generate empty space with consistent width for grid cells
    ///
    /// This is used when a grid cell needs to be empty but must maintain
    /// the same width as cells containing pieces. Essential for:
    /// - Empty cells in the piece grid layout
    /// - Spacing between piece groups
    /// - Maintaining grid column alignment
    ///
    /// **Why not just use empty string:**
    /// - Empty strings would collapse grid columns
    /// - Terminal UI needs explicit space characters for proper alignment
    /// - Consistent cell sizes prevent visual grid distortion
    ///
    /// Returns:
    /// - String of spaces exactly piece_width characters long
    pub fn create_empty_content(&self) -> String {
        " ".repeat(self.piece_width)
    }

    /// Get bounds of a piece shape
    /// Calculate the bounding box of a piece shape
    ///
    /// Given a list of (row, col) coordinates that define a piece, this method
    /// finds the minimum and maximum row/column values. This is essential for
    /// normalization and determining how much space the piece needs.
    ///
    /// **Example:**
    /// Input: [(5,3), (5,4), (6,4), (7,3)]  // Arbitrary coordinates
    /// Output: (5, 7, 3, 4)                 // min_row, max_row, min_col, max_col
    ///
    /// This means the piece spans:
    /// - Rows 5 to 7 (3 rows total)
    /// - Columns 3 to 4 (2 columns total)  
    ///
    /// Args:
    /// - piece_shape: List of (row, col) coordinates
    ///
    /// Returns:
    /// - (min_row, max_row, min_col, max_col): Bounding box coordinates
    fn get_piece_bounds(&self, piece_shape: &[(i32, i32)]) -> (i32, i32, i32, i32) {
        let min_r = piece_shape.iter().map(|p| p.0).min().unwrap_or(0);
        let max_r = piece_shape.iter().map(|p| p.0).max().unwrap_or(0);
        let min_c = piece_shape.iter().map(|p| p.1).min().unwrap_or(0);
        let max_c = piece_shape.iter().map(|p| p.1).max().unwrap_or(0);
        (min_r, max_r, min_c, max_c)
    }

    /// Get the content for a specific line of a piece visualization
    pub fn get_line_content(
        &self,
        label: &str,
        visual_lines: &[String],
        line_index: usize,
        is_selected: bool,
        show_labels: bool,
    ) -> String {
        if line_index == 0 && show_labels {
            // First line: show label
            let label_text = if is_selected {
                format!("[{}]", label)
            } else {
                format!(" {} ", label)
            };
            format!("{:^width$}", label_text, width = self.piece_width)
        } else {
            // Other lines: show piece shape with padding
            let visual_line_index = if show_labels {
                line_index.saturating_sub(1)
            } else {
                line_index
            };

            if visual_line_index < visual_lines.len() {
                let piece_line = &visual_lines[visual_line_index];
                // Pad to exact width
                let current_width = piece_line.chars().count();
                if current_width < self.piece_width {
                    let total_padding = self.piece_width - current_width;
                    let left_padding = total_padding / 2;
                    let right_padding = total_padding - left_padding;
                    format!(
                        "{}{}{}",
                        " ".repeat(left_padding),
                        piece_line,
                        " ".repeat(right_padding)
                    )
                } else if current_width > self.piece_width {
                    piece_line.chars().take(self.piece_width).collect()
                } else {
                    piece_line.to_string()
                }
            } else {
                // Empty line with proper padding
                " ".repeat(self.piece_width)
            }
        }
    }
}
