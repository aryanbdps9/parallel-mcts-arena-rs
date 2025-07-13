//! Grid border rendering utilities for piece grid components.

use ratatui::{
    style::{Color, Style},
    text::{Line, Span},
};

/// Handles rendering of grid borders and separators
pub struct GridBorderRenderer {
    pieces_per_row: usize,
    piece_width: usize,
}

impl GridBorderRenderer {
    pub fn new(pieces_per_row: usize, piece_width: usize) -> Self {
        Self {
            pieces_per_row,
            piece_width,
        }
    }

    /// Add top border of the grid
    pub fn add_top_border(&self, all_lines: &mut Vec<Line>) {
        let border_line = self.create_horizontal_border('┌', '┬', '┐');
        all_lines.push(Line::from(Span::styled(
            border_line,
            Style::default().fg(Color::DarkGray),
        )));
    }

    /// Add bottom border of the grid
    pub fn add_bottom_border(&self, all_lines: &mut Vec<Line>) {
        let border_line = self.create_horizontal_border('└', '┴', '┘');
        all_lines.push(Line::from(Span::styled(
            border_line,
            Style::default().fg(Color::DarkGray),
        )));
    }

    /// Add a horizontal row separator line
    pub fn add_row_separator(&self, all_lines: &mut Vec<Line>) {
        let separator_line = self.create_horizontal_border('├', '┼', '┤');
        all_lines.push(Line::from(Span::styled(
            separator_line,
            Style::default().fg(Color::DarkGray),
        )));
    }

    /// Create a horizontal border line with specified corner and junction characters
    fn create_horizontal_border(
        &self,
        left_char: char,
        junction_char: char,
        right_char: char,
    ) -> String {
        let mut border_chars = Vec::new();
        border_chars.push(left_char);

        for col in 0..self.pieces_per_row {
            // Add horizontal line for this piece cell
            for _ in 0..self.piece_width {
                border_chars.push('─');
            }

            // Add junction or right corner
            if col < self.pieces_per_row - 1 {
                border_chars.push(junction_char);
            } else {
                border_chars.push(right_char);
            }
        }

        border_chars.into_iter().collect()
    }

    /// Add left border to a line
    pub fn add_left_border(line_spans: &mut Vec<Span>) {
        line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
    }

    /// Add right border to a line
    pub fn add_right_border(line_spans: &mut Vec<Span>) {
        line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
    }

    /// Add column separator between pieces
    pub fn add_column_separator(line_spans: &mut Vec<Span>) {
        line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
    }
}
