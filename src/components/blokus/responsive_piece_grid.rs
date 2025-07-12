//! Responsive piece grid component with uniform layout for accurate click detection.

use ratatui::{
    layout::Rect,
    Frame,
    widgets::{Block, Borders, Paragraph},
    style::{Style, Color, Modifier},
    text::{Line, Span},
};
use std::any::Any;
use std::collections::HashSet;
use mcts::GameState;

use crate::app::App;
use crate::game_wrapper::GameWrapper;
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crate::games::blokus::get_blokus_pieces;

/// Configuration for the responsive piece grid
#[derive(Clone)]
pub struct ResponsivePieceGridConfig {
    pub player_color: Color,
    pub min_pieces_per_row: usize,
    pub max_pieces_per_row: usize,
    pub piece_width: usize,
    pub piece_height: usize,
    pub show_borders: bool,
    pub show_labels: bool,
    pub compact_mode: bool,
    pub empty_cell_light: Color,
    pub empty_cell_dark: Color,
    /// Fixed cell height ensures uniform sizing for accurate click detection
    pub uniform_cell_height: usize,
}

impl Default for ResponsivePieceGridConfig {
    fn default() -> Self {
        Self {
            player_color: Color::White,
            min_pieces_per_row: 3,
            max_pieces_per_row: 8,
            piece_width: 8,     // Width for piece display
            piece_height: 4,    // Height for piece shapes
            show_borders: true,
            show_labels: true,
            compact_mode: false,
            empty_cell_light: Color::Rgb(100, 100, 100),
            empty_cell_dark: Color::Rgb(60, 60, 60),
            uniform_cell_height: 5,  // Fixed height: 4 for piece + 1 for label
        }
    }
}

/// Responsive piece grid that optimally arranges pieces in a near-square grid
pub struct ResponsivePieceGridComponent {
    id: ComponentId,
    player: u8,
    config: ResponsivePieceGridConfig,
    available_pieces: Vec<usize>,
    selected_piece: Option<usize>,
    is_current_player: bool,
    area: Option<Rect>,
    pieces_per_row: usize,
    total_rows: usize,
}

impl ResponsivePieceGridComponent {
    pub fn new(player: u8, config: ResponsivePieceGridConfig) -> Self {
        let pieces_per_row = config.max_pieces_per_row;
        Self {
            id: ComponentId::new(),
            player,
            config,
            available_pieces: Vec::new(),
            selected_piece: None,
            is_current_player: false,
            area: None,
            pieces_per_row,
            total_rows: 1,
        }
    }

    pub fn set_available_pieces(&mut self, pieces: Vec<usize>) {
        self.available_pieces = pieces;
        self.update_layout();
    }

    pub fn set_selected_piece(&mut self, piece: Option<usize>) {
        self.selected_piece = piece;
    }

    pub fn set_current_player(&mut self, is_current: bool) {
        self.is_current_player = is_current;
    }

    pub fn get_config(&self) -> &ResponsivePieceGridConfig {
        &self.config
    }

    pub fn get_area(&self) -> Option<Rect> {
        self.area
    }

    pub fn set_area(&mut self, area: Option<Rect>) {
        self.area = area;
    }

    /// Calculate optimal grid layout for near-square arrangement
    fn update_layout(&mut self) {
        let piece_count = self.available_pieces.len();
        if piece_count == 0 {
            self.pieces_per_row = self.config.min_pieces_per_row;
            self.total_rows = 1;
            return;
        }

        // Find the layout that produces the most square-like grid
        let mut best_layout = (self.config.min_pieces_per_row, 1);
        let mut best_ratio = f64::INFINITY;

        for cols in self.config.min_pieces_per_row..=self.config.max_pieces_per_row {
            let rows = (piece_count + cols - 1) / cols; // Ceiling division
            let ratio = if rows > 0 {
                (cols as f64 / rows as f64 - 1.0).abs() // How far from 1:1 ratio
            } else {
                f64::INFINITY
            };

            if ratio < best_ratio {
                best_ratio = ratio;
                best_layout = (cols, rows);
            }
        }

        self.pieces_per_row = best_layout.0;
        self.total_rows = best_layout.1;
    }

    /// Update layout based on available width for responsive design
    fn update_responsive_layout(&mut self, available_width: u16) {
        let separator_width = 1;
        let border_width = if self.config.show_borders { 2 } else { 0 };
        let usable_width = available_width.saturating_sub(border_width);
        
        if usable_width > 0 {
            // Calculate max pieces that can fit
            let max_pieces_that_fit = ((usable_width as usize + separator_width) / (self.config.piece_width + separator_width)).max(1);
            
            // Constrain to config limits
            let old_max = self.config.max_pieces_per_row;
            self.config.max_pieces_per_row = max_pieces_that_fit.min(old_max).max(self.config.min_pieces_per_row);
            
            // Recalculate layout with new constraints
            self.update_layout();
        }
    }

    /// Create visual representation of a piece shape (simplified)
    fn create_visual_piece_shape(&self, piece_shape: &[(i32, i32)]) -> Vec<String> {
        if piece_shape.is_empty() {
            return vec!["██".to_string()];
        }

        let min_r = piece_shape.iter().map(|p| p.0).min().unwrap_or(0);
        let max_r = piece_shape.iter().map(|p| p.0).max().unwrap_or(0);
        let min_c = piece_shape.iter().map(|p| p.1).min().unwrap_or(0);
        let max_c = piece_shape.iter().map(|p| p.1).max().unwrap_or(0);

        let height = (max_r - min_r + 1) as usize;
        let width = (max_c - min_c + 1) as usize;

        // Create a simplified grid
        let mut grid = vec![vec!["  "; width]; height];

        for &(r, c) in piece_shape {
            let gr = (r - min_r) as usize;
            let gc = (c - min_c) as usize;
            if gr < height && gc < width {
                grid[gr][gc] = "██"; // Double block characters
            }
        }

        // Convert to vector of strings
        grid.iter()
            .map(|row| {
                row.iter()
                    .map(|cell| {
                        if *cell == "██" {
                            "██".to_string()
                        } else {
                            "  ".to_string() // Simple empty space
                        }
                    })
                    .collect::<String>()
            })
            .collect()
    }

    /// Get the piece label for display (1-9, 0, a-k)
    fn get_piece_label(&self, piece_idx: usize) -> String {
        if piece_idx < 9 {
            (piece_idx + 1).to_string()
        } else if piece_idx == 9 {
            "0".to_string()
        } else {
            ((b'a' + (piece_idx - 10) as u8) as char).to_string()
        }
    }

    /// Simple, predictable click handling with separator awareness and grid borders
    pub fn handle_piece_click(&mut self, app: &mut App, local_x: u16, local_y: u16) -> Option<usize> {
        if self.area.is_none() { return None; }
        
        // Account for border offset
        let click_x = local_x.saturating_sub(if self.config.show_borders { 1 } else { 0 });
        let click_y = local_y.saturating_sub(if self.config.show_borders { 1 } else { 0 });

        // Account for the top border of the internal grid
        let click_y = click_y.saturating_sub(1); // Top border line

        // Calculate row accounting for row separators
        // Each row takes uniform_cell_height lines + 1 line for row separator (except last row)
        let total_row_height = self.config.uniform_cell_height + 1; // Include row separator
        let row = (click_y as usize) / total_row_height;
        
        // Account for the left border of the internal grid
        let click_x = click_x.saturating_sub(1); // Left border column
        
        // Calculate column based on fixed piece width + separator
        let separator_width = 1;
        let total_cell_width = self.config.piece_width + separator_width;
        let col = (click_x as usize) / total_cell_width;

        // Calculate piece index
        let piece_index = row * self.pieces_per_row + col;
        
        // Check if this piece exists and is available
        if piece_index < self.available_pieces.len() && row < self.total_rows && col < self.pieces_per_row {
            let actual_piece_idx = self.available_pieces[piece_index];
            
            // Only allow selection for current player
            if self.is_current_player {
                app.blokus_ui_config.select_piece(actual_piece_idx);
                return Some(actual_piece_idx);
            }
        }
        
        None
    }

    /// Calculate the height needed for this grid including separators and internal borders
    pub fn calculate_height(&self) -> u16 {
        let content_height = self.total_rows as u16 * self.config.uniform_cell_height as u16;
        // Add height for row separators (one less than total rows)
        let separator_height = if self.total_rows > 1 { self.total_rows as u16 - 1 } else { 0 };
        // Add height for top and bottom internal grid borders
        let internal_border_height = 2;
        let border_height = if self.config.show_borders { 2 } else { 0 };
        content_height + separator_height + internal_border_height + border_height
    }

    /// Render a single row of pieces with uniform cell heights
    fn render_piece_row(
        &self,
        all_lines: &mut Vec<Line>,
        pieces_in_row: &[usize],
        pieces: &[crate::games::blokus::Piece],
        available_set: &HashSet<usize>,
    ) {
        // Create piece data for this row
        let mut pieces_data = Vec::new();
        for piece_idx in pieces_in_row {
            if *piece_idx < pieces.len() {
                let piece = &pieces[*piece_idx];
                let piece_shape = if !piece.transformations.is_empty() {
                    &piece.transformations[0]
                } else {
                    continue;
                };

                let is_available = available_set.contains(piece_idx);
                let is_selected = self.selected_piece == Some(*piece_idx);
                
                let piece_visual_lines = self.create_visual_piece_shape(piece_shape);
                let piece_label = self.get_piece_label(*piece_idx);

                let style = if is_selected {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD).bg(Color::DarkGray)
                } else if is_available {
                    Style::default().fg(self.config.player_color).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM)
                };

                pieces_data.push((piece_label, piece_visual_lines, style));
            }
        }

        if pieces_data.is_empty() {
            return;
        }

        // Render exactly uniform_cell_height lines for this row
        for line_index in 0..self.config.uniform_cell_height {
            let mut line_spans = Vec::new();
            
            // Add left border
            line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
            
            for col in 0..self.pieces_per_row {
                let content = if col < pieces_data.len() {
                    // We have a piece in this column
                    let (piece_label, piece_visual_lines, _) = &pieces_data[col];
                    
                    if line_index == 0 && self.config.show_labels {
                        // First line: show label
                        let label_text = if self.selected_piece == Some(pieces_in_row[col]) {
                            format!("[{}]", piece_label)
                        } else {
                            format!(" {} ", piece_label)
                        };
                        format!("{:^width$}", label_text, width = self.config.piece_width)
                    } else {
                        // Other lines: show piece shape with padding
                        let visual_line_index = if self.config.show_labels {
                            line_index.saturating_sub(1)
                        } else {
                            line_index
                        };
                        
                        if visual_line_index < piece_visual_lines.len() {
                            let piece_line = &piece_visual_lines[visual_line_index];
                            // Pad to exact width
                            let current_width = piece_line.chars().count();
                            if current_width < self.config.piece_width {
                                let total_padding = self.config.piece_width - current_width;
                                let left_padding = total_padding / 2;
                                let right_padding = total_padding - left_padding;
                                format!("{}{}{}", 
                                    " ".repeat(left_padding), 
                                    piece_line, 
                                    " ".repeat(right_padding)
                                )
                            } else if current_width > self.config.piece_width {
                                piece_line.chars().take(self.config.piece_width).collect()
                            } else {
                                piece_line.to_string()
                            }
                        } else {
                            // Empty line with proper padding
                            " ".repeat(self.config.piece_width)
                        }
                    }
                } else {
                    // Empty cell - no piece in this column
                    " ".repeat(self.config.piece_width)
                };

                // Apply styling to the content
                let style = if col < pieces_data.len() {
                    pieces_data[col].2
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                line_spans.push(Span::styled(content, style));
                
                // Add separator between columns (extend to full grid width)
                if col < self.pieces_per_row - 1 {
                    line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
                }
            }
            
            // Add right border
            line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
            
            all_lines.push(Line::from(line_spans));
        }
    }

    /// Add a horizontal row separator line across the full grid width with proper junctions
    fn add_row_separator(&self, all_lines: &mut Vec<Line>) {
        let mut separator_chars = Vec::new();
        separator_chars.push('├'); // Left border junction
        
        // Build the separator character by character to ensure proper alignment
        for col in 0..self.pieces_per_row {
            // Add horizontal line for this piece cell
            for _ in 0..self.config.piece_width {
                separator_chars.push('─');
            }
            
            // Add vertical line separator (junction) after each column except the last
            if col < self.pieces_per_row - 1 {
                separator_chars.push('┼'); // Cross junction for intersection
            }
        }
        
        separator_chars.push('┤'); // Right border junction
        
        let separator_line: String = separator_chars.into_iter().collect();
        all_lines.push(Line::from(Span::styled(separator_line, Style::default().fg(Color::DarkGray))));
    }

    /// Add top border of the grid
    fn add_grid_top_border(&self, all_lines: &mut Vec<Line>) {
        let mut border_chars = Vec::new();
        border_chars.push('┌'); // Top-left corner
        
        for col in 0..self.pieces_per_row {
            // Add horizontal line for this piece cell
            for _ in 0..self.config.piece_width {
                border_chars.push('─');
            }
            
            // Add junction or corner
            if col < self.pieces_per_row - 1 {
                border_chars.push('┬'); // Top junction
            } else {
                border_chars.push('┐'); // Top-right corner
            }
        }
        
        let border_line: String = border_chars.into_iter().collect();
        all_lines.push(Line::from(Span::styled(border_line, Style::default().fg(Color::DarkGray))));
    }

    /// Add bottom border of the grid
    fn add_grid_bottom_border(&self, all_lines: &mut Vec<Line>) {
        let mut border_chars = Vec::new();
        border_chars.push('└'); // Bottom-left corner
        
        for col in 0..self.pieces_per_row {
            // Add horizontal line for this piece cell
            for _ in 0..self.config.piece_width {
                border_chars.push('─');
            }
            
            // Add junction or corner
            if col < self.pieces_per_row - 1 {
                border_chars.push('┴'); // Bottom junction
            } else {
                border_chars.push('┘'); // Bottom-right corner
            }
        }
        
        let border_line: String = border_chars.into_iter().collect();
        all_lines.push(Line::from(Span::styled(border_line, Style::default().fg(Color::DarkGray))));
    }
}

impl Component for ResponsivePieceGridComponent {
    fn id(&self) -> ComponentId {
        self.id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        self.area = Some(area);
        
        // Update responsive layout
        self.update_responsive_layout(area.width);

        // Get current game state
        if let GameWrapper::Blokus(state) = &app.game_wrapper {
            let current_player = app.game_wrapper.get_current_player();
            self.is_current_player = current_player == self.player as i32;
            
            // Update available pieces
            self.available_pieces = state.get_available_pieces(self.player as i32);
            self.update_layout(); // Recalculate layout with new piece count

            // Update selected piece
            if self.is_current_player {
                self.selected_piece = app.blokus_ui_config.selected_piece_idx;
            } else {
                self.selected_piece = None;
            }
        }

        // Create the main block if borders are enabled
        let inner_area = if self.config.show_borders {
            let title = format!("Player {} Pieces", self.player);
            let block = Block::default()
                .borders(Borders::ALL)
                .title(title)
                .border_style(Style::default().fg(self.config.player_color));
            
            let inner = block.inner(area);
            frame.render_widget(block, area);
            inner
        } else {
            area
        };

        if self.available_pieces.is_empty() {
            let no_pieces = Paragraph::new("No pieces available")
                .style(Style::default().fg(Color::DarkGray));
            frame.render_widget(no_pieces, inner_area);
            return Ok(());
        }

        // Get pieces data
        let pieces = get_blokus_pieces();
        let available_set: HashSet<usize> = self.available_pieces.iter().cloned().collect();

        // Create content with uniform cell heights and complete grid structure
        let mut all_lines = Vec::new();

        // Add top border of the grid
        self.add_grid_top_border(&mut all_lines);

        for row in 0..self.total_rows {
            let chunk_start = row * self.pieces_per_row;
            let chunk_end = ((row + 1) * self.pieces_per_row).min(self.available_pieces.len());
            
            if chunk_start >= self.available_pieces.len() {
                break;
            }

            let pieces_in_row: Vec<usize> = self.available_pieces[chunk_start..chunk_end].iter().cloned().collect();

            if pieces_in_row.is_empty() {
                continue;
            }

            // Create uniform cell for each piece in this row
            self.render_piece_row(&mut all_lines, &pieces_in_row, &pieces, &available_set);
            
            // Add row separator after each row except the last one
            if row < self.total_rows - 1 {
                self.add_row_separator(&mut all_lines);
            }
        }

        // Add bottom border of the grid
        self.add_grid_bottom_border(&mut all_lines);

        // Render the content
        let paragraph = Paragraph::new(all_lines);
        frame.render_widget(paragraph, inner_area);

        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::MouseClick { x, y, .. }) => {
                if let Some(area) = self.area {
                    if *x >= area.x && *x < area.x + area.width &&
                       *y >= area.y && *y < area.y + area.height {
                        let local_x = *x - area.x;
                        let local_y = *y - area.y;
                        if let Some(_piece_idx) = self.handle_piece_click(app, local_x, local_y) {
                            return Ok(true);
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(false)
    }
}
