//! Responsive piece grid component with optimal layout calculations.

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
}

impl Default for ResponsivePieceGridConfig {
    fn default() -> Self {
        Self {
            player_color: Color::White,
            min_pieces_per_row: 3,
            max_pieces_per_row: 8,
            piece_width: 8,     // Increased back to accommodate double-width characters
            piece_height: 4,    // Keep same height for piece shapes
            show_borders: true,
            show_labels: true,
            compact_mode: false,
            empty_cell_light: Color::Rgb(100, 100, 100),
            empty_cell_dark: Color::Rgb(60, 60, 60),
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

    /// Create visual representation of a piece shape
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

        // Create a grid with double characters for square appearance
        let mut grid = vec![vec!["  "; width]; height];

        for &(r, c) in piece_shape {
            let gr = (r - min_r) as usize;
            let gc = (c - min_c) as usize;
            if gr < height && gc < width {
                grid[gr][gc] = "██"; // Double block characters for square appearance
            }
        }

        // Convert to vector of strings with checkerboard background
        grid.iter().enumerate()
            .map(|(row_idx, row)| {
                row.iter().enumerate()
                    .map(|(col_idx, cell)| {
                        if *cell == "██" {
                            "██".to_string()
                        } else {
                            // Create checkerboard pattern for empty cells using theme colors
                            if (row_idx + col_idx) % 2 == 0 {
                                "░░".to_string() // Will be styled with light color
                            } else {
                                "▒▒".to_string() // Will be styled with dark color
                            }
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

    /// Handle piece click with grid coordinate mapping
    pub fn handle_piece_click(&mut self, app: &mut App, local_x: u16, local_y: u16) -> Option<usize> {
        let Some(area) = self.area else { return None; };
        
        let _inner_area = if self.config.show_borders {
            Rect::new(
                area.x + 1,
                area.y + 1,
                area.width.saturating_sub(2),
                area.height.saturating_sub(2),
            )
        } else {
            area
        };

        // Account for border offset
        let click_x = local_x.saturating_sub(if self.config.show_borders { 1 } else { 0 });
        let click_y = local_y.saturating_sub(if self.config.show_borders { 1 } else { 0 });

        // Calculate row and column
        let row_height = self.config.piece_height + (if self.config.show_labels { 1 } else { 0 });
        let row = (click_y as usize) / row_height;
        
        let separator_width = 1;
        let col = (click_x as usize) / (self.config.piece_width + separator_width);

        // Calculate piece index
        let piece_index = row * self.pieces_per_row + col;
        
        // Check if this piece exists and is available
        if piece_index < self.available_pieces.len() && row < self.total_rows {
            let actual_piece_idx = self.available_pieces[piece_index];
            
            // Only allow selection for current player
            if self.is_current_player {
                app.blokus_ui_config.select_piece(actual_piece_idx);
                return Some(actual_piece_idx);
            }
        }
        
        None
    }

    /// Calculate the height needed for this grid
    pub fn calculate_height(&self) -> u16 {
        let row_height = self.config.piece_height + (if self.config.show_labels { 1 } else { 0 });
        let content_height = self.total_rows as u16 * row_height as u16;
        let border_height = if self.config.show_borders { 2 } else { 0 };
        content_height + border_height
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
        let (inner_area, _title) = if self.config.show_borders {
            let title = format!("Player {} Pieces", self.player);
            let block = Block::default()
                .borders(Borders::ALL)
                .title(title.clone())
                .border_style(Style::default().fg(self.config.player_color));
            
            let inner = block.inner(area);
            frame.render_widget(block, area);
            (inner, title)
        } else {
            (area, format!("Player {} Pieces", self.player))
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

        // Create content for each row
        let mut all_lines = Vec::new();

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

            // Create piece data for this row
            let mut pieces_in_row_data = Vec::new();
            for piece_idx in &pieces_in_row {
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

                    pieces_in_row_data.push((piece_label, piece_visual_lines, style));
                }
            }

            if pieces_in_row_data.is_empty() {
                continue;
            }

            // Find max height for alignment
            let max_height = pieces_in_row_data.iter()
                .map(|(_, lines, _)| lines.len())
                .max()
                .unwrap_or(1);

            // Add piece labels row if enabled
            if self.config.show_labels {
                let mut key_line_spans = Vec::new();
                for (i, (piece_label, _, style)) in pieces_in_row_data.iter().enumerate() {
                    let label_text = if self.selected_piece == Some(self.available_pieces[chunk_start + i]) {
                        format!("[{}]", piece_label)
                    } else {
                        format!(" {} ", piece_label)
                    };
                    let padded_label = format!("{:^width$}", label_text, width = self.config.piece_width);
                    key_line_spans.push(Span::styled(padded_label, *style));
                    if i < pieces_in_row_data.len() - 1 {
                        key_line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
                    }
                }
                all_lines.push(Line::from(key_line_spans));
            }

            // Add piece visual rows
            for line_idx in 0..max_height {
                let mut shape_line_spans = Vec::new();
                for (i, (_, piece_visual_lines, style)) in pieces_in_row_data.iter().enumerate() {
                    // Get the piece line content, or empty if beyond piece height
                    let piece_line = if line_idx < piece_visual_lines.len() {
                        &piece_visual_lines[line_idx]
                    } else {
                        ""
                    };
                    
                    // Create a properly sized line by padding to exact piece_width
                    let current_width = piece_line.chars().count();
                    let formatted_line = if current_width < self.config.piece_width {
                        let total_padding = self.config.piece_width - current_width;
                        let left_padding = total_padding / 2;
                        let right_padding = total_padding - left_padding;
                        format!("{}{}{}", 
                            " ".repeat(left_padding), 
                            piece_line, 
                            " ".repeat(right_padding)
                        )
                    } else if current_width > self.config.piece_width {
                        // Truncate if too wide
                        piece_line.chars().take(self.config.piece_width).collect()
                    } else {
                        piece_line.to_string()
                    };
                    
                    // Now parse the formatted line and apply styles
                    let chars: Vec<char> = formatted_line.chars().collect();
                    let mut j = 0;
                    
                    while j < chars.len() {
                        if j + 1 < chars.len() {
                            let two_char = format!("{}{}", chars[j], chars[j + 1]);
                            match two_char.as_str() {
                                "██" => {
                                    // Piece block - use player color
                                    shape_line_spans.push(Span::styled("██", *style));
                                }
                                "░░" => {
                                    // Light checkerboard cell
                                    shape_line_spans.push(Span::styled("░░", Style::default().fg(self.config.empty_cell_light)));
                                }
                                "▒▒" => {
                                    // Dark checkerboard cell
                                    shape_line_spans.push(Span::styled("▒▒", Style::default().fg(self.config.empty_cell_dark)));
                                }
                                "  " => {
                                    // Empty space - use default style
                                    shape_line_spans.push(Span::styled("  ", Style::default()));
                                }
                                _ => {
                                    // Fallback for any other two-character combinations
                                    shape_line_spans.push(Span::styled(two_char, *style));
                                }
                            }
                            j += 2;
                        } else {
                            // Handle single character at end (shouldn't happen with double-width approach)
                            shape_line_spans.push(Span::styled(chars[j].to_string(), Style::default()));
                            j += 1;
                        }
                    }
                    
                    // Add vertical separator between pieces
                    if i < pieces_in_row_data.len() - 1 {
                        shape_line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
                    }
                }
                all_lines.push(Line::from(shape_line_spans));
            }

            // Add separator line between rows (except last)
            if row < self.total_rows - 1 && chunk_end < self.available_pieces.len() {
                // Calculate actual width: pieces + separators between pieces
                let num_pieces_in_row = pieces_in_row.len();
                let separator_width = num_pieces_in_row * self.config.piece_width + (num_pieces_in_row.saturating_sub(1));
                let separator_line = "─".repeat(separator_width);
                all_lines.push(Line::from(Span::styled(separator_line, Style::default().fg(Color::DarkGray))));
            }
        }

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
