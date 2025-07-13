//! Component for rendering individual Blokus piece shapes with clean visuals.

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};
use std::any::Any;

use crate::app::App;
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::ComponentEvent;
use crate::games::blokus::get_blokus_pieces;

/// Configuration for piece shape rendering
#[derive(Clone)]
pub struct PieceShapeConfig {
    pub show_label: bool,
    pub show_border: bool,
    pub width: u16,
    pub height: u16,
    pub player_color: Color,
    pub empty_cell_light: Color,
    pub empty_cell_dark: Color,
}

impl Default for PieceShapeConfig {
    fn default() -> Self {
        Self {
            show_label: true,
            show_border: false,
            width: 8,
            height: 4,
            player_color: Color::White,
            empty_cell_light: Color::Rgb(100, 100, 100),
            empty_cell_dark: Color::Rgb(60, 60, 60),
        }
    }
}

/// Component for rendering a single Blokus piece shape
pub struct PieceShapeComponent {
    id: ComponentId,
    piece_idx: usize,
    config: PieceShapeConfig,
    is_available: bool,
    is_selected: bool,
}

impl PieceShapeComponent {
    pub fn new(piece_idx: usize, config: PieceShapeConfig) -> Self {
        Self {
            id: ComponentId::new(),
            piece_idx,
            config,
            is_available: true,
            is_selected: false,
        }
    }

    pub fn set_available(&mut self, available: bool) {
        self.is_available = available;
    }

    pub fn set_selected(&mut self, selected: bool) {
        self.is_selected = selected;
    }

    pub fn get_piece_idx(&self) -> usize {
        self.piece_idx
    }

    /// Create visual representation of a piece shape using board-like characters
    fn create_visual_piece_shape(&self, piece_shape: &[(i32, i32)]) -> Vec<String> {
        if piece_shape.is_empty() {
            return vec!["██".to_string()];
        }

        // Create a 2D visual representation
        let min_r = piece_shape.iter().map(|p| p.0).min().unwrap_or(0);
        let max_r = piece_shape.iter().map(|p| p.0).max().unwrap_or(0);
        let min_c = piece_shape.iter().map(|p| p.1).min().unwrap_or(0);
        let max_c = piece_shape.iter().map(|p| p.1).max().unwrap_or(0);

        let height = (max_r - min_r + 1) as usize;
        let width = (max_c - min_c + 1) as usize;

        // Create a grid to show the shape - use double characters for square appearance
        let mut grid = vec![vec!["  "; width]; height]; // Two spaces for empty cells

        // Fill the grid with the piece shape using double block characters
        for &(r, c) in piece_shape {
            let gr = (r - min_r) as usize;
            let gc = (c - min_c) as usize;
            if gr < height && gc < width {
                grid[gr][gc] = "██"; // Double block characters for square appearance
            }
        }

        // Convert to vector of strings with checkerboard background
        let result: Vec<String> = grid
            .iter()
            .enumerate()
            .map(|(row_idx, row)| {
                row.iter()
                    .enumerate()
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
            .collect();

        // Ensure minimum size for better visibility
        if result.is_empty() {
            vec!["██".to_string()]
        } else {
            result
        }
    }

    /// Get the piece label for display (1-9, 0, a-k)
    fn get_piece_label(&self) -> String {
        if self.piece_idx < 9 {
            (self.piece_idx + 1).to_string()
        } else if self.piece_idx == 9 {
            "0".to_string()
        } else {
            ((b'a' + (self.piece_idx - 10) as u8) as char).to_string()
        }
    }
}

impl Component for PieceShapeComponent {
    fn id(&self) -> ComponentId {
        self.id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, _app: &App) -> ComponentResult<()> {
        // Get piece data
        let pieces = get_blokus_pieces();
        if self.piece_idx >= pieces.len() {
            return Ok(());
        }

        let piece = &pieces[self.piece_idx];
        let piece_shape = if !piece.transformations.is_empty() {
            &piece.transformations[0]
        } else {
            return Ok(());
        };

        // Create visual representation
        let piece_visual_lines = self.create_visual_piece_shape(piece_shape);

        // Determine styling
        let style = if self.is_selected {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else if self.is_available {
            Style::default()
                .fg(self.config.player_color)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM)
        };

        // Create content lines
        let mut content_lines = Vec::new();

        // Add label if enabled
        if self.config.show_label {
            let label = if self.is_selected {
                format!("[{}]", self.get_piece_label())
            } else {
                format!(" {} ", self.get_piece_label())
            };
            let label_line = Line::from(Span::styled(
                format!("{:^width$}", label, width = self.config.width as usize),
                style,
            ));
            content_lines.push(label_line);
        }

        // Add piece visual centered in area
        let available_height = if self.config.show_label {
            self.config.height.saturating_sub(1)
        } else {
            self.config.height
        } as usize;

        for (i, piece_line) in piece_visual_lines.iter().enumerate() {
            if i < available_height {
                // Parse the line to apply different styles to piece blocks vs checkerboard pattern
                let mut line_spans = Vec::new();
                let chars: Vec<char> = piece_line.chars().collect();
                let mut j = 0;

                while j < chars.len() {
                    if j + 1 < chars.len() {
                        let two_char = format!("{}{}", chars[j], chars[j + 1]);
                        match two_char.as_str() {
                            "██" => {
                                // Piece block - use player color
                                line_spans.push(Span::styled("██", style));
                            }
                            "░░" => {
                                // Light checkerboard cell
                                line_spans.push(Span::styled(
                                    "░░",
                                    Style::default().fg(self.config.empty_cell_light),
                                ));
                            }
                            "▒▒" => {
                                // Dark checkerboard cell
                                line_spans.push(Span::styled(
                                    "▒▒",
                                    Style::default().fg(self.config.empty_cell_dark),
                                ));
                            }
                            _ => {
                                // Fallback for any other characters
                                line_spans.push(Span::styled(two_char, style));
                            }
                        }
                        j += 2;
                    } else {
                        // Handle single character at end
                        line_spans.push(Span::styled(chars[j].to_string(), style));
                        j += 1;
                    }
                }

                // Center the line
                let total_width = line_spans.iter().map(|s| s.content.len()).sum::<usize>();
                let padding = (self.config.width as usize).saturating_sub(total_width) / 2;

                let mut centered_spans = Vec::new();
                if padding > 0 {
                    centered_spans.push(Span::styled(" ".repeat(padding), style));
                }
                centered_spans.extend(line_spans);
                if padding > 0 {
                    let remaining_padding =
                        (self.config.width as usize).saturating_sub(total_width + padding);
                    if remaining_padding > 0 {
                        centered_spans.push(Span::styled(" ".repeat(remaining_padding), style));
                    }
                }

                content_lines.push(Line::from(centered_spans));
            }
        }

        // Fill remaining space
        let needed_lines = self.config.height as usize;
        while content_lines.len() < needed_lines {
            let empty_line =
                Line::from(Span::styled(" ".repeat(self.config.width as usize), style));
            content_lines.push(empty_line);
        }

        // Render with or without border
        if self.config.show_border && self.is_selected {
            let block = Block::default().borders(Borders::ALL).border_style(style);
            frame.render_widget(block, area);

            // Render content inside border
            let inner_area = Rect::new(
                area.x + 1,
                area.y + 1,
                area.width.saturating_sub(2),
                area.height.saturating_sub(2),
            );
            if inner_area.width > 0 && inner_area.height > 0 {
                let paragraph = Paragraph::new(content_lines);
                frame.render_widget(paragraph, inner_area);
            }
        } else {
            // Render content directly
            let paragraph = Paragraph::new(content_lines);
            frame.render_widget(paragraph, area);
        }

        Ok(())
    }

    fn handle_event(&mut self, _event: &ComponentEvent, _app: &mut App) -> EventResult {
        // Individual piece shapes don't handle events directly
        // The parent grid component handles clicks and forwards them
        Ok(false)
    }
}
