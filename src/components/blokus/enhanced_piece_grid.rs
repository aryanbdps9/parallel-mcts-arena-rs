//! Enhanced responsive piece grid component with clean borders and layout.

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

/// Configuration for the enhanced piece grid
#[derive(Clone)]
pub struct EnhancedPieceGridConfig {
    pub player_color: Color,
    pub pieces_per_row: usize,
    pub piece_width: usize,
    pub piece_height: usize,
    pub show_borders: bool,
    pub show_labels: bool,
    pub responsive: bool,
}

impl Default for EnhancedPieceGridConfig {
    fn default() -> Self {
        Self {
            player_color: Color::White,
            pieces_per_row: 7, // Default from original implementation
            piece_width: 6,    // Slightly smaller than original for better fit
            piece_height: 3,   // Height for piece visual
            show_borders: true,
            show_labels: true,
            responsive: true,
        }
    }
}

/// Enhanced piece grid component with clean borders like the original
pub struct EnhancedPieceGridComponent {
    id: ComponentId,
    player: u8,
    config: EnhancedPieceGridConfig,
    available_pieces: Vec<usize>,
    selected_piece: Option<usize>,
    is_current_player: bool,
    area: Option<Rect>,
}

impl EnhancedPieceGridComponent {
    pub fn new(player: u8, config: EnhancedPieceGridConfig) -> Self {
        Self {
            id: ComponentId::new(),
            player,
            config,
            available_pieces: Vec::new(),
            selected_piece: None,
            is_current_player: false,
            area: None,
        }
    }

    pub fn set_available_pieces(&mut self, pieces: Vec<usize>) {
        self.available_pieces = pieces;
    }

    pub fn set_selected_piece(&mut self, piece: Option<usize>) {
        self.selected_piece = piece;
    }

    pub fn set_current_player(&mut self, is_current: bool) {
        self.is_current_player = is_current;
    }

    /// Update pieces per row based on available width for responsive design
    fn update_responsive_layout(&mut self, available_width: u16) {
        if !self.config.responsive {
            return;
        }

        // Calculate how many pieces can fit per row
        // Each piece needs piece_width + 1 for separator, except the last one
        let separator_width = 1;
        let border_width = 2; // Left and right borders
        let usable_width = available_width.saturating_sub(border_width);
        
        if usable_width > 0 {
            let pieces_that_fit = ((usable_width as usize + separator_width) / (self.config.piece_width + separator_width)).max(1);
            self.config.pieces_per_row = pieces_that_fit.min(7); // Cap at 7 for readability
        }
    }

    /// Create visual representation of a piece shape
    fn create_visual_piece_shape(&self, piece_shape: &[(i32, i32)]) -> Vec<String> {
        if piece_shape.is_empty() {
            return vec!["▢".to_string()];
        }

        // Create a 2D visual representation
        let min_r = piece_shape.iter().map(|p| p.0).min().unwrap_or(0);
        let max_r = piece_shape.iter().map(|p| p.0).max().unwrap_or(0);
        let min_c = piece_shape.iter().map(|p| p.1).min().unwrap_or(0);
        let max_c = piece_shape.iter().map(|p| p.1).max().unwrap_or(0);

        let height = (max_r - min_r + 1) as usize;
        let width = (max_c - min_c + 1) as usize;

        // Create a grid to show the shape
        let mut grid = vec![vec![' '; width]; height];

        // Fill the grid with the piece shape
        for &(r, c) in piece_shape {
            let gr = (r - min_r) as usize;
            let gc = (c - min_c) as usize;
            if gr < height && gc < width {
                grid[gr][gc] = '▢'; // Use empty square like in the original
            }
        }

        // Convert to vector of strings
        let result: Vec<String> = grid.iter()
            .map(|row| row.iter().collect::<String>())
            .collect();

        result
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

    /// Handle click on a piece in the grid
    pub fn handle_piece_click(&mut self, app: &mut App, local_x: u16, local_y: u16) -> Option<usize> {
        let Some(_area) = self.area else { return None; };
        
        // Account for borders
        let inner_x = local_x.saturating_sub(1);
        let inner_y = local_y.saturating_sub(1);
        
        // Calculate which piece was clicked
        let col = (inner_x as usize) / (self.config.piece_width + 1); // +1 for separator
        let row_height = self.config.piece_height + (if self.config.show_labels { 1 } else { 0 });
        let row = (inner_y as usize) / row_height;
        
        let piece_index = row * self.config.pieces_per_row + col;
        
        // Check if this piece exists and is available
        if piece_index < self.available_pieces.len() {
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
        let pieces_to_show = if self.is_current_player { 
            21 
        } else { 
            self.available_pieces.len().min(10) 
        };
        
        let rows = ((pieces_to_show + self.config.pieces_per_row - 1) / self.config.pieces_per_row).max(1);
        let row_height = self.config.piece_height + (if self.config.show_labels { 1 } else { 0 });
        (rows * row_height) as u16 + 2 // Add borders
    }
}

impl Component for EnhancedPieceGridComponent {
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

            // Update selected piece
            if self.is_current_player {
                self.selected_piece = app.blokus_ui_config.selected_piece_idx;
            } else {
                self.selected_piece = None;
            }
        }

        // Create the main block
        let title = format!("Player {} Pieces", self.player);
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(Style::default().fg(self.config.player_color));
        
        let inner_area = block.inner(area);
        frame.render_widget(block, area);

        // Determine pieces to show
        let pieces_to_display = if self.is_current_player {
            // Show all 21 pieces for current player
            (0..21).collect::<Vec<_>>()
        } else {
            // Show only available pieces for other players, limited to first 10
            self.available_pieces.iter().take(10).cloned().collect()
        };

        if pieces_to_display.is_empty() {
            let no_pieces = Paragraph::new("No pieces available")
                .style(Style::default().fg(Color::DarkGray));
            frame.render_widget(no_pieces, inner_area);
            return Ok(());
        }

        let available_set: HashSet<usize> = self.available_pieces.iter().cloned().collect();
        let pieces = get_blokus_pieces();

        // Calculate grid layout
        let rows = ((pieces_to_display.len() + self.config.pieces_per_row - 1) / self.config.pieces_per_row).max(1);
        let _row_height = self.config.piece_height + (if self.config.show_labels { 1 } else { 0 });

        // Create content for each row
        let mut all_lines = Vec::new();

        for row in 0..rows {
            let chunk_start = row * self.config.pieces_per_row;
            let chunk_end = ((row + 1) * self.config.pieces_per_row).min(pieces_to_display.len());
            let pieces_in_row: Vec<usize> = pieces_to_display[chunk_start..chunk_end].iter().cloned().collect();

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
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
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
                    let padded_label = format!("{:^width$}", piece_label, width = self.config.piece_width);
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
                    let piece_line = if line_idx < piece_visual_lines.len() {
                        format!("{:^width$}", piece_visual_lines[line_idx], width = self.config.piece_width)
                    } else {
                        " ".repeat(self.config.piece_width)
                    };
                    shape_line_spans.push(Span::styled(piece_line, *style));
                    if i < pieces_in_row_data.len() - 1 {
                        shape_line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
                    }
                }
                all_lines.push(Line::from(shape_line_spans));
            }

            // Add separator line between rows (except last)
            if row < rows - 1 {
                let separator_width = self.config.pieces_per_row * self.config.piece_width + (self.config.pieces_per_row - 1);
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
