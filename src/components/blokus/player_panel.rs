//! Player panel component for Blokus piece selector.

use ratatui::{
    layout::Rect,
    Frame,
    widgets::Paragraph,
    style::{Style, Color, Modifier},
    text::{Line, Span},
};
use std::collections::HashSet;
use std::any::Any;
use mcts::GameState;

use crate::app::App;
use crate::game_wrapper::GameWrapper;
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crate::components::blokus::piece_cell::PieceCellComponent;

/// Component representing a single player's panel in the Blokus piece selector
pub struct BlokusPlayerPanelComponent {
    id: ComponentId,
    player: u8,
    is_expanded: bool,
    piece_cells: Vec<PieceCellComponent>,
    area: Option<Rect>,
}

impl BlokusPlayerPanelComponent {
    pub fn new(player: u8, is_expanded: bool) -> Self {
        Self {
            id: ComponentId::new(),
            player,
            is_expanded,
            piece_cells: Vec::new(),
            area: None,
        }
    }

    pub fn set_expanded(&mut self, expanded: bool) {
        self.is_expanded = expanded;
    }

    pub fn get_player(&self) -> u8 {
        self.player
    }

    pub fn is_expanded(&self) -> bool {
        self.is_expanded
    }

    pub fn set_area(&mut self, area: Rect) {
        self.area = Some(area);
    }

    pub fn get_area(&self) -> Option<Rect> {
        self.area
    }

    /// Check if a point is within this component's area
    pub fn contains_point(&self, x: u16, y: u16) -> bool {
        if let Some(area) = self.area {
            x >= area.x && x < area.x + area.width &&
            y >= area.y && y < area.y + area.height
        } else {
            false
        }
    }

    /// Handle click on header to expand/collapse
    pub fn handle_header_click(&mut self, x: u16, y: u16) -> bool {
        if let Some(area) = self.area {
            // Check if click is on the first line (header)
            if x >= area.x && x < area.x + area.width && y == area.y {
                self.is_expanded = !self.is_expanded;
                return true;
            }
        }
        false
    }

    /// Handle click on a piece cell
    pub fn handle_piece_click(&mut self, _app: &mut App, x: u16, y: u16) -> Option<usize> {
        for cell in &self.piece_cells {
            if cell.contains_point(x, y) {
                return Some(cell.get_piece_index());
            }
        }
        None
    }

    /// Update piece cells based on current game state
    pub fn update_piece_cells(&mut self, app: &App) {
        self.piece_cells.clear();

        if let GameWrapper::Blokus(blokus_state) = &app.game_wrapper {
            let available_pieces = blokus_state.get_available_pieces(self.player.into());
            let available_set: HashSet<usize> = available_pieces.iter().cloned().collect();
            let selected_info = app.blokus_ui_config.get_selected_piece_info();

            // Show all pieces (0-20), indicating availability
            for piece_idx in 0..21 {
                let is_available = available_set.contains(&piece_idx);
                let is_selected = if let Some((selected_piece, _)) = selected_info {
                    selected_piece == piece_idx && app.game_wrapper.get_current_player() == self.player.into()
                } else {
                    false
                };

                self.piece_cells.push(PieceCellComponent::new(
                    piece_idx,
                    self.player,
                    is_available,
                    is_selected,
                ));
            }
        }
    }

    /// Calculate the number of lines this panel will occupy when rendered
    pub fn calculate_height(&self, app: &App) -> u16 {
        if !self.is_expanded {
            return 1; // Just the header
        }

        let current_player = app.game_wrapper.get_current_player();
        let is_current = self.player == current_player as u8;
        
        // Current player shows more pieces
        let visible_pieces = if is_current { 21 } else { 10 };
        let pieces_per_row = 5;
        let piece_rows = (visible_pieces + pieces_per_row - 1) / pieces_per_row;
        
        1 + piece_rows as u16 // Header + piece rows
    }
}

impl Component for BlokusPlayerPanelComponent {
    fn id(&self) -> ComponentId {
        self.id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::MouseClick { x, y, .. }) => {
                // Check if it's a header click (expand/collapse)
                if self.handle_header_click(*x, *y) {
                    return Ok(true);
                }

                // Check if it's a piece click (only if expanded and current player)
                if self.is_expanded && app.game_wrapper.get_current_player() == self.player.into() {
                    if let Some(piece_idx) = self.handle_piece_click(app, *x, *y) {
                        // Select this piece
                        app.blokus_ui_config.select_piece(piece_idx);
                        return Ok(true);
                    }
                }
            }
            _ => {}
        }
        Ok(false)
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        self.set_area(area);
        self.update_piece_cells(app);

        if let GameWrapper::Blokus(blokus_state) = &app.game_wrapper {
            let available_pieces = blokus_state.get_available_pieces(self.player.into());
            let available_count = available_pieces.len();
            let available_set: HashSet<usize> = available_pieces.iter().cloned().collect();
            
            let player_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
            let player_names = ["P1", "P2", "P3", "P4"];
            let color = player_colors[(self.player - 1) as usize];
            let current_player = app.game_wrapper.get_current_player();
            let is_current = self.player == current_player as u8;

            let mut current_line = 0;

            // Render header
            let expand_indicator = if self.is_expanded { "▼" } else { "▶" };
            let header_style = if is_current {
                Style::default().fg(color).add_modifier(Modifier::BOLD).bg(Color::DarkGray)
            } else {
                Style::default().fg(color).add_modifier(Modifier::BOLD)
            };
            
            let header_text = if is_current {
                format!("{} ► {} ({} pieces) ◄", expand_indicator, player_names[(self.player - 1) as usize], available_count)
            } else {
                format!("{}   {} ({} pieces)", expand_indicator, player_names[(self.player - 1) as usize], available_count)
            };

            if current_line < area.height {
                let header_area = Rect::new(area.x, area.y + current_line, area.width, 1);
                let paragraph = Paragraph::new(Line::from(Span::styled(header_text, header_style)));
                frame.render_widget(paragraph, header_area);
                current_line += 1;
            }

            // Render pieces if expanded
            if self.is_expanded && current_line < area.height {
                let pieces_per_row = 5;
                let visible_pieces = if is_current { 21 } else { 10 };
                
                // Show pieces in rows
                for chunk_start in (0..visible_pieces).step_by(pieces_per_row) {
                    if current_line >= area.height {
                        break;
                    }

                    let mut line_spans = Vec::new();
                    let chunk_end = std::cmp::min(chunk_start + pieces_per_row, visible_pieces);
                    
                    for piece_idx in chunk_start..chunk_end {
                        let is_available = available_set.contains(&piece_idx);
                        let selected_info = app.blokus_ui_config.get_selected_piece_info();
                        let is_selected = if let Some((selected_piece, _)) = selected_info {
                            selected_piece == piece_idx && is_current
                        } else {
                            false
                        };

                        let piece_char = std::char::from_u32(('A' as u32) + (piece_idx as u32)).unwrap_or('?');
                        
                        let style = if is_selected {
                            Style::default()
                                .fg(Color::Black)
                                .bg(color)
                                .add_modifier(Modifier::BOLD)
                        } else if is_available {
                            Style::default()
                                .fg(color)
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default()
                                .fg(Color::DarkGray)
                        };

                        line_spans.push(Span::styled(format!("{} ", piece_char), style));
                    }

                    let pieces_area = Rect::new(area.x, area.y + current_line, area.width, 1);
                    let paragraph = Paragraph::new(Line::from(line_spans));
                    frame.render_widget(paragraph, pieces_area);
                    current_line += 1;
                }
            }
        }

        Ok(())
    }
}
