//! Individual piece cell component for Blokus UI.

use ratatui::{
    layout::Rect,
    Frame,
    widgets::Paragraph,
    style::{Style, Color, Modifier},
    text::{Line, Span},
};
use std::any::Any;

use crate::app::App;
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};

/// Component representing a single piece cell in the Blokus piece selector
pub struct PieceCellComponent {
    id: ComponentId,
    piece_index: usize,
    player: u8,
    is_available: bool,
    is_selected: bool,
    area: Option<Rect>,
}

impl PieceCellComponent {
    pub fn new(piece_index: usize, player: u8, is_available: bool, is_selected: bool) -> Self {
        Self {
            id: ComponentId::new(),
            piece_index,
            player,
            is_available,
            is_selected,
            area: None,
        }
    }

    pub fn update_state(&mut self, is_available: bool, is_selected: bool) {
        self.is_available = is_available;
        self.is_selected = is_selected;
    }

    pub fn set_area(&mut self, area: Rect) {
        self.area = Some(area);
    }

    pub fn get_area(&self) -> Option<Rect> {
        self.area
    }

    pub fn get_piece_index(&self) -> usize {
        self.piece_index
    }

    pub fn get_player(&self) -> u8 {
        self.player
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
}

impl Component for PieceCellComponent {
    fn id(&self) -> ComponentId {
        self.id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn handle_event(&mut self, event: &ComponentEvent, _app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::MouseClick { x, y, .. }) => {
                if self.contains_point(*x, *y) && self.is_available {
                    // Piece cell was clicked - this should be handled by parent component
                    return Ok(true);
                }
            }
            _ => {}
        }
        Ok(false)
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, _app: &App) -> ComponentResult<()> {
        self.set_area(area);

        let player_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
        let color = player_colors[(self.player - 1) as usize];

        // Determine piece representation
        let piece_char = if self.piece_index < 21 {
            std::char::from_u32(('A' as u32) + (self.piece_index as u32)).unwrap_or('?')
        } else {
            '?'
        };

        // Determine style based on state
        let style = if self.is_selected {
            Style::default()
                .fg(Color::Black)
                .bg(color)
                .add_modifier(Modifier::BOLD)
        } else if self.is_available {
            Style::default()
                .fg(color)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::DarkGray)
        };

        let text = format!("{}", piece_char);
        let paragraph = Paragraph::new(Line::from(Span::styled(text, style)));
        frame.render_widget(paragraph, area);

        Ok(())
    }
}
