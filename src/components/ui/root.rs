//! Root component implementation.

use ratatui::{
    layout::Rect,
    Frame,
    widgets::{Block, Borders, Paragraph},
    style::Style,
};

use crate::components::core::{Component, ComponentId};
use crate::components::events::ComponentEvent;

/// The root component that serves as the top-level container
pub struct RootComponent {
    id: ComponentId,
}

impl RootComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
        }
    }
}

impl Component for RootComponent {
    fn id(&self) -> ComponentId {
        self.id
    }
    
    fn render(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .title("MCTS Arena - Component System")
            .borders(Borders::ALL)
            .style(Style::default());
            
        let paragraph = Paragraph::new("Welcome to the new component-based UI!\n\nThis is the root component.")
            .block(block)
            .style(Style::default());
            
        frame.render_widget(paragraph, area);
    }
    
    fn handle_event(&mut self, event: &ComponentEvent) -> bool {
        match event {
            ComponentEvent::Input(_) => {
                // Root component can handle global shortcuts here
                false // Don't consume events by default
            }
            _ => false,
        }
    }
    
    crate::impl_component_base!(RootComponent);
}