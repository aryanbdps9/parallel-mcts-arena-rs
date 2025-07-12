//! Piece selector component for Blokus UI.

use ratatui::{
    layout::{Rect, Direction},
    Frame,
    widgets::{Block, Borders},
};
use std::any::Any;

use crate::app::App;
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crate::components::ui::{ResponsiveLayoutComponent, ResponsiveLayoutType};
use crate::components::blokus::player_panel::BlokusPlayerPanelComponent;

/// Component managing the piece selector for all players in Blokus
pub struct BlokusPieceSelectorComponent {
    id: ComponentId,
    player_panels: Vec<BlokusPlayerPanelComponent>,
    scroll_offset: u16,
    area: Option<Rect>,
    responsive_layout: ResponsiveLayoutComponent,
}

impl BlokusPieceSelectorComponent {
    pub fn new() -> Self {
        let mut player_panels = Vec::new();
        for player in 1..=4 {
            player_panels.push(BlokusPlayerPanelComponent::new(player, true));
        }

        let mut responsive_layout = ResponsiveLayoutComponent::new(
            ResponsiveLayoutType::ContentDriven, 
            Direction::Vertical
        );
        
        // Configure responsive layout for 4 player panels
        for _ in 0..4 {
            responsive_layout.add_panel(5, 15, 25); // min: 5, preferred: 15, max: 25 lines per player
        }

        Self {
            id: ComponentId::new(),
            player_panels,
            scroll_offset: 0,
            area: None,
            responsive_layout,
        }
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

    /// Toggle expand/collapse for a player
    pub fn toggle_player_expanded(&mut self, player: u8) {
        if let Some(panel) = self.player_panels.get_mut((player - 1) as usize) {
            panel.set_expanded(!panel.is_expanded());
        }
    }

    /// Get expanded state for a player
    pub fn is_player_expanded(&self, player: u8) -> bool {
        if let Some(panel) = self.player_panels.get((player - 1) as usize) {
            panel.is_expanded()
        } else {
            true
        }
    }

    /// Calculate total content height
    pub fn calculate_total_height(&self, _app: &App) -> u16 {
        self.player_panels.iter().map(|panel| panel.calculate_height()).sum()
    }

    /// Handle mouse scroll
    pub fn handle_scroll(&mut self, app: &App, up: bool) {
        let total_height = self.calculate_total_height(app);
        let visible_height = if let Some(area) = self.area {
            area.height.saturating_sub(2) // Account for borders
        } else {
            return;
        };

        if total_height > visible_height {
            if !up {
                // Scroll down
                let max_scroll = total_height.saturating_sub(visible_height);
                self.scroll_offset = std::cmp::min(self.scroll_offset + 1, max_scroll);
            } else {
                // Scroll up
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
            }
        }
    }

    /// Auto-scroll to show current player's expanded section
    pub fn auto_scroll_to_current_player(&mut self, app: &App) {
        use mcts::GameState;
        let current_player = app.game_wrapper.get_current_player();
        
        // Calculate position of current player's panel
        let mut current_player_start = 0;
        for (_i, panel) in self.player_panels.iter().enumerate() {
            if panel.get_player() == current_player as u8 {
                break;
            }
            current_player_start += panel.calculate_height();
        }

        let current_player_height = if let Some(panel) = self.player_panels.get((current_player - 1) as usize) {
            panel.calculate_height()
        } else {
            return;
        };

        let visible_height = if let Some(area) = self.area {
            area.height.saturating_sub(2) // Account for borders
        } else {
            return;
        };

        // Check if current player's panel is fully visible
        let panel_end = current_player_start + current_player_height;
        let view_end = self.scroll_offset + visible_height;

        if current_player_start < self.scroll_offset {
            // Panel starts above visible area - scroll up to show it
            self.scroll_offset = current_player_start;
        } else if panel_end > view_end {
            // Panel ends below visible area - scroll down to show it
            self.scroll_offset = panel_end.saturating_sub(visible_height);
        }
    }

    /// Handle mouse click with proper coordinate mapping
    pub fn handle_click(&mut self, app: &mut App, x: u16, y: u16) -> bool {
        if !self.contains_point(x, y) {
            return false;
        }

        let Some(area) = self.area else { return false; };
        
        // Convert to relative coordinates within the content area
        let content_area = Rect::new(
            area.x + 1, // Account for left border
            area.y + 1, // Account for top border
            area.width.saturating_sub(2), // Account for borders
            area.height.saturating_sub(2),
        );

        if x < content_area.x || x >= content_area.x + content_area.width ||
           y < content_area.y || y >= content_area.y + content_area.height {
            return false;
        }

        let rel_x = x - content_area.x;
        let rel_y = y - content_area.y;
        
        // Adjust for scroll offset
        let content_y = rel_y + self.scroll_offset;

        // Find which player panel this click belongs to
        let mut current_y = 0;
        for panel in &mut self.player_panels {
            let panel_height = panel.calculate_height();
            if content_y >= current_y && content_y < current_y + panel_height {
                // Create panel area for this render
                let panel_area = Rect::new(
                    content_area.x,
                    content_area.y + current_y.saturating_sub(self.scroll_offset),
                    content_area.width,
                    panel_height,
                );

                // Check if the panel area is visible
                if panel_area.y < content_area.y + content_area.height &&
                   panel_area.y + panel_height > content_area.y {
                    
                    // Handle the event with adjusted coordinates
                    let event = ComponentEvent::Input(InputEvent::MouseClick {
                        x: rel_x + content_area.x,
                        y: content_y - current_y + content_area.y,
                        button: 1, // Left button
                    });

                    if let Ok(true) = panel.handle_event(&event, app) {
                        return true;
                    }
                }
                break;
            }
            current_y += panel_height;
        }

        false
    }
}

impl Component for BlokusPieceSelectorComponent {
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
        // First, try to delegate to child components (player panels)
        for panel in &mut self.player_panels {
            if panel.handle_event(event, app)? {
                return Ok(true); // Event was handled by a child
            }
        }

        // If no child handled it, handle it ourselves
        match event {
            ComponentEvent::Input(InputEvent::MouseClick { x, y, .. }) => {
                if self.handle_click(app, *x, *y) {
                    return Ok(true);
                }
            }
            ComponentEvent::Input(InputEvent::MouseScroll { x, y, up }) => {
                if self.contains_point(*x, *y) {
                    self.handle_scroll(app, *up);
                    return Ok(true);
                }
            }
            _ => {}
        }
        Ok(false)
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        self.set_area(area);

        // Draw border
        let block = Block::default()
            .title("Available Pieces (All Players)")
            .borders(Borders::ALL);
        frame.render_widget(block, area);

        // Calculate content area
        let content_area = Rect::new(
            area.x + 1,
            area.y + 1, 
            area.width.saturating_sub(2),
            area.height.saturating_sub(2),
        );

        if content_area.width == 0 || content_area.height == 0 {
            return Ok(());
        }

        // Use responsive layout to calculate optimal panel heights
        let panel_areas = self.responsive_layout.calculate_layout(content_area);
        
        // Render each player panel in its allocated area
        for (i, panel) in self.player_panels.iter_mut().enumerate() {
            if i < panel_areas.len() {
                panel.render(frame, panel_areas[i], app)?;
            }
        }

        Ok(())
    }

    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        self.player_panels.iter_mut().map(|p| p as &mut dyn Component).collect()
    }

    fn children(&self) -> Vec<&dyn Component> {
        self.player_panels.iter().map(|p| p as &dyn Component).collect()
    }
}
