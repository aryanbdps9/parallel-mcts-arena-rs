//! Piece selector component for Blokus UI.

use ratatui::{
    layout::Rect,
    Frame,
    widgets::{Block, Borders, Scrollbar, ScrollbarOrientation, ScrollbarState},
};
use std::any::Any;

use crate::app::App;
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crate::components::blokus::player_panel::BlokusPlayerPanelComponent;

/// Component managing the piece selector for all players in Blokus
pub struct BlokusPieceSelectorComponent {
    id: ComponentId,
    player_panels: Vec<BlokusPlayerPanelComponent>,
    scroll_offset: u16,
    area: Option<Rect>,
}

impl BlokusPieceSelectorComponent {
    pub fn new() -> Self {
        let mut player_panels = Vec::new();
        for player in 1..=4 {
            player_panels.push(BlokusPlayerPanelComponent::new(player, true));
        }

        Self {
            id: ComponentId::new(),
            player_panels,
            scroll_offset: 0,
            area: None,
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
    pub fn calculate_total_height(&self, app: &App) -> u16 {
        self.player_panels.iter().map(|panel| panel.calculate_height(app)).sum()
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
            current_player_start += panel.calculate_height(app);
        }

        let current_player_height = if let Some(panel) = self.player_panels.get((current_player - 1) as usize) {
            panel.calculate_height(app)
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
            let panel_height = panel.calculate_height(app);
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

        // Calculate total height and whether we need scrolling
        let total_height = self.calculate_total_height(app);
        let needs_scrollbar = total_height > content_area.height;

        // Adjust content area if scrollbar is needed
        let (panel_area, scrollbar_area) = if needs_scrollbar && content_area.width > 1 {
            let panel_width = content_area.width.saturating_sub(1);
            (
                Rect::new(content_area.x, content_area.y, panel_width, content_area.height),
                Rect::new(content_area.x + panel_width, content_area.y, 1, content_area.height),
            )
        } else {
            (content_area, Rect::new(0, 0, 0, 0))
        };

        // Render player panels with clipping
        let mut current_y = 0;
        for panel in &mut self.player_panels {
            let panel_height = panel.calculate_height(app);
            
            // Check if this panel is visible in the current scroll view
            let panel_start = current_y;
            let panel_end = current_y + panel_height;
            let view_start = self.scroll_offset;
            let view_end = self.scroll_offset + panel_area.height;

            if panel_end > view_start && panel_start < view_end {
                // Calculate visible portion of the panel
                let visible_start = std::cmp::max(panel_start, view_start);
                let visible_end = std::cmp::min(panel_end, view_end);
                let visible_height = visible_end - visible_start;

                if visible_height > 0 {
                    let render_y = panel_area.y + visible_start.saturating_sub(view_start);
                    
                    let render_area = Rect::new(
                        panel_area.x,
                        render_y,
                        panel_area.width,
                        visible_height,
                    );

                    // Create a clipped area for the panel
                    if render_area.height > 0 {
                        panel.render(frame, render_area, app)?;
                    }
                }
            }

            current_y += panel_height;
        }

        // Render scrollbar if needed
        if needs_scrollbar && scrollbar_area.width > 0 {
            let mut scrollbar_state = ScrollbarState::default()
                .content_length(total_height as usize)
                .viewport_content_length(content_area.height as usize)
                .position(self.scroll_offset as usize);

            let scrollbar = Scrollbar::default()
                .orientation(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("↑"))
                .end_symbol(Some("↓"));

            frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
        }

        Ok(())
    }
}
