//! Improved piece selector component with scrollable column layout and better UX.

use ratatui::{
    layout::Rect,
    Frame,
    widgets::{Block, Borders, Paragraph},
    style::{Style, Color, Modifier},
    text::{Line, Span},
};
use std::any::Any;
use mcts::GameState;

use crate::app::App;
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crate::components::blokus::{ResponsivePieceGridComponent, ResponsivePieceGridConfig};

/// Configuration for the improved piece selector
#[derive(Clone)]
pub struct ImprovedPieceSelectorConfig {
    pub show_all_players: bool,
    pub current_player_priority: bool,
    pub show_scrollbar: bool,
    pub auto_scroll_to_current: bool,
    pub compact_other_players: bool,
}

impl Default for ImprovedPieceSelectorConfig {
    fn default() -> Self {
        Self {
            show_all_players: true,
            current_player_priority: true,
            show_scrollbar: false,
            auto_scroll_to_current: true,
            compact_other_players: true,
        }
    }
}

/// Player panel wrapper for the improved selector
struct ImprovedPlayerPanel {
    player: u8,
    grid: ResponsivePieceGridComponent,
    is_expanded: bool,
    is_current_player: bool,
    height: u16,
}

impl ImprovedPlayerPanel {
    fn new(player: u8, expanded: bool) -> Self {
        let player_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
        let theme = crate::components::ui::theme::UITheme::default();
        let (empty_light, empty_dark) = theme.empty_cell_colors();
        
        let mut config = ResponsivePieceGridConfig::default();
        config.player_color = player_colors.get((player - 1) as usize).cloned().unwrap_or(Color::White);
        config.min_pieces_per_row = 4;
        config.max_pieces_per_row = 7;
        config.piece_width = 8;
        config.piece_height = 4;
        config.show_borders = true;
        config.show_labels = true;
        config.compact_mode = false;
        
        // Use theme colors for checkerboard pattern (same as board)
        config.empty_cell_light = empty_light;
        config.empty_cell_dark = empty_dark;

        Self {
            player,
            grid: ResponsivePieceGridComponent::new(player, config),
            is_expanded: expanded,
            is_current_player: false,
            height: 0,
        }
    }

    fn update_state(&mut self, app: &App) {
        let current_player = app.game_wrapper.get_current_player();
        self.is_current_player = current_player == self.player as i32;
        
        // Auto-expand current player, keep others as configured
        if self.is_current_player {
            self.is_expanded = true;
        }
    }

    fn calculate_height(&mut self, _area_width: u16, config: &ImprovedPieceSelectorConfig) -> u16 {
        if !self.is_expanded {
            // Collapsed: just show header
            self.height = 3; // Header + borders
            return self.height;
        }

        if config.compact_other_players && !self.is_current_player {
            // Compact mode for other players - smaller grid
            self.height = 8; // Fixed compact height
        } else {
            // Full size for current player
            self.height = self.grid.calculate_height().max(12);
        }

        self.height
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App, config: &ImprovedPieceSelectorConfig) -> ComponentResult<()> {
        let player_name = ["Red", "Blue", "Green", "Yellow"][(self.player - 1) as usize];
        let player_color = [Color::Red, Color::Blue, Color::Green, Color::Yellow][(self.player - 1) as usize];

        if !self.is_expanded {
            // Render collapsed header
            let expand_indicator = if self.is_current_player { "►" } else { "▷" };
            let header_text = format!("{} {} Player {} (collapsed)", expand_indicator, player_name, self.player);
            
            let style = if self.is_current_player {
                Style::default().fg(player_color).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(player_color)
            };

            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(style);
            let inner = block.inner(area);
            frame.render_widget(block, area);

            let header = Paragraph::new(Line::from(Span::styled(header_text, style)));
            frame.render_widget(header, inner);
            return Ok(());
        }

        // Render expanded content
        if config.compact_other_players && !self.is_current_player {
            // Render compact grid for other players
            let mut compact_config = self.grid.get_config().clone();
            compact_config.piece_width = 6;
            compact_config.piece_height = 3;
            compact_config.min_pieces_per_row = 5;
            compact_config.max_pieces_per_row = 8;
            compact_config.show_labels = false;
            compact_config.compact_mode = true;

            let mut compact_grid = ResponsivePieceGridComponent::new(self.player, compact_config);
            compact_grid.render(frame, area, app)?;
        } else {
            // Render full grid
            self.grid.render(frame, area, app)?;
        }

        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::MouseClick { x, y, .. }) => {
                if let Some(area) = self.get_area() {
                    if *x >= area.x && *x < area.x + area.width &&
                       *y >= area.y && *y < area.y + area.height {
                        
                        // Check if clicking on header to toggle expansion
                        if !self.is_expanded && *y == area.y + 1 {
                            self.is_expanded = true;
                            return Ok(true);
                        }
                        
                        // Forward to grid if expanded
                        if self.is_expanded {
                            return self.grid.handle_event(event, app);
                        }
                    }
                }
            }
            _ => {
                if self.is_expanded {
                    return self.grid.handle_event(event, app);
                }
            }
        }
        Ok(false)
    }

    fn get_area(&self) -> Option<Rect> {
        self.grid.get_area()
    }
}

/// Improved piece selector with better UX and responsive design
pub struct ImprovedBlokusPieceSelectorComponent {
    id: ComponentId,
    config: ImprovedPieceSelectorConfig,
    player_panels: Vec<ImprovedPlayerPanel>,
    scroll_offset: u16,
    max_scroll: u16,
    area: Option<Rect>,
}

impl ImprovedBlokusPieceSelectorComponent {
    pub fn new() -> Self {
        let config = ImprovedPieceSelectorConfig::default();
        let mut player_panels = Vec::new();
        
        for player in 1..=4 {
            player_panels.push(ImprovedPlayerPanel::new(player, true));
        }

        Self {
            id: ComponentId::new(),
            config,
            player_panels,
            scroll_offset: 0,
            max_scroll: 0,
            area: None,
        }
    }

    pub fn with_config(mut self, config: ImprovedPieceSelectorConfig) -> Self {
        self.config = config;
        self
    }

    /// Toggle expansion for a specific player
    pub fn toggle_player_expanded(&mut self, player: u8) {
        if let Some(panel) = self.player_panels.get_mut((player - 1) as usize) {
            panel.is_expanded = !panel.is_expanded;
        }
    }

    /// Calculate total content height and update scroll bounds
    fn update_scroll_bounds(&mut self, available_height: u16) {
        let total_content_height: u16 = self.player_panels.iter().map(|p| p.height).sum();
        
        if total_content_height > available_height {
            self.max_scroll = total_content_height.saturating_sub(available_height);
        } else {
            self.max_scroll = 0;
            self.scroll_offset = 0;
        }

        // Update scrollbar state
        // Scrollbar removed to prevent jittery behavior
    }

    /// Auto-scroll to show current player if configured
    fn auto_scroll_to_current_player(&mut self, app: &App) {
        if !self.config.auto_scroll_to_current {
            return;
        }

        let current_player = app.game_wrapper.get_current_player();
        let mut current_player_start = 0;
        let mut current_player_height = 0;

        for panel in &self.player_panels {
            if panel.player == current_player as u8 {
                current_player_height = panel.height;
                break;
            }
            current_player_start += panel.height;
        }

        let visible_height = if let Some(area) = self.area {
            area.height.saturating_sub(2) // Account for borders
        } else {
            return;
        };

        let panel_end = current_player_start + current_player_height;
        let view_end = self.scroll_offset + visible_height;

        if current_player_start < self.scroll_offset {
            // Panel starts above visible area - scroll up
            self.scroll_offset = current_player_start;
        } else if panel_end > view_end {
            // Panel ends below visible area - scroll down
            self.scroll_offset = panel_end.saturating_sub(visible_height);
        }

        // Ensure we don't scroll beyond bounds
        self.scroll_offset = self.scroll_offset.min(self.max_scroll);
    }

    /// Handle mouse scroll
    pub fn handle_scroll(&mut self, up: bool) {
        if up {
            self.scroll_offset = self.scroll_offset.saturating_sub(3);
        } else {
            self.scroll_offset = (self.scroll_offset + 3).min(self.max_scroll);
        }

        // Scrollbar position update removed since scrollbar is disabled
    }
}

impl Component for ImprovedBlokusPieceSelectorComponent {
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

        // Update all player states
        for panel in &mut self.player_panels {
            panel.update_state(app);
        }

        // Split area for content only (no scrollbar)
        let content_area = area;

        // Draw main border
        let block = Block::default()
            .title("Blokus Pieces")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));
        frame.render_widget(block, content_area);

        // Calculate inner area
        let inner_area = Rect::new(
            content_area.x + 1,
            content_area.y + 1,
            content_area.width.saturating_sub(2),
            content_area.height.saturating_sub(2),
        );

        if inner_area.width == 0 || inner_area.height == 0 {
            return Ok(());
        }

        // Calculate panel heights based on available space
        let available_width = inner_area.width;
        for panel in &mut self.player_panels {
            panel.calculate_height(available_width, &self.config);
        }

        // Update scroll bounds and auto-scroll
        self.update_scroll_bounds(inner_area.height);
        self.auto_scroll_to_current_player(app);

        // Render visible panels
        let mut current_y = 0i32;
        let scroll_offset = self.scroll_offset as i32;
        let visible_end = scroll_offset + inner_area.height as i32;

        for panel in &mut self.player_panels {
            let panel_height = panel.height as i32;
            let panel_start = current_y;
            let panel_end = current_y + panel_height;

            // Check if panel is visible
            if panel_end > scroll_offset && panel_start < visible_end {
                // Calculate visible portion
                let visible_start = panel_start.max(scroll_offset);
                let visible_height = (panel_end.min(visible_end) - visible_start) as u16;
                
                if visible_height > 0 {
                    let panel_area = Rect::new(
                        inner_area.x,
                        inner_area.y + (visible_start - scroll_offset) as u16,
                        inner_area.width,
                        visible_height.min(panel.height),
                    );

                    panel.render(frame, panel_area, app, &self.config)?;
                }
            }

            current_y += panel_height;
        }

        // Remove scrollbar rendering since it's disabled
        // Scrollbar was causing jittery behavior

        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::MouseScroll { x, y, up }) => {
                if let Some(area) = self.area {
                    if *x >= area.x && *x < area.x + area.width &&
                       *y >= area.y && *y < area.y + area.height {
                        self.handle_scroll(*up);
                        return Ok(true);
                    }
                }
            }
            ComponentEvent::Input(InputEvent::MouseClick { .. }) => {
                // Calculate visible panel areas and forward clicks
                let Some(area) = self.area else { return Ok(false); };
                let inner_area = Rect::new(
                    area.x + 1,
                    area.y + 1,
                    area.width.saturating_sub(2),
                    area.height.saturating_sub(2),
                );

                let mut current_y = 0i32;
                let scroll_offset = self.scroll_offset as i32;
                let visible_end = scroll_offset + inner_area.height as i32;

                for panel in &mut self.player_panels {
                    let panel_height = panel.height as i32;
                    let panel_start = current_y;
                    let panel_end = current_y + panel_height;

                    // Check if panel is visible
                    if panel_end > scroll_offset && panel_start < visible_end {
                        let visible_start = panel_start.max(scroll_offset);
                        let visible_height = (panel_end.min(visible_end) - visible_start) as u16;
                        
                        if visible_height > 0 {
                            let panel_area = Rect::new(
                                inner_area.x,
                                inner_area.y + (visible_start - scroll_offset) as u16,
                                inner_area.width,
                                visible_height.min(panel.height),
                            );

                            // Set the panel area and forward event
                            panel.grid.set_area(Some(panel_area));
                            if panel.handle_event(event, app)? {
                                return Ok(true);
                            }
                        }
                    }

                    current_y += panel_height;
                }
            }
            _ => {}
        }
        Ok(false)
    }
}
