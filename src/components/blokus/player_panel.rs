//! Modular player panel component for Blokus piece selector using EnhancedPieceGridComponent

use mcts::GameState;
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};
use std::any::Any;

use crate::app::App;
use crate::components::blokus::{EnhancedPieceGridComponent, EnhancedPieceGridConfig};
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crate::game_wrapper::GameWrapper;

/// Component representing a single player's panel in the Blokus piece selector
pub struct BlokusPlayerPanelComponent {
    id: ComponentId,
    player: u8,
    is_expanded: bool,
    piece_grid: Option<EnhancedPieceGridComponent>,
    area: Option<Rect>,
}

impl BlokusPlayerPanelComponent {
    pub fn new(player: u8, is_expanded: bool) -> Self {
        Self {
            id: ComponentId::new(),
            player,
            is_expanded,
            piece_grid: None,
            area: None,
        }
    }

    pub fn set_expanded(&mut self, expanded: bool) {
        self.is_expanded = expanded;
    }

    pub fn is_expanded(&self) -> bool {
        self.is_expanded
    }

    pub fn get_player(&self) -> u8 {
        self.player
    }

    /// Handle click on a piece within the grid
    pub fn handle_piece_click(&mut self, app: &mut App, x: u16, y: u16) -> Option<usize> {
        if let Some(ref mut grid) = self.piece_grid {
            // Convert global coordinates to local grid coordinates
            if let Some(area) = self.area {
                let local_x = x.saturating_sub(area.x);
                let local_y = y.saturating_sub(area.y + 1); // Account for header
                return grid.handle_piece_click(app, local_x, local_y);
            }
        }
        None
    }

    /// Initialize the piece grid if needed
    fn ensure_piece_grid(&mut self) {
        if self.piece_grid.is_none() {
            let player_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
            let mut config = EnhancedPieceGridConfig::default();
            config.player_color = player_colors
                .get((self.player - 1) as usize)
                .cloned()
                .unwrap_or(Color::White);
            config.pieces_per_row = 5; // Smaller for individual panels
            config.piece_width = 5;
            config.piece_height = 3;
            config.show_borders = false; // No individual borders in panel view
            config.show_labels = true;
            config.responsive = true;

            self.piece_grid = Some(EnhancedPieceGridComponent::new(self.player, config));
        }
    }

    /// Calculate the height needed for this panel
    pub fn calculate_height(&self) -> u16 {
        if !self.is_expanded {
            return 3; // Just header when collapsed
        }

        // Header + grid content
        if let Some(ref grid) = self.piece_grid {
            3 + grid.calculate_height()
        } else {
            8 // Default height
        }
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

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        self.area = Some(area);

        // Ensure piece grid is initialized
        self.ensure_piece_grid();

        // Get player color
        let player_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
        let player_color = player_colors
            .get((self.player - 1) as usize)
            .cloned()
            .unwrap_or(Color::White);

        // Create header
        let expansion_indicator = if self.is_expanded { "▼" } else { "▶" };
        let title = format!("{} Player {}", expansion_indicator, self.player);

        let mut header_style = Style::default()
            .fg(player_color)
            .add_modifier(Modifier::BOLD);

        // Highlight current player
        if let GameWrapper::Blokus(state) = &app.game_wrapper {
            let current_player = state.get_current_player();
            if current_player == self.player as i32 {
                header_style = header_style.add_modifier(Modifier::UNDERLINED);
            }
        }

        if self.is_expanded {
            // Render expanded panel with piece grid
            let block = Block::default()
                .borders(Borders::ALL)
                .title(title)
                .border_style(header_style);

            let inner_area = block.inner(area);
            frame.render_widget(block, area);

            // Render piece grid
            if let Some(ref mut grid) = self.piece_grid {
                grid.render(frame, inner_area, app)?;
            }
        } else {
            // Render collapsed panel (header only)
            let header_text = Line::from(Span::styled(title, header_style));
            let paragraph = Paragraph::new(vec![header_text]).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(header_style),
            );
            frame.render_widget(paragraph, area);
        }

        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::MouseClick { x, y, button }) => {
                if let Some(area) = self.area {
                    if *x >= area.x
                        && *x < area.x + area.width
                        && *y >= area.y
                        && *y < area.y + area.height
                    {
                        match button {
                            1 => {
                                // Left click
                                if self.is_expanded {
                                    // Forward to piece grid
                                    if let Some(ref mut grid) = self.piece_grid {
                                        return grid.handle_event(event, app);
                                    }
                                } else {
                                    // Expand on click when collapsed
                                    self.set_expanded(true);
                                    return Ok(true);
                                }
                            }
                            3 => {
                                // Right click - toggle expansion
                                self.set_expanded(!self.is_expanded);
                                return Ok(true);
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {
                // Forward other events to piece grid if expanded
                if self.is_expanded {
                    if let Some(ref mut grid) = self.piece_grid {
                        return grid.handle_event(event, app);
                    }
                }
            }
        }
        Ok(false)
    }
}
