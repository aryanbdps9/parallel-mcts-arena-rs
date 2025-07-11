//! Enhanced piece selector component combining multiple player grids with clean layout.

use ratatui::{
    layout::{Rect, Constraint, Direction, Layout},
    Frame,
    widgets::Paragraph,
    style::{Style, Color},
};
use std::any::Any;
use mcts::GameState;

use crate::app::App;
use crate::game_wrapper::GameWrapper;
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::ComponentEvent;
use crate::components::blokus::{EnhancedPieceGridComponent, EnhancedPieceGridConfig};

/// Enhanced piece selector component that provides a clean, responsive view of all players' pieces
pub struct EnhancedBlokusPieceSelectorComponent {
    id: ComponentId,
    player_grids: Vec<EnhancedPieceGridComponent>,
    area: Option<Rect>,
}

impl EnhancedBlokusPieceSelectorComponent {
    pub fn new() -> Self {
        let player_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
        let mut player_grids = Vec::new();

        for player in 0..4 {
            let mut config = EnhancedPieceGridConfig::default();
            config.player_color = player_colors[player];
            config.pieces_per_row = 7; // Start with 7, will be responsive
            config.piece_width = 6;
            config.piece_height = 3;
            config.show_borders = true;
            config.show_labels = true;
            config.responsive = true;
            
            player_grids.push(EnhancedPieceGridComponent::new((player + 1) as u8, config));
        }

        Self {
            id: ComponentId::new(),
            player_grids,
            area: None,
        }
    }

    /// Calculate the optimal layout for player grids based on available space
    fn calculate_layout(&self, area: Rect) -> Vec<Rect> {
        // Try different layouts based on available space
        let total_height = area.height;
        let total_width = area.width;

        // Calculate minimum height needed for each player
        let min_height_per_player = 8; // Minimum viable height
        let preferred_height_per_player = 12; // Comfortable height

        // Determine layout strategy
        if total_height >= preferred_height_per_player * 4 {
            // Vertical stack - plenty of space
            let constraints = vec![Constraint::Percentage(25); 4];
            Layout::default()
                .direction(Direction::Vertical)
                .constraints(constraints)
                .split(area)
                .to_vec()
        } else if total_height >= min_height_per_player * 2 && total_width >= 60 {
            // 2x2 grid layout
            let rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(area);
            
            let mut areas = Vec::new();
            for row in rows.iter() {
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(*row);
                areas.extend(cols.iter().cloned());
            }
            areas
        } else {
            // Horizontal layout for very limited space
            let constraints = vec![Constraint::Percentage(25); 4];
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints(constraints)
                .split(area)
                .to_vec()
        }
    }

    /// Update all player grids with current game state
    fn update_player_grids(&mut self, app: &App) {
        if let GameWrapper::Blokus(state) = &app.game_wrapper {
            let current_player = app.game_wrapper.get_current_player();
            
            for (i, grid) in self.player_grids.iter_mut().enumerate() {
                let player_id = (i + 1) as u32;
                let available_pieces = state.get_available_pieces(player_id as i32);
                
                grid.set_available_pieces(available_pieces);
                grid.set_current_player(current_player == player_id as i32);
                
                // Set selected piece for current player
                if current_player == player_id as i32 {
                    grid.set_selected_piece(app.blokus_ui_config.selected_piece_idx);
                } else {
                    grid.set_selected_piece(None);
                }
            }
        }
    }
}

impl Component for EnhancedBlokusPieceSelectorComponent {
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
        
        // Update all player grids with current state
        self.update_player_grids(app);

        // Calculate layout for player grids
        let grid_areas = self.calculate_layout(area);

        // Ensure we have enough areas
        if grid_areas.len() < 4 {
            let error_msg = Paragraph::new("Insufficient space for piece selector")
                .style(Style::default().fg(Color::Red));
            frame.render_widget(error_msg, area);
            return Ok(());
        }

        // Render each player grid
        for (i, grid) in self.player_grids.iter_mut().enumerate() {
            if i < grid_areas.len() {
                grid.render(frame, grid_areas[i], app)?;
            }
        }

        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        // Forward events to all player grids
        for grid in &mut self.player_grids {
            if let Ok(true) = grid.handle_event(event, app) {
                return Ok(true);
            }
        }
        Ok(false)
    }
}
