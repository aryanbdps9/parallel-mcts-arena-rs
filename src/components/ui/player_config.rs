//! Player configuration component.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    Frame,
    widgets::{Block, Borders, List, ListItem, Paragraph},
    style::{Style, Color},
};

use crate::app::{App, AppMode, Player};
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crossterm::event::KeyCode;

/// Component for player configuration
pub struct PlayerConfigComponent {
    id: ComponentId,
}

impl PlayerConfigComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
        }
    }
}

impl Component for PlayerConfigComponent {
    fn id(&self) -> ComponentId {
        self.id
    }
    
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(3),
            ])
            .split(area);

        // Title
        let title = Paragraph::new("Configure Players")
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Cyan));
        frame.render_widget(title, chunks[0]);

        // Player list
        let mut items: Vec<ListItem> = app
            .player_options
            .iter()
            .enumerate()
            .map(|(i, (player_id, player_type))| {
                let type_str = match player_type {
                    Player::Human => "Human",
                    Player::AI => "AI",
                };
                let highlight = if i == app.selected_player_config_index {
                    " <--"
                } else {
                    ""
                };
                ListItem::new(format!("Player {}: {}{}", player_id, type_str, highlight))
            })
            .collect();

        // Add "Start Game" option
        let start_highlight = if app.selected_player_config_index >= app.player_options.len() {
            " <--"
        } else {
            ""
        };
        items.push(ListItem::new(format!("Start Game{}", start_highlight)));

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Player Configuration"));

        frame.render_widget(list, chunks[1]);

        // Instructions
        let instructions = Paragraph::new("Use ↑/↓ to navigate, ←/→ or Space to change player type, Enter to start/confirm")
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(instructions, chunks[2]);

        Ok(())
    }
    
    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::KeyPress(key)) => {
                match key {
                    KeyCode::Char('q') => {
                        app.should_quit = true;
                        Ok(true)
                    }
                    KeyCode::Left | KeyCode::Right | KeyCode::Char(' ') => {
                        if app.selected_player_config_index < app.player_options.len() {
                            app.cycle_player_type();
                        }
                        Ok(true)
                    }
                    KeyCode::Up => {
                        app.select_prev_player_config();
                        Ok(true)
                    }
                    KeyCode::Down => {
                        app.select_next_player_config();
                        Ok(true)
                    }
                    KeyCode::Enter => {
                        if app.selected_player_config_index < app.player_options.len() {
                            app.cycle_player_type();
                        } else {
                            app.confirm_player_config();
                        }
                        Ok(true)
                    }
                    KeyCode::Esc => {
                        app.mode = AppMode::GameSelection;
                        Ok(true)
                    }
                    _ => Ok(false)
                }
            }
            _ => Ok(false),
        }
    }

    crate::impl_component_base!(PlayerConfigComponent);
}
