//! Game selection component.

use ratatui::{
    layout::Rect,
    Frame,
    widgets::{Block, Borders, List, ListItem},
    style::{Style, Color, Modifier},
};

use crate::app::App;
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crossterm::event::KeyCode;
use mcts::GameState;

/// Component for game selection menu
pub struct GameSelectionComponent {
    id: ComponentId,
}

impl GameSelectionComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
        }
    }
}

impl Component for GameSelectionComponent {
    fn id(&self) -> ComponentId {
        self.id
    }
    
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        let mut items: Vec<ListItem> = app
            .games
            .iter()
            .map(|(name, _)| ListItem::new(*name))
            .collect();

        items.push(ListItem::new("Settings"));
        items.push(ListItem::new("Quit"));

        let title = if app.ai_only {
            "Select a Game (AI-Only Mode)"
        } else {
            "Select a Game"
        };

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(title))
            .highlight_style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .fg(Color::Yellow),
            )
            .highlight_symbol("> ");

        frame.render_stateful_widget(list, area, &mut app.game_selection_state.clone());
        Ok(())
    }
    
    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::KeyPress(key)) => {
                match key {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        app.should_quit = true;
                        Ok(true)
                    }
                    KeyCode::Up => {
                        app.select_prev_game();
                        Ok(true)
                    }
                    KeyCode::Down => {
                        app.select_next_game();
                        Ok(true)
                    }
                    KeyCode::Enter => {
                        if let Some(selected) = app.game_selection_state.selected() {
                            if selected < app.games.len() {
                                // Selected a game - initialize it with current settings and go to player config
                                let game_name = app.games[selected].0;
                                app.game_wrapper = app.create_game_with_current_settings(game_name);
                                app.game_status = crate::app::GameStatus::InProgress;
                                app.last_search_stats = None;
                                app.move_history.clear();

                                let num_players = app.game_wrapper.get_num_players();
                                
                                // Only reset player options if we don't have the right number of players
                                if app.player_options.is_empty() || app.player_options.len() != num_players as usize {
                                    app.player_options = (1..=num_players).map(|i| (i, crate::app::Player::Human)).collect();
                                    app.selected_player_config_index = 0;
                                }

                                // If AI-only mode is enabled, skip player config and go straight to game
                                if app.ai_only {
                                    // Set all players to AI
                                    for (_, player_type) in &mut app.player_options {
                                        *player_type = crate::app::Player::AI;
                                    }
                                    app.confirm_player_config();
                                } else {
                                    app.mode = crate::app::AppMode::PlayerConfig;
                                }
                            } else if selected == app.games.len() {
                                // Selected Settings
                                app.mode = crate::app::AppMode::Settings;
                            } else if selected == app.games.len() + 1 {
                                // Selected Quit
                                app.should_quit = true;
                            }
                        }
                        Ok(true)
                    }
                    _ => Ok(false)
                }
            }
            _ => Ok(false),
        }
    }

    crate::impl_component_base!(GameSelectionComponent);
}
