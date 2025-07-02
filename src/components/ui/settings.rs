//! Settings component.

use ratatui::{
    layout::Rect,
    Frame,
    widgets::{Block, Borders, List, ListItem},
    style::{Style, Modifier},
};

use crate::app::{App, AppMode};
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crossterm::event::KeyCode;

/// Component for settings menu
pub struct SettingsComponent {
    id: ComponentId,
}

impl SettingsComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
        }
    }
}

impl Component for SettingsComponent {
    fn id(&self) -> ComponentId {
        self.id
    }
    
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        let settings_items = vec![
            format!("Board Size: {}", app.settings_board_size),
            format!("Line Size: {}", app.settings_line_size),
            format!("AI Threads: {}", app.settings_ai_threads),
            format!("Max Nodes: {}", app.settings_max_nodes),
            format!("Search Iterations: {}", app.settings_search_iterations),
            format!("Exploration Constant: {:.2}", app.settings_exploration_constant),
            format!("Timeout (secs): {}", app.timeout_secs),
            format!("Stats Interval (secs): {}", app.stats_interval_secs),
            format!("AI Only Mode: {}", if app.ai_only { "Yes" } else { "No" }),
            format!("Shared Tree: {}", if app.shared_tree { "Yes" } else { "No" }),
            "".to_string(), // Separator
            "Back".to_string(),
        ];

        let items: Vec<ListItem> = settings_items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let style = if i == app.selected_settings_index {
                    Style::default().add_modifier(Modifier::REVERSED)
                } else {
                    Style::default()
                };
                ListItem::new(item.as_str()).style(style)
            })
            .collect();

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Settings"))
            .highlight_symbol("> ");

        frame.render_widget(list, area);
        Ok(())
    }
    
    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::MouseClick { x: _, y: _, button: _ }) => {
                // Calculate which setting was clicked based on mouse position
                // For now, just handle it as basic click without precise position mapping
                Ok(false) // TODO: Add precise mouse click handling for settings
            }
            ComponentEvent::Input(InputEvent::KeyPress(key)) => {
                match key {
                    KeyCode::Char('q') => {
                        app.should_quit = true;
                        Ok(true)
                    }
                    KeyCode::Up => {
                        app.select_prev_setting();
                        Ok(true)
                    }
                    KeyCode::Down => {
                        app.select_next_setting();
                        Ok(true)
                    }
                    KeyCode::Left => {
                        app.decrease_setting();
                        Ok(true)
                    }
                    KeyCode::Right => {
                        app.increase_setting();
                        Ok(true)
                    }
                    KeyCode::Enter => {
                        if app.selected_settings_index == 11 { // "Back" option
                            app.mode = AppMode::GameSelection;
                            app.apply_settings_to_current_game();
                        }
                        Ok(true)
                    }
                    KeyCode::Esc => {
                        app.mode = AppMode::GameSelection;
                        app.apply_settings_to_current_game();
                        Ok(true)
                    }
                    _ => Ok(false)
                }
            }
            _ => Ok(false),
        }
    }

    crate::impl_component_base!(SettingsComponent);
}
