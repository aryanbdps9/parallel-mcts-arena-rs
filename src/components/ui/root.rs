//! Root component implementation.

use ratatui::{
    layout::Rect,
    Frame,
    widgets::{Block, Borders, List, ListItem},
    style::{Style, Color, Modifier},
};

use crate::app::{App, AppMode};
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crossterm::event::KeyCode;

/// The root component that serves as the top-level container
/// This component manages the overall application layout and delegates to child components
pub struct RootComponent {
    id: ComponentId,
}

impl RootComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
        }
    }
    
    /// Render the appropriate UI based on the current app mode
    pub fn render_with_app(&self, frame: &mut Frame, area: Rect, app: &mut App) {
        // For modes we've migrated to components, use the component system
        // For others, fall back to the legacy widget system
        match app.mode {
            AppMode::GameSelection | AppMode::Settings => {
                let _ = self.render_component_based(frame, area, app);
            }
            AppMode::PlayerConfig | AppMode::InGame | AppMode::GameOver => {
                // Use legacy widget system for these modes until we migrate them
                crate::tui::widgets::render(app, frame);
            }
        }
    }
    
    /// Render using the component system
    fn render_component_based(&self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        match app.mode {
            AppMode::GameSelection => self.render_game_selection(frame, area, app),
            AppMode::Settings => self.render_settings(frame, area, app),
            AppMode::PlayerConfig => self.render_player_config(frame, area, app),
            AppMode::InGame | AppMode::GameOver => self.render_game_view(frame, area, app),
        }
    }
    
    /// Render game selection menu
    fn render_game_selection(&self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
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
    
    /// Render settings menu
    fn render_settings(&self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
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
            .map(|item| ListItem::new(item.as_str()))
            .collect();

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Settings"))
            .highlight_style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .fg(Color::Yellow),
            )
            .highlight_symbol("> ");

        frame.render_stateful_widget(list, area, &mut app.settings_state.clone());
        Ok(())
    }
    
    /// Render player configuration
    fn render_player_config(&self, frame: &mut Frame, area: Rect, _app: &App) -> ComponentResult<()> {
        // Placeholder - TODO: Implement proper component-based player config
        let block = Block::default()
            .title("Player Configuration")
            .borders(Borders::ALL);
            
        let paragraph = ratatui::widgets::Paragraph::new("Player configuration will be handled here.")
            .block(block);
            
        frame.render_widget(paragraph, area);
        Ok(())
    }
    
    /// Render game view
    fn render_game_view(&self, frame: &mut Frame, area: Rect, _app: &App) -> ComponentResult<()> {
        // Placeholder - TODO: Implement proper component-based game view
        let block = Block::default()
            .title("Game View")
            .borders(Borders::ALL);
            
        let paragraph = ratatui::widgets::Paragraph::new("Game view will be handled here.")
            .block(block);
            
        frame.render_widget(paragraph, area);
        Ok(())
    }
}

impl Component for RootComponent {
    fn id(&self) -> ComponentId {
        self.id
    }
    
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        match app.mode {
            AppMode::GameSelection | AppMode::Settings => {
                self.render_component_based(frame, area, app)
            }
            AppMode::PlayerConfig | AppMode::InGame | AppMode::GameOver => {
                // Show a message that these modes are handled by legacy system
                let block = Block::default()
                    .title("Legacy UI Mode")
                    .borders(Borders::ALL);
                    
                let message = match app.mode {
                    AppMode::PlayerConfig => "Player Configuration (Legacy)",
                    AppMode::InGame => "Game View (Legacy)",
                    AppMode::GameOver => "Game Over (Legacy)",
                    _ => "Unknown Mode",
                };
                
                let paragraph = ratatui::widgets::Paragraph::new(message)
                    .block(block);
                    
                frame.render_widget(paragraph, area);
                Ok(())
            }
        }
    }
    
    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::KeyPress(key)) => {
                match app.mode {
                    AppMode::GameSelection => self.handle_game_selection_input(*key, app),
                    AppMode::Settings => self.handle_settings_input(*key, app),
                    _ => Ok(false), // Let legacy system handle other modes for now
                }
            }
            _ => Ok(false),
        }
    }
    
    crate::impl_component_base!(RootComponent);
}

impl RootComponent {
    /// Handle input for game selection
    fn handle_game_selection_input(&mut self, key: KeyCode, app: &mut App) -> EventResult {
        match key {
            KeyCode::Up => {
                if app.game_selection_state.selected().unwrap_or(0) > 0 {
                    app.game_selection_state.select(Some(app.game_selection_state.selected().unwrap() - 1));
                }
                Ok(true)
            }
            KeyCode::Down => {
                let total_items = app.games.len() + 2; // games + settings + quit
                let current = app.game_selection_state.selected().unwrap_or(0);
                if current < total_items - 1 {
                    app.game_selection_state.select(Some(current + 1));
                }
                Ok(true)
            }
            KeyCode::Enter => {
                let selected = app.game_selection_state.selected().unwrap_or(0);
                if selected < app.games.len() {
                    // Game selected
                    app.game_wrapper = app.games[selected].1();
                    app.mode = if app.ai_only { AppMode::InGame } else { AppMode::PlayerConfig };
                } else if selected == app.games.len() {
                    // Settings selected
                    app.mode = AppMode::Settings;
                } else {
                    // Quit selected
                    app.should_quit = true;
                }
                Ok(true)
            }
            KeyCode::Esc | KeyCode::Char('q') => {
                app.should_quit = true;
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    
    /// Handle input for settings
    fn handle_settings_input(&mut self, key: KeyCode, app: &mut App) -> EventResult {
        match key {
            KeyCode::Up => {
                if app.settings_state.selected().unwrap_or(0) > 0 {
                    app.settings_state.select(Some(app.settings_state.selected().unwrap() - 1));
                }
                Ok(true)
            }
            KeyCode::Down => {
                let total_items = 12; // Total settings items including "Back"
                let current = app.settings_state.selected().unwrap_or(0);
                if current < total_items - 1 {
                    app.settings_state.select(Some(current + 1));
                }
                Ok(true)
            }
            KeyCode::Enter | KeyCode::Esc => {
                let selected = app.settings_state.selected().unwrap_or(0);
                if selected == 11 { // "Back" option
                    app.mode = AppMode::GameSelection;
                }
                // TODO: Handle editing individual settings
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}