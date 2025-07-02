//! # Menu Components
//!
//! UI components for the various menus (game selection, settings, player config).

use crate::app::{App, AppMode, Player};
use crate::components::core::{Component, ComponentId, ComponentResult, UpdateResult};
use crate::components::events::{ComponentEvent, EventResult, InputEvent};
use crate::components::ui::common::List;
use crate::impl_component_base;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

/// Game selection menu component
pub struct GameSelectionMenu {
    id: ComponentId,
    game_list: List,
    visible: bool,
}

impl GameSelectionMenu {
    pub fn new() -> Self {
        let game_names = vec![
            "Gomoku".to_string(),
            "Connect4".to_string(), 
            "Othello".to_string(),
            "Blokus".to_string(),
            "Settings".to_string(),
            "Quit".to_string(),
        ];

        let mut game_list = List::new()
            .with_items(game_names)
            .with_on_select(|index, app| {
                match index {
                    0 => {
                        // Gomoku selected
                        app.game_wrapper = app.games[0].1();
                        app.mode = if app.ai_only { AppMode::InGame } else { AppMode::PlayerConfig };
                    }
                    1 => {
                        // Connect4 selected
                        app.game_wrapper = app.games[1].1();
                        app.mode = if app.ai_only { AppMode::InGame } else { AppMode::PlayerConfig };
                    }
                    2 => {
                        // Othello selected
                        app.game_wrapper = app.games[2].1();
                        app.mode = if app.ai_only { AppMode::InGame } else { AppMode::PlayerConfig };
                    }
                    3 => {
                        // Blokus selected
                        app.game_wrapper = app.games[3].1();
                        app.mode = if app.ai_only { AppMode::InGame } else { AppMode::PlayerConfig };
                    }
                    4 => {
                        // Settings selected
                        app.mode = AppMode::Settings;
                    }
                    5 => {
                        // Quit selected
                        app.should_quit = true;
                    }
                    _ => {}
                }
            });

        game_list.set_focus(true); // Start with focus

        Self {
            id: ComponentId::new(),
            game_list,
            visible: true,
        }
    }
}

impl Component for GameSelectionMenu {
    impl_component_base!(Self, "GameSelectionMenu");

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        if !self.visible {
            return Ok(());
        }

        // Create layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(10), Constraint::Length(3)])
            .split(area);

        // Create title
        let title = if app.ai_only {
            "Select a Game (AI-Only Mode)"
        } else {
            "Select a Game"
        };

        // Render game list with title
        let list_area = chunks[0];
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title);
        let inner_area = block.inner(list_area);
        frame.render_widget(block, list_area);
        
        self.game_list.render(frame, inner_area, app)?;

        // Instructions
        let instructions = Paragraph::new("Use Up/Down to navigate, Enter to select, F10 to quit")
            .block(Block::default().borders(Borders::ALL).title("Instructions"));
        frame.render_widget(instructions, chunks[1]);

        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        self.game_list.handle_event(event, app)
    }

    fn update(&mut self, app: &mut App) -> UpdateResult {
        self.game_list.update(app)
    }

    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        vec![&mut self.game_list]
    }

    fn children(&self) -> Vec<&dyn Component> {
        vec![&self.game_list]
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn on_mount(&mut self, _app: &mut App) {
        self.game_list.set_focus(true);
    }

    fn on_unmount(&mut self, _app: &mut App) {
        self.game_list.set_focus(false);
    }
}

/// Settings menu component
pub struct SettingsMenu {
    id: ComponentId,
    settings_list: List,
    visible: bool,
}

impl SettingsMenu {
    pub fn new() -> Self {
        let mut settings_list = List::new()
            .with_on_select(|index, app| {
                match index {
                    11 => {
                        // Back option
                        app.mode = AppMode::GameSelection;
                    }
                    _ => {
                        // TODO: Handle other settings
                    }
                }
            });

        settings_list.set_focus(true);

        Self {
            id: ComponentId::new(),
            settings_list,
            visible: true,
        }
    }

    fn update_settings_items(&mut self, app: &App) {
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
        self.settings_list.set_items(settings_items);
    }
}

impl Component for SettingsMenu {
    impl_component_base!(Self, "SettingsMenu");

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        if !self.visible {
            return Ok(());
        }

        self.update_settings_items(app);

        // Create layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(10), Constraint::Length(3)])
            .split(area);

        // Render settings list
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Settings");
        let inner_area = block.inner(chunks[0]);
        frame.render_widget(block, chunks[0]);
        
        self.settings_list.render(frame, inner_area, app)?;

        // Instructions
        let instructions = Paragraph::new("Use Up/Down to navigate, Left/Right to adjust values, Enter to confirm, Esc to go back")
            .block(Block::default().borders(Borders::ALL).title("Instructions"));
        frame.render_widget(instructions, chunks[1]);

        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        // Handle Escape key
        if let ComponentEvent::Input(InputEvent::KeyPress(crossterm::event::KeyCode::Esc)) = event {
            app.mode = AppMode::GameSelection;
            return EventResult::Handled;
        }

        // TODO: Handle Left/Right for adjusting values
        
        self.settings_list.handle_event(event, app)
    }

    fn update(&mut self, app: &mut App) -> UpdateResult {
        self.settings_list.update(app)
    }

    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        vec![&mut self.settings_list]
    }

    fn children(&self) -> Vec<&dyn Component> {
        vec![&self.settings_list]
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn on_mount(&mut self, _app: &mut App) {
        self.settings_list.set_focus(true);
    }

    fn on_unmount(&mut self, _app: &mut App) {
        self.settings_list.set_focus(false);
    }
}

/// Player configuration menu component
pub struct PlayerConfigMenu {
    id: ComponentId,
    player_list: List,
    visible: bool,
}

impl PlayerConfigMenu {
    pub fn new() -> Self {
        let mut player_list = List::new()
            .with_on_select(|index, app| {
                if index >= app.player_options.len() {
                    // Start Game option
                    app.mode = AppMode::InGame;
                } else {
                    // Toggle player type
                    let (player_id, current_type) = app.player_options[index];
                    let new_type = match current_type {
                        Player::Human => Player::AI,
                        Player::AI => Player::Human,
                    };
                    app.player_options[index] = (player_id, new_type);
                }
            });

        player_list.set_focus(true);

        Self {
            id: ComponentId::new(),
            player_list,
            visible: true,
        }
    }

    fn update_player_items(&mut self, app: &App) {
        let mut items: Vec<String> = app
            .player_options
            .iter()
            .map(|(id, p_type)| {
                let type_str = match p_type {
                    Player::Human => "Human",
                    Player::AI => "AI",
                };
                format!("Player {}: {}", id, type_str)
            })
            .collect();

        items.push("Start Game".to_string());
        self.player_list.set_items(items);
    }
}

impl Component for PlayerConfigMenu {
    impl_component_base!(Self, "PlayerConfigMenu");

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        if !self.visible {
            return Ok(());
        }

        self.update_player_items(app);

        // Create layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(10), Constraint::Length(3)])
            .split(area);

        // Create title
        let title = if app.ai_only {
            format!("{} - Player Configuration (AI Only Mode)", app.get_selected_game_name())
        } else {
            format!("{} - Player Configuration", app.get_selected_game_name())
        };

        // Render player list
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title);
        let inner_area = block.inner(chunks[0]);
        frame.render_widget(block, chunks[0]);
        
        self.player_list.render(frame, inner_area, app)?;

        // Instructions
        let instructions_text = if app.ai_only {
            "AI Only Mode: All players will be set to AI automatically. Enter to start game, Esc to go back"
        } else {
            "Use Up/Down to navigate, Enter to toggle Human/AI or start game, Esc to go back"
        };

        let instructions = Paragraph::new(instructions_text)
            .block(Block::default().borders(Borders::ALL).title("Instructions"));
        frame.render_widget(instructions, chunks[1]);

        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        // Handle Escape key
        if let ComponentEvent::Input(InputEvent::KeyPress(crossterm::event::KeyCode::Esc)) = event {
            app.mode = AppMode::GameSelection;
            return EventResult::Handled;
        }

        self.player_list.handle_event(event, app)
    }

    fn update(&mut self, app: &mut App) -> UpdateResult {
        self.player_list.update(app)
    }

    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        vec![&mut self.player_list]
    }

    fn children(&self) -> Vec<&dyn Component> {
        vec![&self.player_list]
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn on_mount(&mut self, _app: &mut App) {
        self.player_list.set_focus(true);
    }

    fn on_unmount(&mut self, _app: &mut App) {
        self.player_list.set_focus(false);
    }
}
