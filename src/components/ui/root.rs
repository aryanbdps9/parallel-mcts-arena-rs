//! Root component implementation.

use ratatui::{
    layout::Rect,
    Frame,
};

use crate::app::{App, AppMode};
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::ComponentEvent;
use crate::components::ui::{
    game_selection::GameSelectionComponent,
    settings::SettingsComponent,
    player_config::PlayerConfigComponent,
    in_game::InGameComponent,
    game_over::GameOverComponent,
};

/// The root component that serves as the top-level container
/// This component manages the overall application layout and delegates to child components
pub struct RootComponent {
    id: ComponentId,
    game_selection: GameSelectionComponent,
    settings: SettingsComponent,
    player_config: PlayerConfigComponent,
    in_game: InGameComponent,
    game_over: GameOverComponent,
}

impl RootComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            game_selection: GameSelectionComponent::new(),
            settings: SettingsComponent::new(),
            player_config: PlayerConfigComponent::new(),
            in_game: InGameComponent::new(),
            game_over: GameOverComponent::new(),
        }
    }
}

impl Component for RootComponent {
    fn id(&self) -> ComponentId {
        self.id
    }
    
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        match app.mode {
            AppMode::GameSelection => self.game_selection.render(frame, area, app),
            AppMode::Settings => self.settings.render(frame, area, app),
            AppMode::PlayerConfig => self.player_config.render(frame, area, app),
            AppMode::InGame => self.in_game.render(frame, area, app),
            AppMode::GameOver => self.game_over.render(frame, area, app),
        }
    }
    
    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match app.mode {
            AppMode::GameSelection => self.game_selection.handle_event(event, app),
            AppMode::Settings => self.settings.handle_event(event, app),
            AppMode::PlayerConfig => self.player_config.handle_event(event, app),
            AppMode::InGame => self.in_game.handle_event(event, app),
            AppMode::GameOver => self.game_over.handle_event(event, app),
        }
    }

    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        vec![
            &mut self.game_selection,
            &mut self.settings,
            &mut self.player_config,
            &mut self.in_game,
            &mut self.game_over,
        ]
    }
    
    fn children(&self) -> Vec<&dyn Component> {
        vec![
            &self.game_selection,
            &self.settings,
            &self.player_config,
            &self.in_game,
            &self.game_over,
        ]
    }
    
    crate::impl_component_base!(RootComponent);
}