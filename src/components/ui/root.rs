//! # Root Component - Application Shell
//!
//! This module implements the top-level RootComponent that serves as the application shell.
//! It's responsible for coordinating between different major UI modes and delegating
//! rendering and event handling to the appropriate child components.
//!
//! ## Design Philosophy
//! The RootComponent follows a simple delegation pattern where it determines which
//! child component should be active based on the current AppMode, then forwards
//! all rendering and event handling to that component.
//!
//! ## Component Hierarchy
//! ```text
//! RootComponent (this file)
//! ├── GameSelectionComponent (main menu)
//! ├── SettingsComponent (configuration screens)
//! ├── PlayerConfigComponent (player setup)
//! ├── InGameComponent (active gameplay)
//! └── GameOverComponent (end game results)
//! ```
//!
//! ## State Management
//! The RootComponent itself is stateless - it doesn't maintain any internal state
//! beyond its child components. All application state is managed in the central
//! App struct and passed down to child components as needed.
//!
//! ## Thread Safety
//! This component runs entirely on the main UI thread and doesn't need special
//! thread safety considerations. All AI computation happens in background threads
//! and communicates via message passing handled at the App level.

use ratatui::{Frame, layout::Rect};

use crate::app::{App, AppMode};
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::ComponentEvent;
use crate::components::ui::{
    game_over::GameOverComponent, game_selection::GameSelectionComponent,
    how_to_play::HowToPlayComponent, in_game::InGameComponent,
    player_config::PlayerConfigComponent, settings::SettingsComponent,
};

/// The root component that serves as the top-level container for the entire application
///
/// This component acts as the application shell, managing the overall UI hierarchy
/// and routing rendering/events to the appropriate child component based on the
/// current application mode.
///
/// ## Architecture
/// The RootComponent uses a simple delegation pattern:
/// 1. Examines the current AppMode to determine which child should be active
/// 2. Forwards all render() calls to the active child component
/// 3. Forwards all handle_event() calls to the active child component
/// 4. Manages the lifecycle of all child components
///
/// ## Memory Management
/// All child components are created once during initialization and persist
/// for the entire application lifetime. This avoids the overhead of creating
/// and destroying components during mode transitions while maintaining
/// component state across transitions.
///
/// ## Error Handling
/// The RootComponent propagates any errors from child components upward
/// to the application level, where they can be handled appropriately.
pub struct RootComponent {
    /// Unique identifier for this component instance
    /// Used by the component system for event routing and debugging
    id: ComponentId,

    /// Main menu component for game selection and application entry point
    /// Active when AppMode::GameSelection
    game_selection: GameSelectionComponent,

    /// Settings and configuration component for all application parameters
    /// Active when AppMode::Settings
    settings: SettingsComponent,

    /// Player configuration component for setting up human/AI players
    /// Active when AppMode::PlayerConfig
    player_config: PlayerConfigComponent,

    /// Main gameplay component handling the active game interface
    /// Active when AppMode::InGame
    in_game: InGameComponent,

    /// End-game results and continuation options component
    /// Active when AppMode::GameOver
    game_over: GameOverComponent,

    /// Help screen showing how to play the current game
    /// Active when AppMode::HowToPlay
    how_to_play: HowToPlayComponent,
}

impl RootComponent {
    /// Creates a new RootComponent with all child components initialized
    ///
    /// This constructor initializes all child components immediately rather than
    /// creating them on-demand. This design choice ensures:
    /// - Consistent memory usage throughout application lifetime
    /// - No allocation delays during mode transitions
    /// - Preservation of component state across mode changes
    /// - Simpler error handling (all allocation errors occur at startup)
    ///
    /// # Returns
    /// A new RootComponent instance ready to handle application shell duties
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            game_selection: GameSelectionComponent::new(),
            settings: SettingsComponent::new(),
            player_config: PlayerConfigComponent::new(),
            in_game: InGameComponent::new(),
            game_over: GameOverComponent::new(),
            how_to_play: HowToPlayComponent::new(),
        }
    }
}

impl Component for RootComponent {
    /// Returns the unique identifier for this component instance
    ///
    /// This ID is used by the component system for event routing, debugging,
    /// and component lifecycle management.
    fn id(&self) -> ComponentId {
        self.id
    }

    /// Renders the currently active child component based on application mode
    ///
    /// This method implements the core delegation pattern of the RootComponent.
    /// It examines the current application mode and forwards the render call
    /// to the appropriate child component.
    ///
    /// # Arguments
    /// * `frame` - The Ratatui frame for rendering terminal content
    /// * `area` - The rectangular area available for rendering (full terminal)
    /// * `app` - Current application state containing mode and other data
    ///
    /// # Returns
    /// Result indicating success or rendering error from child component
    ///
    /// # Design Notes
    /// The full terminal area is passed to each child component, allowing them
    /// to manage their own layout. This provides maximum flexibility for
    /// different UI designs across application modes.
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        match app.mode {
            AppMode::GameSelection => self.game_selection.render(frame, area, app),
            AppMode::Settings => self.settings.render(frame, area, app),
            AppMode::PlayerConfig => self.player_config.render(frame, area, app),
            AppMode::InGame => self.in_game.render(frame, area, app),
            AppMode::GameOver => self.game_over.render(frame, area, app),
            AppMode::HowToPlay => self.how_to_play.render(frame, area, app),
        }
    }

    /// Routes events to the currently active child component
    ///
    /// Events are forwarded to the child component corresponding to the current
    /// application mode. This ensures that only the visible/active component
    /// processes user input and system events.
    ///
    /// # Arguments
    /// * `event` - The component event to process (keyboard, mouse, etc.)
    /// * `app` - Mutable application state that can be modified by event handling
    ///
    /// # Returns
    /// EventResult indicating whether the event was handled and if a re-render is needed
    ///
    /// # Event Flow
    /// 1. Determine active child based on app.mode
    /// 2. Forward event to that child's handle_event method
    /// 3. Child component processes event and may modify app state
    /// 4. Return result indicating whether UI update is needed
    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match app.mode {
            AppMode::GameSelection => self.game_selection.handle_event(event, app),
            AppMode::Settings => self.settings.handle_event(event, app),
            AppMode::PlayerConfig => self.player_config.handle_event(event, app),
            AppMode::InGame => self.in_game.handle_event(event, app),
            AppMode::GameOver => self.game_over.handle_event(event, app),
            AppMode::HowToPlay => self.how_to_play.handle_event(event, app),
        }
    }

    /// Provides mutable access to all child components
    ///
    /// This method is used by the component system for advanced operations
    /// like bulk updates, debugging, or component tree traversal.
    ///
    /// # Returns
    /// Vector of mutable references to all child components
    ///
    /// # Usage
    /// Primarily used by the component manager for operations that need
    /// to access multiple components simultaneously or for debugging purposes.
    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        vec![
            &mut self.game_selection,
            &mut self.settings,
            &mut self.player_config,
            &mut self.in_game,
            &mut self.game_over,
            &mut self.how_to_play,
        ]
    }

    /// Provides immutable access to all child components
    ///
    /// This method is used for read-only operations on child components,
    /// such as debugging, state inspection, or component tree analysis.
    ///
    /// # Returns
    /// Vector of immutable references to all child components
    fn children(&self) -> Vec<&dyn Component> {
        vec![
            &self.game_selection,
            &self.settings,
            &self.player_config,
            &self.in_game,
            &self.game_over,
            &self.how_to_play,
        ]
    }

    // Implement default component base functionality using the macro
    // This provides common component methods like focus management, update, etc.
    crate::impl_component_base!(RootComponent);
}
