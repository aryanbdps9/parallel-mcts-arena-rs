//! Event system for component communication.

use crossterm::event::KeyCode;
use std::any::Any;
use std::collections::HashMap;

/// Input events that can be sent to components
#[derive(Debug, Clone)]
pub enum InputEvent {
    KeyPress(KeyCode),
    MouseClick { x: u16, y: u16, button: u8 },
    MouseMove { x: u16, y: u16 },
    MouseScroll { x: u16, y: u16, up: bool },
    Resize { width: u16, height: u16 },
}

/// Custom event data that can be attached to events
pub type CustomEventData = HashMap<String, Box<dyn Any + Send + Sync>>;

/// Main event type for the component system
#[derive(Debug)]
pub enum ComponentEvent {
    /// Input events (keyboard, mouse, etc.)
    Input(InputEvent),

    /// Focus events
    Focus(FocusEvent),

    /// Custom events with arbitrary data
    Custom {
        event_type: String,
        data: CustomEventData,
    },

    /// Update event (sent every frame)
    Update,
}

/// Focus-related events
#[derive(Debug, Clone)]
pub enum FocusEvent {
    /// Component gained focus
    Gained,
    /// Component lost focus
    Lost,
    /// Request to move focus to next component
    Next,
    /// Request to move focus to previous component
    Previous,
}

impl ComponentEvent {
    /// Create a custom event
    pub fn custom(event_type: impl Into<String>) -> Self {
        ComponentEvent::Custom {
            event_type: event_type.into(),
            data: HashMap::new(),
        }
    }

    /// Create a custom event with data
    pub fn custom_with_data(event_type: impl Into<String>, data: CustomEventData) -> Self {
        ComponentEvent::Custom {
            event_type: event_type.into(),
            data,
        }
    }
}
