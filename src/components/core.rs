//! Core component definitions and traits.

use std::any::Any;
use ratatui::{
    layout::Rect,
    Frame,
};

/// Unique identifier for components
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentId(pub u64);

impl ComponentId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        ComponentId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Result type for component operations
pub type ComponentResult<T> = Result<T, ComponentError>;

/// Result type for event handling
pub type EventResult = Result<bool, ComponentError>;

/// Result type for update operations
pub type UpdateResult = Result<(), ComponentError>;

/// Component error types
#[derive(Debug)]
pub enum ComponentError {
    RenderError(String),
    EventError(String),
    UpdateError(String),
}

impl std::fmt::Display for ComponentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComponentError::RenderError(msg) => write!(f, "Render error: {}", msg),
            ComponentError::EventError(msg) => write!(f, "Event error: {}", msg),
            ComponentError::UpdateError(msg) => write!(f, "Update error: {}", msg),
        }
    }
}

impl std::error::Error for ComponentError {}

/// Core trait that all components must implement
pub trait Component: Any + Send + Sync {
    /// Get the unique ID of this component
    fn id(&self) -> ComponentId;
    
    /// Get the type name of this component
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
    
    /// Render the component to the frame with app access
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &crate::app::App) -> ComponentResult<()>;
    
    /// Handle component-specific events
    fn handle_event(&mut self, event: &crate::components::events::ComponentEvent, app: &mut crate::app::App) -> EventResult {
        let _ = (event, app);
        Ok(false) // Default: don't consume events
    }
    
    /// Update component state (called every frame)
    fn update(&mut self, app: &mut crate::app::App) -> UpdateResult {
        let _ = app;
        Ok(()) // Default: no-op
    }
    
    /// Get mutable child components
    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        Vec::new() // Default: no children
    }
    
    /// Get child components
    fn children(&self) -> Vec<&dyn Component> {
        Vec::new() // Default: no children
    }
    
    /// Check if component is visible
    fn is_visible(&self) -> bool {
        true // Default: visible
    }
    
    /// Called when component is mounted
    fn on_mount(&mut self, _app: &mut crate::app::App) {
        // Default: no-op
    }
    
    /// Called when component is unmounted
    fn on_unmount(&mut self, _app: &mut crate::app::App) {
        // Default: no-op
    }
    
    /// Get component as Any for downcasting
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Macro to help implement Component trait
#[macro_export]
macro_rules! impl_component_base {
    ($type:ty) => {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        
        fn type_name(&self) -> &'static str {
            std::any::type_name::<$type>()
        }
    };
    
    ($type:ty, $name:expr) => {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        
        fn type_name(&self) -> &'static str {
            $name
        }
    };
}