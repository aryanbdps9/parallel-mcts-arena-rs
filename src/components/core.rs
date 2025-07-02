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

/// Core trait that all components must implement
pub trait Component: Any + Send + Sync {
    /// Get the unique ID of this component
    fn id(&self) -> ComponentId;
    
    /// Get the type name of this component
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
    
    /// Render the component to the frame
    fn render(&self, frame: &mut Frame, area: Rect);
    
    /// Handle component-specific events
    fn handle_event(&mut self, _event: &crate::components::events::ComponentEvent) -> bool {
        false // Default: don't consume events
    }
    
    /// Update component state (called every frame)
    fn update(&mut self) {}
    
    /// Get child components
    fn children(&self) -> Vec<ComponentId> {
        Vec::new() // Default: no children
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
}