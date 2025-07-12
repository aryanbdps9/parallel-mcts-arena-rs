//! # Component System Core - Foundation for Modular UI
//!
//! This module defines the fundamental traits, types, and infrastructure that
//! power the entire component-based UI system. It provides the core abstractions
//! that enable composable, reusable, and maintainable user interface components.
//!
//! ## Design Philosophy
//! The component system is built around several key principles:
//! - **Composability**: Components can be nested and combined arbitrarily
//! - **Separation of Concerns**: Each component handles one specific UI responsibility
//! - **Event-Driven Architecture**: Components communicate through a centralized event system
//! - **Lifecycle Management**: Proper initialization, update, and cleanup patterns
//! - **Type Safety**: Strong typing prevents many common UI programming errors
//!
//! ## Component Lifecycle
//! ```text
//! Creation → Initialization → Active Phase → Cleanup
//!     ↓           ↓              ↓           ↓
//!   new()     initialize()   render()     drop()
//!                              ↕
//!                         handle_event()
//!                              ↕
//!                           update()
//! ```
//!
//! ## Error Handling Strategy
//! The component system uses a comprehensive error handling approach:
//! - **Graceful Degradation**: Errors in one component don't crash the application
//! - **Error Propagation**: Errors bubble up through the component hierarchy
//! - **Recovery Mechanisms**: Components can often recover from transient errors
//! - **User Feedback**: Errors are logged and optionally displayed to users
//!
//! ## Performance Considerations
//! - **Minimal Allocations**: Most operations work with borrowed data
//! - **Efficient Rendering**: Only re-render components that have changed
//! - **Event Filtering**: Events are only sent to components that can handle them
//! - **Component Caching**: Expensive computations are cached when possible

use std::any::Any;
use ratatui::{
    layout::Rect,
    Frame,
};

/// Unique identifier for components in the UI system
///
/// Each component instance gets a unique ID that's used for:
/// - **Event Routing**: Directing events to specific components
/// - **Focus Management**: Tracking which component currently has focus
/// - **Debugging**: Identifying components in logs and error messages
/// - **Component Lookup**: Finding specific components in the hierarchy
///
/// IDs are generated atomically to ensure uniqueness across threads.
/// The ID space is large enough (u64) to never realistically overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentId(pub u64);

impl ComponentId {
    /// Generate a new unique component ID
    ///
    /// Uses an atomic counter to ensure each ID is unique, even when
    /// components are created concurrently from multiple threads.
    /// The counter starts at 1 (not 0) to make debugging easier.
    ///
    /// # Returns
    /// A new ComponentId that's guaranteed to be unique
    ///
    /// # Thread Safety
    /// This method is safe to call from any thread. The atomic operations
    /// ensure that concurrent calls will receive different IDs.
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        ComponentId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Standard result type for component operations that may fail
///
/// This type alias provides consistency across all component methods
/// that can fail. It uses ComponentError as the error type, which
/// provides detailed information about what went wrong.
///
/// # Usage
/// Most component methods return this type, allowing for consistent
/// error handling throughout the UI system.
pub type ComponentResult<T> = Result<T, ComponentError>;

/// Result type specifically for event handling operations
///
/// Event handlers return a boolean indicating whether the event was
/// handled and a re-render is needed. The boolean semantics are:
/// - `true`: Event was handled, UI should be re-rendered
/// - `false`: Event was not handled or no re-render needed
///
/// # Design Rationale
/// This specific result type makes event handling more explicit and
/// helps optimize rendering by only updating when necessary.
pub type EventResult = Result<bool, ComponentError>;

/// Result type for component update operations
///
/// Update operations typically don't return data but may fail.
/// This type alias provides consistency for these operations.
pub type UpdateResult = Result<(), ComponentError>;

/// Comprehensive error types for component system failures
///
/// This enum categorizes different types of errors that can occur
/// during component operations, providing detailed context for
/// debugging and error recovery.
///
/// # Error Categories
/// - **RenderError**: Failures during UI rendering (drawing, layout, etc.)
/// - **EventError**: Problems processing user input or system events
/// - **UpdateError**: Issues during component state synchronization
#[derive(Debug)]
pub enum ComponentError {
    /// Error occurred during component rendering
    ///
    /// This includes failures in:
    /// - Drawing operations to the terminal
    /// - Layout calculation errors
    /// - Resource loading failures
    /// - Invalid rendering state
    RenderError(String),
    
    /// Error occurred during event processing
    ///
    /// This includes failures in:
    /// - Input validation
    /// - Event routing
    /// - State transition errors
    /// - Invalid user input handling
    EventError(String),
    
    /// Error occurred during component update
    ///
    /// This includes failures in:
    /// - State synchronization with the app
    /// - Data validation during updates
    /// - Resource cleanup failures
    /// - Configuration update errors
    UpdateError(String),
}

impl std::fmt::Display for ComponentError {
    /// Format component errors for user-friendly display
    ///
    /// Provides clear, descriptive error messages that can be shown
    /// to users or logged for debugging purposes.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComponentError::RenderError(msg) => write!(f, "Render error: {}", msg),
            ComponentError::EventError(msg) => write!(f, "Event error: {}", msg),
            ComponentError::UpdateError(msg) => write!(f, "Update error: {}", msg),
        }
    }
}

// Implement standard Error trait for integration with error handling crates
impl std::error::Error for ComponentError {}

/// Core trait that all components must implement
///
/// This trait defines the fundamental interface that every UI component
/// must provide. It establishes the contract for component behavior and
/// enables the component system to work with any component type uniformly.
///
/// ## Design Principles
/// - **Minimal Required Interface**: Only essential methods are required
/// - **Sensible Defaults**: Most methods have default implementations
/// - **Flexibility**: Components can override defaults for custom behavior
/// - **Composability**: Components can contain other components easily
///
/// ## Trait Bounds
/// - `Any`: Enables runtime type checking and dynamic casting
/// - `Send + Sync`: Allows components to be used across thread boundaries
///
/// ## Lifecycle Integration
/// The trait methods map to specific phases of the component lifecycle:
/// - `render()`: Active phase drawing
/// - `handle_event()`: Active phase interaction
/// - `update()`: State synchronization
/// - `children()`: Hierarchy management
///
/// ## Memory Management
/// Components are responsible for managing their own resources.
/// The trait design encourages RAII patterns and automatic cleanup.
pub trait Component: Any + Send + Sync {
    /// Get the unique identifier for this component instance
    ///
    /// This ID is used throughout the component system for:
    /// - Event routing to specific components
    /// - Focus management and tab order
    /// - Debugging and error reporting
    /// - Component hierarchy traversal
    ///
    /// # Returns
    /// The ComponentId assigned to this instance during creation
    fn id(&self) -> ComponentId;
    
    /// Get the compile-time type name of this component
    ///
    /// This is primarily used for debugging, logging, and error reporting.
    /// The default implementation uses Rust's built-in type_name function
    /// to provide accurate type information.
    ///
    /// # Returns
    /// A string slice containing the full type name (including module path)
    ///
    /// # Usage
    /// Mainly used in debug builds and error messages to identify which
    /// component type was involved in an operation or error.
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
    
    /// Render the component to the terminal frame
    ///
    /// This is the core method that every component must implement. It's
    /// responsible for drawing the component's visual representation to
    /// the terminal using the Ratatui framework.
    ///
    /// # Arguments
    /// * `frame` - Mutable reference to the Ratatui frame for drawing
    /// * `area` - The rectangular area allocated for this component
    /// * `app` - Immutable reference to the application state
    ///
    /// # Returns
    /// ComponentResult indicating success or rendering failure
    ///
    /// # Implementation Guidelines
    /// - Use the provided area efficiently and don't draw outside it
    /// - Access only the app state you need for rendering
    /// - Handle gracefully if the area is too small for your content
    /// - Consider caching expensive layout calculations
    /// - Use appropriate styling and colors for consistency
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &crate::app::App) -> ComponentResult<()>;
    
    /// Handle events sent to this component
    ///
    /// This method processes user input and system events. The default
    /// implementation does nothing and returns false (event not handled).
    /// Components should override this to process relevant events.
    ///
    /// # Arguments
    /// * `event` - The event to process (keyboard, mouse, system, etc.)
    /// * `app` - Mutable reference to application state for updates
    ///
    /// # Returns
    /// EventResult with boolean indicating if a re-render is needed:
    /// - `Ok(true)`: Event handled, please re-render the UI
    /// - `Ok(false)`: Event handled, no re-render needed
    /// - `Err(...)`: Error processing the event
    ///
    /// # Event Handling Best Practices
    /// - Only handle events that are relevant to your component
    /// - Validate all input before updating application state
    /// - Return true only when the UI actually needs to be updated
    /// - Use the app reference sparingly and only for necessary updates
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