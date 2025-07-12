//! Component manager for handling component lifecycle and events.

use std::collections::HashMap;
use ratatui::{layout::Rect, Frame};

use crate::components::core::{Component, ComponentId};
use crate::components::events::ComponentEvent;

/// Manages the lifecycle and event routing for components
pub struct ComponentManager {
    components: HashMap<ComponentId, Box<dyn Component>>,
    root_component: Option<ComponentId>,
    focused_component: Option<ComponentId>,
}

impl ComponentManager {
    /// Create a new component manager
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
            root_component: None,
            focused_component: None,
        }
    }
    
    /// Register a component with the manager
    pub fn register_component(&mut self, component: Box<dyn Component>) -> ComponentId {
        let id = component.id();
        self.components.insert(id, component);
        id
    }
    
    /// Set the root component
    pub fn set_root_component(&mut self, component: Box<dyn Component>) -> ComponentId {
        let id = self.register_component(component);
        self.root_component = Some(id);
        id
    }
    
    /// Get a component by ID
    pub fn get_component(&self, id: ComponentId) -> Option<&dyn Component> {
        self.components.get(&id).map(|c| c.as_ref())
    }
    
    /// Get a mutable component by ID
    pub fn get_component_mut(&mut self, id: ComponentId) -> Option<&mut dyn Component> {
        self.components.get_mut(&id).map(|c| c.as_mut())
    }
    
    /// Send an event to a specific component
    pub fn send_event_to_component(&mut self, id: ComponentId, event: &ComponentEvent, app: &mut crate::app::App) -> bool {
        if let Some(component) = self.components.get_mut(&id) {
            component.handle_event(event, app).unwrap_or(false)
        } else {
            false
        }
    }
    
    /// Broadcast an event to all components
    pub fn broadcast_event(&mut self, event: &ComponentEvent, app: &mut crate::app::App) {
        for component in self.components.values_mut() {
            let _ = component.handle_event(event, app);
        }
    }
    
    /// Send an event to the focused component first, then broadcast if not consumed
    pub fn handle_event(&mut self, event: &ComponentEvent, app: &mut crate::app::App) -> bool {
        // Try focused component first
        if let Some(focused_id) = self.focused_component {
            if self.send_event_to_component(focused_id, event, app) {
                return true; // Event was consumed
            }
        }
        
        // If not consumed by focused component, try root component
        if let Some(root_id) = self.root_component {
            if root_id != self.focused_component.unwrap_or(ComponentId(0)) {
                if self.send_event_to_component(root_id, event, app) {
                    return true;
                }
            }
        }
        
        // If still not consumed, broadcast to all components
        self.broadcast_event(event, app);
        false
    }
    
    /// Update all components
    pub fn update(&mut self, app: &mut crate::app::App) {
        for component in self.components.values_mut() {
            let _ = component.update(app);
        }
    }
    
    /// Render all components starting from root
    pub fn render(&mut self, frame: &mut Frame, area: Rect, app: &crate::app::App) {
        if let Some(root_id) = self.root_component {
            if let Some(root_component) = self.components.get_mut(&root_id) {
                let _ = root_component.render(frame, area, app);
            }
        }
    }

    /// Set the focused component
    pub fn set_focus(&mut self, id: Option<ComponentId>, app: &mut crate::app::App) {
        if let Some(old_focus) = self.focused_component {
            let lose_focus = ComponentEvent::Focus(crate::components::events::FocusEvent::Lost);
            self.send_event_to_component(old_focus, &lose_focus, app);
        }
        
        self.focused_component = id;
        
        if let Some(new_focus) = id {
            let gain_focus = ComponentEvent::Focus(crate::components::events::FocusEvent::Gained);
            self.send_event_to_component(new_focus, &gain_focus, app);
        }
    }
    
    /// Get the currently focused component ID
    pub fn get_focused(&self) -> Option<ComponentId> {
        self.focused_component
    }
    
    /// Get the root component ID
    pub fn get_root_component_id(&self) -> Option<ComponentId> {
        self.root_component
    }
    
    /// Get a reference to a component
    pub fn get_component_ref(&self, id: ComponentId) -> Option<&dyn Component> {
        self.components.get(&id).map(|c| c.as_ref())
    }
}

impl Default for ComponentManager {
    fn default() -> Self {
        Self::new()
    }
}
