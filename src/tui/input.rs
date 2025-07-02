//! # Input Handling Module (Legacy)
//!
//! This module contains legacy input handling functions that are kept for
//! backward compatibility and fallback scenarios. The main input handling
//! is now done through the component system.
//!
//! Note: This module is deprecated in favor of component-based event handling.

use crate::app::App;
use crate::tui::mouse;
use crossterm::event::{KeyCode, MouseEventKind};
use ratatui::layout::Rect;

/// Legacy keyboard input handler - kept for fallback scenarios
/// 
/// # Arguments
/// * `app` - Mutable reference to the application state
/// * `key_code` - The key that was pressed
/// 
/// # Note
/// This function is deprecated. Input handling should be done through
/// the component system instead.
pub fn handle_key_press(app: &mut App, key_code: KeyCode) {
    // Legacy fallback - in a fully migrated system, this should rarely be called
    match key_code {
        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
        _ => {
            // Log that legacy input handling was triggered
            #[cfg(debug_assertions)]
            eprintln!("Legacy input handling triggered for key: {:?}", key_code);
        }
    }
}

/// Legacy mouse event handler - kept for fallback scenarios
/// 
/// # Arguments
/// * `app` - Mutable reference to the application state
/// * `kind` - Type of mouse event (click, drag, scroll, etc.)
/// * `col` - Column position of the mouse event
/// * `row` - Row position of the mouse event
/// * `terminal_size` - Size of the terminal for coordinate calculations
/// 
/// # Note
/// This function is deprecated. Mouse handling should be done through
/// the component system instead.
pub fn handle_mouse_event(app: &mut App, kind: MouseEventKind, col: u16, row: u16, terminal_size: Rect) {
    // Delegate to the mouse module for complex mouse operations
    mouse::handle_mouse_event(app, kind, col, row, terminal_size);
}
