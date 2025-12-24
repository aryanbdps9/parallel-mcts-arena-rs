//! # Terminal User Interface Module
//!
//! This module provides the complete terminal-based user interface for the game engine,
//! built using the Ratatui library. It handles all user interactions, display rendering,
//! and input processing for an interactive gaming experience.
//!
//! ## Key Components
//! - **Terminal Management**: Initialization and cleanup of raw terminal mode
//! - **Event Loop**: Main application loop handling input and rendering
//! - **Input Processing**: Keyboard and mouse event handling
//! - **Widget Rendering**: Game-specific UI components and layouts
//! - **Mouse Support**: Click-and-drag interactions for resizable panels
//!
//! ## Supported Input Methods
//! - Keyboard navigation and game control
//! - Mouse clicks for menu selection and piece placement
//! - Drag-and-drop for panel resizing
//! - Scrolling for long content areas
//!
//! The interface adapts dynamically to different game types and supports both
//! human players and AI players with real-time statistics display.

use crate::app::App;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend, layout::Rect};
use std::{io, time::Duration};

pub mod games;
pub mod input;
pub mod layout;
pub mod mouse;
pub mod widgets;

/// Main entry point for the terminal user interface
///
/// Initializes the terminal, runs the main event loop, and handles cleanup.
/// The event loop processes keyboard and mouse input, updates the application state,
/// and renders the UI at 10 FPS. Supports drag-and-drop panel resizing and
/// real-time game updates.
///
/// # Arguments
/// * `app` - Mutable reference to the application state
///
/// # Returns
/// IO result indicating success or failure of terminal operations
///
/// # Errors
/// Returns an error if terminal initialization, event handling, or cleanup fails
pub fn run(app: &mut App) -> io::Result<()> {
    let mut terminal = init_terminal()?;

    loop {
        if app.should_quit {
            app.shutdown();
            break;
        }

        app.update();

        // Update the component system
        // Temporarily take ownership to avoid borrowing issues
        let mut temp_manager = std::mem::take(&mut app.component_manager);
        temp_manager.update(app);
        app.component_manager = temp_manager;

        terminal.draw(|f| {
            // Use the component system for all modes
            let terminal_size = f.area();
            let mut temp_manager = std::mem::take(&mut app.component_manager);

            if temp_manager.get_root_component_id().is_some() {
                temp_manager.render(f, terminal_size, app);
            } else {
                // Fallback to legacy if no root component (should not happen)
                crate::tui::widgets::render(app, f);
            }

            app.component_manager = temp_manager;
        })?;

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => {
                    if key.kind == KeyEventKind::Press {
                        // Create component event from keyboard input
                        let component_event = crate::components::events::ComponentEvent::Input(
                            crate::components::events::InputEvent::KeyPress(key.code),
                        );

                        // Try component system first
                        let mut temp_manager = std::mem::take(&mut app.component_manager);
                        let consumed = temp_manager.handle_event(&component_event, app);
                        app.component_manager = temp_manager;

                        if !consumed {
                            // Fallback to legacy input handling if event not consumed
                            crate::tui::input::handle_key_press(app, key.code);
                        }
                    }
                }
                Event::Mouse(mouse) => {
                    // Create component event from mouse input
                    let component_event = match mouse.kind {
                        crossterm::event::MouseEventKind::Down(_) => {
                            Some(crate::components::events::ComponentEvent::Input(
                                crate::components::events::InputEvent::MouseClick {
                                    x: mouse.column,
                                    y: mouse.row,
                                    button: 1, // Left button by default
                                },
                            ))
                        }
                        crossterm::event::MouseEventKind::Drag(_) => {
                            Some(crate::components::events::ComponentEvent::Input(
                                crate::components::events::InputEvent::MouseMove {
                                    x: mouse.column,
                                    y: mouse.row,
                                },
                            ))
                        }
                        crossterm::event::MouseEventKind::ScrollUp => {
                            Some(crate::components::events::ComponentEvent::Input(
                                crate::components::events::InputEvent::MouseScroll {
                                    x: mouse.column,
                                    y: mouse.row,
                                    up: true,
                                },
                            ))
                        }
                        crossterm::event::MouseEventKind::ScrollDown => {
                            Some(crate::components::events::ComponentEvent::Input(
                                crate::components::events::InputEvent::MouseScroll {
                                    x: mouse.column,
                                    y: mouse.row,
                                    up: false,
                                },
                            ))
                        }
                        _ => {
                            // For other mouse events, use legacy handling
                            let terminal_size = terminal.size()?;
                            let terminal_rect =
                                Rect::new(0, 0, terminal_size.width, terminal_size.height);
                            input::handle_mouse_event(
                                app,
                                mouse.kind,
                                mouse.column,
                                mouse.row,
                                terminal_rect,
                            );
                            None
                        }
                    };

                    // Try component system first if we have a component event
                    if let Some(component_event) = component_event {
                        let mut temp_manager = std::mem::take(&mut app.component_manager);
                        let consumed = temp_manager.handle_event(&component_event, app);
                        app.component_manager = temp_manager;

                        if !consumed {
                            // Legacy mouse handling as fallback
                            let terminal_size = terminal.size()?;
                            let terminal_rect =
                                Rect::new(0, 0, terminal_size.width, terminal_size.height);
                            input::handle_mouse_event(
                                app,
                                mouse.kind,
                                mouse.column,
                                mouse.row,
                                terminal_rect,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }

    restore_terminal(&mut terminal)
}

/// Initializes the terminal for raw mode operation
///
/// Sets up the terminal for interactive use by enabling raw mode, switching to
/// alternate screen, enabling mouse capture, and hiding the cursor.
///
/// # Returns
/// Terminal instance ready for rendering, or IO error if setup fails
fn init_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    execute!(
        handle,
        EnterAlternateScreen,
        EnableMouseCapture,
        crossterm::cursor::Hide
    )?;
    Terminal::new(CrosstermBackend::new(stdout))
}

/// Restores the terminal to normal operation mode
///
/// Cleans up terminal state by showing the cursor, disabling raw mode,
/// leaving alternate screen, and disabling mouse capture.
///
/// # Arguments
/// * `terminal` - Terminal instance to restore
///
/// # Returns
/// IO result indicating success or failure of cleanup operations
fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    terminal.show_cursor()?;
    disable_raw_mode()?;
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    execute!(
        handle,
        LeaveAlternateScreen,
        DisableMouseCapture,
        crossterm::cursor::Show
    )?;
    Ok(())
}
