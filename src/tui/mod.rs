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
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::Rect,
    Terminal,
};
use std::{io, time::Duration};

pub mod input;
pub mod widgets;
pub mod layout;
pub mod mouse;
pub mod blokus_ui;

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

        terminal.draw(|f| widgets::render(app, f))?;

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => {
                    if key.kind == KeyEventKind::Press {
                        input::handle_key_press(app, key.code);
                    }
                }
                Event::Mouse(mouse) => {
                    let terminal_size = terminal.size()?;
                    let terminal_rect = Rect::new(0, 0, terminal_size.width, terminal_size.height);
                    input::handle_mouse_event(app, mouse.kind, mouse.column, mouse.row, terminal_rect);
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
