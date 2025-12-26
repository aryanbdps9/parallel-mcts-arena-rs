//! # Windows GUI Module
//!
//! This module provides a native Windows GUI implementation using the windows-rs crate.
//! It offers an alternative to the TUI (Terminal User Interface) for a more visual experience.
//!
//! ## Architecture Overview
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                         GUI Application                          │
//! ├──────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
//! │  │   Window    │  │   Renderer   │  │    Event Handler       │   │
//! │  │  (HWND)     │◄─┤  (Direct2D)  │  │  (WM_* messages)       │   │
//! │  └─────────────┘  └──────────────┘  └────────────────────────┘   │
//! │                           │                    │                 │
//! │  ┌─────────────────────────────────────────────────────────────┐ │
//! │  │                   Game Renderer Trait                       │ │
//! │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐    │ │
//! │  │  │ Gomoku   │ │ Connect4 │ │ Othello  │ │   Blokus     │    │ │
//! │  │  │ Renderer │ │ Renderer │ │ Renderer │ │   Renderer   │    │ │
//! │  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘    │ │
//! │  └─────────────────────────────────────────────────────────────┘ │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Design Principles
//! 1. **Readability**: Clear separation between window management, rendering, and game logic
//! 2. **Extensibility**: Adding new games requires only implementing `GameRenderer` trait
//! 3. **Reusability**: Common UI elements (buttons, panels) are shared across games
//! 4. **Performance**: Direct2D hardware acceleration with efficient redraw

pub mod app;
pub mod colors;
pub mod game_renderers;
pub mod renderer;
pub mod window;

pub use app::GuiApp;
pub use window::run_gui;
