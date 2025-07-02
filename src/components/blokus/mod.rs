//! Blokus-specific UI components module.

pub mod board;
pub mod piece_cell;
pub mod player_panel;
pub mod piece_selector;
pub mod game_stats;
pub mod instruction_panel;

pub use board::BlokusBoardComponent;
pub use piece_cell::PieceCellComponent;
pub use player_panel::BlokusPlayerPanelComponent;
pub use piece_selector::BlokusPieceSelectorComponent;
pub use game_stats::BlokusGameStatsComponent;
pub use instruction_panel::BlokusInstructionPanelComponent;

// Re-export for convenience
pub use crate::components::events::{InputEvent, ComponentEvent};
pub use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
