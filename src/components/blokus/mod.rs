//! Blokus-specific UI components module.

pub mod board;
pub mod piece_cell;
pub mod piece_shape;
pub mod enhanced_piece_grid;
pub mod enhanced_piece_selector;
pub mod responsive_piece_grid;
pub mod improved_piece_selector;
pub mod player_panel;
pub mod piece_selector;
pub mod game_stats;
pub mod instruction_panel;

// Utility modules for modular piece grid functionality
pub mod grid_border;
pub mod piece_visualizer;
pub mod click_handler;
pub mod grid_layout;

pub use board::BlokusBoardComponent;
pub use piece_cell::PieceCellComponent;
pub use piece_shape::{PieceShapeComponent, PieceShapeConfig};
pub use enhanced_piece_grid::{EnhancedPieceGridComponent, EnhancedPieceGridConfig};
pub use enhanced_piece_selector::EnhancedBlokusPieceSelectorComponent;
pub use responsive_piece_grid::{ResponsivePieceGridComponent, ResponsivePieceGridConfig};
pub use improved_piece_selector::{ImprovedBlokusPieceSelectorComponent, ImprovedPieceSelectorConfig};
pub use player_panel::BlokusPlayerPanelComponent;
pub use piece_selector::BlokusPieceSelectorComponent;
pub use game_stats::BlokusGameStatsComponent;
pub use instruction_panel::BlokusInstructionPanelComponent;

// Re-export for convenience
pub use crate::components::events::{InputEvent, ComponentEvent};
pub use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
