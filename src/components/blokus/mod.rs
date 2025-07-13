//! Blokus-specific UI components module.

pub mod board;
pub mod enhanced_piece_grid;
pub mod game_stats;
pub mod improved_piece_selector;
pub mod instruction_panel;
pub mod piece_cell;
pub mod piece_shape;
pub mod player_panel;
pub mod responsive_piece_grid;

// Utility modules for modular piece grid functionality
pub mod click_handler;
pub mod grid_border;
pub mod grid_layout;
pub mod piece_visualizer;

pub use board::BlokusBoardComponent;
pub use enhanced_piece_grid::{EnhancedPieceGridComponent, EnhancedPieceGridConfig};
pub use game_stats::BlokusGameStatsComponent;
pub use improved_piece_selector::{
    ImprovedBlokusPieceSelectorComponent, ImprovedPieceSelectorConfig,
};
pub use instruction_panel::BlokusInstructionPanelComponent;
pub use piece_cell::PieceCellComponent;
pub use piece_shape::{PieceShapeComponent, PieceShapeConfig};
pub use player_panel::BlokusPlayerPanelComponent;
pub use responsive_piece_grid::{ResponsivePieceGridComponent, ResponsivePieceGridConfig};

// Re-export for convenience
pub use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
pub use crate::components::events::{ComponentEvent, InputEvent};
