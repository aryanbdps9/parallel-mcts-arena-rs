//! UI component implementations.

pub mod root;
pub mod game_selection;
pub mod settings;
pub mod player_config;
pub mod in_game;
pub mod game_over;
pub mod board_cell;
pub mod responsive_layout;
pub mod scrollable;
pub mod theme;
pub mod move_history;

// Re-export reusable components
pub use board_cell::{BoardCellComponent, BoardCellGameType};
pub use responsive_layout::{ResponsiveLayoutComponent, ResponsiveLayoutType};
pub use scrollable::ScrollableComponent;
pub use theme::UITheme;
pub use move_history::MoveHistoryComponent;