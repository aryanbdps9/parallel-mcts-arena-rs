//! UI component implementations.

pub mod board_cell;
pub mod game_over;
pub mod game_selection;
pub mod in_game;
pub mod move_history;
pub mod player_config;
pub mod responsive_layout;
pub mod root;
pub mod scrollable;
pub mod settings;
pub mod theme;

// Re-export reusable components
pub use board_cell::{BoardCellComponent, BoardCellGameType};
pub use move_history::MoveHistoryComponent;
pub use responsive_layout::{ResponsiveLayoutComponent, ResponsiveLayoutType};
pub use scrollable::ScrollableComponent;
pub use theme::UITheme;
