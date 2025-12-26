//! # Color Palette for GUI
//!
//! Centralized color definitions for consistent theming across the application.
//! Uses Direct2D compatible color format (D2D1_COLOR_F).

use windows::Win32::Graphics::Direct2D::Common::D2D1_COLOR_F;

/// Application color palette with semantic naming
pub struct Colors;

impl Colors {
    // Background colors
    pub const BACKGROUND: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.12, g: 0.12, b: 0.15, a: 1.0 };
    pub const PANEL_BG: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.16, g: 0.16, b: 0.20, a: 1.0 };
    pub const BOARD_BG: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.85, g: 0.75, b: 0.55, a: 1.0 }; // Warm wood color
    
    // Grid and borders
    pub const GRID_LINE: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.3, g: 0.3, b: 0.35, a: 1.0 };
    pub const BOARD_GRID: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.2, g: 0.15, b: 0.1, a: 1.0 };
    pub const HIGHLIGHT: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.3, g: 0.7, b: 0.9, a: 0.5 };
    pub const LAST_MOVE: D2D1_COLOR_F = D2D1_COLOR_F { r: 1.0, g: 0.8, b: 0.2, a: 0.6 };

    // Player colors
    pub const PLAYER_1: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.3, g: 0.3, b: 0.3, a: 1.0 }; // Black
    pub const PLAYER_2: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.95, g: 0.95, b: 0.95, a: 1.0 }; // White
    pub const PLAYER_3: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.9, g: 0.3, b: 0.3, a: 1.0 }; // Red
    pub const PLAYER_4: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.3, g: 0.5, b: 0.9, a: 1.0 }; // Blue

    // Text colors
    pub const TEXT_PRIMARY: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.95, g: 0.95, b: 0.95, a: 1.0 };
    pub const TEXT_SECONDARY: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.7, g: 0.7, b: 0.75, a: 1.0 };
    pub const TEXT_ACCENT: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.4, g: 0.8, b: 1.0, a: 1.0 };

    // Button colors
    pub const BUTTON_BG: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.25, g: 0.25, b: 0.30, a: 1.0 };
    pub const BUTTON_HOVER: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.35, g: 0.35, b: 0.40, a: 1.0 };
    pub const BUTTON_PRESSED: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.2, g: 0.2, b: 0.25, a: 1.0 };
    pub const BUTTON_SELECTED: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.3, g: 0.6, b: 0.9, a: 1.0 };

    // Status colors
    pub const STATUS_WIN: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.3, g: 0.8, b: 0.4, a: 1.0 };
    pub const STATUS_DRAW: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.8, g: 0.7, b: 0.3, a: 1.0 };
    pub const STATUS_ERROR: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.9, g: 0.3, b: 0.3, a: 1.0 };
    pub const AI_THINKING: D2D1_COLOR_F = D2D1_COLOR_F { r: 0.5, g: 0.5, b: 0.9, a: 1.0 };
}

/// Get player color by player ID
pub fn player_color(player_id: i32) -> D2D1_COLOR_F {
    match player_id {
        1 => Colors::PLAYER_1,
        -1 | 2 => Colors::PLAYER_2,
        3 => Colors::PLAYER_3,
        4 => Colors::PLAYER_4,
        _ => Colors::TEXT_SECONDARY,
    }
}

/// Create a color with modified alpha
pub fn with_alpha(color: D2D1_COLOR_F, alpha: f32) -> D2D1_COLOR_F {
    D2D1_COLOR_F { a: alpha, ..color }
}
