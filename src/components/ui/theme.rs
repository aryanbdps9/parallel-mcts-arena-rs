//! Visual theme and styling component for consistent UI appearance.

use ratatui::style::{Style, Color, Modifier};

/// A centralized theme component that provides consistent styling across all UI elements
pub struct UITheme {
    // Player colors for consistent game representation
    player_colors: [Color; 4],
    // UI element colors
    background_color: Color,
    border_color: Color,
    text_color: Color,
    highlight_color: Color,
    cursor_color: Color,
    error_color: Color,
    success_color: Color,
    warning_color: Color,
    // Board styling
    empty_cell_light: Color,
    empty_cell_dark: Color,
    ghost_legal_color: Color,
    ghost_illegal_color: Color,
}

impl Default for UITheme {
    fn default() -> Self {
        Self {
            player_colors: [Color::Red, Color::Blue, Color::Green, Color::Yellow],
            background_color: Color::Black,
            border_color: Color::White,
            text_color: Color::White,
            highlight_color: Color::Yellow,
            cursor_color: Color::Yellow,
            error_color: Color::Red,
            success_color: Color::Green,
            warning_color: Color::Cyan,
            empty_cell_light: Color::Rgb(100, 100, 100),
            empty_cell_dark: Color::Rgb(60, 60, 60),
            ghost_legal_color: Color::Cyan,
            ghost_illegal_color: Color::Red,
        }
    }
}

impl UITheme {
    /// Get the color for a specific player (1-4)
    pub fn player_color(&self, player: u8) -> Color {
        if player >= 1 && player <= 4 {
            self.player_colors[(player - 1) as usize]
        } else {
            self.text_color
        }
    }

    /// Get the highlighted version of a player color
    pub fn player_color_highlighted(&self, player: u8) -> Color {
        match player {
            1 => Color::LightRed,
            2 => Color::LightBlue,
            3 => Color::LightGreen,
            4 => Color::LightYellow,
            _ => self.highlight_color,
        }
    }

    /// Get style for regular text
    pub fn text_style(&self) -> Style {
        Style::default().fg(self.text_color)
    }

    /// Get style for highlighted text
    pub fn highlighted_text_style(&self) -> Style {
        Style::default().fg(self.highlight_color).add_modifier(Modifier::BOLD)
    }

    /// Get style for borders
    pub fn border_style(&self) -> Style {
        Style::default().fg(self.border_color)
    }

    /// Get style for player text
    pub fn player_text_style(&self, player: u8) -> Style {
        Style::default().fg(self.player_color(player)).add_modifier(Modifier::BOLD)
    }

    /// Get style for current player (with background)
    pub fn current_player_style(&self, player: u8) -> Style {
        Style::default()
            .fg(self.player_color(player))
            .bg(Color::DarkGray)
            .add_modifier(Modifier::BOLD)
    }

    /// Get style for error messages
    pub fn error_style(&self) -> Style {
        Style::default().fg(self.error_color).add_modifier(Modifier::BOLD)
    }

    /// Get style for success messages
    pub fn success_style(&self) -> Style {
        Style::default().fg(self.success_color).add_modifier(Modifier::BOLD)
    }

    /// Get style for warning messages
    pub fn warning_style(&self) -> Style {
        Style::default().fg(self.warning_color).add_modifier(Modifier::BOLD)
    }

    /// Get cursor style
    pub fn cursor_style(&self) -> Style {
        Style::default().bg(self.cursor_color)
    }

    /// Get empty cell colors for checkerboard pattern (returns tuple of light, dark)
    pub fn empty_cell_colors(&self) -> (Color, Color) {
        (self.empty_cell_light, self.empty_cell_dark)
    }

    /// Get ghost piece colors (returns tuple of legal, illegal)
    pub fn ghost_piece_colors(&self) -> (Color, Color) {
        (self.ghost_legal_color, self.ghost_illegal_color)
    }

    /// Get board cell style for Blokus
    pub fn blokus_cell_style(&self, player: u8, is_highlighted: bool, row: usize, col: usize) -> (&'static str, Style) {
        match player {
            1..=4 => {
                let color = if is_highlighted {
                    self.player_color_highlighted(player)
                } else {
                    self.player_color(player)
                };
                let modifier = if is_highlighted { Modifier::BOLD } else { Modifier::empty() };
                ("██", Style::default().fg(color).add_modifier(modifier))
            }
            _ => {
                // Empty cell with checkerboard pattern
                let is_light_square = (row + col) % 2 == 0;
                let color = if is_light_square { self.empty_cell_light } else { self.empty_cell_dark };
                ("░░", Style::default().fg(color))
            }
        }
    }

    /// Get ghost piece style for Blokus
    pub fn blokus_ghost_style(&self, is_legal: bool) -> (&'static str, Style) {
        if is_legal {
            ("▓▓", Style::default().fg(self.ghost_legal_color).add_modifier(Modifier::BOLD))
        } else {
            ("▓▓", Style::default().fg(self.ghost_illegal_color).add_modifier(Modifier::DIM))
        }
    }

    /// Get board cell style for Othello
    pub fn othello_cell_style(&self, player: u8) -> (&'static str, Style) {
        match player {
            1 => ("●●", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            2 => ("●●", Style::default().fg(Color::Black).add_modifier(Modifier::BOLD)),
            _ => ("··", Style::default().fg(Color::DarkGray)),
        }
    }

    /// Get board cell style for Connect4
    pub fn connect4_cell_style(&self, player: u8) -> (&'static str, Style) {
        match player {
            1 => ("●●", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            2 => ("●●", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            _ => ("··", Style::default().fg(Color::DarkGray)),
        }
    }

    /// Get board cell style for Gomoku
    pub fn gomoku_cell_style(&self, player: u8) -> (&'static str, Style) {
        match player {
            1 => ("●●", Style::default().fg(Color::Black).add_modifier(Modifier::BOLD)),
            2 => ("○○", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            _ => ("++", Style::default().fg(Color::DarkGray)),
        }
    }

    /// Get piece availability style (for piece selectors)
    pub fn piece_availability_style(&self, is_available: bool, is_selected: bool, player_color: Color) -> Style {
        if is_selected {
            Style::default()
                .fg(Color::Black)
                .bg(player_color)
                .add_modifier(Modifier::BOLD)
        } else if is_available {
            Style::default()
                .fg(player_color)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM)
        }
    }
}
