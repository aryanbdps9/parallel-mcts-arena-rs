//! Reusable board cell component for consistent rendering across games.

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::Span,
};
use std::any::Any;

use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};

/// A reusable board cell component that can render different game pieces consistently
pub struct BoardCellComponent {
    id: ComponentId,
    area: Option<Rect>,
    cell_value: u8,
    is_cursor: bool,
    is_highlighted: bool,
    is_ghost: bool,
    ghost_legal: bool,
    game_type: BoardCellGameType,
}

/// Different game types that require different cell rendering styles
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoardCellGameType {
    Blokus,
    Othello,
    Connect4,
    Gomoku,
}

impl BoardCellComponent {
    pub fn new(game_type: BoardCellGameType) -> Self {
        Self {
            id: ComponentId::new(),
            area: None,
            cell_value: 0,
            is_cursor: false,
            is_highlighted: false,
            is_ghost: false,
            ghost_legal: true,
            game_type,
        }
    }

    pub fn set_cell_value(&mut self, value: u8) {
        self.cell_value = value;
    }

    pub fn set_cursor(&mut self, is_cursor: bool) {
        self.is_cursor = is_cursor;
    }

    pub fn set_highlighted(&mut self, is_highlighted: bool) {
        self.is_highlighted = is_highlighted;
    }

    pub fn set_ghost(&mut self, is_ghost: bool, is_legal: bool) {
        self.is_ghost = is_ghost;
        self.ghost_legal = is_legal;
    }

    /// Get the symbol and style for this cell based on game type and state
    fn get_cell_appearance(&self) -> (&'static str, Style) {
        match self.game_type {
            BoardCellGameType::Blokus => self.get_blokus_appearance(),
            BoardCellGameType::Othello => self.get_othello_appearance(),
            BoardCellGameType::Connect4 => self.get_connect4_appearance(),
            BoardCellGameType::Gomoku => self.get_gomoku_appearance(),
        }
    }

    fn get_blokus_appearance(&self) -> (&'static str, Style) {
        if self.is_ghost {
            if self.ghost_legal {
                (
                    "▓▓",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                (
                    "▓▓",
                    Style::default().fg(Color::Red).add_modifier(Modifier::DIM),
                )
            }
        } else {
            match self.cell_value {
                1 => {
                    let color = if self.is_highlighted {
                        Color::LightRed
                    } else {
                        Color::Red
                    };
                    (
                        "██",
                        Style::default()
                            .fg(color)
                            .add_modifier(if self.is_highlighted {
                                Modifier::BOLD
                            } else {
                                Modifier::empty()
                            }),
                    )
                }
                2 => {
                    let color = if self.is_highlighted {
                        Color::LightBlue
                    } else {
                        Color::Blue
                    };
                    (
                        "██",
                        Style::default()
                            .fg(color)
                            .add_modifier(if self.is_highlighted {
                                Modifier::BOLD
                            } else {
                                Modifier::empty()
                            }),
                    )
                }
                3 => {
                    let color = if self.is_highlighted {
                        Color::LightGreen
                    } else {
                        Color::Green
                    };
                    (
                        "██",
                        Style::default()
                            .fg(color)
                            .add_modifier(if self.is_highlighted {
                                Modifier::BOLD
                            } else {
                                Modifier::empty()
                            }),
                    )
                }
                4 => {
                    let color = if self.is_highlighted {
                        Color::LightYellow
                    } else {
                        Color::Yellow
                    };
                    (
                        "██",
                        Style::default()
                            .fg(color)
                            .add_modifier(if self.is_highlighted {
                                Modifier::BOLD
                            } else {
                                Modifier::empty()
                            }),
                    )
                }
                _ => {
                    // Chess-like pattern for empty squares - alternating light and dark
                    // Note: This requires row/col info which we don't have here
                    // For now, use a simple empty pattern
                    ("░░", Style::default().fg(Color::Rgb(100, 100, 100)))
                }
            }
        }
    }

    fn get_othello_appearance(&self) -> (&'static str, Style) {
        match self.cell_value {
            1 => (
                "●●",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            2 => (
                "●●",
                Style::default()
                    .fg(Color::Black)
                    .add_modifier(Modifier::BOLD),
            ),
            _ => {
                if self.is_cursor {
                    ("··", Style::default().fg(Color::Yellow).bg(Color::DarkGray))
                } else {
                    ("··", Style::default().fg(Color::DarkGray))
                }
            }
        }
    }

    fn get_connect4_appearance(&self) -> (&'static str, Style) {
        match self.cell_value {
            1 => (
                "●●",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            2 => (
                "●●",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            _ => {
                if self.is_cursor {
                    ("··", Style::default().fg(Color::Cyan).bg(Color::DarkGray))
                } else {
                    ("··", Style::default().fg(Color::DarkGray))
                }
            }
        }
    }

    fn get_gomoku_appearance(&self) -> (&'static str, Style) {
        match self.cell_value {
            1 => (
                "●●",
                Style::default()
                    .fg(Color::Black)
                    .add_modifier(Modifier::BOLD),
            ),
            2 => (
                "○○",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            _ => {
                if self.is_cursor {
                    ("++", Style::default().fg(Color::Yellow).bg(Color::DarkGray))
                } else {
                    ("++", Style::default().fg(Color::DarkGray))
                }
            }
        }
    }

    fn set_area(&mut self, area: Rect) {
        self.area = Some(area);
    }
}

impl Component for BoardCellComponent {
    fn id(&self) -> ComponentId {
        self.id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn render(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        _app: &crate::app::App,
    ) -> ComponentResult<()> {
        self.set_area(area);

        let (symbol, mut style) = self.get_cell_appearance();

        // Apply cursor background if needed
        if self.is_cursor && self.cell_value == 0 {
            style = style.bg(Color::Yellow);
        }

        let span = Span::styled(symbol, style);
        let paragraph = ratatui::widgets::Paragraph::new(span);
        frame.render_widget(paragraph, area);

        Ok(())
    }

    fn handle_event(
        &mut self,
        event: &crate::components::events::ComponentEvent,
        _app: &mut crate::app::App,
    ) -> EventResult {
        let _ = event;
        Ok(false)
    }
}
