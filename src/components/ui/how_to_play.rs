//! # How To Play Component - Help Screen
//!
//! This module implements the help screen that displays game rules and controls
//! when the user presses 'H' during gameplay.

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap},
};

use crate::app::{App, AppMode};
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crate::game_wrapper::GameWrapper;
use crossterm::event::KeyCode;
use mcts::GameState;

/// How to play content for each game type
const GOMOKU_HELP: &str = include_str!("../../../docs/how_to_play/gomoku.txt");
const CONNECT4_HELP: &str = include_str!("../../../docs/how_to_play/connect4.txt");
const OTHELLO_HELP: &str = include_str!("../../../docs/how_to_play/othello.txt");
const BLOKUS_HELP: &str = include_str!("../../../docs/how_to_play/blokus.txt");

/// Component for displaying how to play information
pub struct HowToPlayComponent {
    id: ComponentId,
    /// Current scroll position
    scroll: u16,
}

impl HowToPlayComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            scroll: 0,
        }
    }

    /// Get the help content for the current game with placeholders replaced
    fn get_help_content(&self, app: &App) -> String {
        let template = match &app.game_wrapper {
            GameWrapper::Gomoku(_) => GOMOKU_HELP,
            GameWrapper::Connect4(_) => CONNECT4_HELP,
            GameWrapper::Othello(_) => OTHELLO_HELP,
            GameWrapper::Blokus(_) => BLOKUS_HELP,
        };

        // Replace placeholders with actual values from app settings
        let board = app.game_wrapper.get_board();
        let board_height = board.len();
        let board_width = if board_height > 0 { board[0].len() } else { 0 };

        template
            .replace("{LINE_SIZE}", &app.settings_line_size.to_string())
            .replace("{BOARD_SIZE}", &app.settings_board_size.to_string())
            .replace("{BOARD_HEIGHT}", &board_height.to_string())
            .replace("{BOARD_WIDTH}", &board_width.to_string())
    }

    /// Get the game name for the title
    fn get_game_name(&self, app: &App) -> &'static str {
        match &app.game_wrapper {
            GameWrapper::Gomoku(_) => "Gomoku",
            GameWrapper::Connect4(_) => "Connect 4",
            GameWrapper::Othello(_) => "Othello",
            GameWrapper::Blokus(_) => "Blokus",
        }
    }

    /// Reset scroll position when showing help
    pub fn reset_scroll(&mut self) {
        self.scroll = 0;
    }
}

impl Component for HowToPlayComponent {
    fn id(&self) -> ComponentId {
        self.id
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        // Create main layout with title bar at top
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Title bar
                Constraint::Min(0),    // Content area
                Constraint::Length(3), // Footer with controls
            ])
            .split(area);

        // Title bar
        let game_name = self.get_game_name(app);
        let title = Paragraph::new(Line::from(vec![
            Span::styled(
                "📖 ",
                Style::default().fg(Color::Yellow),
            ),
            Span::styled(
                format!("How to Play: {}", game_name),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]))
        .block(Block::default().borders(Borders::ALL))
        .alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(title, chunks[0]);

        // Content area
        let content = self.get_help_content(app);
        let lines: Vec<Line> = content
            .lines()
            .map(|line| {
                // Style the lines based on content
                if line.starts_with('╔') || line.starts_with('╚') || line.starts_with('║') {
                    Line::from(Span::styled(line, Style::default().fg(Color::Cyan)))
                } else if line.chars().all(|c| c == '─' || c == ' ') && line.contains('─') {
                    Line::from(Span::styled(line, Style::default().fg(Color::DarkGray)))
                } else if line.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && !line.contains(' ')
                    || line.ends_with("TIPS")
                    || line.ends_with("RULES")
                    || line.ends_with("CONTROLS")
                    || line.ends_with("SELECTION")
                    || line.ends_with("GENERAL")
                    || line.ends_with("SCORING")
                    || line.ends_with("PIECES (by size)")
                    || line == "OBJECTIVE"
                    || line == "GAMEPLAY"
                    || line == "WINNING"
                    || line == "CONTROLS"
                    || line == "CAPTURING"
                    || line == "SCORING"
                {
                    Line::from(Span::styled(
                        line,
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ))
                } else if line.starts_with("  ") && line.contains("  ") {
                    // Control lines with key and description
                    let parts: Vec<&str> = line.splitn(2, "  ").collect();
                    if parts.len() == 2 {
                        let key = parts[0].trim();
                        let desc = parts[1].trim();
                        Line::from(vec![
                            Span::styled(
                                format!("  {:14}", key),
                                Style::default()
                                    .fg(Color::Green)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(desc, Style::default().fg(Color::White)),
                        ])
                    } else {
                        Line::from(Span::styled(line, Style::default().fg(Color::White)))
                    }
                } else if line.starts_with('•') || line.starts_with('✓') || line.starts_with('✗') {
                    let color = if line.starts_with('✓') {
                        Color::Green
                    } else if line.starts_with('✗') {
                        Color::Red
                    } else {
                        Color::White
                    };
                    Line::from(Span::styled(line, Style::default().fg(color)))
                } else {
                    Line::from(Span::styled(line, Style::default().fg(Color::White)))
                }
            })
            .collect();

        let total_lines = lines.len() as u16;
        let visible_height = chunks[1].height.saturating_sub(2); // Account for borders
        let max_scroll = total_lines.saturating_sub(visible_height);
        
        // Clamp scroll to valid range
        self.scroll = self.scroll.min(max_scroll);

        let content_widget = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL))
            .wrap(Wrap { trim: false })
            .scroll((self.scroll, 0));
        frame.render_widget(content_widget, chunks[1]);

        // Scrollbar
        if total_lines > visible_height {
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("▲"))
                .end_symbol(Some("▼"));
            let mut scrollbar_state = ScrollbarState::new(max_scroll as usize)
                .position(self.scroll as usize);
            frame.render_stateful_widget(
                scrollbar,
                chunks[1].inner(ratatui::layout::Margin { vertical: 1, horizontal: 0 }),
                &mut scrollbar_state,
            );
        }

        // Footer with controls
        let footer = Paragraph::new(Line::from(vec![
            Span::styled("Press ", Style::default().fg(Color::Gray)),
            Span::styled(
                "ESC",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" or ", Style::default().fg(Color::Gray)),
            Span::styled(
                "H",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" to return to game  │  ", Style::default().fg(Color::Gray)),
            Span::styled(
                "↑/↓",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" or ", Style::default().fg(Color::Gray)),
            Span::styled(
                "PgUp/PgDn",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" to scroll", Style::default().fg(Color::Gray)),
        ]))
        .block(Block::default().borders(Borders::ALL))
        .alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(footer, chunks[2]);

        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::KeyPress(key)) => match key {
                KeyCode::Esc | KeyCode::Char('h') | KeyCode::Char('H') => {
                    // Return to the game
                    app.mode = AppMode::InGame;
                    Ok(true)
                }
                KeyCode::Up => {
                    self.scroll = self.scroll.saturating_sub(1);
                    Ok(true)
                }
                KeyCode::Down => {
                    self.scroll = self.scroll.saturating_add(1);
                    Ok(true)
                }
                KeyCode::PageUp => {
                    self.scroll = self.scroll.saturating_sub(10);
                    Ok(true)
                }
                KeyCode::PageDown => {
                    self.scroll = self.scroll.saturating_add(10);
                    Ok(true)
                }
                KeyCode::Home => {
                    self.scroll = 0;
                    Ok(true)
                }
                KeyCode::End => {
                    self.scroll = u16::MAX; // Will be clamped in render
                    Ok(true)
                }
                KeyCode::Char('q') => {
                    app.should_quit = true;
                    Ok(true)
                }
                _ => Ok(false),
            },
            ComponentEvent::Input(InputEvent::MouseScroll { up, .. }) => {
                if *up {
                    self.scroll = self.scroll.saturating_sub(3);
                } else {
                    self.scroll = self.scroll.saturating_add(3);
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    crate::impl_component_base!(HowToPlayComponent);
}
