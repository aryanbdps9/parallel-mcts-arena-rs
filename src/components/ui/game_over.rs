//! Game over component.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    Frame,
    widgets::{Block, Borders, Paragraph, Wrap},
    style::{Style, Color, Modifier},
    text::{Line, Span},
};

use crate::app::{App, AppMode, GameStatus};
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crossterm::event::KeyCode;

/// Component for game over screen
pub struct GameOverComponent {
    id: ComponentId,
}

impl GameOverComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
        }
    }
}

impl Component for GameOverComponent {
    fn id(&self) -> ComponentId {
        self.id
    }
    
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),
                Constraint::Min(0),
                Constraint::Length(3),
            ])
            .split(area);

        // Game result
        let (result_text, result_color) = match app.game_status {
            GameStatus::Win(winner) => (
                format!("🎉 Player {} Wins! 🎉", winner),
                Color::Green
            ),
            GameStatus::Draw => (
                "🤝 It's a Draw! 🤝".to_string(),
                Color::Yellow
            ),
            GameStatus::InProgress => (
                "Game in Progress".to_string(),
                Color::White
            ),
        };

        let result_lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                result_text,
                Style::default()
                    .fg(result_color)
                    .add_modifier(Modifier::BOLD)
            )),
            Line::from(""),
        ];

        let result = Paragraph::new(result_lines)
            .block(Block::default().borders(Borders::ALL).title("Game Result"))
            .wrap(Wrap { trim: true });
        frame.render_widget(result, chunks[0]);

        // Game summary (move count, etc.)
        let move_count = app.move_history.len();
        let summary_text = if move_count > 0 {
            format!("Game completed in {} moves", move_count)
        } else {
            "No moves were made".to_string()
        };

        let summary = Paragraph::new(summary_text)
            .block(Block::default().borders(Borders::ALL).title("Game Summary"))
            .style(Style::default().fg(Color::Cyan));
        frame.render_widget(summary, chunks[1]);

        // Instructions
        let instructions = vec![
            Line::from(vec![
                Span::styled("R", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::raw(" - Restart game  "),
                Span::styled("ESC", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::raw(" - Game selection  "),
                Span::styled("Q", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::raw(" - Quit"),
            ]),
        ];

        let instructions_widget = Paragraph::new(instructions)
            .block(Block::default().borders(Borders::ALL).title("Controls"));
        frame.render_widget(instructions_widget, chunks[2]);

        Ok(())
    }
    
    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::KeyPress(key)) => {
                match key {
                    KeyCode::Char('q') => {
                        app.should_quit = true;
                        Ok(true)
                    }
                    KeyCode::Char('r') | KeyCode::Enter => {
                        app.reset_game();
                        Ok(true)
                    }
                    KeyCode::Esc => {
                        app.mode = AppMode::GameSelection;
                        Ok(true)
                    }
                    _ => Ok(false)
                }
            }
            _ => Ok(false),
        }
    }

    crate::impl_component_base!(GameOverComponent);
}
