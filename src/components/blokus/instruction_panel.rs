//! Instruction panel component for Blokus UI.

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use std::any::Any;

use crate::app::App;
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::ComponentEvent;

/// Component for displaying game instructions and controls
pub struct BlokusInstructionPanelComponent {
    id: ComponentId,
    area: Option<Rect>,
}

impl BlokusInstructionPanelComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            area: None,
        }
    }

    pub fn set_area(&mut self, area: Rect) {
        self.area = Some(area);
    }

    pub fn get_area(&self) -> Option<Rect> {
        self.area
    }

    /// Generate instruction text based on current game state
    fn get_instructions(&self, app: &App) -> Vec<Line> {
        let mut lines = Vec::new();

        // Game-specific instructions
        lines.push(Line::from(vec![
            Span::styled(
                "Blokus: ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "Place pieces to corner-connect with your own pieces",
                Style::default().fg(Color::White),
            ),
        ]));

        lines.push(Line::from(""));

        // Controls
        lines.push(Line::from(vec![Span::styled(
            "Controls: ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )]));

        if app.is_current_player_ai() {
            lines.push(Line::from(vec![
                Span::styled("• ", Style::default().fg(Color::White)),
                Span::styled(
                    "AI is thinking... Please wait",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::ITALIC),
                ),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled("• ", Style::default().fg(Color::White)),
                Span::styled("Arrow keys: ", Style::default().fg(Color::Green)),
                Span::styled("Move cursor", Style::default().fg(Color::White)),
            ]));

            lines.push(Line::from(vec![
                Span::styled("• ", Style::default().fg(Color::White)),
                Span::styled("Enter/Space: ", Style::default().fg(Color::Green)),
                Span::styled("Place piece / Make move", Style::default().fg(Color::White)),
            ]));

            lines.push(Line::from(vec![
                Span::styled("• ", Style::default().fg(Color::White)),
                Span::styled("R: ", Style::default().fg(Color::Green)),
                Span::styled("Rotate selected piece", Style::default().fg(Color::White)),
            ]));

            lines.push(Line::from(vec![
                Span::styled("• ", Style::default().fg(Color::White)),
                Span::styled("F: ", Style::default().fg(Color::Green)),
                Span::styled("Flip selected piece", Style::default().fg(Color::White)),
            ]));

            lines.push(Line::from(vec![
                Span::styled("• ", Style::default().fg(Color::White)),
                Span::styled("P: ", Style::default().fg(Color::Green)),
                Span::styled("Pass turn", Style::default().fg(Color::White)),
            ]));

            lines.push(Line::from(vec![
                Span::styled("• ", Style::default().fg(Color::White)),
                Span::styled("Mouse: ", Style::default().fg(Color::Green)),
                Span::styled(
                    "Click to select pieces/place",
                    Style::default().fg(Color::White),
                ),
            ]));

            lines.push(Line::from(vec![
                Span::styled("• ", Style::default().fg(Color::White)),
                Span::styled("Right click: ", Style::default().fg(Color::Green)),
                Span::styled("Rotate piece", Style::default().fg(Color::White)),
            ]));

            lines.push(Line::from(vec![
                Span::styled("• ", Style::default().fg(Color::White)),
                Span::styled("Scroll: ", Style::default().fg(Color::Green)),
                Span::styled("Scroll piece selector", Style::default().fg(Color::White)),
            ]));
        }

        lines.push(Line::from(""));

        // General controls
        lines.push(Line::from(vec![
            Span::styled("• ", Style::default().fg(Color::White)),
            Span::styled("Tab: ", Style::default().fg(Color::Green)),
            Span::styled(
                "Switch between Stats/History",
                Style::default().fg(Color::White),
            ),
        ]));

        lines.push(Line::from(vec![
            Span::styled("• ", Style::default().fg(Color::White)),
            Span::styled("Q: ", Style::default().fg(Color::Green)),
            Span::styled("Quit game", Style::default().fg(Color::White)),
        ]));

        lines
    }
}

impl Component for BlokusInstructionPanelComponent {
    fn id(&self) -> ComponentId {
        self.id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn handle_event(&mut self, _event: &ComponentEvent, _app: &mut App) -> EventResult {
        // Instruction panel is read-only, doesn't handle events
        Ok(false)
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        self.set_area(area);

        // Draw border with title
        let block = Block::default()
            .title("Instructions & Controls")
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::White));

        frame.render_widget(block, area);

        // Calculate content area
        let content_area = Rect::new(
            area.x + 1,
            area.y + 1,
            area.width.saturating_sub(2),
            area.height.saturating_sub(2),
        );

        if content_area.width == 0 || content_area.height == 0 {
            return Ok(());
        }

        // Get instruction lines
        let instruction_lines = self.get_instructions(app);

        // Render instructions
        let visible_lines = std::cmp::min(instruction_lines.len(), content_area.height as usize);
        for (i, line) in instruction_lines.iter().take(visible_lines).enumerate() {
            let line_area = Rect::new(
                content_area.x,
                content_area.y + i as u16,
                content_area.width,
                1,
            );
            let paragraph = Paragraph::new(line.clone()).wrap(Wrap { trim: true });
            frame.render_widget(paragraph, line_area);
        }

        Ok(())
    }
}
