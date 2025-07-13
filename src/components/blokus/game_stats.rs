//! Game stats component for Blokus UI.

use mcts::GameState;
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Tabs},
};
use std::any::Any;

use crate::app::App;
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crate::game_wrapper::GameWrapper;

/// Component for displaying game statistics and history
pub struct BlokusGameStatsComponent {
    id: ComponentId,
    current_tab: usize,
    area: Option<Rect>,
}

impl BlokusGameStatsComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            current_tab: 0,
            area: None,
        }
    }

    pub fn set_area(&mut self, area: Rect) {
        self.area = Some(area);
    }

    pub fn get_area(&self) -> Option<Rect> {
        self.area
    }

    /// Check if a point is within this component's area
    pub fn contains_point(&self, x: u16, y: u16) -> bool {
        if let Some(area) = self.area {
            x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height
        } else {
            false
        }
    }

    /// Switch to the next tab
    pub fn next_tab(&mut self) {
        self.current_tab = (self.current_tab + 1) % 2; // 0: Stats, 1: History
    }

    /// Switch to the previous tab
    pub fn prev_tab(&mut self) {
        self.current_tab = if self.current_tab == 0 { 1 } else { 0 };
    }

    /// Render the statistics tab content
    fn render_stats_content(
        &self,
        frame: &mut Frame,
        area: Rect,
        app: &App,
    ) -> ComponentResult<()> {
        let mut lines = Vec::new();

        if let GameWrapper::Blokus(state) = &app.game_wrapper {
            let player_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
            let player_names = ["P1", "P2", "P3", "P4"];

            // Current player info
            let current_player = app.game_wrapper.get_current_player();
            lines.push(Line::from(vec![
                Span::styled("Current Player: ", Style::default().fg(Color::White)),
                Span::styled(
                    player_names[(current_player - 1) as usize],
                    Style::default()
                        .fg(player_colors[(current_player - 1) as usize])
                        .add_modifier(Modifier::BOLD),
                ),
            ]));

            lines.push(Line::from(""));

            // Player statistics
            for player in 1..=4 {
                let available_pieces = state.get_available_pieces(player);
                let pieces_count = available_pieces.len();

                // For now, use pieces count as a simple score approximation
                // TODO: Implement proper scoring when available
                let score = 21 - pieces_count; // Simple approximation

                let color = player_colors[(player - 1) as usize];
                let is_current = player == current_player;

                let style = if is_current {
                    Style::default()
                        .fg(color)
                        .add_modifier(Modifier::BOLD)
                        .bg(Color::DarkGray)
                } else {
                    Style::default().fg(color)
                };

                lines.push(Line::from(vec![
                    Span::styled(format!("{}: ", player_names[(player - 1) as usize]), style),
                    Span::styled(format!("{} pieces, ~{} points", pieces_count, score), style),
                ]));
            }

            lines.push(Line::from(""));

            // Game progress
            let total_moves = app.move_history.len();
            lines.push(Line::from(vec![
                Span::styled("Moves played: ", Style::default().fg(Color::White)),
                Span::styled(total_moves.to_string(), Style::default().fg(Color::Yellow)),
            ]));

            // AI thinking info
            if app.is_current_player_ai() {
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "AI is thinking...",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::ITALIC),
                )));
            }
        }

        // Render the lines
        for (i, line) in lines.iter().enumerate() {
            if i < area.height as usize {
                let line_area = Rect::new(area.x, area.y + i as u16, area.width, 1);
                let paragraph = Paragraph::new(line.clone());
                frame.render_widget(paragraph, line_area);
            }
        }

        Ok(())
    }

    /// Render the history tab content
    fn render_history_content(
        &self,
        frame: &mut Frame,
        area: Rect,
        app: &App,
    ) -> ComponentResult<()> {
        let mut lines = Vec::new();
        let player_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
        let player_names = ["P1", "P2", "P3", "P4"];

        // Recent move history (show last moves that fit in the area)
        let visible_moves = std::cmp::min(app.move_history.len(), area.height as usize);
        let start_idx = if app.move_history.len() > visible_moves {
            app.move_history.len() - visible_moves
        } else {
            0
        };

        for (i, entry) in app.move_history.iter().enumerate().skip(start_idx) {
            let player = entry.player;
            let color = if player <= 4 {
                player_colors[(player - 1) as usize]
            } else {
                Color::White
            };

            let move_desc = match &entry.a_move {
                crate::game_wrapper::MoveWrapper::Blokus(blokus_move) => {
                    if blokus_move.0 == usize::MAX {
                        "Pass".to_string()
                    } else {
                        let piece_char = std::char::from_u32(('A' as u32) + (blokus_move.0 as u32))
                            .unwrap_or('?');
                        format!("{}@({},{})", piece_char, blokus_move.2, blokus_move.3)
                    }
                }
                _ => "Unknown".to_string(),
            };

            lines.push(Line::from(vec![
                Span::styled(format!("{}. ", i + 1), Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!(
                        "{}: ",
                        player_names.get((player - 1) as usize).unwrap_or(&"??")
                    ),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(move_desc, Style::default().fg(Color::White)),
            ]));
        }

        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "No moves yet",
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            )));
        }

        // Render the lines
        for (i, line) in lines.iter().enumerate() {
            if i < area.height as usize {
                let line_area = Rect::new(area.x, area.y + i as u16, area.width, 1);
                let paragraph = Paragraph::new(line.clone());
                frame.render_widget(paragraph, line_area);
            }
        }

        Ok(())
    }
}

impl Component for BlokusGameStatsComponent {
    fn id(&self) -> ComponentId {
        self.id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn handle_event(&mut self, event: &ComponentEvent, _app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::KeyPress(key)) => match key {
                crossterm::event::KeyCode::Tab => {
                    self.next_tab();
                    return Ok(true);
                }
                crossterm::event::KeyCode::BackTab => {
                    self.prev_tab();
                    return Ok(true);
                }
                _ => {}
            },
            ComponentEvent::Input(InputEvent::MouseClick { x, y, .. }) => {
                if self.contains_point(*x, *y) {
                    // Handle tab switching via mouse clicks on tab area
                    if let Some(area) = self.area {
                        // Check if click is on the tab area (first line)
                        if *y == area.y + 1 {
                            // Account for border
                            // Simple tab switching - click left half for stats, right half for history
                            let mid_x = area.x + area.width / 2;
                            if *x < mid_x {
                                self.current_tab = 0; // Stats
                            } else {
                                self.current_tab = 1; // History
                            }
                            return Ok(true);
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(false)
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        self.set_area(area);

        // Create tabs
        let tab_titles = vec!["Stats", "History"];
        let tabs = Tabs::new(tab_titles)
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::White))
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
            .select(self.current_tab);

        frame.render_widget(tabs, area);

        // Calculate content area (inside borders and below tabs)
        let content_area = Rect::new(
            area.x + 1,
            area.y + 2, // Account for border and tab line
            area.width.saturating_sub(2),
            area.height.saturating_sub(3), // Account for borders and tab line
        );

        if content_area.width == 0 || content_area.height == 0 {
            return Ok(());
        }

        // Render content based on current tab
        match self.current_tab {
            0 => self.render_stats_content(frame, content_area, app)?,
            1 => self.render_history_content(frame, content_area, app)?,
            _ => {}
        }

        Ok(())
    }
}
