//! # Game Components
//!
//! UI components for the in-game interface including game board, stats, and history.

use crate::app::App;
use crate::components::core::{Component, ComponentId, ComponentResult, UpdateResult};
use crate::components::events::{ComponentEvent, EventResult};
use crate::components::ui::common::{Panel, ScrollableArea};
use crate::impl_component_base;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

/// Main game view component that contains the board, stats, and other game elements
pub struct GameView {
    id: ComponentId,
    board_component: BoardComponent,
    stats_panel: StatsPanel,
    history_panel: HistoryPanel,
    visible: bool,
}

impl GameView {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            board_component: BoardComponent::new(),
            stats_panel: StatsPanel::new(),
            history_panel: HistoryPanel::new(),
            visible: true,
        }
    }
}

impl Component for GameView {
    impl_component_base!(Self, "GameView");

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        if !self.visible {
            return Ok(());
        }

        // Create basic layout for now - this will be much more sophisticated later
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(area);

        // Render board on the left
        self.board_component.render(frame, chunks[0], app)?;

        // Split right side for stats and history
        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[1]);

        // Render stats and history
        self.stats_panel.render(frame, right_chunks[0], app)?;
        self.history_panel.render(frame, right_chunks[1], app)?;

        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        // Try board first
        let result = self.board_component.handle_event(event, app);
        if result.was_handled() {
            return result;
        }

        // Try stats panel
        let result = self.stats_panel.handle_event(event, app);
        if result.was_handled() {
            return result;
        }

        // Try history panel
        self.history_panel.handle_event(event, app)
    }

    fn update(&mut self, app: &mut App) -> UpdateResult {
        let mut result = UpdateResult::None;

        // Update all components
        let board_result = self.board_component.update(app);
        let stats_result = self.stats_panel.update(app);
        let history_result = self.history_panel.update(app);

        // Combine results
        for r in [board_result, stats_result, history_result] {
            match r {
                UpdateResult::RequestRedraw => result = UpdateResult::RequestRedraw,
                UpdateResult::StateChanged => {
                    if matches!(result, UpdateResult::None) {
                        result = UpdateResult::StateChanged;
                    }
                }
                UpdateResult::None => {}
            }
        }

        result
    }

    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        vec![
            &mut self.board_component,
            &mut self.stats_panel,
            &mut self.history_panel,
        ]
    }

    fn children(&self) -> Vec<&dyn Component> {
        vec![
            &self.board_component,
            &self.stats_panel,
            &self.history_panel,
        ]
    }

    fn is_visible(&self) -> bool {
        self.visible
    }
}

/// Component for displaying the game board
pub struct BoardComponent {
    id: ComponentId,
    visible: bool,
    focused: bool,
}

impl BoardComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            visible: true,
            focused: false,
        }
    }
}

impl Component for BoardComponent {
    impl_component_base!(Self, "BoardComponent");

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        if !self.visible {
            return Ok(());
        }

        // For now, just render a placeholder
        // TODO: Implement game-specific board rendering
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Game Board")
            .border_style(if self.focused {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            });

        let content = match app.game_wrapper {
            crate::game_wrapper::GameWrapper::Gomoku(_) => "Gomoku Board\n(Component-based rendering coming soon)",
            crate::game_wrapper::GameWrapper::Connect4(_) => "Connect4 Board\n(Component-based rendering coming soon)",
            crate::game_wrapper::GameWrapper::Othello(_) => "Othello Board\n(Component-based rendering coming soon)",
            crate::game_wrapper::GameWrapper::Blokus(_) => "Blokus Board\n(Component-based rendering coming soon)",
        };

        let paragraph = Paragraph::new(content)
            .block(block)
            .alignment(Alignment::Center);

        frame.render_widget(paragraph, area);
        Ok(())
    }

    fn handle_event(&mut self, _event: &ComponentEvent, _app: &mut App) -> EventResult {
        // TODO: Implement board interaction
        EventResult::NotHandled
    }

    fn update(&mut self, _app: &mut App) -> UpdateResult {
        UpdateResult::None
    }

    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        Vec::new() // TODO: Add board cells, labels, etc.
    }

    fn children(&self) -> Vec<&dyn Component> {
        Vec::new()
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn can_focus(&self) -> bool {
        true
    }

    fn has_focus(&self) -> bool {
        self.focused
    }

    fn set_focus(&mut self, focused: bool) {
        self.focused = focused;
    }
}

/// Component for displaying game statistics and AI info
pub struct StatsPanel {
    id: ComponentId,
    scrollable_area: ScrollableArea,
    visible: bool,
}

impl StatsPanel {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            scrollable_area: ScrollableArea::new(),
            visible: true,
        }
    }

    fn update_stats_content(&mut self, app: &App) {
        let mut content = vec![
            "Game Statistics".to_string(),
            "".to_string(),
        ];

        if let Some(stats) = &app.last_search_stats {
            content.extend(vec![
                format!("Nodes explored: {}", stats.nodes_explored),
                format!("Search depth: {}", stats.max_depth),
                format!("Elapsed time: {:.2}s", stats.elapsed_time.as_secs_f64()),
                format!("Nodes per second: {:.0}", stats.nodes_per_second()),
            ]);
        } else {
            content.push("No AI statistics available".to_string());
        }

        self.scrollable_area.set_content(content);
    }
}

impl Component for StatsPanel {
    impl_component_base!(Self, "StatsPanel");

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        if !self.visible {
            return Ok(());
        }

        self.update_stats_content(app);

        let block = Block::default()
            .borders(Borders::ALL)
            .title("Debug Stats");

        let inner_area = block.inner(area);
        frame.render_widget(block, area);

        self.scrollable_area.render(frame, inner_area, app)
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        self.scrollable_area.handle_event(event, app)
    }

    fn update(&mut self, app: &mut App) -> UpdateResult {
        self.scrollable_area.update(app)
    }

    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        vec![&mut self.scrollable_area]
    }

    fn children(&self) -> Vec<&dyn Component> {
        vec![&self.scrollable_area]
    }

    fn is_visible(&self) -> bool {
        self.visible
    }
}

/// Component for displaying move history
pub struct HistoryPanel {
    id: ComponentId,
    scrollable_area: ScrollableArea,
    visible: bool,
}

impl HistoryPanel {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            scrollable_area: ScrollableArea::new(),
            visible: true,
        }
    }

    fn update_history_content(&mut self, app: &App) {
        let content: Vec<String> = app.move_history
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                format!("{}. Player {}: {:?}", i + 1, entry.player, entry.a_move)
            })
            .collect();

        self.scrollable_area.set_content(content);
        
        // Auto-scroll to bottom for new moves
        if app.history_auto_scroll {
            self.scrollable_area.scroll_to_bottom();
        }
    }
}

impl Component for HistoryPanel {
    impl_component_base!(Self, "HistoryPanel");

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        if !self.visible {
            return Ok(());
        }

        self.update_history_content(app);

        let block = Block::default()
            .borders(Borders::ALL)
            .title("Move History");

        let inner_area = block.inner(area);
        frame.render_widget(block, area);

        self.scrollable_area.render(frame, inner_area, app)
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        self.scrollable_area.handle_event(event, app)
    }

    fn update(&mut self, app: &mut App) -> UpdateResult {
        self.scrollable_area.update(app)
    }

    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        vec![&mut self.scrollable_area]
    }

    fn children(&self) -> Vec<&dyn Component> {
        vec![&self.scrollable_area]
    }

    fn is_visible(&self) -> bool {
        self.visible
    }
}
