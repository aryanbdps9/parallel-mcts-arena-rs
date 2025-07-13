//! Move history component that displays and manages game move history.

use ratatui::{Frame, layout::Rect, text::Line};
use std::any::Any;

use crate::app::{App, MoveHistoryEntry};
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::ComponentEvent;
use crate::components::ui::ScrollableComponent;

/// Component that manages and displays move history
pub struct MoveHistoryComponent {
    id: ComponentId,
    area: Option<Rect>,
    move_history: Vec<MoveHistoryEntry>,
    scrollable: ScrollableComponent,
    auto_scroll: bool,
}

impl MoveHistoryComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            area: None,
            move_history: Vec::new(),
            scrollable: ScrollableComponent::new()
                .with_title("Move History".to_string())
                .with_border(true)
                .with_auto_scroll(true),
            auto_scroll: true,
        }
    }

    pub fn add_move(&mut self, entry: MoveHistoryEntry) {
        self.move_history.push(entry);
        self.update_scrollable_content();

        if self.auto_scroll {
            self.scrollable.scroll_to_bottom();
        }
    }

    pub fn clear_history(&mut self) {
        self.move_history.clear();
        self.update_scrollable_content();
    }

    pub fn get_move_history(&self) -> &[MoveHistoryEntry] {
        &self.move_history
    }

    pub fn set_auto_scroll(&mut self, auto_scroll: bool) {
        self.auto_scroll = auto_scroll;
        // Note: ScrollableComponent doesn't have set_auto_scroll, it's set during construction
        if auto_scroll {
            self.scrollable.scroll_to_bottom();
        }
    }

    fn update_scrollable_content(&mut self) {
        let lines: Vec<Line<'static>> = self
            .move_history
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let player_name = format!("Player {}", entry.player + 1);
                let move_str = format!("{}. {}: {:?}", i + 1, player_name, entry.a_move);
                Line::from(move_str)
            })
            .collect();

        self.scrollable.set_content(lines);
    }
}

impl Component for MoveHistoryComponent {
    fn id(&self) -> ComponentId {
        self.id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        self.area = Some(area);

        // Delegate rendering to the scrollable component
        self.scrollable.render(frame, area, app)
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        // Delegate event handling to the scrollable component
        self.scrollable.handle_event(event, app)
    }

    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        vec![&mut self.scrollable]
    }

    fn children(&self) -> Vec<&dyn Component> {
        vec![&self.scrollable]
    }
}

impl Default for MoveHistoryComponent {
    fn default() -> Self {
        Self::new()
    }
}
