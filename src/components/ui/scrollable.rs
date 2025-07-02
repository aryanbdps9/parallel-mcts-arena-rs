//! Scrollable component for consistent scrolling behavior across UI elements.

use ratatui::{
    layout::Rect,
    Frame,
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
    text::Line,
};
use std::any::Any;

use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};

/// A reusable scrollable component that handles content overflow with scrollbars
pub struct ScrollableComponent {
    id: ComponentId,
    area: Option<Rect>,
    content: Vec<Line<'static>>,
    scroll_offset: usize,
    auto_scroll: bool,
    show_scrollbar: bool,
    title: Option<String>,
    border: bool,
}

impl ScrollableComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            area: None,
            content: Vec::new(),
            scroll_offset: 0,
            auto_scroll: false,
            show_scrollbar: true,
            title: None,
            border: true,
        }
    }

    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    pub fn with_border(mut self, border: bool) -> Self {
        self.border = border;
        self
    }

    pub fn with_auto_scroll(mut self, auto_scroll: bool) -> Self {
        self.auto_scroll = auto_scroll;
        self
    }

    pub fn with_scrollbar(mut self, show_scrollbar: bool) -> Self {
        self.show_scrollbar = show_scrollbar;
        self
    }

    pub fn set_content(&mut self, content: Vec<Line<'static>>) {
        self.content = content;
        if self.auto_scroll {
            self.scroll_to_bottom();
        }
    }

    pub fn add_line(&mut self, line: Line<'static>) {
        self.content.push(line);
        if self.auto_scroll {
            self.scroll_to_bottom();
        }
    }

    pub fn clear_content(&mut self) {
        self.content.clear();
        self.scroll_offset = 0;
    }

    pub fn scroll_up(&mut self, lines: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
    }

    pub fn scroll_down(&mut self, lines: usize) {
        self.scroll_offset = self.scroll_offset.saturating_add(lines);
        self.clamp_scroll();
    }

    pub fn scroll_to_top(&mut self) {
        self.scroll_offset = 0;
    }

    pub fn scroll_to_bottom(&mut self) {
        if let Some(area) = self.area {
            let visible_height = self.get_visible_height(area);
            if self.content.len() > visible_height {
                self.scroll_offset = self.content.len() - visible_height;
            } else {
                self.scroll_offset = 0;
            }
        }
    }

    fn get_visible_height(&self, area: Rect) -> usize {
        let height = if self.border {
            area.height.saturating_sub(2) // Account for borders
        } else {
            area.height
        };
        height as usize
    }

    fn clamp_scroll(&mut self) {
        if let Some(area) = self.area {
            let visible_height = self.get_visible_height(area);
            let max_scroll = self.content.len().saturating_sub(visible_height);
            self.scroll_offset = self.scroll_offset.min(max_scroll);
        }
    }

    fn get_layout_areas(&self, area: Rect) -> (Rect, Option<Rect>) {
        if self.show_scrollbar && area.width > 5 {
            let chunks = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Horizontal)
                .constraints([
                    ratatui::layout::Constraint::Min(0),
                    ratatui::layout::Constraint::Length(1)
                ])
                .split(area);
            (chunks[0], Some(chunks[1]))
        } else {
            (area, None)
        }
    }

    fn set_area(&mut self, area: Rect) {
        self.area = Some(area);
        self.clamp_scroll();
    }

    /// Get the current scroll position as a percentage (0.0 to 1.0)
    pub fn get_scroll_percentage(&self) -> f32 {
        if self.content.is_empty() {
            return 0.0;
        }

        if let Some(area) = self.area {
            let visible_height = self.get_visible_height(area);
            if self.content.len() <= visible_height {
                return 0.0;
            }
            
            let max_scroll = self.content.len() - visible_height;
            if max_scroll == 0 {
                0.0
            } else {
                self.scroll_offset as f32 / max_scroll as f32
            }
        } else {
            0.0
        }
    }

    /// Set scroll position from percentage (0.0 to 1.0)
    pub fn set_scroll_percentage(&mut self, percentage: f32) {
        if let Some(area) = self.area {
            let visible_height = self.get_visible_height(area);
            if self.content.len() > visible_height {
                let max_scroll = self.content.len() - visible_height;
                self.scroll_offset = (max_scroll as f32 * percentage.clamp(0.0, 1.0)) as usize;
            }
        }
    }
}

impl Component for ScrollableComponent {
    fn id(&self) -> ComponentId {
        self.id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, _app: &crate::app::App) -> ComponentResult<()> {
        self.set_area(area);

        let (content_area, scrollbar_area) = self.get_layout_areas(area);

        // Create the main block if border is enabled
        let inner_area = if self.border {
            let block = if let Some(ref title) = self.title {
                Block::default().borders(Borders::ALL).title(title.as_str())
            } else {
                Block::default().borders(Borders::ALL)
            };
            let inner = block.inner(content_area);
            frame.render_widget(block, content_area);
            inner
        } else {
            content_area
        };

        // Calculate visible content
        let visible_height = inner_area.height as usize;
        let max_scroll = self.content.len().saturating_sub(visible_height);
        self.scroll_offset = self.scroll_offset.min(max_scroll);

        let visible_lines: Vec<Line> = if self.content.len() > visible_height && self.scroll_offset < self.content.len() {
            self.content.iter()
                .skip(self.scroll_offset)
                .take(visible_height)
                .cloned()
                .collect()
        } else {
            self.content.iter()
                .take(visible_height)
                .cloned()
                .collect()
        };

        // Render content
        let paragraph = Paragraph::new(visible_lines);
        frame.render_widget(paragraph, inner_area);

        // Render scrollbar if needed and we have space for it
        if let Some(scrollbar_area) = scrollbar_area {
            if max_scroll > 0 && scrollbar_area.height > 2 {
                let mut scrollbar_state = ScrollbarState::default()
                    .content_length(self.content.len())
                    .viewport_content_length(visible_height)
                    .position(self.scroll_offset);

                let scrollbar = Scrollbar::default()
                    .orientation(ScrollbarOrientation::VerticalRight)
                    .begin_symbol(Some("↑"))
                    .end_symbol(Some("↓"));

                frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
            }
        }

        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, _app: &mut crate::app::App) -> EventResult {
        match event {
            ComponentEvent::Input(input_event) => {
                match input_event {
                    InputEvent::KeyPress(key) => {
                        match key {
                            crossterm::event::KeyCode::Up => {
                                self.scroll_up(1);
                                Ok(true) // Handled
                            }
                            crossterm::event::KeyCode::Down => {
                                self.scroll_down(1);
                                Ok(true) // Handled
                            }
                            crossterm::event::KeyCode::PageUp => {
                                self.scroll_up(10);
                                Ok(true) // Handled
                            }
                            crossterm::event::KeyCode::PageDown => {
                                self.scroll_down(10);
                                Ok(true) // Handled
                            }
                            crossterm::event::KeyCode::Home => {
                                self.scroll_to_top();
                                Ok(true) // Handled
                            }
                            crossterm::event::KeyCode::End => {
                                self.scroll_to_bottom();
                                Ok(true) // Handled
                            }
                            _ => Ok(false) // Not handled
                        }
                    }
                    InputEvent::MouseScroll { up, .. } => {
                        if *up {
                            self.scroll_up(3);
                        } else {
                            self.scroll_down(3);
                        }
                        Ok(true) // Handled
                    }
                    _ => Ok(false) // Not handled
                }
            }
            _ => Ok(false) // Not handled
        }
    }
}

impl Default for ScrollableComponent {
    fn default() -> Self {
        Self::new()
    }
}
