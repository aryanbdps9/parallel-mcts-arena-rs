//! # Common UI Components
//!
//! Reusable UI components that can be composed to build larger interfaces.

use crate::app::App;
use crate::components::core::{Component, ComponentId, ComponentResult, UpdateResult};
use crate::components::events::{ComponentEvent, EventResult, InputEvent, UIEvent};
use crate::impl_component_base;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

/// A generic panel component that can contain other components
pub struct Panel {
    id: ComponentId,
    title: Option<String>,
    borders: Borders,
    children: Vec<Box<dyn Component>>,
    visible: bool,
    focused: bool,
}

impl Panel {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            title: None,
            borders: Borders::ALL,
            children: Vec::new(),
            visible: true,
            focused: false,
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_borders(mut self, borders: Borders) -> Self {
        self.borders = borders;
        self
    }

    pub fn add_child(&mut self, child: Box<dyn Component>) {
        self.children.push(child);
    }

    pub fn clear_children(&mut self) {
        self.children.clear();
    }
}

impl Component for Panel {
    impl_component_base!(Self, "Panel");

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        if !self.visible {
            return Ok(());
        }

        // Create the block
        let mut block = Block::default().borders(self.borders);
        
        if let Some(title) = &self.title {
            block = block.title(title.as_str());
        }

        if self.focused {
            block = block.border_style(Style::default().fg(Color::Yellow));
        }

        // Render the block
        let inner_area = block.inner(area);
        frame.render_widget(block, area);

        // Render children within the inner area
        for child in &mut self.children {
            child.render(frame, inner_area, app)?;
        }

        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        // Pass events to children first
        for child in &mut self.children {
            let result = child.handle_event(event, app);
            if result.was_handled() {
                return result;
            }
        }

        EventResult::NotHandled
    }

    fn update(&mut self, app: &mut App) -> UpdateResult {
        let mut result = UpdateResult::None;

        for child in &mut self.children {
            let child_result = child.update(app);
            match child_result {
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
        self.children.iter_mut().map(|c| c.as_mut()).collect()
    }

    fn children(&self) -> Vec<&dyn Component> {
        self.children.iter().map(|c| c.as_ref()).collect()
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

/// A clickable button component
pub struct Button {
    id: ComponentId,
    text: String,
    callback: Option<Box<dyn Fn(&mut App) + Send>>,
    focused: bool,
    visible: bool,
    enabled: bool,
}

impl Button {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            id: ComponentId::new(),
            text: text.into(),
            callback: None,
            focused: false,
            visible: true,
            enabled: true,
        }
    }

    pub fn with_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&mut App) + Send + 'static,
    {
        self.callback = Some(Box::new(callback));
        self
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn trigger_click(&mut self, app: &mut App) {
        if self.enabled {
            if let Some(callback) = &self.callback {
                callback(app);
            }
        }
    }
}

impl Component for Button {
    impl_component_base!(Self, "Button");

    fn render(&mut self, frame: &mut Frame, area: Rect, _app: &App) -> ComponentResult<()> {
        if !self.visible {
            return Ok(());
        }

        let mut style = Style::default();
        
        if !self.enabled {
            style = style.fg(Color::DarkGray);
        } else if self.focused {
            style = style
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD);
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(if self.focused { 
                Style::default().fg(Color::Yellow) 
            } else { 
                Style::default() 
            });

        let paragraph = Paragraph::new(self.text.as_str())
            .style(style)
            .block(block)
            .alignment(Alignment::Center);

        frame.render_widget(paragraph, area);
        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::KeyPress(key)) => {
                if self.focused && (*key == crossterm::event::KeyCode::Enter || *key == crossterm::event::KeyCode::Char(' ')) {
                    self.trigger_click(app);
                    return EventResult::Handled;
                }
            }
            ComponentEvent::Input(InputEvent::MouseClick { .. }) => {
                // TODO: Check if click is within button bounds
                self.trigger_click(app);
                return EventResult::Handled;
            }
            _ => {}
        }

        EventResult::NotHandled
    }

    fn update(&mut self, _app: &mut App) -> UpdateResult {
        UpdateResult::None
    }

    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        Vec::new()
    }

    fn children(&self) -> Vec<&dyn Component> {
        Vec::new()
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn can_focus(&self) -> bool {
        self.enabled
    }

    fn has_focus(&self) -> bool {
        self.focused
    }

    fn set_focus(&mut self, focused: bool) {
        self.focused = focused;
    }
}

/// A scrollable area that can contain content larger than its display area
pub struct ScrollableArea {
    id: ComponentId,
    content: Vec<String>,
    scroll_position: u16,
    max_scroll: u16,
    visible: bool,
    focused: bool,
}

impl ScrollableArea {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            content: Vec::new(),
            scroll_position: 0,
            max_scroll: 0,
            visible: true,
            focused: false,
        }
    }

    pub fn set_content(&mut self, content: Vec<String>) {
        self.content = content;
        self.update_max_scroll();
    }

    pub fn add_line(&mut self, line: String) {
        self.content.push(line);
        self.update_max_scroll();
    }

    pub fn clear(&mut self) {
        self.content.clear();
        self.scroll_position = 0;
        self.max_scroll = 0;
    }

    fn update_max_scroll(&mut self) {
        self.max_scroll = self.content.len().saturating_sub(1) as u16;
    }

    pub fn scroll_up(&mut self) {
        self.scroll_position = self.scroll_position.saturating_sub(1);
    }

    pub fn scroll_down(&mut self) {
        if self.scroll_position < self.max_scroll {
            self.scroll_position += 1;
        }
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll_position = self.max_scroll;
    }
}

impl Component for ScrollableArea {
    impl_component_base!(Self, "ScrollableArea");

    fn render(&mut self, frame: &mut Frame, area: Rect, _app: &App) -> ComponentResult<()> {
        if !self.visible {
            return Ok(());
        }

        let visible_lines = area.height as usize;
        let start_idx = self.scroll_position as usize;
        let end_idx = (start_idx + visible_lines).min(self.content.len());

        let visible_content: Vec<Line> = self.content[start_idx..end_idx]
            .iter()
            .map(|s| Line::from(s.as_str()))
            .collect();

        let paragraph = Paragraph::new(visible_content)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(if self.focused {
                        Style::default().fg(Color::Yellow)
                    } else {
                        Style::default()
                    })
            );

        frame.render_widget(paragraph, area);
        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, _app: &mut App) -> EventResult {
        if !self.focused {
            return EventResult::NotHandled;
        }

        match event {
            ComponentEvent::Input(InputEvent::KeyPress(key)) => {
                match key {
                    crossterm::event::KeyCode::Up => {
                        self.scroll_up();
                        return EventResult::Handled;
                    }
                    crossterm::event::KeyCode::Down => {
                        self.scroll_down();
                        return EventResult::Handled;
                    }
                    crossterm::event::KeyCode::Home => {
                        self.scroll_position = 0;
                        return EventResult::Handled;
                    }
                    crossterm::event::KeyCode::End => {
                        self.scroll_to_bottom();
                        return EventResult::Handled;
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        EventResult::NotHandled
    }

    fn update(&mut self, _app: &mut App) -> UpdateResult {
        UpdateResult::None
    }

    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        Vec::new()
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

/// A list component for displaying selectable items
pub struct List {
    id: ComponentId,
    items: Vec<String>,
    selected_index: usize,
    visible: bool,
    focused: bool,
    on_select: Option<Box<dyn Fn(usize, &mut App) + Send>>,
}

impl List {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            items: Vec::new(),
            selected_index: 0,
            visible: true,
            focused: false,
            on_select: None,
        }
    }

    pub fn with_items(mut self, items: Vec<String>) -> Self {
        self.items = items;
        self
    }

    pub fn with_on_select<F>(mut self, callback: F) -> Self
    where
        F: Fn(usize, &mut App) + Send + 'static,
    {
        self.on_select = Some(Box::new(callback));
        self
    }

    pub fn set_items(&mut self, items: Vec<String>) {
        self.items = items;
        if self.selected_index >= self.items.len() && !self.items.is_empty() {
            self.selected_index = self.items.len() - 1;
        }
    }

    pub fn selected_index(&self) -> usize {
        self.selected_index
    }

    pub fn set_selected_index(&mut self, index: usize) {
        if index < self.items.len() {
            self.selected_index = index;
        }
    }

    pub fn move_up(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
        }
    }

    pub fn move_down(&mut self) {
        if self.selected_index + 1 < self.items.len() {
            self.selected_index += 1;
        }
    }

    fn trigger_select(&mut self, app: &mut App) {
        if let Some(callback) = &self.on_select {
            callback(self.selected_index, app);
        }
    }
}

impl Component for List {
    impl_component_base!(Self, "List");

    fn render(&mut self, frame: &mut Frame, area: Rect, _app: &App) -> ComponentResult<()> {
        if !self.visible {
            return Ok(());
        }

        let items: Vec<ratatui::widgets::ListItem> = self.items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let style = if i == self.selected_index {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ratatui::widgets::ListItem::new(item.as_str()).style(style)
            })
            .collect();

        let list = ratatui::widgets::List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(if self.focused {
                        Style::default().fg(Color::Yellow)
                    } else {
                        Style::default()
                    })
            )
            .highlight_symbol("> ");

        frame.render_widget(list, area);
        Ok(())
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        if !self.focused {
            return EventResult::NotHandled;
        }

        match event {
            ComponentEvent::Input(InputEvent::KeyPress(key)) => {
                match key {
                    crossterm::event::KeyCode::Up => {
                        self.move_up();
                        return EventResult::Handled;
                    }
                    crossterm::event::KeyCode::Down => {
                        self.move_down();
                        return EventResult::Handled;
                    }
                    crossterm::event::KeyCode::Enter => {
                        self.trigger_select(app);
                        return EventResult::Handled;
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        EventResult::NotHandled
    }

    fn update(&mut self, _app: &mut App) -> UpdateResult {
        UpdateResult::None
    }

    fn children_mut(&mut self) -> Vec<&mut dyn Component> {
        Vec::new()
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
