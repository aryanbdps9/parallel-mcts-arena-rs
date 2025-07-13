//! Responsive layout component for dynamic UI adaptation.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use std::any::Any;

use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::ComponentEvent;

/// A responsive layout component that adapts to different screen sizes
pub struct ResponsiveLayoutComponent {
    id: ComponentId,
    area: Option<Rect>,
    layout_type: ResponsiveLayoutType,
    min_sizes: Vec<u16>,
    preferred_sizes: Vec<u16>,
    max_sizes: Vec<u16>,
    direction: Direction,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResponsiveLayoutType {
    /// Adaptive layout that adjusts based on content and screen size
    Adaptive,
    /// Fixed proportional layout with minimum/maximum constraints
    FixedRatio,
    /// Content-driven layout that sizes based on actual content needs
    ContentDriven,
}

impl ResponsiveLayoutComponent {
    pub fn new(layout_type: ResponsiveLayoutType, direction: Direction) -> Self {
        Self {
            id: ComponentId::new(),
            area: None,
            layout_type,
            min_sizes: Vec::new(),
            preferred_sizes: Vec::new(),
            max_sizes: Vec::new(),
            direction,
        }
    }

    /// Add a panel with minimum, preferred, and maximum sizes
    pub fn add_panel(&mut self, min_size: u16, preferred_size: u16, max_size: u16) {
        self.min_sizes.push(min_size);
        self.preferred_sizes.push(preferred_size);
        self.max_sizes.push(max_size);
    }

    /// Calculate responsive layout based on available space and content requirements
    pub fn calculate_layout(&self, area: Rect) -> Vec<Rect> {
        if self.min_sizes.is_empty() {
            return vec![area];
        }

        let available_space = match self.direction {
            Direction::Horizontal => area.width,
            Direction::Vertical => area.height,
        };

        let constraints = self.calculate_constraints(available_space);

        Layout::default()
            .direction(self.direction)
            .constraints(constraints)
            .split(area)
            .to_vec()
    }

    fn calculate_constraints(&self, available_space: u16) -> Vec<Constraint> {
        match self.layout_type {
            ResponsiveLayoutType::Adaptive => self.calculate_adaptive_constraints(available_space),
            ResponsiveLayoutType::FixedRatio => {
                self.calculate_fixed_ratio_constraints(available_space)
            }
            ResponsiveLayoutType::ContentDriven => {
                self.calculate_content_driven_constraints(available_space)
            }
        }
    }

    fn calculate_adaptive_constraints(&self, available_space: u16) -> Vec<Constraint> {
        let mut constraints = Vec::new();
        let panel_count = self.min_sizes.len();

        // Calculate total minimum space needed
        let total_min: u16 = self.min_sizes.iter().sum();

        if total_min >= available_space {
            // Use minimum sizes when space is tight
            for &min_size in &self.min_sizes {
                constraints.push(Constraint::Length(min_size));
            }
        } else {
            // Try to use preferred sizes, but scale down if needed
            let total_preferred: u16 = self.preferred_sizes.iter().sum();

            if total_preferred <= available_space {
                // Use preferred sizes if they fit
                for &preferred_size in &self.preferred_sizes {
                    constraints.push(Constraint::Length(preferred_size));
                }
            } else {
                // Scale between minimum and preferred based on available space
                let extra_space = available_space - total_min;
                let preferred_extra: u16 = total_preferred - total_min;

                for i in 0..panel_count {
                    let min_size = self.min_sizes[i];
                    let preferred_size = self.preferred_sizes[i];
                    let panel_extra = preferred_size - min_size;

                    let allocated_extra = if preferred_extra > 0 {
                        (panel_extra as f32 * extra_space as f32 / preferred_extra as f32) as u16
                    } else {
                        0
                    };

                    let final_size = min_size + allocated_extra;
                    constraints.push(Constraint::Length(final_size));
                }
            }
        }

        constraints
    }

    fn calculate_fixed_ratio_constraints(&self, _available_space: u16) -> Vec<Constraint> {
        // For fixed ratio, use percentages based on preferred sizes
        let total_preferred: u16 = self.preferred_sizes.iter().sum();

        self.preferred_sizes
            .iter()
            .map(|&size| {
                let percentage = if total_preferred > 0 {
                    (size as f32 * 100.0 / total_preferred as f32) as u16
                } else {
                    100 / self.preferred_sizes.len() as u16
                };
                Constraint::Percentage(percentage)
            })
            .collect()
    }

    fn calculate_content_driven_constraints(&self, available_space: u16) -> Vec<Constraint> {
        let mut constraints = Vec::new();
        let panel_count = self.min_sizes.len();

        // Allocate minimum sizes first
        let mut remaining_space = available_space;
        let mut sizes = self.min_sizes.clone();

        for &min_size in &self.min_sizes {
            remaining_space = remaining_space.saturating_sub(min_size);
        }

        // Distribute remaining space proportionally
        if remaining_space > 0 && panel_count > 0 {
            let space_per_panel = remaining_space / panel_count as u16;
            for i in 0..panel_count {
                let max_additional = self.max_sizes[i].saturating_sub(sizes[i]);
                let additional = space_per_panel.min(max_additional);
                sizes[i] += additional;
            }
        }

        for size in sizes {
            constraints.push(Constraint::Length(size));
        }

        constraints
    }

    /// Get the calculated layout areas for child components
    pub fn get_layout_areas(&self, area: Rect) -> Vec<Rect> {
        self.calculate_layout(area)
    }

    fn set_area(&mut self, area: Rect) {
        self.area = Some(area);
    }
}

impl Component for ResponsiveLayoutComponent {
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
        _frame: &mut ratatui::Frame,
        area: Rect,
        _app: &crate::app::App,
    ) -> ComponentResult<()> {
        self.set_area(area);
        // Layout components typically don't render themselves, just manage child positioning
        Ok(())
    }

    fn handle_event(&mut self, _event: &ComponentEvent, _app: &mut crate::app::App) -> EventResult {
        Ok(false) // Not handled
    }
}
