//! # Rotatable Board Component
//!
//! A reusable component for 3D-like board views with tilt and rotation.
//! Supports right-click drag to adjust viewing angle interactively.

use crate::gui::renderer::Renderer;
use super::{GameInput, InputResult};

/// Default isometric tilt factor (0.0 = side view, 1.0 = top-down)
const DEFAULT_TILT: f32 = 1.0;
const MIN_TILT: f32 = 0.2;
const MAX_TILT: f32 = 1.0;

/// Default rotation angle in radians
const DEFAULT_ROTATION: f32 = 0.0;

/// Sensitivity for drag controls
const TILT_SENSITIVITY: f32 = 0.003;
const ROTATION_SENSITIVITY: f32 = 0.005;

/// A 3D rotatable board wrapper component.
///
/// This component manages:
/// - Tilt (Y-axis compression for isometric view)
/// - Rotation (around the center point)
/// - Right-click drag handling for interactive adjustment
/// - D2D transform setup and teardown
///
/// # Usage
/// ```ignore
/// let mut board = RotatableBoard::new();
///
/// // In render:
/// board.begin_draw(renderer, center_x, center_y);
/// // ... draw board contents (hexes, pieces, etc.) ...
/// board.end_draw(renderer);
///
/// // In input handling:
/// if let Some(result) = board.handle_input(&input) {
///     return result;
/// }
/// ```
#[derive(Debug, Clone)]
pub struct RotatableBoard {
    /// Current tilt (Y-axis scale factor)
    tilt: f32,
    /// Current rotation angle in radians
    rotation: f32,
}

impl Default for RotatableBoard {
    fn default() -> Self {
        Self::new()
    }
}

impl RotatableBoard {
    /// Create a new rotatable board with default tilt and rotation
    pub fn new() -> Self {
        Self {
            tilt: DEFAULT_TILT,
            rotation: DEFAULT_ROTATION,
        }
    }

    /// Create with custom initial tilt and rotation
    pub fn with_params(tilt: f32, rotation: f32) -> Self {
        Self {
            tilt: tilt.clamp(MIN_TILT, MAX_TILT),
            rotation,
        }
    }

    /// Get current tilt value
    pub fn tilt(&self) -> f32 {
        self.tilt
    }

    /// Get current rotation value in radians
    pub fn rotation(&self) -> f32 {
        self.rotation
    }

    /// Reset to default view
    pub fn reset_view(&mut self) {
        self.tilt = DEFAULT_TILT;
        self.rotation = DEFAULT_ROTATION;
    }

    /// Begin drawing with the board transform applied.
    /// Call this before drawing any board content.
    /// 
    /// # Arguments
    /// * `renderer` - The renderer to set transform on
    /// * `center_x` - X coordinate of rotation center
    /// * `center_y` - Y coordinate of rotation center
    pub fn begin_draw(&self, renderer: &Renderer, center_x: f32, center_y: f32) {
        renderer.set_board_transform(center_x, center_y, self.tilt, self.rotation);
    }

    /// End drawing and reset the transform.
    /// Call this after drawing board content, before drawing UI elements.
    pub fn end_draw(&self, renderer: &Renderer) {
        renderer.reset_transform();
    }

    /// Handle input for view adjustment.
    /// 
    /// Returns `Some(InputResult::Redraw)` if the view was changed,
    /// `None` if the input wasn't handled by this component.
    pub fn handle_input(&mut self, input: &GameInput) -> Option<InputResult> {
        match input {
            GameInput::Drag { dx, dy } => {
                // Adjust tilt based on vertical drag
                // Dragging up = more tilt, down = less tilt
                let tilt_delta = -dy * TILT_SENSITIVITY;
                self.tilt = (self.tilt + tilt_delta).clamp(MIN_TILT, MAX_TILT);

                // Adjust rotation based on horizontal drag
                // Dragging right = rotate clockwise
                let rotation_delta = dx * ROTATION_SENSITIVITY;
                self.rotation += rotation_delta;

                Some(InputResult::Redraw)
            }
            GameInput::RightDown { .. } | GameInput::RightUp { .. } => {
                // Consumed but no action needed (window handles capture)
                Some(InputResult::None)
            }
            _ => None, // Not handled by this component
        }
    }

    /// Transform a point from local board coordinates to screen coordinates.
    /// Useful for mouse picking when you need to reverse the transform.
    ///
    /// # Arguments
    /// * `x` - Local X coordinate (relative to center)
    /// * `y` - Local Y coordinate (relative to center)
    /// * `center_x` - Center X in screen coordinates
    /// * `center_y` - Center Y in screen coordinates
    pub fn local_to_screen(&self, x: f32, y: f32, center_x: f32, center_y: f32) -> (f32, f32) {
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();

        // Apply tilt then rotation
        let y_tilted = y * self.tilt;
        let x_rotated = x * cos_r - y_tilted * sin_r;
        let y_rotated = x * sin_r + y_tilted * cos_r;

        (center_x + x_rotated, center_y + y_rotated)
    }

    /// Transform a point from screen coordinates to local board coordinates.
    /// Useful for mouse picking to determine which cell was clicked.
    ///
    /// # Arguments
    /// * `screen_x` - Screen X coordinate
    /// * `screen_y` - Screen Y coordinate
    /// * `center_x` - Center X in screen coordinates
    /// * `center_y` - Center Y in screen coordinates
    pub fn screen_to_local(&self, screen_x: f32, screen_y: f32, center_x: f32, center_y: f32) -> (f32, f32) {
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();

        // Translate to center-relative
        let rel_x = screen_x - center_x;
        let rel_y = screen_y - center_y;

        // Reverse rotation
        let x_unrotated = rel_x * cos_r + rel_y * sin_r;
        let y_unrotated = -rel_x * sin_r + rel_y * cos_r;

        // Reverse tilt
        let y_untilted = y_unrotated / self.tilt;

        (x_unrotated, y_untilted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let board = RotatableBoard::new();
        assert!((board.tilt() - 0.55).abs() < 0.001);
        assert!((board.rotation() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_reset_view() {
        let mut board = RotatableBoard::with_params(0.8, 1.5);
        board.reset_view();
        assert!((board.tilt() - 0.55).abs() < 0.001);
        assert!((board.rotation() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_tilt_clamping() {
        let board = RotatableBoard::with_params(0.0, 0.0);
        assert!((board.tilt() - 0.2).abs() < 0.001); // Clamped to MIN_TILT
        
        let board = RotatableBoard::with_params(2.0, 0.0);
        assert!((board.tilt() - 1.0).abs() < 0.001); // Clamped to MAX_TILT
    }
}
