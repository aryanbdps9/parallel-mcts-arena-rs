//! # Direct2D Renderer
//!
//! Provides hardware-accelerated 2D rendering using Direct2D.
//! This module handles:
//! - Direct2D factory and render target management
//! - Brush and resource caching for performance
//! - Common drawing primitives (shapes, text)
//! - Safe wrapper around COM interfaces

use windows::{
    Win32::{
        Foundation::{HWND, RECT},
        Graphics::{
            Direct2D::{
                Common::{D2D1_COLOR_F, D2D_POINT_2F, D2D_SIZE_U},
                D2D1CreateFactory, ID2D1Factory, ID2D1HwndRenderTarget, ID2D1SolidColorBrush,
                ID2D1RenderTarget, D2D1_ANTIALIAS_MODE_PER_PRIMITIVE, D2D1_DRAW_TEXT_OPTIONS_NONE,
                D2D1_ELLIPSE, D2D1_FACTORY_TYPE_SINGLE_THREADED,
                D2D1_HWND_RENDER_TARGET_PROPERTIES, D2D1_PRESENT_OPTIONS_NONE,
                D2D1_RENDER_TARGET_PROPERTIES, D2D1_ROUNDED_RECT,
            },
            DirectWrite::{
                DWriteCreateFactory, IDWriteFactory, IDWriteTextFormat,
                DWRITE_FACTORY_TYPE_SHARED, DWRITE_FONT_STRETCH_NORMAL,
                DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_WEIGHT_NORMAL,
                DWRITE_FONT_WEIGHT_BOLD, DWRITE_PARAGRAPH_ALIGNMENT_CENTER,
                DWRITE_TEXT_ALIGNMENT_CENTER, DWRITE_TEXT_ALIGNMENT_LEADING,
            },
        },
        UI::WindowsAndMessaging::GetClientRect,
    },
    core::{Interface, PCWSTR},
};

/// Rectangle structure for layout calculations
#[derive(Debug, Clone, Copy, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Create a D2D1_RECT_F from this Rect
    pub fn to_d2d(&self) -> windows::Win32::Graphics::Direct2D::Common::D2D_RECT_F {
        windows::Win32::Graphics::Direct2D::Common::D2D_RECT_F {
            left: self.x,
            top: self.y,
            right: self.x + self.width,
            bottom: self.y + self.height,
        }
    }

    /// Check if a point is inside this rectangle
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    /// Get the center point of the rectangle
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Inset the rectangle by a given amount on all sides
    pub fn inset(&self, amount: f32) -> Self {
        Self {
            x: self.x + amount,
            y: self.y + amount,
            width: (self.width - 2.0 * amount).max(0.0),
            height: (self.height - 2.0 * amount).max(0.0),
        }
    }

    /// Get a sub-rectangle with padding
    pub fn with_padding(&self, padding: f32) -> Self {
        self.inset(padding)
    }
}

/// Main renderer that wraps Direct2D functionality
pub struct Renderer {
    factory: ID2D1Factory,
    dwrite_factory: IDWriteFactory,
    render_target: Option<ID2D1HwndRenderTarget>,
    hwnd: HWND,
    
    // Cached text formats
    title_format: Option<IDWriteTextFormat>,
    normal_format: Option<IDWriteTextFormat>,
    small_format: Option<IDWriteTextFormat>,
}

impl Renderer {
    /// Create a new renderer for the given window
    pub fn new(hwnd: HWND) -> windows::core::Result<Self> {
        // Create Direct2D factory
        let factory: ID2D1Factory = unsafe {
            D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, None)?
        };

        // Create DirectWrite factory for text rendering
        let dwrite_factory: IDWriteFactory = unsafe {
            DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)?
        };

        let mut renderer = Self {
            factory,
            dwrite_factory,
            render_target: None,
            hwnd,
            title_format: None,
            normal_format: None,
            small_format: None,
        };

        renderer.create_resources()?;
        Ok(renderer)
    }

    /// Create or recreate device-dependent resources
    pub fn create_resources(&mut self) -> windows::core::Result<()> {
        let size = self.get_window_size();

        // Create render target
        let render_target_properties = D2D1_RENDER_TARGET_PROPERTIES::default();
        let hwnd_render_target_properties = D2D1_HWND_RENDER_TARGET_PROPERTIES {
            hwnd: self.hwnd,
            pixelSize: D2D_SIZE_U {
                width: size.0,
                height: size.1,
            },
            presentOptions: D2D1_PRESENT_OPTIONS_NONE,
        };

        self.render_target = Some(unsafe {
            self.factory.CreateHwndRenderTarget(
                &render_target_properties,
                &hwnd_render_target_properties,
            )?
        });

        // Create text formats
        self.title_format = Some(self.create_text_format("Segoe UI", 32.0, true)?);
        self.normal_format = Some(self.create_text_format("Segoe UI", 16.0, false)?);
        self.small_format = Some(self.create_text_format("Segoe UI", 12.0, false)?);

        Ok(())
    }

    /// Get current window size
    fn get_window_size(&self) -> (u32, u32) {
        let mut rect = RECT::default();
        unsafe { let _ = GetClientRect(self.hwnd, &mut rect); }
        ((rect.right - rect.left) as u32, (rect.bottom - rect.top) as u32)
    }

    /// Handle window resize
    pub fn resize(&mut self) -> windows::core::Result<()> {
        let size = self.get_window_size();
        if let Some(rt) = &self.render_target {
            unsafe {
                let _ = rt.Resize(&D2D_SIZE_U {
                    width: size.0,
                    height: size.1,
                });
            }
        }
        Ok(())
    }

    /// Create a text format with specified parameters
    fn create_text_format(
        &self,
        font_family: &str,
        size: f32,
        bold: bool,
    ) -> windows::core::Result<IDWriteTextFormat> {
        let font_family_wide: Vec<u16> = font_family.encode_utf16().chain(std::iter::once(0)).collect();
        let locale: Vec<u16> = "en-US".encode_utf16().chain(std::iter::once(0)).collect();
        
        let weight = if bold { DWRITE_FONT_WEIGHT_BOLD } else { DWRITE_FONT_WEIGHT_NORMAL };
        
        unsafe {
            self.dwrite_factory.CreateTextFormat(
                PCWSTR(font_family_wide.as_ptr()),
                None,
                weight,
                DWRITE_FONT_STYLE_NORMAL,
                DWRITE_FONT_STRETCH_NORMAL,
                size,
                PCWSTR(locale.as_ptr()),
            )
        }
    }

    /// Get the render target client area as a Rect
    pub fn get_client_rect(&self) -> Rect {
        let size = self.get_window_size();
        Rect::new(0.0, 0.0, size.0 as f32, size.1 as f32)
    }

    /// Begin a drawing session
    pub fn begin_draw(&self) {
        if let Some(rt) = &self.render_target {
            unsafe { rt.BeginDraw(); }
        }
    }

    /// End a drawing session
    pub fn end_draw(&self) -> windows::core::Result<()> {
        if let Some(rt) = &self.render_target {
            unsafe { rt.EndDraw(None, None)?; }
        }
        Ok(())
    }

    /// Clear the render target with a color
    pub fn clear(&self, color: D2D1_COLOR_F) {
        if let Some(rt) = &self.render_target {
            unsafe { rt.Clear(Some(&color)); }
        }
    }

    /// Create a solid color brush
    pub fn create_brush(&self, color: D2D1_COLOR_F) -> windows::core::Result<ID2D1SolidColorBrush> {
        if let Some(rt) = &self.render_target {
            // Cast to ID2D1RenderTarget using the ComInterface trait
            let base: ID2D1RenderTarget = rt.cast()?;
            unsafe {
                base.CreateSolidColorBrush(&color, None)
            }
        } else {
            Err(windows::core::Error::from_win32())
        }
    }

    /// Fill a rectangle with a color
    pub fn fill_rect(&self, rect: Rect, color: D2D1_COLOR_F) {
        if let Some(rt) = &self.render_target {
            if let Ok(brush) = self.create_brush(color) {
                unsafe { rt.FillRectangle(&rect.to_d2d(), &brush); }
            }
        }
    }

    /// Draw a rectangle outline
    pub fn draw_rect(&self, rect: Rect, color: D2D1_COLOR_F, stroke_width: f32) {
        if let Some(rt) = &self.render_target {
            if let Ok(brush) = self.create_brush(color) {
                unsafe { rt.DrawRectangle(&rect.to_d2d(), &brush, stroke_width, None); }
            }
        }
    }

    /// Fill a rounded rectangle
    pub fn fill_rounded_rect(&self, rect: Rect, radius: f32, color: D2D1_COLOR_F) {
        if let Some(rt) = &self.render_target {
            if let Ok(brush) = self.create_brush(color) {
                let rounded_rect = D2D1_ROUNDED_RECT {
                    rect: rect.to_d2d(),
                    radiusX: radius,
                    radiusY: radius,
                };
                unsafe { rt.FillRoundedRectangle(&rounded_rect, &brush); }
            }
        }
    }

    /// Draw a rounded rectangle outline
    pub fn draw_rounded_rect(&self, rect: Rect, radius: f32, color: D2D1_COLOR_F, stroke_width: f32) {
        if let Some(rt) = &self.render_target {
            if let Ok(brush) = self.create_brush(color) {
                let rounded_rect = D2D1_ROUNDED_RECT {
                    rect: rect.to_d2d(),
                    radiusX: radius,
                    radiusY: radius,
                };
                unsafe { rt.DrawRoundedRectangle(&rounded_rect, &brush, stroke_width, None); }
            }
        }
    }

    /// Fill an ellipse/circle
    pub fn fill_ellipse(&self, center_x: f32, center_y: f32, radius_x: f32, radius_y: f32, color: D2D1_COLOR_F) {
        if let Some(rt) = &self.render_target {
            if let Ok(brush) = self.create_brush(color) {
                let ellipse = D2D1_ELLIPSE {
                    point: D2D_POINT_2F { x: center_x, y: center_y },
                    radiusX: radius_x,
                    radiusY: radius_y,
                };
                unsafe { rt.FillEllipse(&ellipse, &brush); }
            }
        }
    }

    /// Draw an ellipse/circle outline
    pub fn draw_ellipse(&self, center_x: f32, center_y: f32, radius_x: f32, radius_y: f32, color: D2D1_COLOR_F, stroke_width: f32) {
        if let Some(rt) = &self.render_target {
            if let Ok(brush) = self.create_brush(color) {
                let ellipse = D2D1_ELLIPSE {
                    point: D2D_POINT_2F { x: center_x, y: center_y },
                    radiusX: radius_x,
                    radiusY: radius_y,
                };
                unsafe { rt.DrawEllipse(&ellipse, &brush, stroke_width, None); }
            }
        }
    }

    /// Draw a line
    pub fn draw_line(&self, x1: f32, y1: f32, x2: f32, y2: f32, color: D2D1_COLOR_F, stroke_width: f32) {
        if let Some(rt) = &self.render_target {
            if let Ok(brush) = self.create_brush(color) {
                let p0 = D2D_POINT_2F { x: x1, y: y1 };
                let p1 = D2D_POINT_2F { x: x2, y: y2 };
                unsafe { rt.DrawLine(p0, p1, &brush, stroke_width, None); }
            }
        }
    }

    /// Draw text with title format (large, bold)
    pub fn draw_title(&self, text: &str, rect: Rect, color: D2D1_COLOR_F, centered: bool) {
        self.draw_text_with_format(text, rect, color, self.title_format.as_ref(), centered);
    }

    /// Draw text with normal format
    pub fn draw_text(&self, text: &str, rect: Rect, color: D2D1_COLOR_F, centered: bool) {
        self.draw_text_with_format(text, rect, color, self.normal_format.as_ref(), centered);
    }

    /// Draw text with small format
    pub fn draw_small_text(&self, text: &str, rect: Rect, color: D2D1_COLOR_F, centered: bool) {
        self.draw_text_with_format(text, rect, color, self.small_format.as_ref(), centered);
    }

    /// Internal text drawing with specified format
    fn draw_text_with_format(
        &self,
        text: &str,
        rect: Rect,
        color: D2D1_COLOR_F,
        format: Option<&IDWriteTextFormat>,
        centered: bool,
    ) {
        if let (Some(rt), Some(fmt)) = (&self.render_target, format) {
            if let Ok(brush) = self.create_brush(color) {
                // Set text alignment
                if centered {
                    unsafe {
                        let _ = fmt.SetTextAlignment(DWRITE_TEXT_ALIGNMENT_CENTER);
                        let _ = fmt.SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
                    }
                } else {
                    unsafe {
                        let _ = fmt.SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
                        let _ = fmt.SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
                    }
                }

                let text_wide: Vec<u16> = text.encode_utf16().collect();
                unsafe {
                    rt.DrawText(
                        &text_wide,
                        fmt,
                        &rect.to_d2d(),
                        &brush,
                        D2D1_DRAW_TEXT_OPTIONS_NONE,
                        windows::Win32::Graphics::DirectWrite::DWRITE_MEASURING_MODE_NATURAL,
                    );
                }
            }
        }
    }

    /// Set antialiasing mode
    pub fn set_antialias(&self, enabled: bool) {
        if let Some(rt) = &self.render_target {
            unsafe {
                rt.SetAntialiasMode(if enabled {
                    D2D1_ANTIALIAS_MODE_PER_PRIMITIVE
                } else {
                    windows::Win32::Graphics::Direct2D::D2D1_ANTIALIAS_MODE_ALIASED
                });
            }
        }
    }
}
