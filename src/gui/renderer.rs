//! # Direct2D Renderer with SVG Support
//!
//! Provides hardware-accelerated 2D rendering using Direct2D with native SVG support.
//! Uses ID2D1DeviceContext5 for DrawSvgDocument capability (Windows 10 1703+).

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use windows::{
    core::{Interface, PCWSTR},
    Win32::{
        Foundation::{HWND, RECT},
        Graphics::{
            Direct2D::{
                Common::{D2D1_COLOR_F, D2D_POINT_2F, D2D_SIZE_F,
                         D2D1_FIGURE_BEGIN_FILLED, D2D1_FIGURE_BEGIN_HOLLOW, D2D1_FIGURE_END_CLOSED},
                D2D1CreateFactory, ID2D1Factory1, ID2D1Device, ID2D1DeviceContext,
                ID2D1DeviceContext5, ID2D1SolidColorBrush, ID2D1SvgDocument,
                D2D1_ANTIALIAS_MODE_PER_PRIMITIVE, D2D1_DRAW_TEXT_OPTIONS_NONE,
                D2D1_ELLIPSE, D2D1_FACTORY_TYPE_SINGLE_THREADED,
                D2D1_ROUNDED_RECT, D2D1_DEVICE_CONTEXT_OPTIONS_NONE,
                D2D1_BITMAP_OPTIONS_TARGET, D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
                D2D1_BITMAP_PROPERTIES1, D2D1_UNIT_MODE_DIPS,
            },
            Direct3D::D3D_DRIVER_TYPE_HARDWARE,
            Direct3D11::{
                D3D11CreateDevice, ID3D11Device, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                D3D11_SDK_VERSION,
            },
            DirectWrite::{
                DWriteCreateFactory, IDWriteFactory, IDWriteTextFormat,
                DWRITE_FACTORY_TYPE_SHARED, DWRITE_FONT_STRETCH_NORMAL,
                DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_WEIGHT_NORMAL,
                DWRITE_FONT_WEIGHT_BOLD, DWRITE_PARAGRAPH_ALIGNMENT_CENTER,
                DWRITE_TEXT_ALIGNMENT_CENTER, DWRITE_TEXT_ALIGNMENT_LEADING,
                DWRITE_MEASURING_MODE_NATURAL,
            },
            Dxgi::{
                IDXGIDevice, IDXGIFactory2, IDXGISwapChain1, IDXGISurface,
                DXGI_SWAP_CHAIN_DESC1, DXGI_USAGE_RENDER_TARGET_OUTPUT,
                DXGI_SCALING_STRETCH, DXGI_SWAP_EFFECT_FLIP_DISCARD,
                DXGI_PRESENT, CreateDXGIFactory2,
            },
            Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC, DXGI_ALPHA_MODE_IGNORE},
        },
        System::Com::{
            CoInitializeEx, IStream, COINIT_MULTITHREADED,
        },
        UI::Shell::SHCreateMemStream,
        UI::WindowsAndMessaging::GetClientRect,
    },
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

/// Main renderer that wraps Direct2D functionality with SVG support
pub struct Renderer {
    #[allow(dead_code)]
    d3d_device: ID3D11Device,
    #[allow(dead_code)]
    d2d_device: ID2D1Device,
    device_context: ID2D1DeviceContext5,
    dwrite_factory: IDWriteFactory,
    factory: ID2D1Factory1,
    swap_chain: IDXGISwapChain1,
    hwnd: HWND,
    
    // Cached text formats
    title_format: Option<IDWriteTextFormat>,
    normal_format: Option<IDWriteTextFormat>,
    small_format: Option<IDWriteTextFormat>,
    
    // Cached SVG documents
    svg_cache: HashMap<String, ID2D1SvgDocument>,
}

impl Renderer {
    /// Create a new renderer for the given window
    pub fn new(hwnd: HWND) -> windows::core::Result<Self> {
        // Initialize COM
        unsafe { let _ = CoInitializeEx(None, COINIT_MULTITHREADED); }
        
        // Create D3D11 device
        let mut d3d_device: Option<ID3D11Device> = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                windows::Win32::Foundation::HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut d3d_device),
                None,
                None,
            )?;
        }
        let d3d_device = d3d_device.unwrap();
        
        // Get DXGI device
        let dxgi_device: IDXGIDevice = d3d_device.cast()?;
        
        // Create D2D factory
        let factory: ID2D1Factory1 = unsafe {
            D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, None)?
        };
        
        // Create D2D device
        let d2d_device = unsafe { factory.CreateDevice(&dxgi_device)? };
        
        // Create device context
        let device_context: ID2D1DeviceContext = unsafe {
            d2d_device.CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE)?
        };
        
        // Cast to ID2D1DeviceContext5 for SVG support
        let device_context: ID2D1DeviceContext5 = device_context.cast()?;
        
        // Create DXGI factory
        let dxgi_factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(Default::default())? };
        
        // Get window size
        let mut rect = RECT::default();
        unsafe { let _ = GetClientRect(hwnd, &mut rect); }
        let width = (rect.right - rect.left) as u32;
        let height = (rect.bottom - rect.top) as u32;
        
        // Create swap chain
        let swap_chain_desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 2,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_IGNORE,
            Flags: 0,
        };
        
        let swap_chain = unsafe {
            dxgi_factory.CreateSwapChainForHwnd(
                &d3d_device,
                hwnd,
                &swap_chain_desc,
                None,
                None,
            )?
        };
        
        // Create DirectWrite factory
        let dwrite_factory: IDWriteFactory = unsafe {
            DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)?
        };
        
        let mut renderer = Self {
            d3d_device,
            d2d_device,
            device_context,
            dwrite_factory,
            factory,
            swap_chain,
            hwnd,
            title_format: None,
            normal_format: None,
            small_format: None,
            svg_cache: HashMap::new(),
        };
        
        renderer.create_resources()?;
        Ok(renderer)
    }
    
    /// Create or recreate device-dependent resources
    pub fn create_resources(&mut self) -> windows::core::Result<()> {
        // Create render target from swap chain
        self.create_render_target()?;
        
        // Create text formats
        self.title_format = Some(self.create_text_format("Segoe UI", 32.0, true)?);
        self.normal_format = Some(self.create_text_format("Segoe UI", 16.0, false)?);
        self.small_format = Some(self.create_text_format("Segoe UI", 12.0, false)?);
        
        Ok(())
    }
    
    /// Create render target from swap chain back buffer
    fn create_render_target(&mut self) -> windows::core::Result<()> {
        // Get back buffer surface
        let surface: IDXGISurface = unsafe { self.swap_chain.GetBuffer(0)? };
        
        // Create bitmap from surface
        let dpi = 96.0f32;
        let bitmap_props = D2D1_BITMAP_PROPERTIES1 {
            pixelFormat: windows::Win32::Graphics::Direct2D::Common::D2D1_PIXEL_FORMAT {
                format: DXGI_FORMAT_B8G8R8A8_UNORM,
                alphaMode: windows::Win32::Graphics::Direct2D::Common::D2D1_ALPHA_MODE_IGNORE,
            },
            dpiX: dpi,
            dpiY: dpi,
            bitmapOptions: D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
            colorContext: ManuallyDrop::new(None),
        };
        
        let bitmap = unsafe {
            self.device_context.CreateBitmapFromDxgiSurface(&surface, Some(&bitmap_props))?
        };
        
        // Set the bitmap as render target
        unsafe { self.device_context.SetTarget(&bitmap); }
        unsafe { self.device_context.SetUnitMode(D2D1_UNIT_MODE_DIPS); }
        
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
        if size.0 == 0 || size.1 == 0 {
            return Ok(());
        }
        
        // Release the target before resizing
        unsafe { self.device_context.SetTarget(None); }
        
        // Resize swap chain buffers
        unsafe {
            self.swap_chain.ResizeBuffers(
                0,
                size.0,
                size.1,
                DXGI_FORMAT_UNKNOWN,
                Default::default(),
            )?;
        }
        
        // Recreate render target
        self.create_render_target()?;
        
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
        unsafe { self.device_context.BeginDraw(); }
    }
    
    /// End a drawing session
    pub fn end_draw(&self) -> windows::core::Result<()> {
        unsafe { self.device_context.EndDraw(None, None)?; }
        unsafe { self.swap_chain.Present(1, DXGI_PRESENT(0)).ok()?; }
        Ok(())
    }
    
    /// Clear the render target with a color
    pub fn clear(&self, color: D2D1_COLOR_F) {
        unsafe { self.device_context.Clear(Some(&color)); }
    }
    
    /// Create a solid color brush
    fn create_brush(&self, color: D2D1_COLOR_F) -> windows::core::Result<ID2D1SolidColorBrush> {
        unsafe { self.device_context.CreateSolidColorBrush(&color, None) }
    }
    
    /// Fill a rectangle with a color
    pub fn fill_rect(&self, rect: Rect, color: D2D1_COLOR_F) {
        if let Ok(brush) = self.create_brush(color) {
            unsafe { self.device_context.FillRectangle(&rect.to_d2d(), &brush); }
        }
    }
    
    /// Draw a rectangle outline
    pub fn draw_rect(&self, rect: Rect, color: D2D1_COLOR_F, stroke_width: f32) {
        if let Ok(brush) = self.create_brush(color) {
            unsafe { self.device_context.DrawRectangle(&rect.to_d2d(), &brush, stroke_width, None); }
        }
    }
    
    /// Fill a rounded rectangle
    pub fn fill_rounded_rect(&self, rect: Rect, radius: f32, color: D2D1_COLOR_F) {
        if let Ok(brush) = self.create_brush(color) {
            let rounded_rect = D2D1_ROUNDED_RECT {
                rect: rect.to_d2d(),
                radiusX: radius,
                radiusY: radius,
            };
            unsafe { self.device_context.FillRoundedRectangle(&rounded_rect, &brush); }
        }
    }
    
    /// Draw a rounded rectangle outline
    pub fn draw_rounded_rect(&self, rect: Rect, radius: f32, color: D2D1_COLOR_F, stroke_width: f32) {
        if let Ok(brush) = self.create_brush(color) {
            let rounded_rect = D2D1_ROUNDED_RECT {
                rect: rect.to_d2d(),
                radiusX: radius,
                radiusY: radius,
            };
            unsafe { self.device_context.DrawRoundedRectangle(&rounded_rect, &brush, stroke_width, None); }
        }
    }
    
    /// Fill an ellipse/circle
    pub fn fill_ellipse(&self, center_x: f32, center_y: f32, radius_x: f32, radius_y: f32, color: D2D1_COLOR_F) {
        if let Ok(brush) = self.create_brush(color) {
            let ellipse = D2D1_ELLIPSE {
                point: D2D_POINT_2F { x: center_x, y: center_y },
                radiusX: radius_x,
                radiusY: radius_y,
            };
            unsafe { self.device_context.FillEllipse(&ellipse, &brush); }
        }
    }
    
    /// Draw an ellipse/circle outline
    pub fn draw_ellipse(&self, center_x: f32, center_y: f32, radius_x: f32, radius_y: f32, color: D2D1_COLOR_F, stroke_width: f32) {
        if let Ok(brush) = self.create_brush(color) {
            let ellipse = D2D1_ELLIPSE {
                point: D2D_POINT_2F { x: center_x, y: center_y },
                radiusX: radius_x,
                radiusY: radius_y,
            };
            unsafe { self.device_context.DrawEllipse(&ellipse, &brush, stroke_width, None); }
        }
    }
    
    /// Draw a line
    pub fn draw_line(&self, x1: f32, y1: f32, x2: f32, y2: f32, color: D2D1_COLOR_F, stroke_width: f32) {
        if let Ok(brush) = self.create_brush(color) {
            let p0 = D2D_POINT_2F { x: x1, y: y1 };
            let p1 = D2D_POINT_2F { x: x2, y: y2 };
            unsafe { self.device_context.DrawLine(p0, p1, &brush, stroke_width, None); }
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
        if let (Some(fmt), Ok(brush)) = (format, self.create_brush(color)) {
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
                // Cast to ID2D1RenderTarget which has the 6-arg DrawText
                let render_target: windows::Win32::Graphics::Direct2D::ID2D1RenderTarget = self.device_context.cast().unwrap();
                render_target.DrawText(
                    &text_wide,
                    fmt,
                    &rect.to_d2d(),
                    &brush,
                    D2D1_DRAW_TEXT_OPTIONS_NONE,
                    DWRITE_MEASURING_MODE_NATURAL,
                );
            }
        }
    }
    
    /// Set antialiasing mode
    pub fn set_antialias(&self, enabled: bool) {
        unsafe {
            self.device_context.SetAntialiasMode(if enabled {
                D2D1_ANTIALIAS_MODE_PER_PRIMITIVE
            } else {
                windows::Win32::Graphics::Direct2D::D2D1_ANTIALIAS_MODE_ALIASED
            });
        }
    }
    
    /// Get hexagon corner points for a pointy-top hexagon
    pub fn get_hexagon_points(center_x: f32, center_y: f32, size: f32) -> [(f32, f32); 6] {
        let mut points = [(0.0f32, 0.0f32); 6];
        for i in 0..6 {
            let angle_deg = 60.0 * i as f32 - 30.0;
            let angle_rad = std::f32::consts::PI / 180.0 * angle_deg;
            points[i] = (
                center_x + size * angle_rad.cos(),
                center_y + size * angle_rad.sin(),
            );
        }
        points
    }
    
    /// Fill a hexagon (pointy-top orientation)
    pub fn fill_hexagon(&self, center_x: f32, center_y: f32, size: f32, color: D2D1_COLOR_F) {
        if let Ok(brush) = self.create_brush(color) {
            if let Ok(path_geometry) = unsafe { self.factory.CreatePathGeometry() } {
                if let Ok(sink) = unsafe { path_geometry.Open() } {
                    let points = Self::get_hexagon_points(center_x, center_y, size);
                    
                    unsafe {
                        sink.BeginFigure(
                            D2D_POINT_2F { x: points[0].0, y: points[0].1 },
                            D2D1_FIGURE_BEGIN_FILLED,
                        );
                        
                        for i in 1..6 {
                            sink.AddLine(D2D_POINT_2F { x: points[i].0, y: points[i].1 });
                        }
                        
                        sink.EndFigure(D2D1_FIGURE_END_CLOSED);
                        let _ = sink.Close();
                    }
                    
                    unsafe { self.device_context.FillGeometry(&path_geometry, &brush, None); }
                }
            }
        }
    }
    
    /// Draw a hexagon outline (pointy-top orientation)
    pub fn draw_hexagon(&self, center_x: f32, center_y: f32, size: f32, color: D2D1_COLOR_F, stroke_width: f32) {
        if let Ok(brush) = self.create_brush(color) {
            if let Ok(path_geometry) = unsafe { self.factory.CreatePathGeometry() } {
                if let Ok(sink) = unsafe { path_geometry.Open() } {
                    let points = Self::get_hexagon_points(center_x, center_y, size);
                    
                    unsafe {
                        sink.BeginFigure(
                            D2D_POINT_2F { x: points[0].0, y: points[0].1 },
                            D2D1_FIGURE_BEGIN_HOLLOW,
                        );
                        
                        for i in 1..6 {
                            sink.AddLine(D2D_POINT_2F { x: points[i].0, y: points[i].1 });
                        }
                        
                        sink.EndFigure(D2D1_FIGURE_END_CLOSED);
                        let _ = sink.Close();
                    }
                    
                    unsafe { self.device_context.DrawGeometry(&path_geometry, &brush, stroke_width, None); }
                }
            }
        }
    }
    
    /// Get hexagon corner points for a pointy-top hexagon with isometric tilt
    pub fn get_iso_hexagon_points(center_x: f32, center_y: f32, size: f32, y_scale: f32) -> [(f32, f32); 6] {
        let mut points = [(0.0f32, 0.0f32); 6];
        for i in 0..6 {
            let angle_deg = 60.0 * i as f32 - 30.0;
            let angle_rad = std::f32::consts::PI / 180.0 * angle_deg;
            points[i] = (
                center_x + size * angle_rad.cos(),
                center_y + size * angle_rad.sin() * y_scale,
            );
        }
        points
    }
    
    /// Fill an isometric hexagon (pointy-top orientation, compressed Y)
    pub fn fill_iso_hexagon(&self, center_x: f32, center_y: f32, size: f32, y_scale: f32, color: D2D1_COLOR_F) {
        if let Ok(brush) = self.create_brush(color) {
            if let Ok(path_geometry) = unsafe { self.factory.CreatePathGeometry() } {
                if let Ok(sink) = unsafe { path_geometry.Open() } {
                    let points = Self::get_iso_hexagon_points(center_x, center_y, size, y_scale);
                    
                    unsafe {
                        sink.BeginFigure(
                            D2D_POINT_2F { x: points[0].0, y: points[0].1 },
                            D2D1_FIGURE_BEGIN_FILLED,
                        );
                        
                        for i in 1..6 {
                            sink.AddLine(D2D_POINT_2F { x: points[i].0, y: points[i].1 });
                        }
                        
                        sink.EndFigure(D2D1_FIGURE_END_CLOSED);
                        let _ = sink.Close();
                    }
                    
                    unsafe { self.device_context.FillGeometry(&path_geometry, &brush, None); }
                }
            }
        }
    }
    
    /// Draw an isometric hexagon outline (pointy-top orientation, compressed Y)
    pub fn draw_iso_hexagon(&self, center_x: f32, center_y: f32, size: f32, y_scale: f32, color: D2D1_COLOR_F, stroke_width: f32) {
        if let Ok(brush) = self.create_brush(color) {
            if let Ok(path_geometry) = unsafe { self.factory.CreatePathGeometry() } {
                if let Ok(sink) = unsafe { path_geometry.Open() } {
                    let points = Self::get_iso_hexagon_points(center_x, center_y, size, y_scale);
                    
                    unsafe {
                        sink.BeginFigure(
                            D2D_POINT_2F { x: points[0].0, y: points[0].1 },
                            D2D1_FIGURE_BEGIN_HOLLOW,
                        );
                        
                        for i in 1..6 {
                            sink.AddLine(D2D_POINT_2F { x: points[i].0, y: points[i].1 });
                        }
                        
                        sink.EndFigure(D2D1_FIGURE_END_CLOSED);
                        let _ = sink.Close();
                    }
                    
                    unsafe { self.device_context.DrawGeometry(&path_geometry, &brush, stroke_width, None); }
                }
            }
        }
    }
    
    // ========== SVG Support ==========
    
    /// Load an SVG document from a string and cache it
    pub fn load_svg(&mut self, name: &str, svg_content: &str, viewport_width: f32, viewport_height: f32) -> windows::core::Result<()> {
        let svg_doc = self.create_svg_document(svg_content, viewport_width, viewport_height)?;
        self.svg_cache.insert(name.to_string(), svg_doc);
        Ok(())
    }
    
    /// Create an SVG document from string content
    fn create_svg_document(&self, svg_content: &str, viewport_width: f32, viewport_height: f32) -> windows::core::Result<ID2D1SvgDocument> {
        // Create IStream from SVG content
        let stream = self.create_stream_from_string(svg_content)?;
        
        let viewport_size = D2D_SIZE_F {
            width: viewport_width,
            height: viewport_height,
        };
        
        unsafe {
            self.device_context.CreateSvgDocument(&stream, viewport_size)
        }
    }
    
    /// Create an IStream from a string
    fn create_stream_from_string(&self, content: &str) -> windows::core::Result<IStream> {
        let bytes = content.as_bytes();
        
        // Create a memory stream from the bytes
        let stream = unsafe { SHCreateMemStream(Some(bytes)) };
        
        stream.ok_or_else(|| windows::core::Error::from_win32())
    }
    
    /// Draw a cached SVG document at the specified position and size
    pub fn draw_svg(&self, name: &str, center_x: f32, center_y: f32, width: f32, height: f32) {
        self.draw_svg_tilted(name, center_x, center_y, width, height, 1.0)
    }
    
    /// Draw a cached SVG document with isometric tilt (Y-axis compression)
    /// tilt: 1.0 = no tilt, 0.5 = 50% Y compression, etc.
    pub fn draw_svg_tilted(&self, name: &str, center_x: f32, center_y: f32, width: f32, height: f32, tilt: f32) {
        if let Some(svg_doc) = self.svg_cache.get(name) {
            // Get the viewport size of the SVG
            let viewport = unsafe { svg_doc.GetViewportSize() };
            
            // Calculate scale (apply tilt to Y axis)
            let scale_x = width / viewport.width;
            let scale_y = (height / viewport.height) * tilt;
            
            // Calculate offset - center the (possibly tilted) SVG
            let offset_x = center_x - width / 2.0;
            let offset_y = center_y - (height * tilt) / 2.0;
            
            // Create transform matrix - use the type from windows crate's re-export
            #[repr(C)]
            #[allow(non_snake_case)]
            struct Matrix3x2 {
                M11: f32, M12: f32,
                M21: f32, M22: f32,
                M31: f32, M32: f32,
            }
            
            let transform = Matrix3x2 {
                M11: scale_x,
                M12: 0.0,
                M21: 0.0,
                M22: scale_y,
                M31: offset_x,
                M32: offset_y,
            };
            
            // Save current transform
            let mut old_transform = Matrix3x2 {
                M11: 1.0, M12: 0.0,
                M21: 0.0, M22: 1.0,
                M31: 0.0, M32: 0.0,
            };
            unsafe { 
                self.device_context.GetTransform(
                    &mut old_transform as *mut Matrix3x2 as *mut _
                ); 
            }
            
            // Apply new transform
            unsafe { 
                self.device_context.SetTransform(
                    &transform as *const Matrix3x2 as *const _
                ); 
            }
            
            // Draw the SVG
            unsafe { self.device_context.DrawSvgDocument(svg_doc); }
            
            // Restore transform
            unsafe { 
                self.device_context.SetTransform(
                    &old_transform as *const Matrix3x2 as *const _
                ); 
            }
        }
    }
    
    /// Check if an SVG is loaded
    pub fn has_svg(&self, name: &str) -> bool {
        self.svg_cache.contains_key(name)
    }
    
    /// Get reference to the factory for geometry creation
    pub fn factory(&self) -> &ID2D1Factory1 {
        &self.factory
    }
}
