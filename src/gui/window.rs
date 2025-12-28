//! # Windows GUI Window Management
//!
//! This module handles Win32 window creation, message processing, and rendering loop.
//! Uses Direct2D for hardware-accelerated rendering.

use std::cell::RefCell;
use std::rc::Rc;

use mcts::GameState;
use windows::{
    Win32::{
        Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, WPARAM, POINT},
        Graphics::Gdi::{BeginPaint, EndPaint, InvalidateRect, ScreenToClient, PAINTSTRUCT},
        System::LibraryLoader::GetModuleHandleW,
        UI::WindowsAndMessaging::{
            CreateWindowExW, DefWindowProcW, DispatchMessageW, GetClientRect, GetMessageW,
            LoadCursorW, PostQuitMessage, RegisterClassW, ShowWindow, TranslateMessage,
            CS_HREDRAW, CS_VREDRAW, CW_USEDEFAULT, IDC_ARROW, IDC_SIZEWE, MSG,
            WM_CLOSE, WM_DESTROY, WM_KEYDOWN, WM_LBUTTONDOWN, WM_LBUTTONUP,
            WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_PAINT, WM_SIZE, WM_TIMER, WNDCLASSW, WS_OVERLAPPEDWINDOW,
            WM_RBUTTONDOWN, WM_RBUTTONUP,
            SetTimer, KillTimer, SetCursor,
        },
        UI::Input::KeyboardAndMouse::{VK_ESCAPE, VK_RETURN, VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT, VK_TAB, VK_SPACE, VK_BACK, VK_PRIOR, VK_NEXT, SetCapture, ReleaseCapture},
    },
    core::PCWSTR,
};

const MK_CONTROL: u16 = 0x0008;
const MK_SHIFT: u16 = 0x0004;

use super::app::{GuiApp, GuiMode, PlayerType, GameStatus, ActiveTab};
use super::colors::Colors;
use super::game_renderers::{GameInput, InputResult};
use super::renderer::{Rect, Renderer};

// Timer ID for update loop
const UPDATE_TIMER_ID: usize = 1;
const UPDATE_INTERVAL_MS: u32 = 100;

// Store app state in thread-local for window procedure access
thread_local! {
    static APP_STATE: RefCell<Option<Rc<RefCell<GuiApp>>>> = RefCell::new(None);
    static RENDERER: RefCell<Option<Renderer>> = RefCell::new(None);
}

/// Main entry point for the GUI application
pub fn run_gui(app: GuiApp) -> windows::core::Result<()> {
    unsafe {
        // Store app in thread-local
        let app_rc = Rc::new(RefCell::new(app));
        APP_STATE.with(|state| {
            *state.borrow_mut() = Some(app_rc.clone());
        });

        // Register window class
        let instance = GetModuleHandleW(None)?;
        let class_name = wide_string("ParallelMCTSArenaWindow");

        let wc = WNDCLASSW {
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(window_proc),
            hInstance: std::mem::transmute(instance),
            lpszClassName: PCWSTR(class_name.as_ptr()),
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };

        RegisterClassW(&wc);

        // Create window
        let title = wide_string("Parallel MCTS Arena");
        let hinstance: HINSTANCE = std::mem::transmute(instance);
        let hwnd = CreateWindowExW(
            Default::default(),
            PCWSTR(class_name.as_ptr()),
            PCWSTR(title.as_ptr()),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            1280,
            800,
            None,
            None,
            Some(hinstance),
            None,
        )?;

        // Create renderer
        let mut renderer = Renderer::new(hwnd)?;
        
        // Load Hive SVG icons
        load_hive_svgs(&mut renderer);
        
        RENDERER.with(|r| {
            *r.borrow_mut() = Some(renderer);
        });

        // Show window
        let _ = ShowWindow(hwnd, windows::Win32::UI::WindowsAndMessaging::SW_SHOWMAXIMIZED);

        // Set up update timer
        SetTimer(Some(hwnd), UPDATE_TIMER_ID, UPDATE_INTERVAL_MS, None);

        // Message loop
        let mut msg = MSG::default();
        loop {
            let result = GetMessageW(&mut msg, None, 0, 0);
            if result.0 == 0 || result.0 == -1 {
                break;
            }
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);

            // Check if app wants to quit
            let should_quit = APP_STATE.with(|state| {
                state.borrow()
                    .as_ref()
                    .map(|app| app.borrow().should_quit)
                    .unwrap_or(true)
            });

            if should_quit {
                PostQuitMessage(0);
            }
        }

        let _ = KillTimer(Some(hwnd), UPDATE_TIMER_ID);
        Ok(())
    }
}

/// Convert string to wide string for Win32 APIs
fn wide_string(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Window procedure for handling messages
unsafe extern "system" fn window_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_PAINT => {
            let mut ps = PAINTSTRUCT::default();
            unsafe { let _ = BeginPaint(hwnd, &mut ps); }
            
            // Render
            RENDERER.with(|r| {
                if let Some(renderer) = r.borrow().as_ref() {
                    APP_STATE.with(|state| {
                        if let Some(app) = state.borrow().as_ref() {
                            render(renderer, &app.borrow());
                        }
                    });
                }
            });

            unsafe { let _ = EndPaint(hwnd, &ps); }
            LRESULT(0)
        }

        WM_SIZE => {
            RENDERER.with(|r| {
                if let Some(renderer) = r.borrow_mut().as_mut() {
                    let _ = renderer.resize();
                }
            });
            unsafe { let _ = InvalidateRect(Some(hwnd), None, false); }
            LRESULT(0)
        }

        WM_TIMER => {
            // Update app state
            let needs_redraw = APP_STATE.with(|state| {
                if let Some(app) = state.borrow().as_ref() {
                    let mut app = app.borrow_mut();
                    app.update();
                    let redraw = app.needs_redraw;
                    app.needs_redraw = false;
                    redraw
                } else {
                    false
                }
            });

            if needs_redraw {
                unsafe { let _ = InvalidateRect(Some(hwnd), None, false); }
            }
            LRESULT(0)
        }

        WM_LBUTTONDOWN => {
            let x = (lparam.0 & 0xFFFF) as f32;
            let y = ((lparam.0 >> 16) & 0xFFFF) as f32;
            
            let (needs_redraw, capture) = APP_STATE.with(|state| {
                if let Some(app) = state.borrow().as_ref() {
                    let mut rect = windows::Win32::Foundation::RECT::default();
                    unsafe { let _ = GetClientRect(hwnd, &mut rect); }
                    let width = (rect.right - rect.left) as f32;
                    let height = (rect.bottom - rect.top) as f32;
                    let (redraw, start_drag) = handle_click(&mut app.borrow_mut(), x, y, width, height);
                    (redraw, start_drag)
                } else {
                    (false, false)
                }
            });

            if capture {
                unsafe { let _ = SetCapture(hwnd); }
            }
            if needs_redraw {
                unsafe { let _ = InvalidateRect(Some(hwnd), None, false); }
            }
            LRESULT(0)
        }

        WM_LBUTTONUP => {
            APP_STATE.with(|state| {
                if let Some(app) = state.borrow().as_ref() {
                    let mut app = app.borrow_mut();
                    if app.is_dragging_splitter {
                        app.is_dragging_splitter = false;
                        app.needs_redraw = true;
                    }
                }
            });
            unsafe { let _ = ReleaseCapture(); }
            unsafe { let _ = InvalidateRect(Some(hwnd), None, false); }
            LRESULT(0)
        }

        WM_RBUTTONDOWN => {
            let x = (lparam.0 & 0xFFFF) as f32;
            let y = ((lparam.0 >> 16) & 0xFFFF) as f32;
            
            let capture = APP_STATE.with(|state| {
                if let Some(app) = state.borrow().as_ref() {
                    let mut app = app.borrow_mut();
                    if matches!(app.mode, GuiMode::InGame) {
                        app.is_right_dragging = true;
                        app.last_drag_pos = Some((x, y));
                        return true;
                    }
                }
                false
            });

            if capture {
                unsafe { let _ = SetCapture(hwnd); }
            }
            LRESULT(0)
        }

        WM_RBUTTONUP => {
            APP_STATE.with(|state| {
                if let Some(app) = state.borrow().as_ref() {
                    let mut app = app.borrow_mut();
                    app.is_right_dragging = false;
                    app.last_drag_pos = None;
                }
            });
            
            unsafe { let _ = ReleaseCapture(); }
            LRESULT(0)
        }

        WM_MOUSEMOVE => {
            let x = (lparam.0 & 0xFFFF) as f32;
            let y = ((lparam.0 >> 16) & 0xFFFF) as f32;
            let keys = wparam.0 as u32;
            let shift_pressed = (keys & (MK_SHIFT as u32)) != 0;
            let ctrl_pressed = (keys & (MK_CONTROL as u32)) != 0;
            
            let (needs_redraw, cursor_change) = RENDERER.with(|r| {
                if let Some(renderer) = r.borrow().as_ref() {
                    APP_STATE.with(|state| {
                        if let Some(app) = state.borrow().as_ref() {
                            let mut rect = windows::Win32::Foundation::RECT::default();
                            unsafe { let _ = GetClientRect(hwnd, &mut rect); }
                            let width = (rect.right - rect.left) as f32;
                            let height = (rect.bottom - rect.top) as f32;
                            handle_mouse_move(&mut app.borrow_mut(), renderer, x, y, width, height, shift_pressed, ctrl_pressed)
                        } else {
                            (false, false)
                        }
                    })
                } else {
                    (false, false)
                }
            });

            // Change cursor when over splitter
            if cursor_change {
                unsafe {
                    if let Ok(cursor) = LoadCursorW(None, IDC_SIZEWE) {
                        let _ = SetCursor(Some(cursor));
                    }
                }
            }

            if needs_redraw {
                unsafe { let _ = InvalidateRect(Some(hwnd), None, false); }
            }
            LRESULT(0)
        }

        WM_MOUSEWHEEL => {
            let delta = (wparam.0 >> 16) as i16;
            let keys = (wparam.0 & 0xFFFF) as u16;
            let ctrl_pressed = (keys & MK_CONTROL) != 0;

            let x_screen = (lparam.0 & 0xFFFF) as i32;
            let y_screen = ((lparam.0 >> 16) & 0xFFFF) as i32;
            
            let mut point = POINT { x: x_screen, y: y_screen };
            unsafe { let _ = ScreenToClient(hwnd, &mut point); }
            let x = point.x as f32;
            let y = point.y as f32;

            let needs_redraw = APP_STATE.with(|state| {
                if let Some(app) = state.borrow().as_ref() {
                    let mut app_guard = app.borrow_mut();
                    let app = &mut *app_guard;
                    
                    match app.mode {
                        GuiMode::InGame => {
                            let mut rect = windows::Win32::Foundation::RECT::default();
                            unsafe { let _ = GetClientRect(hwnd, &mut rect); }
                            let width = (rect.right - rect.left) as f32;
                            let height = (rect.bottom - rect.top) as f32;
                            
                            let info_area = get_info_area(&app, width, height);
                            
                            // Check if mouse is over info panel
                            if info_area.contains(x, y) {
                                match app.active_tab {
                                    ActiveTab::DebugStats => {
                                        if delta > 0 { app.scroll_debug_up(); } else { app.scroll_debug_down(); }
                                    },
                                    ActiveTab::MoveHistory => {
                                        if delta > 0 { app.scroll_history_up(); } else { app.scroll_history_down(); }
                                    }
                                }
                                return true;
                            } else {
                                // Dispatch to game
                                let input = GameInput::Wheel { 
                                    delta: delta as f32, 
                                    x, 
                                    y, 
                                    ctrl: ctrl_pressed 
                                };
                                
                                let board_area = get_board_area(&*app, width, height);
                                if let InputResult::Redraw = app.game_renderer.handle_input(input, &app.game, board_area) {
                                    return true;
                                }
                            }
                        },
                        GuiMode::HowToPlay => {
                            if delta > 0 { app.scroll_how_to_play_up(); } else { app.scroll_how_to_play_down(); }
                            return true;
                        },
                        GuiMode::GameSelection => {
                            if delta > 0 { app.game_selection_scroll -= 40; } else { app.game_selection_scroll += 40; }
                            return true;
                        },
                        GuiMode::Settings => {
                            if delta > 0 { app.settings_scroll -= 40; } else { app.settings_scroll += 40; }
                            return true;
                        },
                        _ => {}
                    }
                    false
                } else {
                    false
                }
            });

            if needs_redraw {
                unsafe { let _ = InvalidateRect(Some(hwnd), None, false); }
            }
            LRESULT(0)
        }

        WM_KEYDOWN => {
            let vk = wparam.0 as u16;
            
            let should_quit = APP_STATE.with(|state| {
                if let Some(app) = state.borrow().as_ref() {
                    handle_key(&mut app.borrow_mut(), vk, hwnd)
                } else {
                    false
                }
            });

            if should_quit {
                unsafe { PostQuitMessage(0); }
            } else {
                unsafe { let _ = InvalidateRect(Some(hwnd), None, false); }
            }
            LRESULT(0)
        }

        WM_CLOSE => {
            APP_STATE.with(|state| {
                if let Some(app) = state.borrow().as_ref() {
                    app.borrow_mut().should_quit = true;
                }
            });
            unsafe { PostQuitMessage(0); }
            LRESULT(0)
        }

        WM_DESTROY => {
            unsafe { PostQuitMessage(0); }
            LRESULT(0)
        }

        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

/// Handle keyboard input
fn handle_key(app: &mut GuiApp, vk: u16, hwnd: HWND) -> bool {
    let num_settings = 14; // 12 settings + separator + Back
    let num_games = super::app::GameType::all().len() + 2; // games + Settings + Quit
    
    // Get window dimensions for layout calculations
    let (width, height) = unsafe {
        let mut rect = windows::Win32::Foundation::RECT::default();
        let _ = GetClientRect(hwnd, &mut rect);
        ((rect.right - rect.left) as f32, (rect.bottom - rect.top) as f32)
    };

    if vk == VK_ESCAPE.0 || vk == VK_BACK.0 {
        app.go_back();
        return app.should_quit;
    }

    match app.mode {
        GuiMode::GameSelection => {
            let games_len = super::app::GameType::all().len();
            if vk == VK_UP.0 {
                if app.selected_game_index > 0 {
                    app.selected_game_index -= 1;
                    // Auto-scroll
                    if app.selected_game_index < games_len {
                        let item_height = 90.0; // 80 + 10
                        let target_y = app.selected_game_index as f32 * item_height;
                        if target_y < app.game_selection_scroll as f32 {
                            app.game_selection_scroll = target_y as i32;
                        }
                    }
                }
            } else if vk == VK_DOWN.0 {
                if app.selected_game_index < num_games - 1 {
                    app.selected_game_index += 1;
                    // Auto-scroll
                    if app.selected_game_index < games_len {
                        let item_height = 90.0;
                        let list_height = height - 140.0 - 160.0;
                        let target_y = (app.selected_game_index + 1) as f32 * item_height;
                        if target_y > app.game_selection_scroll as f32 + list_height {
                            app.game_selection_scroll = (target_y - list_height) as i32;
                        }
                    }
                }
            } else if vk == VK_RETURN.0 {
                let games = super::app::GameType::all();
                if app.selected_game_index < games.len() {
                    app.select_game(app.selected_game_index);
                } else if app.selected_game_index == games.len() {
                    // Settings
                    app.mode = GuiMode::Settings;
                    app.selected_settings_index = 0;
                } else if app.selected_game_index == games.len() + 1 {
                    // How To Play
                    app.mode = GuiMode::HowToPlay;
                }
            }
        }
        GuiMode::Settings => {
            let settings_len = app.get_settings_items().len();
            if vk == VK_UP.0 {
                if app.selected_settings_index > 0 {
                    app.selected_settings_index -= 1;
                    // Auto-scroll
                    if app.selected_settings_index < settings_len {
                        let item_height = 45.0; // 40 + 5
                        let target_y = app.selected_settings_index as f32 * item_height;
                        if target_y < app.settings_scroll as f32 {
                            app.settings_scroll = target_y as i32;
                        }
                    }
                }
            } else if vk == VK_DOWN.0 {
                if app.selected_settings_index < num_settings - 1 {
                    app.selected_settings_index += 1;
                    // Auto-scroll
                    if app.selected_settings_index < settings_len {
                        let item_height = 45.0;
                        let list_height = height - 100.0 - 100.0;
                        let target_y = (app.selected_settings_index + 1) as f32 * item_height;
                        if target_y > app.settings_scroll as f32 + list_height {
                            app.settings_scroll = (target_y - list_height) as i32;
                        }
                    }
                }
            } else if vk == VK_LEFT.0 {
                if app.selected_settings_index < 12 {
                    app.adjust_setting(app.selected_settings_index, -1);
                }
            } else if vk == VK_RIGHT.0 || vk == VK_SPACE.0 {
                if app.selected_settings_index < 12 {
                    app.adjust_setting(app.selected_settings_index, 1);
                }
            } else if vk == VK_RETURN.0 {
                if app.selected_settings_index == num_settings - 1 {
                    // Back
                    app.mode = GuiMode::GameSelection;
                } else if app.selected_settings_index >= 9 && app.selected_settings_index <= 10 {
                    // Toggle bool settings
                    app.adjust_setting(app.selected_settings_index, 1);
                }
            }
        }
        GuiMode::PlayerConfig => {
            if vk == VK_UP.0 {
                if app.selected_player_index > 0 {
                    app.selected_player_index -= 1;
                }
            } else if vk == VK_DOWN.0 {
                if app.selected_player_index < app.player_types.len() {
                    app.selected_player_index += 1;
                }
            } else if vk == VK_LEFT.0 || vk == VK_RIGHT.0 || vk == VK_SPACE.0 {
                if app.selected_player_index < app.player_types.len() {
                    app.toggle_player(app.selected_player_index);
                }
            } else if vk == VK_RETURN.0 {
                if app.selected_player_index == app.player_types.len() {
                    app.start_game();
                } else {
                    app.toggle_player(app.selected_player_index);
                }
            }
        }
        GuiMode::InGame => {
            if vk == VK_TAB.0 {
                app.toggle_tab();
            } else if vk == 0x43 { // 'C' key - Copy move history to clipboard
                app.copy_history_to_clipboard();
            } else if vk == VK_PRIOR.0 { // Page Up
                match app.active_tab {
                    ActiveTab::DebugStats => app.scroll_debug_up(),
                    ActiveTab::MoveHistory => app.scroll_history_up(),
                }
            } else if vk == VK_NEXT.0 { // Page Down
                match app.active_tab {
                    ActiveTab::DebugStats => app.scroll_debug_down(),
                    ActiveTab::MoveHistory => app.scroll_history_down(),
                }
            } else {
                // Forward other keys to game renderer for game-specific controls
                // (e.g., R=Rotate, X=Flip, P=Pass, Arrow keys, Enter for Blokus)
                let current_player = app.game.get_current_player();
                let is_human = app.player_types
                    .iter()
                    .find(|(id, _)| *id == current_player)
                    .map(|(_, pt)| *pt == PlayerType::Human)
                    .unwrap_or(false);

                if is_human && !app.ai_thinking {
                    let board_area = get_board_area(app, width, height);
                    let input = GameInput::Key { code: vk as u32, pressed: true };
                    
                    match app.game_renderer.handle_input(input, &app.game, board_area) {
                        InputResult::Move(mv) => {
                            app.make_move(mv);
                        }
                        InputResult::Redraw => {}
                        InputResult::None => {}
                    }
                }
            }
        }
        GuiMode::GameOver => {
            if vk == VK_RETURN.0 || vk == VK_SPACE.0 {
                app.go_back();
            } else if vk == 0x43 { // 'C' key - Copy move history to clipboard
                app.copy_history_to_clipboard();
            }
        }
        GuiMode::HowToPlay => {
            if vk == VK_LEFT.0 {
                if app.selected_how_to_play_game > 0 {
                    app.selected_how_to_play_game -= 1;
                    app.how_to_play_scroll = 0;
                }
            } else if vk == VK_RIGHT.0 {
                if app.selected_how_to_play_game < 4 {
                    app.selected_how_to_play_game += 1;
                    app.how_to_play_scroll = 0;
                }
            } else if vk == VK_UP.0 {
                app.scroll_how_to_play_up();
            } else if vk == VK_DOWN.0 {
                app.scroll_how_to_play_down();
            } else if vk == VK_PRIOR.0 { // Page Up
                for _ in 0..5 { app.scroll_how_to_play_up(); }
            } else if vk == VK_NEXT.0 { // Page Down
                for _ in 0..5 { app.scroll_how_to_play_down(); }
            }
        }
    }
    app.needs_redraw = true;
    false
}

/// Handle mouse click
/// Handle mouse click
/// Returns (needs_redraw, start_capture) tuple
fn handle_click(app: &mut GuiApp, x: f32, y: f32, width: f32, height: f32) -> (bool, bool) {
    app.needs_redraw = true;

    match app.mode {
        GuiMode::GameSelection => {
            // Layout constants (must match render_game_selection)
            let bottom_area_height = 160.0;
            let list_area = Rect::new(0.0, 140.0, width, height - 140.0 - bottom_area_height);
            let item_height = 80.0;
            let item_width = 480.0_f32.min(width * 0.8);
            let spacing = 10.0;
            
            // Calculate scroll exactly as in render_game_selection
            let games = super::app::GameType::all();
            let total_content_height = games.len() as f32 * (item_height + spacing);
            let max_scroll = (total_content_height - list_area.height).max(0.0);
            let scroll = (app.game_selection_scroll as f32).clamp(0.0, max_scroll);

            // Check game list
            if list_area.contains(x, y) {
                for (i, _) in games.iter().enumerate() {
                    let item_y = list_area.y + (i as f32 * (item_height + spacing)) - scroll;
                    let item_x = (width - item_width) / 2.0;
                    let button_rect = Rect::new(item_x, item_y, item_width, item_height);
                    
                    // Only check if visible
                    if item_y + item_height >= list_area.y && item_y <= list_area.y + list_area.height {
                        if button_rect.contains(x, y) {
                            app.select_game(i);
                            return (true, false);
                        }
                    }
                }
            }

            // Check fixed buttons
            let bottom_start_y = height - bottom_area_height + 20.0;
            
            // Settings
            let settings_rect = Rect::new((width - item_width) / 2.0, bottom_start_y, item_width, 50.0);
            if settings_rect.contains(x, y) {
                app.mode = GuiMode::Settings;
                app.selected_settings_index = 0;
                return (true, false);
            }
            
            // How To Play
            let help_rect = Rect::new((width - item_width) / 2.0, bottom_start_y + 60.0, item_width, 50.0);
            if help_rect.contains(x, y) {
                app.mode = GuiMode::HowToPlay;
                app.selected_how_to_play_game = 0;
                app.how_to_play_scroll = 0;
                return (true, false);
            }
        }
        GuiMode::PlayerConfig => {
            // Check player toggle buttons
            for i in 0..app.player_types.len() {
                let button_rect = get_player_button_rect(i, width, height);
                if button_rect.contains(x, y) {
                    app.toggle_player(i);
                    return (true, false);
                }
            }
            
            // Check start button
            let start_rect = get_start_button_rect(width, height);
            if start_rect.contains(x, y) {
                app.start_game();
                return (true, false);
            }
        }
        GuiMode::InGame => {
            // Check if clicking on splitter
            let splitter = get_splitter_rect(app, width, height);
            if splitter.contains(x, y) {
                app.is_dragging_splitter = true;
                return (true, true); // Start capture
            }

            // Check if current player is human
            let current_player = app.game.get_current_player();
            let is_human = app.player_types
                .iter()
                .find(|(id, _)| *id == current_player)
                .map(|(_, pt)| *pt == PlayerType::Human)
                .unwrap_or(false);

            // Check for tab clicks (always allowed, even during AI thinking)
            let tab_area = get_tab_area(app, width, height);
            if tab_area.contains(x, y) {
                app.toggle_tab();
                return (true, false);
            }

            if is_human && !app.ai_thinking {
                // Use the same area for input hit-testing as we use for rendering.
                // Rendering uses `board_area` (left of the splitter); using the full
                // `game_area` here inflates offsets and misaligns clicks/hover.
                let board_area = get_board_area(app, width, height);
                let input = GameInput::Click { x, y };
                
                match app.game_renderer.handle_input(input, &app.game, board_area) {
                    InputResult::Move(mv) => {
                        app.make_move(mv);
                        return (true, false);
                    }
                    InputResult::Redraw => return (true, false),
                    InputResult::None => {}
                }
            }
        }
        GuiMode::Settings => {
            // Layout constants (must match render_settings)
            let bottom_area_height = 100.0;
            let list_area = Rect::new(0.0, 100.0, width, height - 100.0 - bottom_area_height);
            let item_height = 40.0;
            let item_width = 600.0_f32.min(width * 0.9);
            let spacing = 5.0;
            
            // Calculate scroll exactly as in render_settings
            let settings = app.get_settings_items();
            let total_content_height = (settings.len() + 1) as f32 * (item_height + spacing);
            let max_scroll = (total_content_height - list_area.height).max(0.0);
            let scroll = (app.settings_scroll as f32).clamp(0.0, max_scroll);

            // Check settings list
            if list_area.contains(x, y) {
                for i in 0..settings.len() {
                    let item_y = list_area.y + (i as f32 * (item_height + spacing)) - scroll;
                    let item_x = (width - item_width) / 2.0;
                    let item_rect = Rect::new(item_x, item_y, item_width, item_height);
                    
                    if item_y + item_height >= list_area.y && item_y <= list_area.y + list_area.height {
                        if item_rect.contains(x, y) {
                            app.selected_settings_index = i;
                            if i >= 8 && i <= 9 {
                                // Toggle bool settings
                                app.adjust_setting(i, 1);
                            }
                            return (true, false);
                        }
                    }
                }
            }
            
            // Check Back button
            let back_y = height - bottom_area_height + 20.0;
            let back_rect = Rect::new((width - item_width) / 2.0, back_y, item_width, 50.0);
            if back_rect.contains(x, y) {
                app.go_back();
                return (true, false);
            }
        }
        GuiMode::HowToPlay => {
            // Check game tab clicks
            let games = ["Gomoku", "Connect4", "Othello", "Blokus", "Hive"];
            for (i, _) in games.iter().enumerate() {
                let tab_rect = get_how_to_play_tab_rect(i, width, height);
                if tab_rect.contains(x, y) {
                    app.selected_how_to_play_game = i;
                    app.how_to_play_scroll = 0;
                    return (true, false);
                }
            }
        }
        GuiMode::GameOver => {
            // Click anywhere to go back
            app.go_back();
            return (true, false);
        }
    }

    (false, false)
}

/// Handle mouse movement (for hover effects and splitter dragging)
/// Returns (needs_redraw, show_resize_cursor) tuple
fn handle_mouse_move(app: &mut GuiApp, _renderer: &Renderer, x: f32, y: f32, width: f32, height: f32, shift_pressed: bool, ctrl_pressed: bool) -> (bool, bool) {
    let mut needs_redraw = false;
    let mut show_resize_cursor = false;

    if app.mode == GuiMode::InGame {
        // Handle right-drag for tilt adjustment
        if app.is_right_dragging {
            if let Some((last_x, last_y)) = app.last_drag_pos {
                let dx = x - last_x;
                let dy = y - last_y;
                
                // Update last position
                app.last_drag_pos = Some((x, y));
                
                // Send drag event to game renderer
                let board_area = get_board_area(app, width, height);
                let input = GameInput::Drag { dx, dy, shift: shift_pressed, ctrl: ctrl_pressed };
                if let InputResult::Redraw = app.game_renderer.handle_input(input, &app.game, board_area) {
                    needs_redraw = true;
                }
            }
            return (needs_redraw, show_resize_cursor);
        }
        
        // Handle splitter dragging
        if app.is_dragging_splitter {
            let game_area = get_game_area(width, height);
            // Calculate new panel ratio based on mouse position
            // x position relative to the right edge of game area gives us the panel width
            let panel_width = (game_area.x + game_area.width) - x;
            let new_ratio = (panel_width / game_area.width).clamp(
                app.min_panel_width / game_area.width,
                app.max_panel_ratio,
            );
            if (new_ratio - app.info_panel_ratio).abs() > 0.001 {
                app.info_panel_ratio = new_ratio;
                needs_redraw = true;
            }
            show_resize_cursor = true;
        } else {
            // Check if hovering over splitter
            let splitter = get_splitter_rect(app, width, height);
            if splitter.contains(x, y) {
                show_resize_cursor = true;
            }
        }

        // Handle game hover
        let board_area = get_board_area(app, width, height);
        let input = GameInput::Hover { x, y };
        
        if let InputResult::Redraw = app.game_renderer.handle_input(input, &app.game, board_area) {
            needs_redraw = true;
        }
    }
    
    (needs_redraw, show_resize_cursor)
}

/// Main render function
fn render(renderer: &Renderer, app: &GuiApp) {
    renderer.begin_draw();
    renderer.clear(Colors::BACKGROUND);

    match app.mode {
        GuiMode::GameSelection => render_game_selection(renderer, app),
        GuiMode::Settings => render_settings(renderer, app),
        GuiMode::PlayerConfig => render_player_config(renderer, app),
        GuiMode::InGame => render_in_game(renderer, app),
        GuiMode::GameOver => render_game_over(renderer, app),
        GuiMode::HowToPlay => render_how_to_play(renderer, app),
    }

    let _ = renderer.end_draw();
}

/// Render game selection screen
fn render_game_selection(renderer: &Renderer, app: &GuiApp) {
    let client = renderer.get_client_rect();
    
    // Title
    let title_rect = Rect::new(0.0, 40.0, client.width, 60.0);
    renderer.draw_title("Parallel MCTS Arena", title_rect, Colors::TEXT_PRIMARY, true);
    
    // Subtitle
    let subtitle_text = if app.ai_only { "AI-Only Mode - Select a game" } else { "Select a game to play" };
    let subtitle_rect = Rect::new(0.0, 100.0, client.width, 30.0);
    renderer.draw_text(subtitle_text, subtitle_rect, Colors::TEXT_SECONDARY, true);

    // Fixed bottom area for Settings and How To Play
    let bottom_area_height = 160.0;
    let list_area = Rect::new(
        0.0, 
        140.0, 
        client.width, 
        client.height - 140.0 - bottom_area_height
    );

    // Game List
    let games = super::app::GameType::all();
    let item_height = 80.0;
    let item_width = 480.0_f32.min(client.width * 0.8);
    let spacing = 10.0;
    
    // Calculate visible range
    let total_content_height = games.len() as f32 * (item_height + spacing);
    let max_scroll = (total_content_height - list_area.height).max(0.0);
    
    // Clamp scroll
    let scroll = (app.game_selection_scroll as f32).clamp(0.0, max_scroll);

    renderer.set_clip(list_area);
    
    for (i, game) in games.iter().enumerate() {
        let y = list_area.y + (i as f32 * (item_height + spacing)) - scroll;
        
        // Optimization: Skip if out of view
        if y + item_height < list_area.y || y > list_area.y + list_area.height {
            continue;
        }

        let x = (client.width - item_width) / 2.0;
        let button_rect = Rect::new(x, y, item_width, item_height);
        
        let is_selected = i == app.selected_game_index;
        let bg_color = if is_selected { Colors::BUTTON_SELECTED } else { Colors::BUTTON_BG };
        
        renderer.fill_rounded_rect(button_rect, 8.0, bg_color);
        
        // Game name
        let name_rect = Rect::new(button_rect.x, button_rect.y + 15.0, button_rect.width, 30.0);
        renderer.draw_text(game.name(), name_rect, Colors::TEXT_PRIMARY, true);
        
        // Description
        let desc_rect = Rect::new(button_rect.x, button_rect.y + 45.0, button_rect.width, 25.0);
        renderer.draw_small_text(game.description(), desc_rect, Colors::TEXT_SECONDARY, true);
    }
    
    renderer.remove_clip();

    // Scrollbar
    if max_scroll > 0.0 {
        let scroll_pct = scroll / max_scroll;
        let bar_height = (list_area.height * (list_area.height / total_content_height)).max(30.0);
        let bar_y = list_area.y + (list_area.height - bar_height) * scroll_pct;
        let bar_rect = Rect::new(client.width - 10.0, bar_y, 6.0, bar_height);
        renderer.fill_rounded_rect(bar_rect, 3.0, Colors::GRID_LINE);
    }

    // Bottom buttons
    let bottom_start_y = client.height - bottom_area_height + 20.0;
    
    // Settings button
    let settings_rect = Rect::new((client.width - item_width) / 2.0, bottom_start_y, item_width, 50.0);
    let is_settings_selected = app.selected_game_index == games.len();
    let settings_bg = if is_settings_selected { Colors::BUTTON_SELECTED } else { Colors::PANEL_BG };
    renderer.fill_rounded_rect(settings_rect, 8.0, settings_bg);
    if !is_settings_selected {
        renderer.draw_rounded_rect(settings_rect, 8.0, Colors::GRID_LINE, 2.0);
    }
    renderer.draw_text("Settings", Rect::new(settings_rect.x, settings_rect.y + 10.0, settings_rect.width, 30.0), Colors::TEXT_PRIMARY, true);

    // How To Play button
    let help_rect = Rect::new((client.width - item_width) / 2.0, bottom_start_y + 60.0, item_width, 50.0);
    let is_help_selected = app.selected_game_index == games.len() + 1;
    let help_bg = if is_help_selected { Colors::BUTTON_SELECTED } else { Colors::BUTTON_BG };
    renderer.fill_rounded_rect(help_rect, 8.0, help_bg);
    renderer.draw_text("How To Play", Rect::new(help_rect.x, help_rect.y + 10.0, help_rect.width, 30.0), Colors::TEXT_PRIMARY, true);

    // Instructions
    let instr_rect = Rect::new(0.0, client.height - 30.0, client.width, 30.0);
    renderer.draw_small_text("↑↓ Navigate • Enter Select • Escape Quit", instr_rect, Colors::TEXT_SECONDARY, true);
}

/// Render settings screen
fn render_settings(renderer: &Renderer, app: &GuiApp) {
    let client = renderer.get_client_rect();
    
    // Title
    let title_rect = Rect::new(0.0, 30.0, client.width, 50.0);
    renderer.draw_title("Settings", title_rect, Colors::TEXT_PRIMARY, true);
    
    // Fixed bottom area for Back button
    let bottom_area_height = 100.0;
    let list_area = Rect::new(
        0.0, 
        100.0, 
        client.width, 
        client.height - 100.0 - bottom_area_height
    );

    // Settings List
    let settings = app.get_settings_items();
    let item_height = 40.0;
    let item_width = 600.0_f32.min(client.width * 0.9);
    let spacing = 5.0;
    
    // Calculate visible range
    // +1 for separator
    let total_content_height = (settings.len() + 1) as f32 * (item_height + spacing);
    let max_scroll = (total_content_height - list_area.height).max(0.0);
    
    // Clamp scroll
    let scroll = (app.settings_scroll as f32).clamp(0.0, max_scroll);

    renderer.set_clip(list_area);
    
    for (i, (name, value)) in settings.iter().enumerate() {
        let y = list_area.y + (i as f32 * (item_height + spacing)) - scroll;
        
        // Optimization: Skip if out of view
        if y + item_height < list_area.y || y > list_area.y + list_area.height {
            continue;
        }

        let x = (client.width - item_width) / 2.0;
        let item_rect = Rect::new(x, y, item_width, item_height);
        
        let is_selected = i == app.selected_settings_index;
        let bg_color = if is_selected { Colors::BUTTON_SELECTED } else { Colors::BUTTON_BG };
        
        renderer.fill_rounded_rect(item_rect, 5.0, bg_color);
        
        // Setting name
        let name_rect = Rect::new(item_rect.x + 20.0, item_rect.y, item_rect.width * 0.5, item_rect.height);
        renderer.draw_text(name, name_rect, Colors::TEXT_PRIMARY, false);
        
        // Setting value
        let value_rect = Rect::new(item_rect.x + item_rect.width * 0.5, item_rect.y, item_rect.width * 0.5 - 20.0, item_rect.height);
        renderer.draw_text(value, value_rect, Colors::TEXT_ACCENT, false);
    }
    
    // Separator
    let sep_index = settings.len();
    let sep_y = list_area.y + (sep_index as f32 * (item_height + spacing)) - scroll;
    if sep_y + item_height >= list_area.y && sep_y <= list_area.y + list_area.height {
        let x = (client.width - item_width) / 2.0;
        let sep_rect = Rect::new(x, sep_y, item_width, item_height);
        renderer.draw_line(sep_rect.x + 50.0, sep_rect.y + sep_rect.height / 2.0, 
                           sep_rect.x + sep_rect.width - 50.0, sep_rect.y + sep_rect.height / 2.0,
                           Colors::GRID_LINE, 1.0);
    }
    
    renderer.remove_clip();

    // Scrollbar
    if max_scroll > 0.0 {
        let scroll_pct = scroll / max_scroll;
        let bar_height = (list_area.height * (list_area.height / total_content_height)).max(30.0);
        let bar_y = list_area.y + (list_area.height - bar_height) * scroll_pct;
        let bar_rect = Rect::new(client.width - 10.0, bar_y, 6.0, bar_height);
        renderer.fill_rounded_rect(bar_rect, 3.0, Colors::GRID_LINE);
    }
    
    // Back button
    let back_index = settings.len() + 1;
    let back_y = client.height - bottom_area_height + 20.0;
    let back_rect = Rect::new((client.width - item_width) / 2.0, back_y, item_width, 50.0);
    
    let is_back_selected = app.selected_settings_index == back_index;
    let back_bg = if is_back_selected { Colors::BUTTON_SELECTED } else { Colors::BUTTON_BG };
    renderer.fill_rounded_rect(back_rect, 5.0, back_bg);
    renderer.draw_text("Back", back_rect, Colors::TEXT_PRIMARY, true);
    
    // Instructions
    let help_rect = Rect::new(0.0, client.height - 30.0, client.width, 30.0);
    renderer.draw_small_text("↑↓ Navigate • ←→ Adjust • Enter Confirm • Escape Back", help_rect, Colors::TEXT_SECONDARY, true);
}

/// Render how to play screen
fn render_how_to_play(renderer: &Renderer, app: &GuiApp) {
    let client = renderer.get_client_rect();
    
    // Title
    let title_rect = Rect::new(0.0, 30.0, client.width, 50.0);
    renderer.draw_title("How To Play", title_rect, Colors::TEXT_PRIMARY, true);
    
    // Game tabs
    let games = ["Gomoku", "Connect4", "Othello", "Blokus", "Hive"];
    for (i, name) in games.iter().enumerate() {
        let tab_rect = get_how_to_play_tab_rect(i, client.width, client.height);
        let is_selected = i == app.selected_how_to_play_game;
        let bg_color = if is_selected { Colors::BUTTON_SELECTED } else { Colors::BUTTON_BG };
        renderer.fill_rounded_rect(tab_rect, 5.0, bg_color);
        renderer.draw_text(*name, tab_rect, Colors::TEXT_PRIMARY, true);
    }
    
    // Instructions content
    let content_rect = Rect::new(100.0, 150.0, client.width - 200.0, client.height - 250.0);
    renderer.fill_rounded_rect(content_rect, 10.0, Colors::PANEL_BG);
    
    let instructions = get_game_instructions(app.selected_how_to_play_game);
    let lines: Vec<&str> = instructions.lines().collect();
    
    let line_height = 20.0;
    let text_area = content_rect.with_padding(20.0);
    let max_visible = (text_area.height / line_height) as usize;
    
    let start_index = (app.how_to_play_scroll as usize).min(lines.len().saturating_sub(1));
    
    for (i, line) in lines.iter().skip(start_index).enumerate() {
        if i >= max_visible {
            break;
        }
        let y = text_area.y + i as f32 * line_height;
        let line_rect = Rect::new(text_area.x, y, text_area.width, line_height);
        renderer.draw_small_text(line, line_rect, Colors::TEXT_PRIMARY, false);
    }
    
    // Scrollbar
    if lines.len() > max_visible {
        let max_scroll = lines.len().saturating_sub(max_visible);
        if max_scroll > 0 {
            let scroll_pct = (start_index as f32 / max_scroll as f32).min(1.0);
            let bar_height = content_rect.height * (max_visible as f32 / lines.len() as f32);
            let bar_y = content_rect.y + (content_rect.height - bar_height) * scroll_pct;
            let bar_rect = Rect::new(content_rect.x + content_rect.width - 6.0, bar_y, 6.0, bar_height);
            renderer.fill_rounded_rect(bar_rect, 3.0, Colors::GRID_LINE);
        }
    }
    
    // Navigation help
    let help_rect = Rect::new(0.0, client.height - 50.0, client.width, 30.0);
    renderer.draw_small_text("←→ Switch Game • ↑↓ Scroll • Escape Back", help_rect, Colors::TEXT_SECONDARY, true);
}

/// Get game instructions text
fn get_game_instructions(game_index: usize) -> String {
    match game_index {
        0 => "GOMOKU (Five in a Row)\n\n\
              Objective: Get exactly 5 of your pieces in a row\n\
              (horizontally, vertically, or diagonally).\n\n\
              Rules:\n\
              • Players take turns placing pieces on the board\n\
              • First player to get 5 in a row wins\n\
              • If the board fills up, it's a draw\n\n\
              Strategy Tips:\n\
              • Build multiple threats at once\n\
              • Block opponent's lines of 3 or more\n\
              • Control the center of the board".to_string(),
        1 => "CONNECT 4\n\n\
              Objective: Connect 4 of your pieces in a row\n\
              (horizontally, vertically, or diagonally).\n\n\
              Rules:\n\
              • Click on a column to drop your piece\n\
              • Pieces fall to the lowest available space\n\
              • First player to connect 4 wins\n\n\
              Strategy Tips:\n\
              • Control the center column\n\
              • Set up multiple winning possibilities\n\
              • Force opponent into defensive moves".to_string(),
        2 => "OTHELLO (Reversi)\n\n\
              Objective: Have the most pieces of your color\n\
              when the game ends.\n\n\
              Rules:\n\
              • Place a piece to surround opponent pieces\n\
              • Surrounded pieces flip to your color\n\
              • Must make a legal capturing move if possible\n\
              • Game ends when neither player can move\n\n\
              Strategy Tips:\n\
              • Corners are very valuable (can't be flipped)\n\
              • Edges are generally strong positions\n\
              • Don't just maximize pieces early on".to_string(),
        3 => "BLOKUS\n\n\
              Objective: Place as many of your pieces as possible.\n\
              Player with fewest remaining squares wins.\n\n\
              Rules:\n\
              • Each player has 21 unique polyomino pieces\n\
              • First piece must cover a corner square\n\
              • Subsequent pieces must touch your own pieces\n\
                by corner only (not edge-to-edge)\n\
              • Pieces cannot overlap\n\n\
              Strategy Tips:\n\
              • Use larger pieces early\n\
              • Block opponent's expansion paths\n\
              • Keep options open in multiple directions".to_string(),
        4 => "HIVE\n\n\
              Objective: Completely surround the opponent's Queen Bee.\n\n\
              Rules:\n\
              • Place tiles adjacent to your own pieces (except first move)\n\
              • Queen Bee must be placed by turn 4\n\
              • Pieces move only after Queen is placed\n\
              • Hive must remain connected at all times\n\n\
              Piece Movements:\n\
              • Queen: 1 space\n\
              • Beetle: 1 space, can climb on top\n\
              • Spider: Exactly 3 spaces\n\
              • Grasshopper: Jumps over pieces\n\
              • Ant: Any number of spaces around edge".to_string(),
        _ => "Select a game to see instructions.".to_string(),
    }
}

/// Render player configuration screen
fn render_player_config(renderer: &Renderer, app: &GuiApp) {
    let client = renderer.get_client_rect();
    
    // Title
    let title_rect = Rect::new(0.0, 40.0, client.width, 60.0);
    let title = if app.ai_only {
        format!("{} - AI Only Mode", app.selected_game_type.name())
    } else {
        format!("{} - Player Setup", app.selected_game_type.name())
    };
    renderer.draw_title(&title, title_rect, Colors::TEXT_PRIMARY, true);

    // Player type buttons
    for (i, (player_id, player_type)) in app.player_types.iter().enumerate() {
        let button_rect = get_player_button_rect(i, client.width, client.height);
        let is_selected = i == app.selected_player_index;
        
        let bg_color = match player_type {
            PlayerType::Human => if is_selected { Colors::BUTTON_SELECTED } else { Colors::STATUS_WIN },
            PlayerType::AiCpu | PlayerType::AiGpu => if is_selected { Colors::BUTTON_SELECTED } else { Colors::BUTTON_BG },
            PlayerType::AiGpuNative => if is_selected { Colors::BUTTON_SELECTED } else { Colors::AI_THINKING },
        };
        renderer.fill_rounded_rect(button_rect, 8.0, bg_color);
        
        let label = match app.selected_game_type {
            super::app::GameType::Blokus => {
                let color_name = match player_id {
                    1 => "Blue",
                    2 => "Yellow", 
                    3 => "Red",
                    4 => "Green",
                    _ => "Player",
                };
                format!("{}: {}", color_name, match player_type {
                    PlayerType::Human => "Human",
                    PlayerType::AiCpu => "AI (CPU)",
                    PlayerType::AiGpu => "AI (GPU)",
                    PlayerType::AiGpuNative => "AI (GPU-Native)",
                })
            }
            _ => {
                let player_name = if *player_id == 1 { "Player 1" } else { "Player 2" };
                format!("{}: {}", player_name, match player_type {
                    PlayerType::Human => "Human",
                    PlayerType::AiCpu => "AI (CPU)",
                    PlayerType::AiGpu => "AI (GPU)",
                    PlayerType::AiGpuNative => "AI (GPU-Native)",
                })
            }
        };
        
        renderer.draw_text(&label, button_rect, Colors::TEXT_PRIMARY, true);
    }

    // Start button
    let start_rect = get_start_button_rect(client.width, client.height);
    renderer.fill_rounded_rect(start_rect, 8.0, Colors::BUTTON_SELECTED);
    renderer.draw_text("Start Game", start_rect, Colors::TEXT_PRIMARY, true);

    // Instructions
    let help_rect = Rect::new(0.0, client.height - 50.0, client.width, 30.0);
    renderer.draw_small_text("Click to toggle Human/AI • Enter to start • Escape to go back", help_rect, Colors::TEXT_SECONDARY, true);
}

/// Render in-game screen
fn render_in_game(renderer: &Renderer, app: &GuiApp) {
    let client = renderer.get_client_rect();
    
    // Header with game info
    let header_rect = Rect::new(0.0, 0.0, client.width, 50.0);
    renderer.fill_rect(header_rect, Colors::PANEL_BG);
    
    let title = app.game_renderer.game_name();
    let title_rect = Rect::new(10.0, 0.0, 200.0, 50.0);
    renderer.draw_text(title, title_rect, Colors::TEXT_PRIMARY, false);

    // Current player indicator
    let current_player = app.game.get_current_player();
    let player_name = app.game_renderer.player_name(current_player);
    
    let player_type = app.player_types
        .iter()
        .find(|(id, _)| *id == current_player)
        .map(|(_, pt)| *pt)
        .unwrap_or(PlayerType::Human);

    let type_suffix = match player_type {
        PlayerType::Human => "",
        PlayerType::AiCpu => " (CPU)",
        PlayerType::AiGpu => " (GPU)",
        PlayerType::AiGpuNative => " (GPU-Native)",
    };

    let status_text = if app.ai_thinking {
        let elapsed = app.ai_thinking_start.map(|t| t.elapsed().as_secs()).unwrap_or(0);
        format!("{}{}{} (thinking... {}s)", player_name, type_suffix, if type_suffix.is_empty() { "AI " } else { "" }, elapsed)
    } else {
        format!("{}{}'s turn", player_name, type_suffix)
    };
    let status_rect = Rect::new(client.width - 350.0, 0.0, 340.0, 50.0);
    let status_color = if app.ai_thinking { Colors::AI_THINKING } else { Colors::TEXT_PRIMARY };
    renderer.draw_text(&status_text, status_rect, status_color, false);

    if app.ai_thinking {
        // Draw loading bar below the header
        let bar_rect = Rect::new(status_rect.x, 52.0, 200.0, 4.0);
        renderer.fill_rounded_rect(bar_rect, 2.0, Colors::BUTTON_BG);
        
        // Animate bar
        let time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as f32;
        let fill_width = bar_rect.width * 0.3;
        let x_pos = bar_rect.x + (bar_rect.width - fill_width) * ((time / 500.0).sin().abs());
        
        let fill_rect = Rect::new(x_pos, bar_rect.y, fill_width, bar_rect.height);
        renderer.fill_rounded_rect(fill_rect, 2.0, Colors::AI_THINKING);
    }

    // Main area: game board on left, splitter in middle, info panel on right
    let board_area = get_board_area(app, client.width, client.height);
    let info_area = get_info_area(app, client.width, client.height);
    let splitter = get_splitter_rect(app, client.width, client.height);
    
    // Render game board
    app.game_renderer.render(renderer, &app.game, board_area);

    // Render splitter (visible resize handle)
    let splitter_color = if app.is_dragging_splitter { 
        Colors::BUTTON_SELECTED 
    } else { 
        Colors::GRID_LINE 
    };
    renderer.fill_rounded_rect(
        Rect::new(splitter.x + 2.0, splitter.y + splitter.height / 3.0, 4.0, splitter.height / 3.0),
        2.0,
        splitter_color
    );

    // Render info panel with tabs
    render_info_panel(renderer, app, info_area);

    // Move count and controls hint at bottom (update hint to include drag)
    let moves_text = format!("Moves: {} | Tab: Switch Panel | C: Copy History | Drag splitter to resize", app.move_history.len());
    let moves_rect = Rect::new(10.0, client.height - 30.0, 700.0, 30.0);
    renderer.draw_small_text(&moves_text, moves_rect, Colors::TEXT_SECONDARY, false);
}

/// Render the info panel with tabs (Debug Stats / Move History)
fn render_info_panel(renderer: &Renderer, app: &GuiApp, area: Rect) {
    // Panel background
    renderer.fill_rounded_rect(area, 8.0, Colors::PANEL_BG);
    
    // Tab bar
    let tab_height = 35.0;
    let tab_width = area.width / 2.0;
    
    // Debug Stats tab
    let debug_tab = Rect::new(area.x, area.y, tab_width, tab_height);
    let debug_bg = if app.active_tab == ActiveTab::DebugStats { Colors::BUTTON_SELECTED } else { Colors::BUTTON_BG };
    renderer.fill_rounded_rect(debug_tab, 5.0, debug_bg);
    renderer.draw_small_text("Debug Stats", debug_tab, Colors::TEXT_PRIMARY, true);
    
    // Move History tab
    let history_tab = Rect::new(area.x + tab_width, area.y, tab_width, tab_height);
    let history_bg = if app.active_tab == ActiveTab::MoveHistory { Colors::BUTTON_SELECTED } else { Colors::BUTTON_BG };
    renderer.fill_rounded_rect(history_tab, 5.0, history_bg);
    renderer.draw_small_text("Move History", history_tab, Colors::TEXT_PRIMARY, true);
    
    // Content area
    let content_area = Rect::new(area.x + 10.0, area.y + tab_height + 10.0, area.width - 20.0, area.height - tab_height - 20.0);
    
    match app.active_tab {
        ActiveTab::DebugStats => render_debug_stats_panel(renderer, app, content_area),
        ActiveTab::MoveHistory => render_move_history_panel(renderer, app, content_area),
    }
}

/// Render debug statistics panel
fn render_debug_stats_panel(renderer: &Renderer, app: &GuiApp, area: Rect) {
    let lines = app.get_debug_stats_lines();
    let line_height = 18.0;
    let max_visible = (area.height / line_height) as usize;
    
    let start_index = (app.debug_scroll as usize).min(lines.len().saturating_sub(1));
    
    for (i, line) in lines.iter().skip(start_index).enumerate() {
        if i >= max_visible {
            break;
        }
        let y = area.y + i as f32 * line_height;
        let line_rect = Rect::new(area.x, y, area.width, line_height);
        let color = if line.starts_with("AI Status") || line.contains("Top AI Moves") {
            Colors::TEXT_ACCENT
        } else {
            Colors::TEXT_SECONDARY
        };
        renderer.draw_small_text(line, line_rect, color, false);
    }

    // Scrollbar indicator
    if lines.len() > max_visible {
        let scroll_pct = start_index as f32 / (lines.len() - max_visible) as f32;
        let bar_height = area.height * (max_visible as f32 / lines.len() as f32);
        let bar_y = area.y + (area.height - bar_height) * scroll_pct;
        let bar_rect = Rect::new(area.x + area.width - 4.0, bar_y, 4.0, bar_height);
        renderer.fill_rounded_rect(bar_rect, 2.0, Colors::GRID_LINE);
    }
}

/// Render move history panel
fn render_move_history_panel(renderer: &Renderer, app: &GuiApp, area: Rect) {
    let history = app.get_formatted_history();
    let line_height = 18.0;
    
    if history.is_empty() {
        let empty_rect = Rect::new(area.x, area.y + 20.0, area.width, 30.0);
        renderer.draw_small_text("No moves yet", empty_rect, Colors::TEXT_SECONDARY, true);
        return;
    }
    
    let max_visible = ((area.height / line_height) as usize).max(1);
    let max_scroll = history.len().saturating_sub(max_visible);
    
    // When history_scroll is i32::MAX (auto-scroll mode), show the end of the list
    // Otherwise use the scroll position as starting index
    let start_index = if app.history_scroll >= max_scroll as i32 {
        max_scroll
    } else {
        (app.history_scroll as usize).min(max_scroll)
    };
    
    for (i, line) in history.iter().skip(start_index).enumerate() {
        if i >= max_visible {
            break;
        }
        let y = area.y + i as f32 * line_height;
        let line_rect = Rect::new(area.x, y, area.width, line_height);
        renderer.draw_small_text(line, line_rect, Colors::TEXT_SECONDARY, false);
    }

    // Scrollbar indicator
    if history.len() > max_visible && max_scroll > 0 {
        let scroll_pct = (start_index as f32 / max_scroll as f32).min(1.0);
        let bar_height = area.height * (max_visible as f32 / history.len() as f32);
        let bar_y = area.y + (area.height - bar_height) * scroll_pct;
        let bar_rect = Rect::new(area.x + area.width - 4.0, bar_y, 4.0, bar_height);
        renderer.fill_rounded_rect(bar_rect, 2.0, Colors::GRID_LINE);
    }
}

/// Render game over screen
fn render_game_over(renderer: &Renderer, app: &GuiApp) {
    // First render the game state
    render_in_game(renderer, app);
    
    let client = renderer.get_client_rect();
    
    // Overlay
    let overlay = windows::Win32::Graphics::Direct2D::Common::D2D1_COLOR_F {
        r: 0.0, g: 0.0, b: 0.0, a: 0.7,
    };
    renderer.fill_rect(client, overlay);
    
    // Result panel
    let panel_rect = Rect::new(
        client.width / 2.0 - 200.0,
        client.height / 2.0 - 100.0,
        400.0,
        200.0,
    );
    renderer.fill_rounded_rect(panel_rect, 15.0, Colors::PANEL_BG);
    
    // Result text
    let result_text = match app.game_status {
        GameStatus::Win(player) => {
            let winner_name = app.game_renderer.player_name(player);
            
            let player_type = app.player_types
                .iter()
                .find(|(id, _)| *id == player)
                .map(|(_, pt)| *pt)
                .unwrap_or(PlayerType::Human);
                
            let type_suffix = match player_type {
                PlayerType::Human => "",
                PlayerType::AiCpu => " (CPU)",
                PlayerType::AiGpu => " (GPU)",
                PlayerType::AiGpuNative => " (GPU-Native)",
            };
            
            format!("{}{} wins!", winner_name, type_suffix)
        }
        GameStatus::Draw => "It's a draw!".to_string(),
        GameStatus::InProgress => "Game in progress".to_string(),
    };
    
    let result_color = match app.game_status {
        GameStatus::Win(_) => Colors::STATUS_WIN,
        GameStatus::Draw => Colors::STATUS_DRAW,
        _ => Colors::TEXT_PRIMARY,
    };
    
    let result_rect = Rect::new(panel_rect.x, panel_rect.y + 50.0, panel_rect.width, 50.0);
    renderer.draw_title(&result_text, result_rect, result_color, true);
    
    // Instructions
    let help_rect = Rect::new(panel_rect.x, panel_rect.y + 130.0, panel_rect.width, 30.0);
    renderer.draw_text("Click or press Enter to continue", help_rect, Colors::TEXT_SECONDARY, true);
}

// Layout helper functions

fn get_player_button_rect(index: usize, width: f32, height: f32) -> Rect {
    let button_width = 400.0_f32.min(width * 0.8);
    let button_height = 50.0_f32.min(height * 0.08);
    let start_y = height * 0.25;
    let spacing = button_height * 1.2;
    let x = (width - button_width) / 2.0;
    let y = start_y + index as f32 * spacing;
    Rect::new(x, y, button_width, button_height)
}

fn get_start_button_rect(width: f32, height: f32) -> Rect {
    let button_width = 300.0_f32.min(width * 0.6);
    let button_height = 50.0_f32.min(height * 0.08);
    let x = (width - button_width) / 2.0;
    let y = height * 0.7;
    Rect::new(x, y, button_width, button_height)
}

fn get_game_area(width: f32, height: f32) -> Rect {
    // Reserve space for header and margins
    let margin = 20.0;
    let header_height = 60.0;
    Rect::new(margin, header_height, width - 2.0 * margin, height - header_height - margin)
}

fn get_how_to_play_tab_rect(index: usize, width: f32, _height: f32) -> Rect {
    let tab_width = (width - 40.0) / 5.0;
    let tab_height = 40.0;
    let x = 20.0 + index as f32 * tab_width;
    let y = 80.0;
    Rect::new(x, y, tab_width, tab_height)
}

/// Get the info panel width based on the app's panel ratio
fn get_info_panel_width(app: &GuiApp, game_area: &Rect) -> f32 {
    (game_area.width * app.info_panel_ratio).max(app.min_panel_width).min(game_area.width * app.max_panel_ratio)
}

/// Get the splitter rect for hit testing and rendering
fn get_splitter_rect(app: &GuiApp, width: f32, height: f32) -> Rect {
    let game_area = get_game_area(width, height);
    let info_panel_width = get_info_panel_width(app, &game_area);
    let splitter_width = 8.0;
    let splitter_x = game_area.x + game_area.width - info_panel_width - splitter_width;
    Rect::new(splitter_x, game_area.y, splitter_width, game_area.height)
}

/// Get the board area (left of the splitter)
fn get_board_area(app: &GuiApp, width: f32, height: f32) -> Rect {
    let game_area = get_game_area(width, height);
    let info_panel_width = get_info_panel_width(app, &game_area);
    let splitter_width = 8.0;
    Rect::new(game_area.x, game_area.y, game_area.width - info_panel_width - splitter_width - 10.0, game_area.height)
}

/// Get the info panel area (right of the splitter)
fn get_info_area(app: &GuiApp, width: f32, height: f32) -> Rect {
    let game_area = get_game_area(width, height);
    let info_panel_width = get_info_panel_width(app, &game_area);
    Rect::new(game_area.x + game_area.width - info_panel_width, game_area.y, info_panel_width, game_area.height)
}

fn get_tab_area(app: &GuiApp, width: f32, height: f32) -> Rect {
    let info_area = get_info_area(app, width, height);
    Rect::new(info_area.x, info_area.y, info_area.width, 35.0)
}

/// Load all Hive SVG icons into the renderer cache
fn load_hive_svgs(renderer: &mut Renderer) {
    // SVG content for each Hive piece
    const QUEEN_SVG: &str = include_str!("../../assets/hive/queen.svg");
    const BEETLE_SVG: &str = include_str!("../../assets/hive/beetle.svg");
    const SPIDER_SVG: &str = include_str!("../../assets/hive/spider.svg");
    const GRASSHOPPER_SVG: &str = include_str!("../../assets/hive/grasshopper.svg");
    const ANT_SVG: &str = include_str!("../../assets/hive/ant.svg");
    
    // Load each SVG with a 100x100 viewport (matching the SVG viewBox)
    let _ = renderer.load_svg("hive_queen", QUEEN_SVG, 100.0, 100.0);
    let _ = renderer.load_svg("hive_beetle", BEETLE_SVG, 100.0, 100.0);
    let _ = renderer.load_svg("hive_spider", SPIDER_SVG, 100.0, 100.0);
    let _ = renderer.load_svg("hive_grasshopper", GRASSHOPPER_SVG, 100.0, 100.0);
    let _ = renderer.load_svg("hive_ant", ANT_SVG, 100.0, 100.0);
}


