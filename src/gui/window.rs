//! # Windows GUI Window Management
//!
//! This module handles Win32 window creation, message processing, and rendering loop.
//! Uses Direct2D for hardware-accelerated rendering.

use std::cell::RefCell;
use std::rc::Rc;

use mcts::GameState;
use windows::{
    Win32::{
        Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, WPARAM},
        Graphics::Gdi::{BeginPaint, EndPaint, InvalidateRect, PAINTSTRUCT},
        System::LibraryLoader::GetModuleHandleW,
        UI::WindowsAndMessaging::{
            CreateWindowExW, DefWindowProcW, DispatchMessageW, GetMessageW,
            LoadCursorW, PostQuitMessage, RegisterClassW, ShowWindow, TranslateMessage,
            CS_HREDRAW, CS_VREDRAW, CW_USEDEFAULT, IDC_ARROW, MSG,
            SW_SHOW, WM_CLOSE, WM_DESTROY, WM_KEYDOWN, WM_LBUTTONDOWN,
            WM_MOUSEMOVE, WM_PAINT, WM_SIZE, WM_TIMER, WNDCLASSW, WS_OVERLAPPEDWINDOW,
            SetTimer, KillTimer,
        },
        UI::Input::KeyboardAndMouse::{VK_ESCAPE, VK_RETURN},
    },
    core::PCWSTR,
};

use super::app::{GuiApp, GuiMode, PlayerType, GameStatus};
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
        let renderer = Renderer::new(hwnd)?;
        RENDERER.with(|r| {
            *r.borrow_mut() = Some(renderer);
        });

        // Show window
        let _ = ShowWindow(hwnd, SW_SHOW);

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
            
            let needs_redraw = APP_STATE.with(|state| {
                if let Some(app) = state.borrow().as_ref() {
                    handle_click(&mut app.borrow_mut(), x, y)
                } else {
                    false
                }
            });

            if needs_redraw {
                unsafe { let _ = InvalidateRect(Some(hwnd), None, false); }
            }
            LRESULT(0)
        }

        WM_MOUSEMOVE => {
            let x = (lparam.0 & 0xFFFF) as f32;
            let y = ((lparam.0 >> 16) & 0xFFFF) as f32;
            
            let needs_redraw = RENDERER.with(|r| {
                if let Some(renderer) = r.borrow().as_ref() {
                    APP_STATE.with(|state| {
                        if let Some(app) = state.borrow().as_ref() {
                            handle_mouse_move(&mut app.borrow_mut(), renderer, x, y)
                        } else {
                            false
                        }
                    })
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
                    handle_key(&mut app.borrow_mut(), vk)
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
fn handle_key(app: &mut GuiApp, vk: u16) -> bool {
    if vk == VK_ESCAPE.0 {
        app.go_back();
        app.should_quit
    } else if vk == VK_RETURN.0 {
        match app.mode {
            GuiMode::GameSelection => {
                app.select_game(app.selected_game_index);
            }
            GuiMode::PlayerConfig => {
                app.start_game();
            }
            GuiMode::GameOver => {
                app.go_back();
            }
            _ => {}
        }
        false
    } else {
        false
    }
}

/// Handle mouse click
fn handle_click(app: &mut GuiApp, x: f32, y: f32) -> bool {
    app.needs_redraw = true;

    match app.mode {
        GuiMode::GameSelection => {
            // Check button clicks
            let games = super::app::GameType::all();
            for (i, _) in games.iter().enumerate() {
                let button_rect = get_game_button_rect(i);
                if button_rect.contains(x, y) {
                    app.select_game(i);
                    return true;
                }
            }
        }
        GuiMode::PlayerConfig => {
            // Check player toggle buttons
            for i in 0..app.player_types.len() {
                let button_rect = get_player_button_rect(i);
                if button_rect.contains(x, y) {
                    app.toggle_player(i);
                    return true;
                }
            }
            
            // Check start button
            let start_rect = get_start_button_rect();
            if start_rect.contains(x, y) {
                app.start_game();
                return true;
            }
        }
        GuiMode::InGame => {
            if app.ai_thinking {
                return false;
            }

            // Check if current player is human
            let current_player = app.game.get_current_player();
            let is_human = app.player_types
                .iter()
                .find(|(id, _)| *id == current_player)
                .map(|(_, pt)| *pt == PlayerType::Human)
                .unwrap_or(false);

            if is_human {
                // Get game area and pass to renderer
                let game_area = get_game_area();
                let input = GameInput::Click { x, y };
                
                match app.game_renderer.handle_input(input, &app.game, game_area) {
                    InputResult::Move(mv) => {
                        app.make_move(mv);
                        return true;
                    }
                    InputResult::Redraw => return true,
                    InputResult::None => {}
                }
            }
        }
        GuiMode::GameOver => {
            // Click anywhere to go back
            app.go_back();
            return true;
        }
    }

    false
}

/// Handle mouse movement (for hover effects)
fn handle_mouse_move(app: &mut GuiApp, _renderer: &Renderer, x: f32, y: f32) -> bool {
    if app.mode == GuiMode::InGame {
        let game_area = get_game_area();
        let input = GameInput::Hover { x, y };
        
        if let InputResult::Redraw = app.game_renderer.handle_input(input, &app.game, game_area) {
            return true;
        }
    }
    false
}

/// Main render function
fn render(renderer: &Renderer, app: &GuiApp) {
    renderer.begin_draw();
    renderer.clear(Colors::BACKGROUND);

    match app.mode {
        GuiMode::GameSelection => render_game_selection(renderer, app),
        GuiMode::PlayerConfig => render_player_config(renderer, app),
        GuiMode::InGame => render_in_game(renderer, app),
        GuiMode::GameOver => render_game_over(renderer, app),
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
    let subtitle_rect = Rect::new(0.0, 100.0, client.width, 30.0);
    renderer.draw_text("Select a game to play", subtitle_rect, Colors::TEXT_SECONDARY, true);

    // Game buttons
    let games = super::app::GameType::all();
    for (i, game) in games.iter().enumerate() {
        let button_rect = get_game_button_rect(i);
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

    // Instructions
    let help_rect = Rect::new(0.0, client.height - 50.0, client.width, 30.0);
    renderer.draw_small_text("Click to select • Enter to confirm • Escape to quit", help_rect, Colors::TEXT_SECONDARY, true);
}

/// Render player configuration screen
fn render_player_config(renderer: &Renderer, app: &GuiApp) {
    let client = renderer.get_client_rect();
    
    // Title
    let title_rect = Rect::new(0.0, 40.0, client.width, 60.0);
    let title = format!("{} - Player Setup", app.selected_game_type.name());
    renderer.draw_title(&title, title_rect, Colors::TEXT_PRIMARY, true);

    // Player type buttons
    for (i, (player_id, player_type)) in app.player_types.iter().enumerate() {
        let button_rect = get_player_button_rect(i);
        
        let bg_color = match player_type {
            PlayerType::Human => Colors::BUTTON_SELECTED,
            PlayerType::AI => Colors::BUTTON_BG,
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
                format!("{}: {}", color_name, if *player_type == PlayerType::Human { "Human" } else { "AI" })
            }
            _ => {
                let player_name = if *player_id == 1 { "Player 1" } else { "Player 2" };
                format!("{}: {}", player_name, if *player_type == PlayerType::Human { "Human" } else { "AI" })
            }
        };
        
        renderer.draw_text(&label, button_rect, Colors::TEXT_PRIMARY, true);
    }

    // Start button
    let start_rect = get_start_button_rect();
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
    let status_text = if app.ai_thinking {
        format!("{} (AI thinking...)", player_name)
    } else {
        format!("{}'s turn", player_name)
    };
    let status_rect = Rect::new(client.width - 250.0, 0.0, 240.0, 50.0);
    let status_color = if app.ai_thinking { Colors::AI_THINKING } else { Colors::TEXT_PRIMARY };
    renderer.draw_text(&status_text, status_rect, status_color, false);

    // Game area
    let game_area = get_game_area();
    app.game_renderer.render(renderer, &app.game, game_area);

    // Move count
    let moves_text = format!("Moves: {}", app.move_history.len());
    let moves_rect = Rect::new(10.0, client.height - 30.0, 150.0, 30.0);
    renderer.draw_small_text(&moves_text, moves_rect, Colors::TEXT_SECONDARY, false);
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
            format!("{} wins!", winner_name)
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

fn get_game_button_rect(index: usize) -> Rect {
    let y = 160.0 + index as f32 * 100.0;
    Rect::new(400.0, y, 480.0, 80.0)
}

fn get_player_button_rect(index: usize) -> Rect {
    let y = 160.0 + index as f32 * 70.0;
    Rect::new(440.0, y, 400.0, 55.0)
}

fn get_start_button_rect() -> Rect {
    Rect::new(490.0, 450.0, 300.0, 50.0)
}

fn get_game_area() -> Rect {
    // Reserve space for header and margins
    Rect::new(20.0, 60.0, 1240.0, 700.0)
}
