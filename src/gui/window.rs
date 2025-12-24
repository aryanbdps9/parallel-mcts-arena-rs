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
        UI::Input::KeyboardAndMouse::{VK_ESCAPE, VK_RETURN, VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT, VK_TAB, VK_SPACE, VK_BACK},
    },
    core::PCWSTR,
};

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
    let num_settings = 12; // 10 settings + separator + Back
    let num_games = super::app::GameType::all().len() + 2; // games + Settings + Quit

    if vk == VK_ESCAPE.0 || vk == VK_BACK.0 {
        app.go_back();
        return app.should_quit;
    }

    match app.mode {
        GuiMode::GameSelection => {
            if vk == VK_UP.0 {
                if app.selected_game_index > 0 {
                    app.selected_game_index -= 1;
                }
            } else if vk == VK_DOWN.0 {
                if app.selected_game_index < num_games - 1 {
                    app.selected_game_index += 1;
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
            if vk == VK_UP.0 {
                if app.selected_settings_index > 0 {
                    app.selected_settings_index -= 1;
                }
            } else if vk == VK_DOWN.0 {
                if app.selected_settings_index < num_settings - 1 {
                    app.selected_settings_index += 1;
                }
            } else if vk == VK_LEFT.0 {
                if app.selected_settings_index < 10 {
                    app.adjust_setting(app.selected_settings_index, -1);
                }
            } else if vk == VK_RIGHT.0 || vk == VK_SPACE.0 {
                if app.selected_settings_index < 10 {
                    app.adjust_setting(app.selected_settings_index, 1);
                }
            } else if vk == VK_RETURN.0 {
                if app.selected_settings_index == num_settings - 1 {
                    // Back
                    app.mode = GuiMode::GameSelection;
                } else if app.selected_settings_index >= 8 && app.selected_settings_index <= 9 {
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
            }
            // Additional game input would go here (arrow keys for cursor, etc.)
        }
        GuiMode::GameOver => {
            if vk == VK_RETURN.0 || vk == VK_SPACE.0 {
                app.go_back();
            }
        }
        GuiMode::HowToPlay => {
            if vk == VK_LEFT.0 {
                if app.selected_how_to_play_game > 0 {
                    app.selected_how_to_play_game -= 1;
                    app.how_to_play_scroll = 0;
                }
            } else if vk == VK_RIGHT.0 {
                if app.selected_how_to_play_game < 3 {
                    app.selected_how_to_play_game += 1;
                    app.how_to_play_scroll = 0;
                }
            } else if vk == VK_UP.0 {
                app.how_to_play_scroll = (app.how_to_play_scroll - 1).max(0);
            } else if vk == VK_DOWN.0 {
                app.how_to_play_scroll += 1;
            }
        }
    }
    app.needs_redraw = true;
    false
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
            // Check Settings button
            let settings_rect = get_game_button_rect(games.len());
            if settings_rect.contains(x, y) {
                app.mode = GuiMode::Settings;
                app.selected_settings_index = 0;
                return true;
            }
            // Check How To Play button
            let help_rect = get_game_button_rect(games.len() + 1);
            if help_rect.contains(x, y) {
                app.mode = GuiMode::HowToPlay;
                app.selected_how_to_play_game = 0;
                app.how_to_play_scroll = 0;
                return true;
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
            // Check if current player is human
            let current_player = app.game.get_current_player();
            let is_human = app.player_types
                .iter()
                .find(|(id, _)| *id == current_player)
                .map(|(_, pt)| *pt == PlayerType::Human)
                .unwrap_or(false);

            // Check for tab clicks (always allowed, even during AI thinking)
            let tab_area = get_tab_area();
            if tab_area.contains(x, y) {
                app.toggle_tab();
                return true;
            }

            if is_human && !app.ai_thinking {
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
        GuiMode::Settings => {
            // Check settings item clicks
            let settings = app.get_settings_items();
            for i in 0..settings.len() + 2 {
                let item_rect = get_settings_item_rect(i);
                if item_rect.contains(x, y) {
                    app.selected_settings_index = i;
                    if i == settings.len() + 1 {
                        // Back button
                        app.go_back();
                    } else if i >= 8 && i <= 9 {
                        // Toggle bool settings
                        app.adjust_setting(i, 1);
                    }
                    return true;
                }
            }
        }
        GuiMode::HowToPlay => {
            // Check game tab clicks
            let games = ["Gomoku", "Connect4", "Othello", "Blokus"];
            for (i, _) in games.iter().enumerate() {
                let tab_rect = get_how_to_play_tab_rect(i);
                if tab_rect.contains(x, y) {
                    app.selected_how_to_play_game = i;
                    app.how_to_play_scroll = 0;
                    return true;
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

    // Settings button
    let settings_rect = get_game_button_rect(games.len());
    let is_settings_selected = app.selected_game_index == games.len();
    let settings_bg = if is_settings_selected { Colors::BUTTON_SELECTED } else { Colors::BUTTON_BG };
    renderer.fill_rounded_rect(settings_rect, 8.0, settings_bg);
    renderer.draw_text("Settings", Rect::new(settings_rect.x, settings_rect.y + 20.0, settings_rect.width, 30.0), Colors::TEXT_PRIMARY, true);

    // How To Play button
    let help_button_rect = get_game_button_rect(games.len() + 1);
    let is_help_selected = app.selected_game_index == games.len() + 1;
    let help_bg = if is_help_selected { Colors::BUTTON_SELECTED } else { Colors::BUTTON_BG };
    renderer.fill_rounded_rect(help_button_rect, 8.0, help_bg);
    renderer.draw_text("How To Play", Rect::new(help_button_rect.x, help_button_rect.y + 20.0, help_button_rect.width, 30.0), Colors::TEXT_PRIMARY, true);

    // Instructions
    let help_rect = Rect::new(0.0, client.height - 50.0, client.width, 30.0);
    renderer.draw_small_text("↑↓ Navigate • Enter Select • Escape Quit", help_rect, Colors::TEXT_SECONDARY, true);
}

/// Render settings screen
fn render_settings(renderer: &Renderer, app: &GuiApp) {
    let client = renderer.get_client_rect();
    
    // Title
    let title_rect = Rect::new(0.0, 30.0, client.width, 50.0);
    renderer.draw_title("Settings", title_rect, Colors::TEXT_PRIMARY, true);
    
    // Settings list
    let settings = app.get_settings_items();
    for (i, (name, value)) in settings.iter().enumerate() {
        let item_rect = get_settings_item_rect(i);
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
    
    // Separator (empty row)
    let sep_index = settings.len();
    let sep_rect = get_settings_item_rect(sep_index);
    renderer.draw_line(sep_rect.x + 50.0, sep_rect.y + sep_rect.height / 2.0, 
                       sep_rect.x + sep_rect.width - 50.0, sep_rect.y + sep_rect.height / 2.0,
                       Colors::GRID_LINE, 1.0);
    
    // Back button
    let back_index = settings.len() + 1;
    let back_rect = get_settings_item_rect(back_index);
    let is_back_selected = app.selected_settings_index == back_index;
    let back_bg = if is_back_selected { Colors::BUTTON_SELECTED } else { Colors::BUTTON_BG };
    renderer.fill_rounded_rect(back_rect, 5.0, back_bg);
    renderer.draw_text("Back", back_rect, Colors::TEXT_PRIMARY, true);
    
    // Instructions
    let help_rect = Rect::new(0.0, client.height - 50.0, client.width, 30.0);
    renderer.draw_small_text("↑↓ Navigate • ←→ Adjust • Enter Confirm • Escape Back", help_rect, Colors::TEXT_SECONDARY, true);
}

/// Render how to play screen
fn render_how_to_play(renderer: &Renderer, app: &GuiApp) {
    let client = renderer.get_client_rect();
    
    // Title
    let title_rect = Rect::new(0.0, 30.0, client.width, 50.0);
    renderer.draw_title("How To Play", title_rect, Colors::TEXT_PRIMARY, true);
    
    // Game tabs
    let games = ["Gomoku", "Connect4", "Othello", "Blokus"];
    for (i, name) in games.iter().enumerate() {
        let tab_rect = get_how_to_play_tab_rect(i);
        let is_selected = i == app.selected_how_to_play_game;
        let bg_color = if is_selected { Colors::BUTTON_SELECTED } else { Colors::BUTTON_BG };
        renderer.fill_rounded_rect(tab_rect, 5.0, bg_color);
        renderer.draw_text(*name, tab_rect, Colors::TEXT_PRIMARY, true);
    }
    
    // Instructions content
    let content_rect = Rect::new(100.0, 150.0, client.width - 200.0, client.height - 250.0);
    renderer.fill_rounded_rect(content_rect, 10.0, Colors::PANEL_BG);
    
    let instructions = get_game_instructions(app.selected_how_to_play_game);
    let text_rect = content_rect.with_padding(20.0);
    renderer.draw_small_text(&instructions, text_rect, Colors::TEXT_PRIMARY, false);
    
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
        let button_rect = get_player_button_rect(i);
        let is_selected = i == app.selected_player_index;
        
        let bg_color = match player_type {
            PlayerType::Human => if is_selected { Colors::BUTTON_SELECTED } else { Colors::STATUS_WIN },
            PlayerType::AI => if is_selected { Colors::BUTTON_SELECTED } else { Colors::BUTTON_BG },
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
        let elapsed = app.ai_thinking_start.map(|t| t.elapsed().as_secs()).unwrap_or(0);
        format!("{} (AI thinking... {}s)", player_name, elapsed)
    } else {
        format!("{}'s turn", player_name)
    };
    let status_rect = Rect::new(client.width - 350.0, 0.0, 340.0, 50.0);
    let status_color = if app.ai_thinking { Colors::AI_THINKING } else { Colors::TEXT_PRIMARY };
    renderer.draw_text(&status_text, status_rect, status_color, false);

    // Main area: game board on left, info panel on right
    let game_area = get_game_area();
    let info_panel_width = 300.0;
    let board_area = Rect::new(game_area.x, game_area.y, game_area.width - info_panel_width - 20.0, game_area.height);
    let info_area = Rect::new(game_area.x + board_area.width + 20.0, game_area.y, info_panel_width, game_area.height);
    
    // Render game board
    app.game_renderer.render(renderer, &app.game, board_area);

    // Render info panel with tabs
    render_info_panel(renderer, app, info_area);

    // Move count and controls hint at bottom
    let moves_text = format!("Moves: {} | Tab: Switch Panel", app.move_history.len());
    let moves_rect = Rect::new(10.0, client.height - 30.0, 400.0, 30.0);
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
    
    for (i, line) in lines.iter().enumerate() {
        let y = area.y + i as f32 * line_height;
        if y + line_height > area.y + area.height {
            break; // Stop if we run out of space
        }
        let line_rect = Rect::new(area.x, y, area.width, line_height);
        let color = if line.starts_with("AI Status") || line.contains("Top AI Moves") {
            Colors::TEXT_ACCENT
        } else {
            Colors::TEXT_SECONDARY
        };
        renderer.draw_small_text(line, line_rect, color, false);
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
    
    // Show recent moves (scroll to bottom)
    let max_visible = ((area.height / line_height) as usize).max(1);
    let start_index = history.len().saturating_sub(max_visible);
    
    for (i, line) in history.iter().skip(start_index).enumerate() {
        let y = area.y + i as f32 * line_height;
        if y + line_height > area.y + area.height {
            break;
        }
        let line_rect = Rect::new(area.x, y, area.width, line_height);
        renderer.draw_small_text(line, line_rect, Colors::TEXT_SECONDARY, false);
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
    // Use smaller spacing to fit all 6 items (4 games + Settings + How To Play)
    let y = 140.0 + index as f32 * 85.0;
    Rect::new(400.0, y, 480.0, 70.0)
}

fn get_player_button_rect(index: usize) -> Rect {
    let y = 140.0 + index as f32 * 60.0;
    Rect::new(440.0, y, 400.0, 50.0)
}

fn get_start_button_rect() -> Rect {
    // Position below the player buttons, accounting for up to 4 players (Blokus)
    Rect::new(490.0, 400.0, 300.0, 50.0)
}

fn get_game_area() -> Rect {
    // Reserve space for header and margins
    Rect::new(20.0, 60.0, 1240.0, 700.0)
}

fn get_settings_item_rect(index: usize) -> Rect {
    let y = 100.0 + index as f32 * 45.0;
    Rect::new(340.0, y, 600.0, 40.0)
}

fn get_how_to_play_tab_rect(index: usize) -> Rect {
    let x = 200.0 + index as f32 * 220.0;
    Rect::new(x, 90.0, 200.0, 40.0)
}

fn get_tab_area() -> Rect {
    // The tab area in the info panel
    Rect::new(960.0, 60.0, 300.0, 35.0)
}
