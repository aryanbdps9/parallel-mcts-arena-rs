//! # Input Handling Module
//!
//! This module is responsible for handling all user input, including keyboard
//! and mouse events. It translates these events into actions within the application.

use crate::app::{App, AppMode, GameStatus, Player};
use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::games::{gomoku::GomokuMove, connect4::Connect4Move, othello::OthelloMove, blokus::BlokusMove};
use crossterm::event::{KeyCode, MouseEventKind, MouseButton};
use ratatui::layout::Rect;
use mcts::GameState;

pub fn handle_key_press(app: &mut App, key_code: KeyCode) {
    match app.mode {
        AppMode::GameSelection => handle_game_selection_input(key_code, app),
        AppMode::Settings => handle_settings_input(key_code, app),
        AppMode::PlayerConfig => handle_player_config_input(key_code, app),
        AppMode::InGame => handle_ingame_input(key_code, app),
        AppMode::GameOver => handle_game_over_input(key_code, app),
    }
}

pub fn handle_mouse_event(app: &mut App, kind: MouseEventKind, col: u16, row: u16, _terminal_size: Rect) {
    match kind {
        MouseEventKind::Down(MouseButton::Left) => {
            if app.mode == AppMode::InGame {
                handle_board_click(app, col, row);
            }
        }
        MouseEventKind::ScrollUp => {
            if app.mode == AppMode::InGame {
                // Scroll debug stats up
                app.scroll_debug_up();
            }
        }
        MouseEventKind::ScrollDown => {
            if app.mode == AppMode::InGame {
                // Scroll debug stats down  
                app.scroll_debug_down();
            }
        }
        _ => {}
    }
}

fn handle_game_selection_input(key_code: KeyCode, app: &mut App) {
    match key_code {
        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
        KeyCode::Up => app.select_prev_game(),
        KeyCode::Down => app.select_next_game(),
        KeyCode::Enter => {
            if let Some(selected) = app.game_selection_state.selected() {
                if selected < app.games.len() {
                    // Selected a game
                    app.start_game();
                } else if selected == app.games.len() {
                    // Selected Settings (games + settings)
                    app.mode = AppMode::Settings;
                } else if selected == app.games.len() + 1 {
                    // Selected Quit (games + settings + quit)
                    app.should_quit = true;
                }
            }
        }
        _ => {}
    }
}

fn handle_settings_input(key_code: KeyCode, app: &mut App) {
    match key_code {
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Up => app.select_prev_setting(),
        KeyCode::Down => app.select_next_setting(),
        KeyCode::Left => app.decrease_setting(),
        KeyCode::Right => app.increase_setting(),
        KeyCode::Enter => {
            if app.selected_settings_index == 10 { // "Back" option (9 settings + separator + back = index 10)
                app.mode = AppMode::GameSelection;
            }
        }
        KeyCode::Esc => app.mode = AppMode::GameSelection,
        _ => {}
    }
}

fn handle_player_config_input(key_code: KeyCode, app: &mut App) {
    match key_code {
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Left | KeyCode::Right | KeyCode::Char(' ') => app.cycle_player_type(),
        KeyCode::Up => app.select_prev_player_config(),
        KeyCode::Down => app.select_next_player_config(),
        KeyCode::Enter => {
            if app.selected_player_config_index < app.player_options.len() {
                app.cycle_player_type();
            } else {
                app.confirm_player_config();
            }
        }
        KeyCode::Esc => app.mode = AppMode::GameSelection,
        _ => {}
    }
}

fn handle_ingame_input(key_code: KeyCode, app: &mut App) {
    if app.game_status != GameStatus::InProgress {
        return;
    }

    // Only allow human player input
    if !is_current_player_human(app) {
        match key_code {
            KeyCode::Char('q') => app.should_quit = true,
            KeyCode::Char('r') => app.reset_game(),
            KeyCode::Esc => app.mode = AppMode::GameSelection,
            _ => {}
        }
        return;
    }

    match key_code {
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Char('r') => app.reset_game(),
        KeyCode::Esc => app.mode = AppMode::GameSelection,
        KeyCode::Up => move_cursor_up(app),
        KeyCode::Down => move_cursor_down(app),
        KeyCode::Left => move_cursor_left(app),
        KeyCode::Right => move_cursor_right(app),
        KeyCode::Enter | KeyCode::Char(' ') => make_move(app),
        KeyCode::PageUp => app.scroll_debug_up(),
        KeyCode::PageDown => app.scroll_debug_down(),
        KeyCode::Home => app.reset_debug_scroll(),
        KeyCode::End => app.reset_history_scroll(),
        _ => {}
    }
}

fn handle_game_over_input(key_code: KeyCode, app: &mut App) {
    match key_code {
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Char('r') | KeyCode::Enter => app.reset_game(),
        KeyCode::Esc => app.mode = AppMode::GameSelection,
        _ => {}
    }
}

fn handle_board_click(app: &mut App, col: u16, row: u16) {
    if app.game_status != GameStatus::InProgress || !is_current_player_human(app) {
        return;
    }

    // Convert screen coordinates to board coordinates
    // This is a simplified version - in the original it was much more complex
    let board = app.game_wrapper.get_board();
    let board_height = board.len();
    let board_width = if board_height > 0 { board[0].len() } else { 0 };

    // Simple click handling - assume click is within board area
    if (row as usize) < board_height && (col as usize) < board_width {
        app.board_cursor = (row, col);
        make_move(app);
    }
}

fn is_current_player_human(app: &App) -> bool {
    let current_player_id = app.game_wrapper.get_current_player();
    app.player_options
        .iter()
        .any(|(id, p_type)| *id == current_player_id && *p_type == Player::Human)
}

fn move_cursor_up(app: &mut App) {
    if app.board_cursor.0 > 0 {
        app.board_cursor.0 -= 1;
    }
}

fn move_cursor_down(app: &mut App) {
    let board = app.game_wrapper.get_board();
    let max_row = board.len() as u16;
    if app.board_cursor.0 < max_row.saturating_sub(1) {
        app.board_cursor.0 += 1;
    }
}

fn move_cursor_left(app: &mut App) {
    if app.board_cursor.1 > 0 {
        app.board_cursor.1 -= 1;
        // For Connect4, update cursor to lowest available position in new column
        if let GameWrapper::Connect4(_) = app.game_wrapper {
            update_connect4_cursor_row(app);
        }
    }
}

fn move_cursor_right(app: &mut App) {
    let board = app.game_wrapper.get_board();
    let max_col = if !board.is_empty() { board[0].len() as u16 } else { 0 };
    if app.board_cursor.1 < max_col.saturating_sub(1) {
        app.board_cursor.1 += 1;
        // For Connect4, update cursor to lowest available position in new column
        if let GameWrapper::Connect4(_) = app.game_wrapper {
            update_connect4_cursor_row(app);
        }
    }
}

fn update_connect4_cursor_row(app: &mut App) {
    let board = app.game_wrapper.get_board();
    let board_height = board.len();
    let col = app.board_cursor.1 as usize;
    
    if col < board[0].len() {
        // Find the lowest empty row in this column
        for r in (0..board_height).rev() {
            if board[r][col] == 0 {
                app.board_cursor.0 = r as u16;
                return;
            }
        }
        // If column is full, stay at the top
        app.board_cursor.0 = 0;
    }
}

fn make_move(app: &mut App) {
    let (row, col) = (app.board_cursor.0 as usize, app.board_cursor.1 as usize);
    
    let player_move = match &app.game_wrapper {
        GameWrapper::Gomoku(_) => MoveWrapper::Gomoku(GomokuMove(row, col)),
        GameWrapper::Connect4(_) => MoveWrapper::Connect4(Connect4Move(col)),
        GameWrapper::Othello(_) => MoveWrapper::Othello(OthelloMove(row, col)),
        GameWrapper::Blokus(_) => {
            // For Blokus, use a simple pass move for now - this would need more complex handling
            MoveWrapper::Blokus(BlokusMove(usize::MAX, 0, 0, 0))
        }
    };

    if app.game_wrapper.is_legal(&player_move) {
        let current_player = app.game_wrapper.get_current_player();
        app.move_history.push(crate::app::MoveHistoryEntry::new(current_player, player_move.clone()));
        app.game_wrapper.make_move(&player_move);
        
        // Advance the AI worker's MCTS tree root to reflect the move that was just made
        app.ai_worker.advance_root(&player_move);
        
        // Check for game over
        if app.game_wrapper.is_terminal() {
            app.game_status = match app.game_wrapper.get_winner() {
                Some(winner) => GameStatus::Win(winner),
                None => GameStatus::Draw,
            };
            app.mode = AppMode::GameOver;
        }
    }
}


