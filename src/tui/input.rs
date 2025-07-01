//! # Input Handling Module
//!
//! This module is responsible for handling all user input, including keyboard
//! and mouse events. It translates these events into actions within the application.

use crate::app::{App, AppMode, GameStatus, Player};
use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::games::{gomoku::GomokuMove, connect4::Connect4Move, othello::OthelloMove, blokus::BlokusMove};
use crate::games::blokus::get_blokus_pieces;
use crate::tui::mouse;
use crossterm::event::{KeyCode, MouseEventKind};
use ratatui::layout::Rect;
use mcts::GameState;

/// Handles keyboard input based on the current application mode
/// 
/// Routes key presses to the appropriate handler function depending on
/// which screen/menu is currently active.
/// 
/// # Arguments
/// * `app` - Mutable reference to the application state
/// * `key_code` - The key that was pressed
pub fn handle_key_press(app: &mut App, key_code: KeyCode) {
    match app.mode {
        AppMode::GameSelection => handle_game_selection_input(key_code, app),
        AppMode::Settings => handle_settings_input(key_code, app),
        AppMode::PlayerConfig => handle_player_config_input(key_code, app),
        AppMode::InGame => handle_ingame_input(key_code, app),
        AppMode::GameOver => handle_game_over_input(key_code, app),
    }
}

/// Handles mouse events by delegating to the mouse module
/// 
/// # Arguments
/// * `app` - Mutable reference to the application state
/// * `kind` - Type of mouse event (click, drag, scroll, etc.)
/// * `col` - Column position of the mouse event
/// * `row` - Row position of the mouse event
/// * `terminal_size` - Size of the terminal for coordinate calculations
pub fn handle_mouse_event(app: &mut App, kind: MouseEventKind, col: u16, row: u16, terminal_size: Rect) {
    mouse::handle_mouse_event(app, kind, col, row, terminal_size);
}

/// Handles keyboard input in the game selection menu
/// 
/// Supports navigation with arrow keys, selection with Enter,
/// and quitting with Q or Escape.
/// 
/// # Arguments
/// * `key_code` - The key that was pressed
/// * `app` - Mutable reference to the application state
fn handle_game_selection_input(key_code: KeyCode, app: &mut App) {
    match key_code {
        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
        KeyCode::Up => app.select_prev_game(),
        KeyCode::Down => app.select_next_game(),
        KeyCode::Enter => {
            if let Some(selected) = app.game_selection_state.selected() {
                if selected < app.games.len() {
                    // Selected a game - initialize it and go to player config
                    let factory = &app.games[selected].1;
                    app.game_wrapper = factory();
                    app.game_status = crate::app::GameStatus::InProgress;
                    app.last_search_stats = None;
                    app.move_history.clear();

                    let num_players = app.game_wrapper.get_num_players();
                    
                    // Only reset player options if we don't have the right number of players
                    // or if we don't have any player options configured yet
                    if app.player_options.is_empty() || app.player_options.len() != num_players as usize {
                        app.player_options = (1..=num_players).map(|i| (i, crate::app::Player::Human)).collect();
                        app.selected_player_config_index = 0;
                    }

                    // If AI-only mode is enabled, skip player config and go straight to game
                    if app.ai_only {
                        // Set all players to AI
                        for (_, player_type) in &mut app.player_options {
                            *player_type = crate::app::Player::AI;
                        }
                        app.confirm_player_config();
                    } else {
                        app.mode = crate::app::AppMode::PlayerConfig;
                    }
                } else if selected == app.games.len() {
                    // Selected Settings (games + settings)
                    app.mode = crate::app::AppMode::Settings;
                } else if selected == app.games.len() + 1 {
                    // Selected Quit (games + settings + quit)
                    app.should_quit = true;
                }
            }
        }
        _ => {}
    }
}

/// Handles keyboard input in the settings menu
/// 
/// Supports navigation with arrow keys, value adjustment with left/right arrows,
/// and returning to the main menu with Escape.
/// 
/// # Arguments
/// * `key_code` - The key that was pressed
/// * `app` - Mutable reference to the application state
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

/// Handles keyboard input in the player configuration menu
/// 
/// Supports navigation with arrow keys, player type cycling with left/right or space,
/// and starting the game with Enter when "Start Game" is selected.
/// 
/// # Arguments
/// * `key_code` - The key that was pressed
/// * `app` - Mutable reference to the application state
fn handle_player_config_input(key_code: KeyCode, app: &mut App) {
    match key_code {
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Left | KeyCode::Right | KeyCode::Char(' ') => {
            if app.selected_player_config_index < app.player_options.len() {
                app.cycle_player_type();
            }
        }
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

/// Handles keyboard input during active gameplay
/// 
/// Supports move input, game controls, debug toggles, scrolling, and navigation.
/// Only allows move input when it's a human player's turn and the game is in progress.
/// 
/// # Arguments
/// * `key_code` - The key that was pressed
/// * `app` - Mutable reference to the application state
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
        KeyCode::Char('r') => {
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) && is_current_player_human(app) {
                app.blokus_rotate_piece();
            } else {
                app.reset_game();
            }
        },
        KeyCode::Esc => app.mode = AppMode::GameSelection,
        KeyCode::Up => move_cursor_up(app),
        KeyCode::Down => move_cursor_down(app),
        KeyCode::Left => move_cursor_left(app),
        KeyCode::Right => move_cursor_right(app),
        KeyCode::Enter | KeyCode::Char(' ') => make_move(app),
        KeyCode::PageUp => app.scroll_debug_up(),
        KeyCode::PageDown => app.scroll_debug_down(),
        KeyCode::Home => app.reset_debug_scroll(),
        KeyCode::End => app.enable_history_auto_scroll(),
        KeyCode::Char('f') => {
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                app.blokus_select_piece(15);
            }
        },
        KeyCode::Char('p') => {
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                app.blokus_pass_move();
            }
        },
        KeyCode::Char('e') => {
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                app.blokus_select_piece(14);
            }
        },
        KeyCode::Char('+') | KeyCode::Char('=') => {
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                app.blokus_expand_all();
            }
        },
        KeyCode::Char('-') => {
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                app.blokus_collapse_all();
            }
        },
        KeyCode::Char('x') => {
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                app.blokus_flip_piece();
            }
        },
        KeyCode::Char('z') => {
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                let current_player = app.game_wrapper.get_current_player();
                app.blokus_toggle_player_expand((current_player - 1) as usize);
            }
        },
        // Piece selection keys - refactored to use lookup table
        KeyCode::Char(c) => {
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                // Map characters to piece indices
                let piece_index = match c {
                    // Numbers 1-9 map to pieces 0-8, 0 maps to piece 9
                    '1'..='9' => Some((c as u8 - b'1') as usize),
                    '0' => Some(9),
                    // Letters a-d map to pieces 10-13, g-k map to pieces 16-20
                    // (e and f are used for other functions like 'e' and 'p', x and z have specific handlers above)
                    'a' => Some(10),
                    'b' => Some(11),
                    'c' => Some(12),
                    'd' => Some(13),
                    'g' => Some(16),
                    'h' => Some(17),
                    'i' => Some(18),
                    'j' => Some(19),
                    'k' => Some(20),
                    _ => None,
                };
                
                if let Some(index) = piece_index {
                    app.blokus_select_piece(index);
                }
            }
        },
        _ => {}
    }
}

/// Handles keyboard input on the game over screen
/// 
/// Supports restarting the game with R or Enter, returning to game selection
/// with Escape, and quitting with Q.
/// 
/// # Arguments
/// * `key_code` - The key that was pressed
/// * `app` - Mutable reference to the application state
fn handle_game_over_input(key_code: KeyCode, app: &mut App) {
    match key_code {
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Char('r') | KeyCode::Enter => app.reset_game(),
        KeyCode::Esc => app.mode = AppMode::GameSelection,
        _ => {}
    }
}

/// Checks if the current player is human (vs AI)
/// 
/// Maps the game's internal player ID to the UI player configuration
/// to determine if the current player should accept human input.
/// 
/// # Arguments
/// * `app` - Reference to the application state
/// 
/// # Returns
/// true if the current player is human, false if AI
fn is_current_player_human(app: &App) -> bool {
    let game_player_id = app.game_wrapper.get_current_player();
    let ui_player_id = match &app.game_wrapper {
        GameWrapper::Blokus(_) => game_player_id, // Blokus already uses 1,2,3,4
        _ => {
            // For 2-player games, map 1->1 and -1->2
            if game_player_id == 1 {
                1
            } else if game_player_id == -1 {
                2
            } else {
                game_player_id // fallback
            }
        }
    };
    app.player_options
        .iter()
        .any(|(id, p_type)| *id == ui_player_id && *p_type == Player::Human)
}

/// Moves the board cursor up by one row
/// 
/// For Blokus, validates that the selected piece would still fit at the new position.
/// For Connect4, this is disabled since navigation is column-based.
/// For other games, simply moves the cursor if within bounds.
/// 
/// # Arguments
/// * `app` - Mutable reference to the application state
fn move_cursor_up(app: &mut App) {
    // Connect4 uses column-based navigation only
    if matches!(app.game_wrapper, GameWrapper::Connect4(_)) {
        return;
    }

    if app.board_cursor.0 > 0 {
        let new_row = app.board_cursor.0 - 1;
        // For Blokus, check if the selected piece would fit at the new position
        if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
            if would_blokus_piece_fit(app, new_row, app.board_cursor.1) {
                app.board_cursor.0 = new_row;
            }
        } else {
            app.board_cursor.0 = new_row;
        }
    }
}

/// Moves the board cursor down by one row
/// 
/// For Blokus, validates that the selected piece would still fit at the new position.
/// For Connect4, this is disabled since navigation is column-based.
/// For other games, simply moves the cursor if within bounds.
/// 
/// # Arguments
/// * `app` - Mutable reference to the application state
fn move_cursor_down(app: &mut App) {
    // Connect4 uses column-based navigation only
    if matches!(app.game_wrapper, GameWrapper::Connect4(_)) {
        return;
    }

    let board = app.game_wrapper.get_board();
    let max_row = board.len() as u16;
    if app.board_cursor.0 < max_row.saturating_sub(1) {
        let new_row = app.board_cursor.0 + 1;
        // For Blokus, check if the selected piece would fit at the new position
        if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
            if would_blokus_piece_fit(app, new_row, app.board_cursor.1) {
                app.board_cursor.0 = new_row;
            }
        } else {
            app.board_cursor.0 = new_row;
        }
    }
}

/// Moves the board cursor left by one column
/// 
/// For Blokus, validates that the selected piece would still fit at the new position.
/// For other games, simply moves the cursor if within bounds.
/// 
/// # Arguments
/// * `app` - Mutable reference to the application state
fn move_cursor_left(app: &mut App) {
    if app.board_cursor.1 > 0 {
        let new_col = app.board_cursor.1 - 1;
        // For Blokus, check if the selected piece would fit at the new position
        if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
            if would_blokus_piece_fit(app, app.board_cursor.0, new_col) {
                app.board_cursor.1 = new_col;
            }
        } else {
            app.board_cursor.1 = new_col;
            // For Connect4, update cursor to lowest available position in new column
            if let GameWrapper::Connect4(_) = app.game_wrapper {
                update_connect4_cursor_row(app);
            }
        }
    }
}

/// Moves the board cursor right by one column
/// 
/// For Blokus, validates that the selected piece would still fit at the new position.
/// For Connect4, updates the cursor to the lowest available position in the new column.
/// For other games, simply moves the cursor if within bounds.
/// 
/// # Arguments
/// * `app` - Mutable reference to the application state
fn move_cursor_right(app: &mut App) {
    let board = app.game_wrapper.get_board();
    let max_col = if !board.is_empty() { board[0].len() as u16 } else { 0 };
    if app.board_cursor.1 < max_col.saturating_sub(1) {
        let new_col = app.board_cursor.1 + 1;
        // For Blokus, check if the selected piece would fit at the new position
        if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
            if would_blokus_piece_fit(app, app.board_cursor.0, new_col) {
                app.board_cursor.1 = new_col;
            }
        } else {
            app.board_cursor.1 = new_col;
            // For Connect4, update cursor to lowest available position in new column
            if let GameWrapper::Connect4(_) = app.game_wrapper {
                update_connect4_cursor_row(app);
            }
        }
    }
}

/// Updates the Connect4 cursor to the lowest available position in the current column
/// 
/// In Connect4, pieces fall due to gravity, so the cursor should always point
/// to where a piece would actually land if dropped in the current column.
/// 
/// # Arguments
/// * `app` - Mutable reference to the application state
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

/// Attempts to make a move at the current cursor position
/// 
/// Creates the appropriate move type based on the current game and validates
/// it before applying. If the move is legal, it's added to the move history
/// and the game state is updated.
/// 
/// # Arguments
/// * `app` - Mutable reference to the application state
fn make_move(app: &mut App) {
    let (row, col) = (app.board_cursor.0 as usize, app.board_cursor.1 as usize);
    
    let player_move = match &app.game_wrapper {
        GameWrapper::Gomoku(_) => MoveWrapper::Gomoku(GomokuMove(row, col)),
        GameWrapper::Connect4(_) => MoveWrapper::Connect4(Connect4Move(col)),
        GameWrapper::Othello(_) => MoveWrapper::Othello(OthelloMove(row, col)),
        GameWrapper::Blokus(_) => {
            // For Blokus, create a move from the selected piece and cursor position
            if let Some((piece_idx, transformation_idx)) = app.blokus_ui_config.get_selected_piece_info() {
                MoveWrapper::Blokus(BlokusMove(piece_idx, transformation_idx, row, col))
            } else {
                // No piece selected, use pass move
                MoveWrapper::Blokus(BlokusMove(usize::MAX, 0, 0, 0))
            }
        }
    };

    if app.game_wrapper.is_legal(&player_move) {
        let current_player = app.game_wrapper.get_current_player();
        app.move_history.push(crate::app::MoveHistoryEntry::new(current_player, player_move.clone()));
        app.on_move_added(); // Auto-scroll to bottom
        app.game_wrapper.make_move(&player_move);
        
        // Advance the AI worker's MCTS tree root to reflect the move that was just made
        app.ai_worker.advance_root(&player_move);
        
        // Clear selected piece if it becomes unavailable after move (for Blokus)
        if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
            app.clear_selected_piece_if_unavailable();
        }
        
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

/// Check if a Blokus piece would fit within board bounds at the given position
fn would_blokus_piece_fit(app: &App, new_row: u16, new_col: u16) -> bool {
    // If no piece is selected, always allow movement
    let (piece_idx, transformation_idx) = match app.blokus_ui_config.get_selected_piece_info() {
        Some(info) => info,
        None => return true,
    };
    
    // Only check for Blokus game
    if let GameWrapper::Blokus(state) = &app.game_wrapper {
        let board = state.get_board();
        let board_height = board.len();
        let board_width = if board_height > 0 { board[0].len() } else { 0 };
        
        // Check if this piece is available for the current player
        let current_player = state.get_current_player();
        let available_pieces = state.get_available_pieces(current_player);
        if !available_pieces.contains(&piece_idx) {
            return true; // Allow movement if piece is not available anyway
        }
        
        // Get the piece and its transformation
        let pieces = get_blokus_pieces();
        if let Some(piece) = pieces.iter().find(|p| p.id == piece_idx) {
            if transformation_idx < piece.transformations.len() {
                let shape = &piece.transformations[transformation_idx];
                
                // Check if all blocks of the piece would be within bounds
                for &(dr, dc) in shape {
                    let board_r = new_row as i32 + dr;
                    let board_c = new_col as i32 + dc;
                    
                    // If any block would be out of bounds, don't allow this cursor position
                    if board_r < 0 || board_r >= board_height as i32 || board_c < 0 || board_c >= board_width as i32 {
                        return false;
                    }
                }
            }
        }
    }
    
    true
}


