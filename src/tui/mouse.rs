//! # Mouse Module
//!
//! This module handles mouse events including clicking, dragging, and scrolling.
//! It provides a clean interface for mouse interaction with the UI.

use crate::app::{App, AppMode, GameStatus, Player};
use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::games::{gomoku::GomokuMove, connect4::Connect4Move, othello::OthelloMove, blokus::BlokusMove};
use crate::tui::layout::DragBoundary;
use crossterm::event::{MouseButton, MouseEventKind};
use ratatui::layout::Rect;
use mcts::GameState;

/// State for tracking mouse drag operations
#[derive(Debug, Clone)]
pub struct DragState {
    /// Whether we're currently dragging
    pub is_dragging: bool,
    /// Which boundary is being dragged
    pub drag_boundary: Option<DragBoundary>,
    /// Starting position of the drag
    pub drag_start: Option<(u16, u16)>,
    /// Last position during drag
    pub last_drag_pos: Option<(u16, u16)>,
}

impl Default for DragState {
    fn default() -> Self {
        Self {
            is_dragging: false,
            drag_boundary: None,
            drag_start: None,
            last_drag_pos: None,
        }
    }
}

impl DragState {
    /// Start a drag operation
    pub fn start_drag(&mut self, boundary: DragBoundary, col: u16, row: u16) {
        self.is_dragging = true;
        self.drag_boundary = Some(boundary);
        self.drag_start = Some((col, row));
        self.last_drag_pos = Some((col, row));
    }

    /// Update drag position
    pub fn update_drag(&mut self, col: u16, row: u16) -> Option<(i16, i16)> {
        if !self.is_dragging {
            return None;
        }

        if let Some((last_col, last_row)) = self.last_drag_pos {
            let delta_col = col as i16 - last_col as i16;
            let delta_row = row as i16 - last_row as i16;
            self.last_drag_pos = Some((col, row));
            Some((delta_col, delta_row))
        } else {
            self.last_drag_pos = Some((col, row));
            None
        }
    }

    /// Stop drag operation
    pub fn stop_drag(&mut self) {
        self.is_dragging = false;
        self.drag_boundary = None;
        self.drag_start = None;
        self.last_drag_pos = None;
    }
}

/// Handle mouse events for the application
pub fn handle_mouse_event(app: &mut App, kind: MouseEventKind, col: u16, row: u16, terminal_size: Rect) {
    match kind {
        MouseEventKind::Down(MouseButton::Left) => {
            handle_mouse_click(app, col, row, terminal_size);
        }
        MouseEventKind::Drag(MouseButton::Left) => {
            handle_mouse_drag(app, col, row, terminal_size);
        }
        MouseEventKind::Up(MouseButton::Left) => {
            handle_mouse_release(app, col, row, terminal_size);
        }
        MouseEventKind::ScrollUp => {
            handle_mouse_scroll(app, col, row, terminal_size, true);
        }
        MouseEventKind::ScrollDown => {
            handle_mouse_scroll(app, col, row, terminal_size, false);
        }
        _ => {}
    }
}

/// Handle mouse click events
fn handle_mouse_click(app: &mut App, col: u16, row: u16, terminal_size: Rect) {
    // Check if the click is on a drag boundary first
    let is_blokus = matches!(app.game_wrapper, GameWrapper::Blokus(_));
    if let Some(boundary) = app.layout_config.detect_boundary_click(col, row, terminal_size, is_blokus) {
        app.drag_state.start_drag(boundary, col, row);
        return;
    }

    match app.mode {
        AppMode::GameSelection => {
            handle_menu_click(app, col, row, terminal_size);
        }
        AppMode::Settings => {
            handle_settings_click(app, col, row, terminal_size);
        }
        AppMode::InGame => {
            if !app.ai_only {
                if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                    handle_blokus_click(app, col, row, terminal_size);
                } else {
                    handle_board_click(app, col, row, terminal_size);
                }
            }
        }
        AppMode::GameOver => {
            // Could add click handling for game over state if needed
        }
        AppMode::PlayerConfig => {
            // Player configuration clicks handled in the event loop
        }
    }
}

/// Handle mouse drag events
fn handle_mouse_drag(app: &mut App, col: u16, row: u16, terminal_size: Rect) {
    if let Some((delta_col, delta_row)) = app.drag_state.update_drag(col, row) {
        if let Some(boundary) = app.drag_state.drag_boundary {
            let delta = match boundary {
                DragBoundary::StatsHistory |
                DragBoundary::BlokusPieceSelectionLeft |
                DragBoundary::BlokusPieceSelectionRight => delta_col,
                _ => delta_row,
            };
            app.layout_config.handle_drag(boundary, delta, terminal_size);
        }
    }
}

/// Handle mouse release events
fn handle_mouse_release(app: &mut App, _col: u16, _row: u16, _terminal_size: Rect) {
    if app.drag_state.is_dragging {
        app.drag_state.stop_drag();
    }
}

/// Handle mouse scroll events
fn handle_mouse_scroll(app: &mut App, col: u16, row: u16, terminal_size: Rect, scroll_up: bool) {
    match app.mode {
        AppMode::InGame | AppMode::GameOver => {
            // Special handling for Blokus piece selection scrolling
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                let (_, piece_selection_area, _) = app.layout_config.get_blokus_layout(terminal_size);
                
                // Check if mouse is in piece selection area
                if col >= piece_selection_area.x && col < piece_selection_area.x + piece_selection_area.width &&
                   row >= piece_selection_area.y && row < piece_selection_area.y + piece_selection_area.height {
                    if scroll_up {
                        app.blokus_scroll_panel_up();
                    } else {
                        app.blokus_scroll_panel_down();
                    }
                    return;
                }
            }

            // Default scrolling for stats area
            let (_, _, stats_area) = app.layout_config.get_main_layout(terminal_size);
            if row >= stats_area.y {
                let (debug_area, _history_area) = app.layout_config.get_stats_layout(stats_area);
                
                if col < debug_area.x + debug_area.width {
                    // Mouse is in debug stats area
                    if scroll_up {
                        app.scroll_debug_up();
                    } else {
                        app.scroll_debug_down();
                    }
                } else {
                    // Mouse is in move history area
                    if scroll_up {
                        app.scroll_move_history_up();
                    } else {
                        app.scroll_move_history_down();
                    }
                }
            }
        }
        _ => {
            // No scrolling for other modes
        }
    }
}

/// Handle menu clicks
fn handle_menu_click(app: &mut App, _col: u16, row: u16, terminal_size: Rect) {
    let (board_area, _, _) = app.layout_config.get_main_layout(terminal_size);
    
    // Check if click is within the menu area
    if row < board_area.height {
        let menu_start_row = 2; // Account for border and title
        if row >= menu_start_row {
            let clicked_item = (row - menu_start_row) as usize;
            let total_items = app.games.len() + 2; // games + settings + quit
            
            if clicked_item < total_items {
                if clicked_item < app.games.len() {
                    // A game was clicked
                    app.game_selection_state.select(Some(clicked_item));
                    app.start_game();
                } else if clicked_item == app.games.len() {
                    // Settings was clicked
                    app.mode = AppMode::Settings;
                } else {
                    // Quit was clicked
                    app.should_quit = true;
                }
            }
        }
    }
}

/// Handle settings menu clicks
fn handle_settings_click(app: &mut App, _col: u16, row: u16, terminal_size: Rect) {
    let (board_area, _, _) = app.layout_config.get_main_layout(terminal_size);
    
    // Check if click is within the settings area
    if row < board_area.height {
        let settings_area_start = 1; // Top border
        if row >= settings_area_start {
            let clicked_index = (row - settings_area_start) as usize;
            if clicked_index < 11 { // 9 settings + separator + back
                app.selected_settings_index = clicked_index;
                if app.selected_settings_index == 10 { // "Back" option
                    app.mode = AppMode::GameSelection;
                }
            }
        }
    }
}

/// Handle board clicks for standard games
fn handle_board_click(app: &mut App, col: u16, row: u16, terminal_size: Rect) {
    if app.game_status != GameStatus::InProgress || !is_current_player_human(app) {
        return;
    }

    let board = app.game_wrapper.get_board();
    let board_height = board.len();
    let board_width = if board_height > 0 { board[0].len() } else { 0 };

    let (board_area, _, _) = app.layout_config.get_main_layout(terminal_size);
    
    // Check if click is within the board area
    if row < board_area.height {
        // Calculate board cell from click position
        // Account for borders and labels
        let board_start_col = board_area.x + 1; // Border
        let board_start_row = board_area.y + 1; // Border
        
        if col >= board_start_col && row >= board_start_row {
            let cell_width = (board_area.width.saturating_sub(2)) / board_width as u16;
            let cell_height = (board_area.height.saturating_sub(2)) / board_height as u16;
            
            let board_col = ((col - board_start_col) / cell_width.max(1)) as usize;
            let board_row = ((row - board_start_row) / cell_height.max(1)) as usize;
            
            if board_row < board_height && board_col < board_width {
                if board[board_row][board_col] == 0 {
                    app.board_cursor = (board_row as u16, board_col as u16);
                    make_move(app);
                }
            }
        }
    }
}

/// Handle Blokus-specific clicks
fn handle_blokus_click(app: &mut App, col: u16, row: u16, terminal_size: Rect) {
    if app.game_status != GameStatus::InProgress || !is_current_player_human(app) {
        return;
    }

    let (board_area, piece_area, _) = app.layout_config.get_blokus_layout(terminal_size);
    
    if col >= board_area.x && col < board_area.x + board_area.width {
        // Click on board area
        handle_board_click(app, col, row, terminal_size);
    } else if col >= piece_area.x && col < piece_area.x + piece_area.width {
        // Click on piece selection area
        handle_blokus_piece_selection_click(app, col - piece_area.x, row - piece_area.y);
    }
}

/// Handle clicks in Blokus piece selection area
fn handle_blokus_piece_selection_click(app: &mut App, col: u16, row: u16) {
    // Simple implementation - toggle expansion for players
    if col <= 5 { // Click on expand/collapse indicator area
        let estimated_player = match row {
            0..=10 => 1,
            11..=20 => 2,
            21..=30 => 3,
            31..=40 => 4,
            _ => return,
        };
        
        if estimated_player >= 1 && estimated_player <= 4 {
            app.blokus_toggle_player_expand((estimated_player - 1) as usize);
        }
    }
}

/// Check if current player is human
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

/// Make a move at the current cursor position
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
