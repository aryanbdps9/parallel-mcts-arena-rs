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
        MouseEventKind::Down(MouseButton::Right) => {
            handle_mouse_right_click(app, col, row, terminal_size);
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

/// Handle right mouse click events
fn handle_mouse_right_click(app: &mut App, _col: u16, _row: u16, _terminal_size: Rect) {
    // Right-click in Blokus to rotate selected piece
    if matches!(app.game_wrapper, GameWrapper::Blokus(_)) && app.mode == AppMode::InGame {
        if let Some((piece_idx, _)) = app.blokus_ui_config.get_selected_piece_info() {
            // Get the number of transformations for this piece
            let pieces = crate::games::blokus::get_blokus_pieces();
            if piece_idx < pieces.len() {
                let total_transformations = pieces[piece_idx].transformations.len();
                app.blokus_ui_config.rotate_piece(total_transformations);
            }
        }
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
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                handle_blokus_click(app, col, row, terminal_size);
            } else {
                handle_board_click(app, col, row, terminal_size);
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
            // Special handling for Blokus
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                let (board_area, piece_selection_area, _) = app.layout_config.get_blokus_layout(terminal_size);
                
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
                
                // Check if mouse is over board area and a piece is selected - rotate piece
                if col >= board_area.x && col < board_area.x + board_area.width &&
                   row >= board_area.y && row < board_area.y + board_area.height {
                    if let Some((piece_idx, _)) = app.blokus_ui_config.get_selected_piece_info() {
                        // Get the number of transformations for this piece
                        let pieces = crate::games::blokus::get_blokus_pieces();
                        if piece_idx < pieces.len() {
                            let total_transformations = pieces[piece_idx].transformations.len();
                            if scroll_up {
                                app.blokus_ui_config.rotate_piece(total_transformations);
                            } else {
                                // Rotate in reverse direction
                                for _ in 0..(total_transformations - 1) {
                                    app.blokus_ui_config.rotate_piece(total_transformations);
                                }
                            }
                        }
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

    let (board_area, _, _) = app.layout_config.get_main_layout(terminal_size);
    
    // Check if click is within the board area
    if row >= board_area.y && row < board_area.y + board_area.height &&
       col >= board_area.x && col < board_area.x + board_area.width {

        // Convert to relative coordinates within the board area
        let relative_col = col - board_area.x;
        let relative_row = row - board_area.y;
        
        // Account for the border around the board area (1 character on all sides)
        // The actual board content is rendered in the inner area
        if relative_col == 0 || relative_row == 0 || 
           relative_col >= board_area.width - 1 || relative_row >= board_area.height - 1 {
            return; // Click is on the border, ignore
        }
        
        // Adjust for the border offset
        let inner_col = relative_col - 1;
        let inner_row = relative_row - 1;
        
        let (board_height, board_width) = {
            let board = app.game_wrapper.get_board();
            (board.len(), if board.len() > 0 { board[0].len() } else { 0 })
        };

        // Handle Connect4 column label clicks (first row in inner area)
        if matches!(app.game_wrapper, GameWrapper::Connect4(_)) && inner_row == 0 { // First row is column labels
            let col_width = 2; // Match draw_standard_board logic
            let board_col = (inner_col / col_width) as usize;
            
            if board_col < board_width {
                // Update cursor to this column
                app.board_cursor.1 = board_col as u16;
                update_connect4_cursor_row(app);
                
                // Make the move immediately
                let player_move = MoveWrapper::Connect4(Connect4Move(board_col));
                if app.game_wrapper.is_legal(&player_move) {
                    make_move_with_move(app, player_move);
                }
            }
            return;
        }

        // For Connect4, ignore clicks on the board itself (only column labels are clickable)
        if matches!(app.game_wrapper, GameWrapper::Connect4(_)) {
            return;
        }

        // Calculate board cell from click position for Gomoku/Othello
        // Account for borders and labels
        let needs_row_labels = !matches!(app.game_wrapper, GameWrapper::Connect4(_));
        let row_label_width = if needs_row_labels { 2 } else { 0 };
        
        // Skip the column header row
        if inner_row == 0 {
            return;
        }
        
        if inner_col >= row_label_width && inner_row >= 1 {
            // Use the same layout logic as the board rendering to get accurate coordinates
            let col_width = match &app.game_wrapper {
                GameWrapper::Othello(_) => 2,  
                _ => 2, // Standard width for X/O
            };
            
            // Calculate actual board position using the same logic as rendering
            // The key insight: in rendering, board cells start at cell_areas[1] for Gomoku/Othello
            // So when calculating from mouse coordinates, we need to account for this offset
            let adjusted_col = inner_col - row_label_width;
            let board_col = (adjusted_col / col_width) as usize;
            let board_row = (inner_row - 1) as usize; // -1 to account for column header row
            
            if board_row < board_height && board_col < board_width {
                // Update cursor position
                app.board_cursor = (board_row as u16, board_col as u16);
                
                // Check if move is legal before making it
                let player_move = {
                    let board = app.game_wrapper.get_board();
                    match &app.game_wrapper {
                        GameWrapper::Gomoku(_) => {
                            if board[board_row][board_col] == 0 {
                                Some(MoveWrapper::Gomoku(GomokuMove(board_row, board_col)))
                            } else {
                                None
                            }
                        },
                        GameWrapper::Othello(_) => {
                            Some(MoveWrapper::Othello(OthelloMove(board_row, board_col)))
                        },
                        GameWrapper::Blokus(_) => {
                            // Blokus handled separately
                            None
                        }
                        GameWrapper::Connect4(_) => {
                            // Already handled above
                            None
                        }
                    }
                };
                
                if let Some(mv) = player_move {
                    if app.game_wrapper.is_legal(&mv) {
                        make_move_with_move(app, mv);
                    }
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
    
    if col >= board_area.x && col < board_area.x + board_area.width &&
       row >= board_area.y && row < board_area.y + board_area.height {
        // Click on board area - place piece or update cursor
        handle_blokus_board_click(app, col - board_area.x, row - board_area.y, board_area);
    } else if col >= piece_area.x && col < piece_area.x + piece_area.width &&
              row >= piece_area.y && row < piece_area.y + piece_area.height {
        // Click on piece selection area
        handle_blokus_piece_selection_click(app, col - piece_area.x, row - piece_area.y);
    }
}

/// Handle clicks on the Blokus board
fn handle_blokus_board_click(app: &mut App, col: u16, row: u16, _board_area: Rect) {
    let board = app.game_wrapper.get_board();
    let board_height = board.len();
    let board_width = if board_height > 0 { board[0].len() } else { 0 };
    
    // Calculate board cell from click position
    let board_start_col = 1; // Border
    let board_start_row = 1; // Border
    
    if col >= board_start_col && row >= board_start_row {
        // Each board cell is rendered as 2 characters wide (██, ▓▓, ░░/▒▒)
        let cell_width = 2;
        let cell_height = 1;
        
        // Calculate which board cell was clicked
        let board_col = ((col - board_start_col) / cell_width) as usize;
        let board_row = ((row - board_start_row) / cell_height) as usize;
        
        if board_row < board_height && board_col < board_width {
            // Update cursor position
            app.board_cursor = (board_row as u16, board_col as u16);
            
            // If a piece is selected, try to place it
            if let Some((piece_idx, transformation_idx)) = app.blokus_ui_config.get_selected_piece_info() {
                let player_move = MoveWrapper::Blokus(BlokusMove(
                    piece_idx, 
                    transformation_idx, 
                    board_row, 
                    board_col
                ));
                
                if app.game_wrapper.is_legal(&player_move) {
                    make_move_with_move(app, player_move);
                    // Deselect piece after successful placement
                    app.blokus_ui_config.selected_piece_idx = None;
                }
            }
        }
    }
}

/// Handle clicks in Blokus piece selection area
fn handle_blokus_piece_selection_click(app: &mut App, col: u16, row: u16) {
    let current_player = app.game_wrapper.get_current_player();
    
    // Check if clicking on expand/collapse area (only the very first few columns)
    if col <= 1 { // Reduced from 3 to 1 to only catch the very left edge
        // Simple row-based calculation for player headers
        // Each player header is roughly 1 row + content, but we'll use a simple approximation
        let estimated_rows_per_player = 20; // Generous estimate
        let clicked_player = (row / estimated_rows_per_player).min(3) as usize;
        
        if clicked_player < 4 {
            app.blokus_ui_config.toggle_player_expand(clicked_player);
        }
        return;
    }
    
    // Only handle piece selection for the current player
    if let GameWrapper::Blokus(ref state) = app.game_wrapper {
        let available_pieces = state.get_available_pieces(current_player);
        let available_set: std::collections::HashSet<usize> = available_pieces.iter().cloned().collect();
        
        // Very precise column detection based on exact rendering layout
        let pieces_per_row = 5;
        let piece_width = 7;     // Each piece is now 7 characters wide (for better centering)
        let separator_width = 1; // 1 character between pieces (│ separator)
        
        // Calculate the grid positioning (same logic as rendering)
        let content_width = pieces_per_row * piece_width + (pieces_per_row - 1) * separator_width; // 7*5 + 4*1 = 39
        let total_grid_width = content_width + 2; // +2 for left and right borders
        let panel_width = 50; // Approximate panel width, adjust if needed
        let grid_padding = if panel_width > total_grid_width { (panel_width - total_grid_width) / 2 } else { 0 };
        let grid_start_col = grid_padding + 1; // +1 for left border
        
        // Check if click is within the grid area
        if col < grid_start_col as u16 {
            return; // Click is in padding area, ignore
        }
        
        let grid_relative_col = col - grid_start_col as u16;
        
        // Find which piece column was clicked (0-4)
        let mut piece_col = None;
        let mut col_start = 0u16;
        
        for i in 0..pieces_per_row {
            let piece_end = col_start + piece_width as u16;
            
            if grid_relative_col >= col_start && grid_relative_col < piece_end {
                piece_col = Some(i);
                break;
            }
            
            col_start = piece_end + separator_width as u16;
        }
        
        if let Some(col_idx) = piece_col {
            // Account for the top border before calculating row
            let grid_content_start_row = 1; // Top border takes 1 row
            
            // Only process clicks that are after the top border
            if row < grid_content_start_row {
                return; // Click is on the top border, ignore
            }
            
            // Adjust row to be relative to the actual grid content
            let grid_relative_row = row - grid_content_start_row;
            
            // Each piece row consists of approximately:
            // - 1 line for piece keys/names
            // - ~3 lines for piece shapes (varies by piece height)
            // - 1 line for row separator (except for last row)
            // So roughly 5 lines per piece row (including separator)
            let estimated_lines_per_piece_row = 5;
            
            // Calculate piece row, accounting for row separators
            let piece_row = (grid_relative_row / estimated_lines_per_piece_row) as usize;
            let piece_row = piece_row.min(4); // Clamp to 0-4 (5 rows max)
            
            // Calculate piece index
            let piece_index = piece_row * pieces_per_row + col_idx as usize;
            
            // Final validation
            if piece_index < 21 && available_set.contains(&piece_index) {
                app.blokus_ui_config.select_piece(piece_index);
            }
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

/// Update Connect4 cursor to lowest available position in column
fn update_connect4_cursor_row(app: &mut App) {
    let board = app.game_wrapper.get_board();
    let board_height = board.len();
    let col = app.board_cursor.1 as usize;
    
    if col < board[0].len() {
        // Find the lowest available row in this column
        for row in (0..board_height).rev() {
            if board[row][col] == 0 {
                app.board_cursor.0 = row as u16;
                break;
            }
        }
    }
}

/// Make a move with a specific move
fn make_move_with_move(app: &mut App, player_move: MoveWrapper) {
    let current_player = app.game_wrapper.get_current_player();
    app.move_history.push(crate::app::MoveHistoryEntry::new(current_player, player_move.clone()));
    app.on_move_added(); // Auto-scroll to bottom
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
