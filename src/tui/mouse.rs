//! # Mouse Module
//!
//! This module handles mouse events including clicking, dragging, and scrolling.
//! It provides a clean interface for mouse interaction with the UI.

use crate::app::{App, AppMode, GameStatus, Player};
use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::games::{gomoku::GomokuMove, connect4::Connect4Move, othello::OthelloMove, blokus::BlokusMove};
use crate::tui::layout::DragBoundary;
use crossterm::event::{MouseButton, MouseEventKind};
use ratatui::layout::{Rect, Layout, Direction, Constraint};
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
                // For Blokus, we need to manually calculate the stats area since get_blokus_layout doesn't return it
                // The Blokus layout splits vertically: 65% for main game area, 35% for bottom info
                let vertical_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
                    .split(terminal_size);
                
                let main_game_area = vertical_chunks[0];
                let bottom_info_area = vertical_chunks[1];
                
                // Bottom info area is split: 40% instructions, 60% stats
                let bottom_vertical = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
                    .split(bottom_info_area);
                
                let stats_area = bottom_vertical[1];
                
                // Get the three main areas from the top section
                let (board_area, piece_selection_area, _player_area) = app.layout_config.get_blokus_layout(main_game_area);
                
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
                
                // Check if mouse is in stats area (debug/history panels)
                if col >= stats_area.x && col < stats_area.x + stats_area.width &&
                   row >= stats_area.y && row < stats_area.y + stats_area.height {
                    // Split stats area horizontally for debug stats and move history
                    let stats_chunks = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(stats_area);
                    
                    let debug_area = stats_chunks[0];
                    
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
                    return;
                }
                
                // Only scroll piece selection panel if mouse is directly in that area
                // and not in any other specific scrollable area
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

            // Default scrolling for stats area (for non-Blokus games or as fallback)
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
        let menu_start_row = 1; // Ratatui List with borders: row 0=border, row 1=first item
        
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
            if clicked_index < 12 { // 10 settings + separator + back
                app.selected_settings_index = clicked_index;
                if app.selected_settings_index == 11 { // "Back" option
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
        handle_blokus_piece_selection_click(app, col - piece_area.x, row - piece_area.y, piece_area.width);
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
fn handle_blokus_piece_selection_click(app: &mut App, col: u16, row: u16, area_width: u16) {
    if let GameWrapper::Blokus(ref state) = app.game_wrapper {
        let current_player = app.game_wrapper.get_current_player();
        
        // IMPORTANT: Account for scrolling offset when determining click position
        let scroll_offset = app.blokus_ui_config.panel_scroll_offset;
        let absolute_row = row + scroll_offset as u16;
        
        // Get pieces for height calculations (same as rendering)
        let pieces = crate::games::blokus::get_blokus_pieces();
        
        // For debugging: log the click information
        #[cfg(debug_assertions)]
        eprintln!("=== CLICK DEBUG: visual_row={}, scroll_offset={}, absolute_row={}, col={} ===", 
                  row, scroll_offset, absolute_row, col);
        
        #[cfg(debug_assertions)]
        eprintln!("CLICK TYPE TEST: Testing what type of area this click is in...");
        
        // SIMPLIFIED APPROACH: Only handle clicks for the current player accurately
        // For other players, just handle expand/collapse clicks
        let mut content_row = 0u16;
        let pieces_per_row = 5;
        
        // Process each player section
        for player in 1..=4 {
            let is_current = player == current_player;
            let is_expanded = app.blokus_ui_config.players_expanded.get((player - 1) as usize).unwrap_or(&true);
            
            #[cfg(debug_assertions)]
            eprintln!("DEBUG Player {}: content_row={}, is_current={}, is_expanded={}", 
                      player, content_row, is_current, is_expanded);
            
            // Player header line
            if absolute_row == content_row {
                #[cfg(debug_assertions)]
                eprintln!("DEBUG: Click on player {} header", player);
                
                // Check if clicking on expand/collapse area (first few columns)
                if col <= 2 {
                    app.blokus_ui_config.toggle_player_expand((player - 1) as usize);
                }
                return;
            }
            content_row += 1;
            
            if *is_expanded {
                if is_current {
                    // CURRENT PLAYER: Handle piece selection accurately
                    let total_pieces_to_show = 21;
                    let available_pieces = state.get_available_pieces(current_player);
                    let available_set: std::collections::HashSet<usize> = available_pieces.iter().cloned().collect();
                    
                    // Process the current player's piece grid with exact rendering logic
                    // This function handles separators and invalid clicks correctly
                    #[cfg(debug_assertions)]
                    eprintln!("DEBUG: Calling try_select_piece_in_current_player_grid with area_width={}", area_width);
                    
                    match try_select_piece_in_current_player_grid(
                        absolute_row, col, &mut content_row, pieces_per_row,
                        total_pieces_to_show, &pieces, &available_set, area_width
                    ) {
                        Some(selected_piece) => {
                            app.blokus_ui_config.select_piece(selected_piece);
                            return;
                        }
                        None => {
                            // Click was within current player area but not on a valid piece
                            // (separator, border, unavailable piece, etc.) - don't select anything
                            return;
                        }
                    }
                } else {
                    // OTHER PLAYERS: Simulate their content more accurately 
                    let visible_pieces = 10;
                    let total_pieces_to_show = visible_pieces.min(21);
                    
                    // Simulate the exact same logic as rendering for other players
                    let mut other_player_content_rows = 0u16;
                    
                    // Top border (only if total_pieces_to_show > 0)
                    if total_pieces_to_show > 0 {
                        other_player_content_rows += 1; // top border line
                    }
                    
                    // Process each chunk of pieces (same as rendering)
                    for chunk_start in (0..total_pieces_to_show).step_by(pieces_per_row) {
                        let chunk_end = (chunk_start + pieces_per_row).min(total_pieces_to_show);
                        
                        // Calculate max height for this chunk
                        let mut max_height = 1;
                        for display_idx in chunk_start..chunk_end {
                            if display_idx < pieces.len() && !pieces[display_idx].transformations.is_empty() {
                                let piece_shape = &pieces[display_idx].transformations[0];
                                let piece_visual_lines = create_visual_piece_shape(piece_shape);
                                max_height = max_height.max(piece_visual_lines.len());
                            }
                        }
                        
                        // Key/name line
                        other_player_content_rows += 1;
                        
                        // Shape lines
                        other_player_content_rows += max_height as u16;
                        
                        // Row separator (if not last chunk)
                        if chunk_start + pieces_per_row < total_pieces_to_show {
                            other_player_content_rows += 1;
                        }
                    }
                    
                    // Bottom border (only for current player in rendering, but let's be safe)
                    // Actually, looking at the rendering code, bottom border is only for current player
                    // So we don't add it here
                    
                    #[cfg(debug_assertions)]
                    eprintln!("DEBUG: Other player {} simulated content rows: {}", player, other_player_content_rows);
                    
                    // Check if click is within this player's content area
                    if absolute_row >= content_row && absolute_row < content_row + other_player_content_rows {
                        #[cfg(debug_assertions)]
                        eprintln!("DEBUG: Click consumed by other player {} content", player);
                        return;
                    }
                    
                    content_row += other_player_content_rows;
                }
            } else {
                // Collapsed player - just the summary line
                if absolute_row == content_row {
                    #[cfg(debug_assertions)]
                    eprintln!("DEBUG: Click on collapsed player {} summary", player);
                    return;
                }
                content_row += 1;
            }
            
            // Separator between players (empty line)
            if player < 4 {
                content_row += 1;
            }
        }
        
        #[cfg(debug_assertions)]
        eprintln!("DEBUG: Click not handled - absolute_row={}", absolute_row);
    }
}

/// Try to select a piece in the current player's grid
/// 
/// Expected behavior:
/// - Each piece spans multiple rows (key line + shape lines) and should be clickable in ALL those rows
/// - Vertical separators (│) between pieces should NOT be clickable
/// - Horizontal separators (├─────┤) between rows of pieces should NOT be clickable  
/// - Borders around the grid should NOT be clickable
/// - Clicks on unavailable pieces should return None but still consume the click
fn try_select_piece_in_current_player_grid(
    absolute_row: u16,
    col: u16,
    content_row: &mut u16,
    pieces_per_row: usize,
    total_pieces_to_show: usize,
    pieces: &[crate::games::blokus::Piece],
    available_set: &std::collections::HashSet<usize>,
    area_width: u16,
) -> Option<usize> {
    #[cfg(debug_assertions)]
    eprintln!("DEBUG: try_select_piece_in_current_player_grid called with absolute_row={}, col={}, pieces_per_row={}, total_pieces_to_show={}, area_width={}", 
              absolute_row, col, pieces_per_row, total_pieces_to_show, area_width);
              
    // Top border line
    if total_pieces_to_show > 0 {
        if absolute_row == *content_row {
            return None; // Click on border
        }
        *content_row += 1;
    }
    
    // Calculate grid dimensions - use the actual area width from rendering
    let piece_width = 7;
    let separator_width = 1;
    let content_width = pieces_per_row * piece_width + (pieces_per_row - 1) * separator_width;
    let total_grid_width = content_width + 2; // +2 for left and right borders
    
    // Use the actual area_width from the rendering to calculate padding exactly
    let available_width = area_width as usize;
    let padding = if available_width > total_grid_width { 
        (available_width - total_grid_width) / 2 
    } else { 
        0 
    };
    
    // The grid content starts after padding + left border
    let grid_content_start_col = padding + 1;
    
    #[cfg(debug_assertions)]
    eprintln!("DEBUG: Grid params - piece_width={}, content_width={}, padding={}, grid_start={}", 
              piece_width, content_width, padding, grid_content_start_col);
    
    #[cfg(debug_assertions)]
    eprintln!("DEBUG: Click at absolute_row={}, col={}", absolute_row, col);
    
    #[cfg(debug_assertions)]
    eprintln!("DEBUG: Starting chunk processing for {} pieces", total_pieces_to_show);
    
    // Process each row of pieces using direct coordinate mapping
    for chunk_start in (0..total_pieces_to_show).step_by(pieces_per_row) {
        let chunk_end = (chunk_start + pieces_per_row).min(total_pieces_to_show);
        
        #[cfg(debug_assertions)]
        eprintln!("DEBUG: Processing chunk {}-{}, current content_row={}", 
                  chunk_start, chunk_end-1, *content_row);
        
        // Calculate max height for this chunk - MUST match the rendering logic exactly
        let mut pieces_in_row_visual_lines = Vec::new();
        for display_idx in chunk_start..chunk_end {
            if display_idx < pieces.len() && !pieces[display_idx].transformations.is_empty() {
                let piece_shape = &pieces[display_idx].transformations[0];
                let piece_visual_lines = create_visual_piece_shape(piece_shape);
                pieces_in_row_visual_lines.push(piece_visual_lines);
            }
        }
        
        // Calculate max_height exactly like the rendering code
        let max_height = pieces_in_row_visual_lines.iter()
            .map(|lines| lines.len())
            .max()
            .unwrap_or(1);
        
        #[cfg(debug_assertions)]
        eprintln!("DEBUG: Calculated max_height={} for chunk {}-{}, visual_lines_count={}", 
                  max_height, chunk_start, chunk_end-1, pieces_in_row_visual_lines.len());
        
        #[cfg(debug_assertions)]
        for (i, lines) in pieces_in_row_visual_lines.iter().enumerate() {
            eprintln!("DEBUG: Piece {} has {} visual lines", chunk_start + i, lines.len());
        }
        
        // Calculate the row ranges for this chunk
        let chunk_start_row = *content_row;
        // Each chunk has: 1 name line + max_height shape lines + 1 separator line (if not last chunk)
        let has_separator = chunk_start + pieces_per_row < total_pieces_to_show;
        let chunk_total_rows = 1 + max_height + if has_separator { 1 } else { 0 };
        let chunk_end_row = chunk_start_row + chunk_total_rows as u16;
        
        // Define the clickable area - be more generous to improve user experience
        // Include name line + ALL shape lines + one extra row for visual tolerance
        let clickable_start_row = chunk_start_row;
        let clickable_end_row = chunk_start_row + (1 + max_height + 1) as u16; // name + shape + 1 extra
        
        #[cfg(debug_assertions)]
        eprintln!("DEBUG: Chunk rows {}-{} (total: {}), clickable: {}-{}, has_separator={}", 
                  chunk_start_row, chunk_end_row-1, chunk_total_rows, 
                  clickable_start_row, clickable_end_row-1, has_separator);
        
        // Check if click is within this chunk's expanded clickable area
        if absolute_row >= clickable_start_row && absolute_row < clickable_end_row {
            #[cfg(debug_assertions)]
            eprintln!("DEBUG: Click is within chunk row range - processing column detection");
            
            // Click is within this chunk - check column position
            if col >= grid_content_start_col as u16 && col < (grid_content_start_col + content_width) as u16 {
                let grid_col = col - grid_content_start_col as u16;
                
                #[cfg(debug_assertions)]
                eprintln!("DEBUG: Click is within grid column range - grid_col={}, content_width={}", grid_col, content_width);
                
                // Calculate which piece column this click corresponds to
                let pieces_in_chunk = chunk_end - chunk_start;
                
                #[cfg(debug_assertions)]
                eprintln!("DEBUG: Processing {} pieces in chunk", pieces_in_chunk);
                
                // Use the exact same layout as the rendering:
                // Each piece takes piece_width characters, followed by separator_width
                // Grid structure: [piece0][sep][piece1][sep][piece2][sep][piece3][sep][piece4]
                //                  0-6    7   8-14   15  16-22  23  24-30  31  32-38
                
                for piece_col in 0..pieces_in_chunk {
                    // Calculate the exact column range for this piece
                    let piece_start_col = piece_col * (piece_width + separator_width);
                    let piece_end_col = piece_start_col + piece_width;
                    
                    #[cfg(debug_assertions)]
                    eprintln!("DEBUG: Piece {} range: {}-{}, grid_col={}", 
                              piece_col, piece_start_col, piece_end_col-1, grid_col);
                    
                    // Check if click is within this piece's column range (excluding separator)
                    if grid_col >= piece_start_col as u16 && grid_col < piece_end_col as u16 {
                        let piece_idx = chunk_start + piece_col;
                        
                        #[cfg(debug_assertions)]
                        eprintln!("DEBUG: Hit piece {} (available: {})", 
                                  piece_idx, available_set.contains(&piece_idx));
                        
                        if available_set.contains(&piece_idx) {
                            #[cfg(debug_assertions)]
                            eprintln!("DEBUG: Returning Some({})", piece_idx);
                            return Some(piece_idx);
                        } else {
                            // Piece exists but not available
                            #[cfg(debug_assertions)]
                            eprintln!("DEBUG: Piece {} not available, returning None", piece_idx);
                            return None;
                        }
                    }
                }
                
                #[cfg(debug_assertions)]
                eprintln!("DEBUG: Click on vertical separator within chunk - not selecting anything");
                // Click was within the grid but on a vertical separator - return None
                return None;
            }
            // Click was in chunk row range but outside the grid - return None
            #[cfg(debug_assertions)]
            eprintln!("DEBUG: Click outside grid area within chunk - col={}, valid range {}-{}", 
                      col, grid_content_start_col, grid_content_start_col + content_width);
            return None;
        } else {
            #[cfg(debug_assertions)]
            eprintln!("DEBUG: Click NOT within expanded clickable area - absolute_row={}, clickable range {}-{}", 
                      absolute_row, clickable_start_row, clickable_end_row-1);
        }
        
        // Update content_row to after this chunk (including separator if present)
        *content_row = chunk_end_row;
        
        #[cfg(debug_assertions)]
        eprintln!("DEBUG: Updated content_row to {} after chunk", *content_row);
    }
    
    // Bottom border
    if total_pieces_to_show > 0 {
        if absolute_row == *content_row {
            return None; // Click on border
        }
        *content_row += 1;
    }
    
    None
}

/// Create visual piece shape (helper function to match rendering logic)
fn create_visual_piece_shape(piece_shape: &[(i32, i32)]) -> Vec<String> {
    if piece_shape.is_empty() {
        return vec!["".to_string()];
    }
    
    // Find bounds
    let min_row = piece_shape.iter().map(|(r, _)| *r).min().unwrap_or(0);
    let max_row = piece_shape.iter().map(|(r, _)| *r).max().unwrap_or(0);
    let min_col = piece_shape.iter().map(|(_, c)| *c).min().unwrap_or(0);
    let max_col = piece_shape.iter().map(|(_, c)| *c).max().unwrap_or(0);
    
    let height = (max_row - min_row + 1) as usize;
    let width = (max_col - min_col + 1) as usize;
    
    let mut lines = Vec::new();
    for row in 0..height {
        let mut line = String::new();
        for col in 0..width {
            let absolute_row = row as i32 + min_row;
            let absolute_col = col as i32 + min_col;
            if piece_shape.contains(&(absolute_row, absolute_col)) {
                line.push('█');
            } else {
                line.push(' ');
            }
        }
        lines.push(line);
    }
    
    lines
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
