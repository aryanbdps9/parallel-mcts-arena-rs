//! In-game component.

use ratatui::{
    layout::Rect,
    Frame,
};

use crate::app::{App, AppMode, GameStatus, Player};
use crate::game_wrapper::{GameWrapper, MoveWrapper};
use crate::games::{gomoku::GomokuMove, connect4::Connect4Move, othello::OthelloMove, blokus::BlokusMove};
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crossterm::event::KeyCode;
use mcts::GameState;

/// Component for in-game view
pub struct InGameComponent {
    id: ComponentId,
}

impl InGameComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
        }
    }

    /// Checks if the current player is human (vs AI)
    fn is_current_player_human(&self, app: &App) -> bool {
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

    /// Attempts to make a move at the current cursor position
    fn make_move(&self, app: &mut App) {
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

    /// Moves the board cursor up
    fn move_cursor_up(&self, app: &mut App) {
        // Connect4 uses column-based navigation only
        if matches!(app.game_wrapper, GameWrapper::Connect4(_)) {
            return;
        }

        if app.board_cursor.0 > 0 {
            let new_row = app.board_cursor.0 - 1;
            // For Blokus, check if the selected piece would fit at the new position
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                if self.would_blokus_piece_fit(app, new_row, app.board_cursor.1) {
                    app.board_cursor.0 = new_row;
                }
            } else {
                app.board_cursor.0 = new_row;
            }
        }
    }

    /// Moves the board cursor down
    fn move_cursor_down(&self, app: &mut App) {
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
                if self.would_blokus_piece_fit(app, new_row, app.board_cursor.1) {
                    app.board_cursor.0 = new_row;
                }
            } else {
                app.board_cursor.0 = new_row;
            }
        }
    }

    /// Moves the board cursor left
    fn move_cursor_left(&self, app: &mut App) {
        if app.board_cursor.1 > 0 {
            let new_col = app.board_cursor.1 - 1;
            // For Blokus, check if the selected piece would fit at the new position
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                if self.would_blokus_piece_fit(app, app.board_cursor.0, new_col) {
                    app.board_cursor.1 = new_col;
                }
            } else {
                app.board_cursor.1 = new_col;
                // For Connect4, update cursor to lowest available position in new column
                if let GameWrapper::Connect4(_) = app.game_wrapper {
                    self.update_connect4_cursor_row(app);
                }
            }
        }
    }

    /// Moves the board cursor right
    fn move_cursor_right(&self, app: &mut App) {
        let board = app.game_wrapper.get_board();
        let max_col = if !board.is_empty() { board[0].len() as u16 } else { 0 };
        if app.board_cursor.1 < max_col.saturating_sub(1) {
            let new_col = app.board_cursor.1 + 1;
            // For Blokus, check if the selected piece would fit at the new position
            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                if self.would_blokus_piece_fit(app, app.board_cursor.0, new_col) {
                    app.board_cursor.1 = new_col;
                }
            } else {
                app.board_cursor.1 = new_col;
                // For Connect4, update cursor to lowest available position in new column
                if let GameWrapper::Connect4(_) = app.game_wrapper {
                    self.update_connect4_cursor_row(app);
                }
            }
        }
    }

    /// Updates the Connect4 cursor to the lowest available position in the current column
    fn update_connect4_cursor_row(&self, app: &mut App) {
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

    /// Check if a Blokus piece would fit within board bounds at the given position
    fn would_blokus_piece_fit(&self, app: &App, new_row: u16, new_col: u16) -> bool {
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
            let pieces = crate::games::blokus::get_blokus_pieces();
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

    /// Render the game board based on the current game type
    fn render_game_board(&self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        use ratatui::{
            widgets::{Block, Borders, Paragraph},
            style::{Style, Color},
            text::{Line, Span},
        };
        
        let board = app.game_wrapper.get_board();
        let (cursor_row, cursor_col) = (app.board_cursor.0 as usize, app.board_cursor.1 as usize);
        
        // Create board representation
        let mut lines = Vec::new();
        
        match &app.game_wrapper {
            crate::game_wrapper::GameWrapper::Gomoku(_) => {
                for (row_idx, row) in board.iter().enumerate() {
                    let mut spans = Vec::new();
                    for (col_idx, &cell) in row.iter().enumerate() {
                        let symbol = match cell {
                            1 => "●",   // Player 1 (black)
                            -1 => "○",  // Player 2 (white)
                            _ => "·",   // Empty
                        };
                        
                        let style = if row_idx == cursor_row && col_idx == cursor_col {
                            Style::default().bg(Color::Yellow).fg(Color::Black)
                        } else {
                            Style::default().fg(if cell == 1 { Color::White } else if cell == -1 { Color::Gray } else { Color::DarkGray })
                        };
                        
                        spans.push(Span::styled(format!("{} ", symbol), style));
                    }
                    lines.push(Line::from(spans));
                }
            }
            crate::game_wrapper::GameWrapper::Connect4(_) => {
                for (row_idx, row) in board.iter().enumerate() {
                    let mut spans = Vec::new();
                    for (col_idx, &cell) in row.iter().enumerate() {
                        let symbol = match cell {
                            1 => "●",   // Player 1 (red)
                            -1 => "●",  // Player 2 (yellow)
                            _ => "·",   // Empty
                        };
                        
                        let style = if row_idx == cursor_row && col_idx == cursor_col {
                            Style::default().bg(Color::Yellow).fg(Color::Black)
                        } else {
                            match cell {
                                1 => Style::default().fg(Color::Red),
                                -1 => Style::default().fg(Color::Yellow),
                                _ => Style::default().fg(Color::DarkGray),
                            }
                        };
                        
                        spans.push(Span::styled(format!("{} ", symbol), style));
                    }
                    lines.push(Line::from(spans));
                }
            }
            crate::game_wrapper::GameWrapper::Othello(_) => {
                for (row_idx, row) in board.iter().enumerate() {
                    let mut spans = Vec::new();
                    for (col_idx, &cell) in row.iter().enumerate() {
                        let symbol = match cell {
                            1 => "●",   // Player 1 (black)
                            -1 => "○",  // Player 2 (white)
                            _ => "·",   // Empty
                        };
                        
                        let style = if row_idx == cursor_row && col_idx == cursor_col {
                            Style::default().bg(Color::Yellow).fg(Color::Black)
                        } else {
                            Style::default().fg(if cell == 1 { Color::White } else if cell == -1 { Color::Gray } else { Color::DarkGray })
                        };
                        
                        spans.push(Span::styled(format!("{} ", symbol), style));
                    }
                    lines.push(Line::from(spans));
                }
            }
            crate::game_wrapper::GameWrapper::Blokus(_) => {
                // Blokus has a more complex rendering
                for (row_idx, row) in board.iter().enumerate() {
                    let mut spans = Vec::new();
                    for (col_idx, &cell) in row.iter().enumerate() {
                        let symbol = match cell {
                            1 => "■",   // Player 1
                            2 => "■",   // Player 2
                            3 => "■",   // Player 3
                            4 => "■",   // Player 4
                            _ => "·",   // Empty
                        };
                        
                        let style = if row_idx == cursor_row && col_idx == cursor_col {
                            Style::default().bg(Color::Yellow).fg(Color::Black)
                        } else {
                            match cell {
                                1 => Style::default().fg(Color::Red),
                                2 => Style::default().fg(Color::Blue),
                                3 => Style::default().fg(Color::Green),
                                4 => Style::default().fg(Color::Cyan),
                                _ => Style::default().fg(Color::DarkGray),
                            }
                        };
                        
                        spans.push(Span::styled(format!("{}", symbol), style));
                    }
                    lines.push(Line::from(spans));
                }
            }
        }
        
        let game_name = match &app.game_wrapper {
            crate::game_wrapper::GameWrapper::Gomoku(_) => "Gomoku",
            crate::game_wrapper::GameWrapper::Connect4(_) => "Connect 4",
            crate::game_wrapper::GameWrapper::Othello(_) => "Othello",
            crate::game_wrapper::GameWrapper::Blokus(_) => "Blokus",
        };
        
        let board_widget = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title(format!("{} Board", game_name)))
            .style(Style::default().fg(Color::White));
            
        frame.render_widget(board_widget, area);
        Ok(())
    }
    
    /// Render the side panel with stats and move history
    fn render_side_panel(&self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        use ratatui::{
            layout::{Constraint, Direction, Layout},
            widgets::{Block, Borders, Paragraph, List, ListItem},
            style::{Style, Color},
            text::Line,
        };
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),  // AI Stats
                Constraint::Min(0),     // Move History
            ])
            .split(area);

        // AI Stats
        let mut stats_lines = vec![
            Line::from("AI Statistics:"),
            Line::from(""),
        ];
        
        if let Some(stats) = &app.last_search_stats {
            stats_lines.extend(vec![
                Line::from(format!("Total Nodes: {}", stats.total_nodes)),
                Line::from(format!("Root Visits: {}", stats.root_visits)),
                Line::from(format!("Root Value: {:.3}", stats.root_value)),
            ]);
        } else {
            stats_lines.push(Line::from("No stats available"));
        }
        
        let stats_widget = Paragraph::new(stats_lines)
            .block(Block::default().borders(Borders::ALL).title("Stats"))
            .style(Style::default().fg(Color::Cyan));
        frame.render_widget(stats_widget, chunks[0]);

        // Move History
        let history_items: Vec<ListItem> = app.move_history
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let move_str = format!("{}. P{}: {:?}", i + 1, entry.player, entry.a_move);
                ListItem::new(move_str)
            })
            .collect();

        let history_widget = List::new(history_items)
            .block(Block::default().borders(Borders::ALL).title("Move History"))
            .style(Style::default().fg(Color::White));
        frame.render_widget(history_widget, chunks[1]);

        Ok(())
    }
}

impl Component for InGameComponent {
    fn id(&self) -> ComponentId {
        self.id
    }
    
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        use ratatui::{
            layout::{Constraint, Direction, Layout},
            widgets::{Block, Borders, Paragraph},
            style::{Style, Color, Modifier},
        };
        
        // Main layout: top status bar, then game area
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Status bar
                Constraint::Min(0),    // Game area
            ])
            .split(area);

        // Status bar
        let current_player = app.game_wrapper.get_current_player();
        let player_type = if self.is_current_player_human(app) { "Human" } else { "AI" };
        let status_text = match app.game_status {
            GameStatus::InProgress => {
                if self.is_current_player_human(app) {
                    format!("Player {}: {} - Your turn! Use arrow keys to move, Enter/Space to place", current_player, player_type)
                } else {
                    format!("Player {}: {} - AI is thinking...", current_player, player_type)
                }
            }
            GameStatus::Win(winner) => format!("🎉 Player {} wins! Press R to restart, ESC for menu", winner),
            GameStatus::Draw => "🤝 It's a draw! Press R to restart, ESC for menu".to_string(),
        };
        
        let status_style = match app.game_status {
            GameStatus::InProgress => Style::default().fg(Color::Green),
            GameStatus::Win(_) => Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            GameStatus::Draw => Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
        };
        
        let status = Paragraph::new(status_text)
            .block(Block::default().borders(Borders::ALL).title("Game Status"))
            .style(status_style);
        frame.render_widget(status, chunks[0]);

        // Game area layout
        let game_area = chunks[1];
        let game_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(20),   // Game board
                Constraint::Length(30), // Side panel (stats/history)
            ])
            .split(game_area);

        // Render the game board
        self.render_game_board(frame, game_chunks[0], app)?;
        
        // Render side panel
        self.render_side_panel(frame, game_chunks[1], app)?;

        Ok(())
    }
    
    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        if app.game_status != GameStatus::InProgress {
            return Ok(false);
        }

        // Only allow human player input
        if !self.is_current_player_human(app) {
            match event {
                ComponentEvent::Input(InputEvent::KeyPress(key)) => {
                    match key {
                        KeyCode::Char('q') => {
                            app.should_quit = true;
                            Ok(true)
                        }
                        KeyCode::Char('r') => {
                            app.reset_game();
                            Ok(true)
                        }
                        KeyCode::Esc => {
                            app.mode = AppMode::GameSelection;
                            Ok(true)
                        }
                        _ => Ok(false)
                    }
                }
                _ => Ok(false),
            }
        } else {
            match event {
                ComponentEvent::Input(InputEvent::KeyPress(key)) => {
                    match key {
                        KeyCode::Char('q') => {
                            app.should_quit = true;
                            Ok(true)
                        }
                        KeyCode::Char('r') => {
                            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) && self.is_current_player_human(app) {
                                app.blokus_rotate_piece();
                            } else {
                                app.reset_game();
                            }
                            Ok(true)
                        }
                        KeyCode::Esc => {
                            app.mode = AppMode::GameSelection;
                            Ok(true)
                        }
                        KeyCode::Up => {
                            self.move_cursor_up(app);
                            Ok(true)
                        }
                        KeyCode::Down => {
                            self.move_cursor_down(app);
                            Ok(true)
                        }
                        KeyCode::Left => {
                            self.move_cursor_left(app);
                            Ok(true)
                        }
                        KeyCode::Right => {
                            self.move_cursor_right(app);
                            Ok(true)
                        }
                        KeyCode::Enter | KeyCode::Char(' ') => {
                            self.make_move(app);
                            Ok(true)
                        }
                        KeyCode::PageUp => {
                            match app.active_tab {
                                crate::app::ActiveTab::Debug => app.scroll_debug_up(),
                                crate::app::ActiveTab::History => app.scroll_move_history_up(),
                            }
                            Ok(true)
                        }
                        KeyCode::PageDown => {
                            match app.active_tab {
                                crate::app::ActiveTab::Debug => app.scroll_debug_down(),
                                crate::app::ActiveTab::History => app.scroll_move_history_down(),
                            }
                            Ok(true)
                        }
                        KeyCode::Tab => {
                            app.active_tab = app.active_tab.next();
                            Ok(true)
                        }
                        KeyCode::Home => {
                            app.reset_debug_scroll();
                            Ok(true)
                        }
                        KeyCode::End => {
                            app.enable_history_auto_scroll();
                            Ok(true)
                        }
                        // Blokus-specific keys
                        KeyCode::Char('f') => {
                            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                                app.blokus_select_piece(15);
                            }
                            Ok(true)
                        }
                        KeyCode::Char('p') => {
                            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                                app.blokus_pass_move();
                            }
                            Ok(true)
                        }
                        KeyCode::Char('e') => {
                            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                                app.blokus_select_piece(14);
                            }
                            Ok(true)
                        }
                        KeyCode::Char('+') | KeyCode::Char('=') => {
                            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                                app.blokus_expand_all();
                            }
                            Ok(true)
                        }
                        KeyCode::Char('-') => {
                            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                                app.blokus_collapse_all();
                            }
                            Ok(true)
                        }
                        KeyCode::Char('x') => {
                            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                                app.blokus_flip_piece();
                            }
                            Ok(true)
                        }
                        KeyCode::Char('z') => {
                            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                                let current_player = app.game_wrapper.get_current_player();
                                app.blokus_toggle_player_expand((current_player - 1) as usize);
                            }
                            Ok(true)
                        }
                        // Piece selection keys
                        KeyCode::Char(c) => {
                            if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                                // Map characters to piece indices
                                let piece_index = match *c {
                                    '1'..='9' => Some((*c as u8 - b'1') as usize),
                                    '0' => Some(9),
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
                            Ok(true)
                        }
                        _ => Ok(false)
                    }
                }
                _ => Ok(false),
            }
        }
    }

    crate::impl_component_base!(InGameComponent);
}
