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
        !app.is_current_player_ai()
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
            style::{Style, Color, Modifier},
            text::{Line, Span},
            layout::{Constraint, Direction, Layout, Alignment},
        };
        
        let block = Block::default().borders(Borders::ALL).title(format!("{} Board", 
            match &app.game_wrapper {
                crate::game_wrapper::GameWrapper::Gomoku(_) => "Gomoku",
                crate::game_wrapper::GameWrapper::Connect4(_) => "Connect 4",
                crate::game_wrapper::GameWrapper::Othello(_) => "Othello",
                crate::game_wrapper::GameWrapper::Blokus(_) => "Blokus",
            }));
        let inner_area = block.inner(area);
        frame.render_widget(block, area);
        
        let board = app.game_wrapper.get_board();
        let board_height = board.len();
        let board_width = if board_height > 0 { board[0].len() } else { 0 };
        
        if board_height == 0 || board_width == 0 {
            let paragraph = Paragraph::new("No board to display");
            frame.render_widget(paragraph, inner_area);
            return Ok(());
        }
        
        // Handle Blokus differently (no labels for now, more complex rendering)
        if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
            return self.render_blokus_board(frame, inner_area, app);
        }
        
        // Calculate column width and determine if we need row labels
        let col_width = match &app.game_wrapper {
            GameWrapper::Connect4(_) => 2,
            GameWrapper::Othello(_) => 2,
            _ => 2, // Gomoku
        };
        
        let needs_row_labels = !matches!(app.game_wrapper, GameWrapper::Connect4(_));
        let row_label_width = if needs_row_labels { 2 } else { 0 };
        
        // Create layout with space for labels
        let mut layout_constraints = Vec::new();
        layout_constraints.push(Constraint::Length(1)); // Column header row
        for _ in 0..board_height {
            layout_constraints.push(Constraint::Length(1)); // Board rows
        }
        
        let rows_layout = Layout::default()
            .constraints(layout_constraints)
            .split(inner_area);
        
        // Draw column labels
        let col_label_constraints = if needs_row_labels {
            let mut constraints = vec![Constraint::Length(row_label_width)]; // Space for row label
            constraints.extend(vec![Constraint::Length(col_width); board_width]);
            constraints
        } else {
            vec![Constraint::Length(col_width); board_width]
        };
        
        let col_label_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(col_label_constraints)
            .split(rows_layout[0]);
        
        // Draw column labels with cursor for Connect4
        let col_start_idx = if needs_row_labels { 1 } else { 0 };
        for c in 0..board_width {
            let col_letter = char::from(b'A' + (c as u8));
            let is_cursor_col = matches!(app.game_wrapper, GameWrapper::Connect4(_)) &&
                               (c as u16) == app.board_cursor.1 &&
                               self.is_current_player_human(app);
            
            let style = if is_cursor_col {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD).bg(Color::Blue)
            } else {
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            };
            
            let paragraph = Paragraph::new(col_letter.to_string())
                .style(style)
                .alignment(Alignment::Center);
            frame.render_widget(paragraph, col_label_area[col_start_idx + c]);
        }
        
        // Draw board rows with row labels
        for (r, row) in board.iter().enumerate() {
            let row_area = rows_layout[r + 1]; // +1 because first row is column labels
            let row_constraints = if needs_row_labels {
                let mut constraints = vec![Constraint::Length(row_label_width)]; // Space for row label
                constraints.extend(vec![Constraint::Length(col_width); board_width]);
                constraints
            } else {
                vec![Constraint::Length(col_width); board_width]
            };
            
            let cell_areas = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(row_constraints)
                .split(row_area);
            
            // Draw row label if needed
            if needs_row_labels {
                let row_number = (r + 1).to_string();
                let paragraph = Paragraph::new(row_number)
                    .style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD))
                    .alignment(Alignment::Center);
                frame.render_widget(paragraph, cell_areas[0]);
            }
            
            // Draw board cells
            let cell_start_idx = if needs_row_labels { 1 } else { 0 };
            for (c, &cell) in row.iter().enumerate() {
                let is_cursor = !matches!(app.game_wrapper, GameWrapper::Connect4(_)) &&
                               (r as u16, c as u16) == app.board_cursor;
                
                let (symbol, style) = match &app.game_wrapper {
                    GameWrapper::Connect4(_) => {
                        match cell {
                            1 => ("🔴", Style::default().fg(Color::Red)),
                            -1 => ("🟡", Style::default().fg(Color::Yellow)),
                            _ => ("·", Style::default().fg(Color::DarkGray))
                        }
                    }
                    GameWrapper::Othello(_) => {
                        match cell {
                            1 => ("⚫", Style::default().fg(Color::White)),
                            -1 => ("⚪", Style::default().fg(Color::White)),
                            _ => {
                                if is_cursor && self.is_current_player_human(app) {
                                    ("▓", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                                } else {
                                    ("·", Style::default().fg(Color::DarkGray))
                                }
                            }
                        }
                    }
                    _ => { // Gomoku and others
                        match cell {
                            1 => ("X", Style::default().fg(Color::Red)),
                            -1 => ("O", Style::default().fg(Color::Blue)),
                            _ => {
                                if is_cursor && self.is_current_player_human(app) {
                                    ("▓", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                                } else {
                                    ("·", Style::default().fg(Color::DarkGray))
                                }
                            }
                        }
                    }
                };
                
                let final_style = if is_cursor && cell != 0 && self.is_current_player_human(app) {
                    style.bg(Color::Yellow)
                } else {
                    style
                };
                
                let paragraph = Paragraph::new(symbol)
                    .style(final_style)
                    .alignment(Alignment::Center);
                frame.render_widget(paragraph, cell_areas[cell_start_idx + c]);
            }
        }
        
        Ok(())
    }
    
    /// Render Blokus board (simplified version)
    fn render_blokus_board(&self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        use ratatui::{
            widgets::Paragraph,
            style::{Style, Color},
        };
        
        let board = app.game_wrapper.get_board();
        let (cursor_row, cursor_col) = (app.board_cursor.0 as usize, app.board_cursor.1 as usize);
        
        // Create board representation
        let mut lines = Vec::new();
        
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
        
        let board_widget = Paragraph::new(lines)
            .style(Style::default().fg(Color::White));
            
        frame.render_widget(board_widget, area);
        Ok(())
    }
    
    /// Render the game info panel with current game status and player information
    fn render_game_info(&self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        use ratatui::{
            widgets::{Block, Borders, Paragraph},
            style::{Style, Color, Modifier},
            text::{Line, Span},
        };
        
        let mut text = vec![
            Line::from(format!("Game: {}  |  Status: {:?}", 
                match &app.game_wrapper {
                    crate::game_wrapper::GameWrapper::Gomoku(_) => "Gomoku",
                    crate::game_wrapper::GameWrapper::Connect4(_) => "Connect 4", 
                    crate::game_wrapper::GameWrapper::Othello(_) => "Othello",
                    crate::game_wrapper::GameWrapper::Blokus(_) => "Blokus",
                }, 
                app.game_status)),
        ];
        
        // Only show current player info when game is in progress
        if app.game_status == GameStatus::InProgress {
            // Show current player
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
            let player_type = app.player_options
                .iter()
                .find(|(id, _)| *id == ui_player_id)
                .map(|(_, p_type)| p_type)
                .unwrap_or(&Player::Human);

            let current_player_text = match app.game_wrapper {
                GameWrapper::Blokus(_) => format!("Player {} ({:?})", ui_player_id, player_type),
                _ => {
                    let symbol = if ui_player_id == 1 { "X" } else { "O" };
                    format!("{} ({:?})", symbol, player_type)
                }
            };

            // Get player color to match board display
            let player_color = match &app.game_wrapper {
                GameWrapper::Connect4(_) => {
                    if ui_player_id == 1 { Color::Red } else { Color::Yellow }
                }
                GameWrapper::Othello(_) => {
                    if ui_player_id == 1 { Color::White } else { Color::White } // Both use white for contrast
                }
                GameWrapper::Blokus(_) => {
                    match ui_player_id {
                        1 => Color::Red,
                        2 => Color::Blue, 
                        3 => Color::Green,
                        4 => Color::Yellow,
                        _ => Color::White,
                    }
                }
                _ => { // Gomoku and others
                    if ui_player_id == 1 { Color::Red } else { Color::Blue }
                }
            };

            // Add current player indicator with color-coded marker
            let player_marker = match &app.game_wrapper {
                GameWrapper::Connect4(_) => {
                    if ui_player_id == 1 { "🔴" } else { "🟡" }
                }
                GameWrapper::Othello(_) => {
                    if ui_player_id == 1 { "⚫" } else { "⚪" }
                }
                GameWrapper::Blokus(_) => {
                    match ui_player_id {
                        1 => "🟥", // Red square
                        2 => "🟦", // Blue square
                        3 => "🟩", // Green square  
                        4 => "🟨", // Yellow square
                        _ => "⬜",
                    }
                }
                _ => { // Gomoku and others
                    if ui_player_id == 1 { "❌" } else { "⭕" }
                }
            };

            text.push(Line::from(vec![
                Span::styled("Current: ", Style::default().fg(Color::White)),
                Span::styled(player_marker, Style::default()),
                Span::styled(" ", Style::default()),
                Span::styled(current_player_text, Style::default().fg(player_color).add_modifier(Modifier::BOLD)),
            ]));

            // Show AI status - display horizontally to save vertical space
            if app.is_current_player_ai() {
                if let Some(start_time) = app.ai_thinking_start {
                    let elapsed = start_time.elapsed();
                    let elapsed_secs = elapsed.as_secs();
                    let elapsed_millis = elapsed.as_millis() % 1000;
                    let remaining = app.timeout_secs.saturating_sub(elapsed_secs);
                    
                    // Create a compact progress bar
                    let progress = if app.timeout_secs > 0 {
                        (elapsed_secs as f64 / app.timeout_secs as f64 * 10.0) as usize
                    } else {
                        0
                    };
                    let progress_bar = "█".repeat(progress.min(10)) + &"░".repeat(10 - progress.min(10));
                    
                    // Display AI status and timer info on one line
                    let mut line_spans = vec![
                        Span::styled("🤖 AI: ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                        Span::styled("🤔 Thinking ", Style::default().fg(Color::Yellow)),
                        Span::styled(format!("{}.{}s", elapsed_secs, elapsed_millis / 100), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                        Span::styled(format!(" / {}s", app.timeout_secs), Style::default().fg(Color::Gray)),
                        Span::styled("  ⏰ ", Style::default().fg(Color::Yellow)),
                        Span::styled(format!("{}s left", remaining), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    ];
                    
                    // Add pending response indicator if applicable
                    if app.pending_ai_response.is_some() {
                        line_spans.push(Span::styled("  📥 Ready", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)));
                    }
                    
                    text.push(Line::from(line_spans));
                    
                    // Progress bar on second line
                    text.push(Line::from(vec![
                        Span::styled("Progress: [", Style::default().fg(Color::Cyan)),
                        Span::styled(progress_bar, Style::default().fg(Color::Cyan)),
                        Span::styled("]", Style::default().fg(Color::Cyan)),
                        // Add debug info about minimum display time
                        Span::styled(format!("  ⏳ Min: {:.1}s", app.ai_minimum_display_duration.as_secs_f64()), Style::default().fg(Color::Gray)),
                        Span::styled(
                            if elapsed.as_secs_f64() >= app.ai_minimum_display_duration.as_secs_f64() { " ✓" } else { " ⏱️" },
                            Style::default().fg(if elapsed.as_secs_f64() >= app.ai_minimum_display_duration.as_secs_f64() { Color::Green } else { Color::Yellow })
                        ),
                    ]));
                } else {
                    // AI starting search
                    text.push(Line::from(vec![
                        Span::styled("🤖🤔 AI Starting search...", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                    ]));
                }
            }
        }

        // Show basic statistics if available - compact format
        if let Some(stats) = &app.last_search_stats {
            text.push(Line::from(format!("Nodes: {} | Root Value: {:.3}", stats.total_nodes, stats.root_value)));
        }

        // Game-specific instructions - compact
        let instructions = match app.mode {
            crate::app::AppMode::InGame => {
                if app.game_status == GameStatus::InProgress {
                    if app.is_current_player_ai() {
                        "AI is thinking..."
                    } else {
                        "Arrows: move cursor | Enter/Space: make move | PgUp/PgDn: scroll"
                    }
                } else {
                    "Press 'r' to restart | Esc for menu"
                }
            }
            crate::app::AppMode::GameOver => "Press 'r' to restart | Esc for menu",
            _ => "",
        };

        if !instructions.is_empty() {
            text.push(Line::from(instructions));
        }
        
        let paragraph = Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title("Game Info"));
        frame.render_widget(paragraph, area);
        Ok(())
    }
    
    /// Render the combined stats and history pane with tabs
    fn render_stats_history_tabs(&self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        use ratatui::{
            widgets::{Block, Borders, Tabs},
            style::{Style, Color, Modifier},
        };
        
        // Create the main bordered block for the entire pane
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Debug Stats / Move History");
        let inner_area = block.inner(area);
        frame.render_widget(block, area);
        
        // Render the content of the active tab in the full inner area
        match app.active_tab {
            crate::app::ActiveTab::Debug => self.render_debug_stats_content(frame, inner_area, app)?,
            crate::app::ActiveTab::History => self.render_move_history_content(frame, inner_area, app)?,
        }
        
        // Position tabs on the bottom border line
        let tabs_area = ratatui::layout::Rect {
            x: area.x + 1, // Start after left border
            y: area.y + area.height.saturating_sub(1), // Bottom border line
            width: area.width.saturating_sub(2), // Account for left and right borders
            height: 1,
        };
        
        // Create tab titles
        let titles = vec!["Debug Stats", "Move History"];
        let tabs = Tabs::new(titles)
            .block(Block::default().borders(Borders::NONE))
            .select(app.active_tab as usize)
            .style(Style::default()
                .fg(Color::White)
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD))
            .highlight_style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            );
        frame.render_widget(tabs, tabs_area);
        Ok(())
    }
    
    /// Render the debug statistics content
    fn render_debug_stats_content(&self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        use ratatui::{
            layout::{Constraint, Direction, Layout},
            widgets::{Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
            text::Line,
        };
        
        // Split area for content and scrollbar
        let chunks = if area.width > 5 {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(0), Constraint::Length(1)])
                .split(area)
        } else {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(100)])
                .split(area)
        };
        
        let mut text = vec![Line::from("Debug Statistics")];
        if let Some(stats) = &app.last_search_stats {
            text.push(Line::from(""));
            text.push(Line::from("AI Status: Active"));
            text.push(Line::from(format!("Total Nodes: {}", stats.total_nodes)));
            text.push(Line::from(format!("Root Visits: {}", stats.root_visits)));
            text.push(Line::from(format!("Root Value: {:.3}", stats.root_value)));
            text.push(Line::from(""));
            
            // Show top moves with their statistics
            let mut sorted_children: Vec<_> = stats.children_stats.iter().collect();
            sorted_children.sort_by_key(|(_, (_, visits))| *visits);
            sorted_children.reverse();
            text.push(Line::from("Top AI Moves:"));
            for (i, (move_str, (value, visits))) in sorted_children.iter().take(10).enumerate() {
                let line = format!("{}. {}: {:.3} ({})", i + 1, move_str, value, visits);
                text.push(Line::from(line));
            }
        } else {
            text.push(Line::from(""));
            text.push(Line::from("AI Status: Idle"));
            text.push(Line::from("Waiting for MCTS statistics..."));
        }
        
        // Apply scrolling
        let content_height = text.len();
        let visible_height = chunks[0].height as usize;
        let max_scroll = content_height.saturating_sub(visible_height);
        let scroll_offset = (app.debug_scroll as usize).min(max_scroll);
        let visible_lines: Vec<Line> = text
            .into_iter()
            .skip(scroll_offset)
            .take(visible_height)
            .collect();
        
        let paragraph = Paragraph::new(visible_lines);
        frame.render_widget(paragraph, chunks[0]);
        
        // Render scrollbar if content is scrollable and we have space for it
        if max_scroll > 0 && chunks.len() > 1 && chunks[1].height > 0 {
            let mut scrollbar_state = ScrollbarState::default()
                .content_length(content_height)
                .viewport_content_length(visible_height)
                .position(scroll_offset);
            let scrollbar = Scrollbar::default()
                .orientation(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("↑"))
                .end_symbol(Some("↓"));
            frame.render_stateful_widget(scrollbar, chunks[1], &mut scrollbar_state);
        }
        Ok(())
    }
    
    /// Render the move history content  
    fn render_move_history_content(&self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        use ratatui::{
            layout::{Constraint, Direction, Layout},
            widgets::{List, ListItem, Scrollbar, ScrollbarOrientation, ScrollbarState},
        };
        
        // Split area for content and scrollbar
        let chunks = if area.width > 5 {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(0), Constraint::Length(1)])
                .split(area)
        } else {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(100)])
                .split(area)
        };
        
        let history_items: Vec<ListItem> = app.move_history
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let move_str = format!("{}. P{}: {:?}", i + 1, entry.player, entry.a_move);
                ListItem::new(move_str)
            })
            .collect();
        
        let history_widget = List::new(history_items);
        frame.render_widget(history_widget, chunks[0]);
        
        // Add scrollbar for move history if needed
        let content_height = app.move_history.len();
        let visible_height = chunks[0].height as usize;
        let max_scroll = content_height.saturating_sub(visible_height);
        if max_scroll > 0 && chunks.len() > 1 && chunks[1].height > 0 {
            let scroll_offset = app.get_history_scroll_offset(content_height, visible_height);
            let mut scrollbar_state = ScrollbarState::default()
                .content_length(content_height)
                .viewport_content_length(visible_height)
                .position(scroll_offset);
            let scrollbar = Scrollbar::default()
                .orientation(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("↑"))
                .end_symbol(Some("↓"));
            frame.render_stateful_widget(scrollbar, chunks[1], &mut scrollbar_state);
        }
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
        };
        
        // Use the original layout: board at top, game status in middle, stats/history tabs at bottom
        let (board_area, bottom_area) = app.layout_config.get_main_layout(area);
        
        // Split the bottom area into game info and stats/history
        let bottom_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(bottom_area);
        
        let game_info_area = bottom_chunks[0];
        let stats_area = bottom_chunks[1];

        // Render the game board
        self.render_game_board(frame, board_area, app)?;
        
        // Render game info (status)
        self.render_game_info(frame, game_info_area, app)?;
        
        // Render the combined stats and history pane with tabs
        self.render_stats_history_tabs(frame, stats_area, app)?;

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
