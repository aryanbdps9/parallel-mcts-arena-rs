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
use crate::components::blokus::{
    BlokusBoardComponent, BlokusPieceSelectorComponent, BlokusGameStatsComponent, 
    BlokusInstructionPanelComponent
};
use crossterm::event::KeyCode;
use mcts::GameState;

/// Component for in-game view
pub struct InGameComponent {
    id: ComponentId,
    // Modular Blokus components
    blokus_board: Option<BlokusBoardComponent>,
    blokus_piece_selector: Option<BlokusPieceSelectorComponent>,
    blokus_game_stats: Option<BlokusGameStatsComponent>,
    blokus_instruction_panel: Option<BlokusInstructionPanelComponent>,
}

impl InGameComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            blokus_board: None,
            blokus_piece_selector: None,
            blokus_game_stats: None,
            blokus_instruction_panel: None,
        }
    }

    /// Initialize Blokus components when needed
    fn ensure_blokus_components(&mut self) {
        if self.blokus_board.is_none() {
            self.blokus_board = Some(BlokusBoardComponent::new());
        }
        if self.blokus_piece_selector.is_none() {
            self.blokus_piece_selector = Some(BlokusPieceSelectorComponent::new());
        }
        if self.blokus_game_stats.is_none() {
            self.blokus_game_stats = Some(BlokusGameStatsComponent::new());
        }
        if self.blokus_instruction_panel.is_none() {
            self.blokus_instruction_panel = Some(BlokusInstructionPanelComponent::new());
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
        
        // Handle Blokus differently (use specialized view)
        if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
            // This should not be called for Blokus - use render_blokus_game_view instead
            let paragraph = Paragraph::new("Blokus board rendering error - use specialized view");
            frame.render_widget(paragraph, inner_area);
            return Ok(());
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

    /// Render the specialized Blokus game view with proper layout and panels
    fn render_blokus_game_view(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        use ratatui::{
            layout::{Constraint, Direction, Layout},
        };
        
        // Ensure Blokus components are initialized
        self.ensure_blokus_components();
        
        // First split vertically to have the main game area and bottom info area  
        let vertical_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
            .split(area);

        let main_game_area = vertical_chunks[0];
        let bottom_info_area = vertical_chunks[1];

        // Use Blokus-specific layout for the main game area
        let (board_area, piece_area, _player_area) = app.layout_config.get_blokus_layout(main_game_area);

        // Render the Blokus board using the modular component
        if let Some(ref mut board_component) = self.blokus_board {
            board_component.render(frame, board_area, app)?;
        }

        // Render piece selection panel using the modular component
        if let Some(ref mut piece_selector) = self.blokus_piece_selector {
            piece_selector.render(frame, piece_area, app)?;
        }

        // Split the bottom area into instructions and stats/history
        let bottom_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(bottom_info_area);

        let instructions_area = bottom_chunks[0];
        let stats_area = bottom_chunks[1];

        // Render game instructions using the modular component
        if let Some(ref mut instruction_panel) = self.blokus_instruction_panel {
            instruction_panel.render(frame, instructions_area, app)?;
        }

        // Render the combined stats and history using the modular component
        if let Some(ref mut game_stats) = self.blokus_game_stats {
            game_stats.render(frame, stats_area, app)?;
        }

        Ok(())
    }

    /// Render Blokus board with ghost piece preview (proper implementation)
    fn render_blokus_board_with_ghost(&self, frame: &mut Frame, area: Rect, app: &App, state: &crate::games::blokus::BlokusState, selected_piece: Option<(usize, usize, usize, usize)>, show_cursor: bool) -> ComponentResult<()> {
        use ratatui::{
            widgets::{Block, Borders, Paragraph},
            style::{Style, Color, Modifier},
            text::{Line, Span},
        };
        use std::collections::HashSet;
        use mcts::GameState;
        
        let board = state.get_board();
        let board_height = board.len();
        let board_width = if board_height > 0 { board[0].len() } else { 0 };

        if board_height == 0 || board_width == 0 {
            let paragraph = Paragraph::new("No board to display");
            frame.render_widget(paragraph, area);
            return Ok(());
        }

        let block = Block::default().borders(Borders::ALL).title("Blokus Board");
        let inner_area = block.inner(area);
        frame.render_widget(block, area);

        // Create board display with ghost pieces
        let mut board_lines = Vec::new();
        
        // Get ghost piece positions if a piece is selected
        let ghost_positions = if let Some((piece_id, transformation, row, col)) = selected_piece {
            self.get_ghost_piece_positions(state, piece_id, transformation, row, col)
        } else {
            HashSet::new()
        };
        
        // Get last move positions for highlighting
        let last_move_positions: HashSet<(usize, usize)> = state.get_last_move()
            .map(|coords| coords.into_iter().collect())
            .unwrap_or_default();
        
        let cursor_pos = (app.board_cursor.0, app.board_cursor.1);
        
        for (r, row) in board.iter().enumerate() {
            let mut line_spans = Vec::new();
            for (c, &cell) in row.iter().enumerate() {
                let is_cursor = (r as u16, c as u16) == cursor_pos;
                let is_ghost = ghost_positions.contains(&(r, c));
                let is_last_move = last_move_positions.contains(&(r, c));
                
                let (symbol, style) = if is_ghost {
                    // Check if this ghost position would be legal
                    let is_legal = if let Some((piece_id, transformation, cursor_row, cursor_col)) = selected_piece {
                        use crate::games::blokus::BlokusMove;
                        let test_move = BlokusMove(piece_id, transformation, cursor_row, cursor_col);
                        state.is_legal(&test_move)
                    } else {
                        false
                    };
                    
                    if is_legal {
                        // Legal ghost piece preview (cyan)
                        ("▓▓", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                    } else {
                        // Illegal ghost piece preview (red)
                        ("▓▓", Style::default().fg(Color::Red).add_modifier(Modifier::DIM))
                    }
                } else {
                    match cell {                    1 => {
                        let color = if is_last_move { Color::LightRed } else { Color::Red };
                        ("██", Style::default().fg(color).add_modifier(if is_last_move { Modifier::BOLD } else { Modifier::empty() }))
                    }
                    2 => {
                        let color = if is_last_move { Color::LightBlue } else { Color::Blue };
                        ("██", Style::default().fg(color).add_modifier(if is_last_move { Modifier::BOLD } else { Modifier::empty() }))
                    }
                    3 => {
                        let color = if is_last_move { Color::LightGreen } else { Color::Green };
                        ("██", Style::default().fg(color).add_modifier(if is_last_move { Modifier::BOLD } else { Modifier::empty() }))
                    }
                    4 => {
                        let color = if is_last_move { Color::LightYellow } else { Color::Yellow };
                        ("██", Style::default().fg(color).add_modifier(if is_last_move { Modifier::BOLD } else { Modifier::empty() }))
                    }
                        _ => {
                            // Chess-like pattern for empty squares - alternating light and dark
                            let is_light_square = (r + c) % 2 == 0;
                            if is_light_square {
                                ("░░", Style::default().fg(Color::Rgb(100, 100, 100))) // Light gray
                            } else {
                                ("▒▒", Style::default().fg(Color::Rgb(60, 60, 60))) // Dark gray
                            }
                        }
                    }
                };

                let final_style = if is_cursor && cell == 0 && show_cursor {
                    style.bg(Color::Yellow)
                } else {
                    style
                };

                line_spans.push(Span::styled(symbol, final_style));
            }
            board_lines.push(Line::from(line_spans));
        }

        let paragraph = Paragraph::new(board_lines);
        frame.render_widget(paragraph, inner_area);
        Ok(())
    }

    /// Get ghost piece positions for preview
    fn get_ghost_piece_positions(&self, _state: &crate::games::blokus::BlokusState, piece_id: usize, transformation: usize, row: usize, col: usize) -> std::collections::HashSet<(usize, usize)> {
        use std::collections::HashSet;
        let pieces = crate::games::blokus::get_blokus_pieces();
        
        if let Some(piece) = pieces.iter().find(|p| p.id == piece_id) {
            if transformation < piece.transformations.len() {
                let shape = &piece.transformations[transformation];
                let mut positions = HashSet::new();
                
                for &(dr, dc) in shape {
                    let new_r = row as i32 + dr;
                    let new_c = col as i32 + dc;
                    
                    if new_r >= 0 && new_c >= 0 {
                        positions.insert((new_r as usize, new_c as usize));
                    }
                }
                
                return positions;
            }
        }
        
        HashSet::new()
    }

    /// Render Blokus piece selection panel
    fn render_blokus_piece_selection(&self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        use ratatui::{
            widgets::{Block, Borders, Paragraph},
            style::{Style, Color, Modifier},
            text::{Line, Span},
            layout::{Layout, Direction, Constraint},
        };
        use std::collections::HashSet;
        
        if let GameWrapper::Blokus(blokus_state) = &app.game_wrapper {
            // Calculate area for content and scrollbar
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

            let block = Block::default()
                .title("Available Pieces (All Players)")
                .borders(Borders::ALL);
            frame.render_widget(block, area);

            let inner_area = Layout::default()
                .margin(1)
                .constraints([Constraint::Min(0)])
                .split(chunks[0])[0];

            let current_player = app.game_wrapper.get_current_player();
            let pieces = crate::games::blokus::get_blokus_pieces();
            let player_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
            let player_names = ["P1", "P2", "P3", "P4"];

            let mut all_lines = Vec::new();

            // Generate content for all players
            for player in 1..=4 {
                let available_pieces = blokus_state.get_available_pieces(player);
                let available_count = available_pieces.len();
                let color = player_colors[(player - 1) as usize];
                let is_current = player == current_player;
                let is_expanded = app.blokus_ui_config.players_expanded.get((player - 1) as usize).unwrap_or(&true);

                // Convert available pieces to a set for quick lookup
                let available_set: HashSet<usize> = available_pieces.iter().cloned().collect();

                // Player header with expand/collapse indicator
                let expand_indicator = if *is_expanded { "▼" } else { "▶" };
                let header_style = if is_current {
                    Style::default().fg(color).add_modifier(Modifier::BOLD).bg(Color::DarkGray)
                } else {
                    Style::default().fg(color).add_modifier(Modifier::BOLD)
                };
                
                let header_text = if is_current {
                    format!("{} ► {} ({} pieces) ◄", expand_indicator, player_names[(player - 1) as usize], available_count)
                } else {
                    format!("{}   {} ({} pieces)", expand_indicator, player_names[(player - 1) as usize], available_count)
                };
                
                all_lines.push(Line::from(Span::styled(header_text, header_style)));

                // Show pieces for this player only if expanded
                if *is_expanded {
                    let pieces_per_row = 5;
                    let visible_pieces = if is_current { 21 } else { 10 };
                    
                    // Show all pieces (0-20), graying out unavailable ones
                    let total_pieces_to_show = if is_current { 21 } else { visible_pieces.min(21) };
                    
                    // Add top border for the entire grid
                    if is_current && total_pieces_to_show > 0 {
                        let separator_width = 1; // Each separator is 1 character: │
                        let content_width = pieces_per_row * 7 + (pieces_per_row - 1) * separator_width; // 7*5 + 4*1 = 39
                        let total_grid_width = content_width + 2; // +2 for left and right borders
                        let available_width = inner_area.width as usize;
                        let padding = if available_width > total_grid_width { (available_width - total_grid_width) / 2 } else { 0 };
                        
                        let top_border = "┌".to_string() + &"─".repeat(content_width) + "┐";
                        let padded_border = " ".repeat(padding) + &top_border;
                        all_lines.push(Line::from(Span::styled(padded_border, Style::default().fg(Color::DarkGray))));
                    }
                    
                    // Show pieces in rows
                    for chunk_start in (0..total_pieces_to_show).step_by(pieces_per_row) {
                        let chunk_end = (chunk_start + pieces_per_row).min(total_pieces_to_show);
                        
                        let mut pieces_in_row = Vec::new();
                        for display_idx in chunk_start..chunk_end {
                            let piece_idx = display_idx; // Show pieces 0-20 in order
                            let piece = &pieces[piece_idx];
                            let is_available = available_set.contains(&piece_idx);
                            let is_selected = is_current && app.blokus_ui_config.selected_piece_idx == Some(piece_idx);
                            
                            // Create piece shape representation
                            let piece_shape = if !piece.transformations.is_empty() {
                                &piece.transformations[0]
                            } else {
                                continue;
                            };
                            
                            let key_label = if display_idx < 9 {
                                (display_idx + 1).to_string()
                            } else if display_idx == 9 {
                                "0".to_string()
                            } else {
                                ((b'a' + (display_idx - 10) as u8) as char).to_string()
                            };
                            
                            // Create visual shape for this piece
                            let piece_visual_lines = self.create_visual_piece_shape(piece_shape);
                            
                            let piece_name_text = if is_selected {
                                format!("[{}]", key_label)
                            } else {
                                format!(" {} ", key_label)
                            };
                            
                            // Determine style based on availability and selection
                            let style = if !is_available {
                                // Grayed out for used pieces
                                Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM)
                            } else if is_selected {
                                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD).bg(Color::DarkGray)
                            } else if is_current {
                                Style::default().fg(Color::White)
                            } else {
                                Style::default().fg(color)
                            };
                            
                            pieces_in_row.push((piece_name_text, piece_visual_lines, style));
                        }
                        
                        if !pieces_in_row.is_empty() {
                            // Find max height and width for alignment
                            let max_height = pieces_in_row.iter()
                                .map(|(_, lines, _)| lines.len())
                                .max()
                                .unwrap_or(1);
                            let piece_width = 7; // Increased from 6 to 7 for better centering of 3-char strings
                            
                            // First line: piece keys/names with subtle column highlighting
                            let mut key_line_spans = Vec::new();
                            
                            // Calculate padding for centering - use FULL grid width for consistency
                            let separator_width = 1; // Each separator is 1 character: │
                            let content_width = pieces_per_row * piece_width + (pieces_per_row - 1) * separator_width; // total content width
                            let total_grid_width = content_width + 2; // +2 for left and right borders
                            let available_width = inner_area.width as usize;
                            let padding = if available_width > total_grid_width { (available_width - total_grid_width) / 2 } else { 0 };
                            
                            // Add left padding and border
                            let left_padding = " ".repeat(padding);
                            key_line_spans.push(Span::styled(left_padding + "│", Style::default().fg(Color::DarkGray)));
                            
                            for (i, (piece_name, _, style)) in pieces_in_row.iter().enumerate() {
                                let padded_name = format!("{:^width$}", piece_name, width = piece_width);
                                key_line_spans.push(Span::styled(padded_name, *style));
                                if i < pieces_per_row - 1 {
                                    // Add visible column separator
                                    key_line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
                                }
                            }
                            
                            // Fill remaining columns with empty cells if row is incomplete
                            for i in pieces_in_row.len()..pieces_per_row {
                                let empty_cell = " ".repeat(piece_width);
                                key_line_spans.push(Span::styled(empty_cell, Style::default()));
                                if i < pieces_per_row - 1 {
                                    key_line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
                                }
                            }
                            
                            // Add right border
                            key_line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
                            
                            all_lines.push(Line::from(key_line_spans));
                            
                            // Show each line of the pieces with column separators
                            for line_idx in 0..max_height {
                                let mut shape_line_spans = Vec::new();
                                
                                // Add left padding and border
                                let left_padding = " ".repeat(padding);
                                shape_line_spans.push(Span::styled(left_padding + "│", Style::default().fg(Color::DarkGray)));
                                
                                for (i, (_, piece_visual_lines, style)) in pieces_in_row.iter().enumerate() {
                                    let piece_line = if line_idx < piece_visual_lines.len() {
                                        format!("{:^width$}", piece_visual_lines[line_idx], width = piece_width)
                                    } else {
                                        " ".repeat(piece_width)
                                    };
                                    shape_line_spans.push(Span::styled(piece_line, *style));
                                    if i < pieces_per_row - 1 {
                                        // Add visible column separator
                                        shape_line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
                                    }
                                }
                                
                                // Fill remaining columns with empty cells if row is incomplete
                                for i in pieces_in_row.len()..pieces_per_row {
                                    let empty_cell = " ".repeat(piece_width);
                                    shape_line_spans.push(Span::styled(empty_cell, Style::default()));
                                    if i < pieces_per_row - 1 {
                                        shape_line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
                                    }
                                }
                                
                                // Add right border
                                shape_line_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
                                
                                all_lines.push(Line::from(shape_line_spans));
                            }
                            
                            // Add a proper row separator that spans the full width
                            if chunk_start + pieces_per_row < total_pieces_to_show {
                                let left_padding = " ".repeat(padding);
                                let row_separator = "├".to_string() + &"─".repeat(content_width) + "┤";
                                all_lines.push(Line::from(Span::styled(left_padding + &row_separator, Style::default().fg(Color::DarkGray))));
                            }
                        }
                    }
                    
                    // Add bottom border for the entire grid
                    if is_current && total_pieces_to_show > 0 {
                        let separator_width = 1; // Each separator is 1 character: │
                        let content_width = pieces_per_row * 7 + (pieces_per_row - 1) * separator_width; // 7*5 + 4*1 = 39
                        let total_grid_width = content_width + 2; // +2 for left and right borders
                        let available_width = inner_area.width as usize;
                        let padding = if available_width > total_grid_width { (available_width - total_grid_width) / 2 } else { 0 };
                        
                        let bottom_border = "└".to_string() + &"─".repeat(content_width) + "┘";
                        let padded_border = " ".repeat(padding) + &bottom_border;
                        all_lines.push(Line::from(Span::styled(padded_border, Style::default().fg(Color::DarkGray))));
                    }
                } else {
                    // Show compact summary when collapsed
                    let used_count = 21 - available_count;
                    let status_text = if available_count > 0 {
                        if used_count > 0 {
                            format!("  {} available, {} used", available_count, used_count)
                        } else {
                            format!("  All {} pieces available", available_count)
                        }
                    } else {
                        "  All pieces used".to_string()
                    };
                    
                    all_lines.push(Line::from(Span::styled(status_text, Style::default().fg(color))));
                }
            }

            let paragraph = Paragraph::new(all_lines);
            frame.render_widget(paragraph, inner_area);
        }
        
        Ok(())
    }

    /// Render Blokus player status panel
    fn render_blokus_player_status(&self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        use ratatui::{
            widgets::{Block, Borders, Paragraph},
            style::{Style, Color, Modifier},
            text::{Line, Span},
        };
        
        if let GameWrapper::Blokus(blokus_state) = &app.game_wrapper {
            let block = Block::default().title("Players").borders(Borders::ALL);
            let inner_area = block.inner(area);
            frame.render_widget(block, area);
            
            let mut status_lines = Vec::new();
            let current_player = app.game_wrapper.get_current_player();
            let player_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
            let player_names = ["P1", "P2", "P3", "P4"];

            for player in 1..=4 {
                let available_pieces = blokus_state.get_available_pieces(player);
                let piece_count = available_pieces.len();
                let color = player_colors[(player - 1) as usize];
                
                let status_text = if player == current_player {
                    format!("► {} ({} pieces)", player_names[(player - 1) as usize], piece_count)
                } else {
                    format!("  {} ({} pieces)", player_names[(player - 1) as usize], piece_count)
                };
                
                let style = if player == current_player {
                    Style::default().fg(color).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(color)
                };
                
                status_lines.push(Line::from(Span::styled(status_text, style)));
            }

            // Add controls
            status_lines.push(Line::from(""));
            status_lines.push(Line::from(Span::styled("Controls:", Style::default().fg(Color::Gray))));
            status_lines.push(Line::from(Span::styled("1-9,0,a-k: Select", Style::default().fg(Color::Gray))));
            status_lines.push(Line::from(Span::styled("R: Rotate  X: Flip", Style::default().fg(Color::Gray))));
            status_lines.push(Line::from(Span::styled("Enter: Place", Style::default().fg(Color::Gray))));
            status_lines.push(Line::from(Span::styled("P: Pass", Style::default().fg(Color::Gray))));

            let paragraph = Paragraph::new(status_lines);
            frame.render_widget(paragraph, inner_area);
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
        
        // Check if this is Blokus game and use specialized layout
        if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
            return self.render_blokus_game_view(frame, area, app);
        }
        
        // Use the original layout for non-Blokus games: board at top, game status in middle, stats/history tabs at bottom
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

        // For Blokus game, route events to modular components
        if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
            self.ensure_blokus_components();
            
            // Try to handle the event with Blokus components first
            if let Some(ref mut piece_selector) = self.blokus_piece_selector {
                if let Ok(true) = piece_selector.handle_event(event, app) {
                    return Ok(true);
                }
            }
            
            if let Some(ref mut board_component) = self.blokus_board {
                if let Ok(true) = board_component.handle_event(event, app) {
                    return Ok(true);
                }
            }
            
            if let Some(ref mut game_stats) = self.blokus_game_stats {
                if let Ok(true) = game_stats.handle_event(event, app) {
                    return Ok(true);
                }
            }
            
            if let Some(ref mut instruction_panel) = self.blokus_instruction_panel {
                if let Ok(true) = instruction_panel.handle_event(event, app) {
                    return Ok(true);
                }
            }
        }

        // Handle general events
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
                    _ => {
                        // Handle game-specific input if components didn't handle it
                        if self.is_current_player_human(app) {
                            self.handle_keyboard_input(app, key)
                        } else {
                            Ok(false)
                        }
                    }
                }
            }
            ComponentEvent::Input(InputEvent::MouseClick { x, y, button }) => {
                // If components didn't handle mouse click, handle it with legacy logic
                if self.is_current_player_human(app) || matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                    match button {
                        1 => { // Left click
                            self.handle_mouse_click(app, *x, *y);
                            Ok(true)
                        }
                        3 => { // Right click
                            self.handle_mouse_right_click(app, *x, *y);
                            Ok(true)
                        }
                        _ => Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            ComponentEvent::Input(InputEvent::MouseScroll { x: _, y: _, up: _ }) => {
                // Route mouse scroll to piece selector for Blokus
                if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
                    if let Some(ref mut piece_selector) = self.blokus_piece_selector {
                        piece_selector.handle_event(event, app)
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            _ => Ok(false),
        }
    }

    crate::impl_component_base!(InGameComponent);
}

impl InGameComponent {
    /// Handle left mouse click
    fn handle_mouse_click(&self, app: &mut App, col: u16, row: u16) {
        if app.game_status != GameStatus::InProgress {
            return;
        }
        
        match &app.game_wrapper {
            GameWrapper::Blokus(_) => {
                self.handle_blokus_click(app, col, row);
            }
            _ => {
                self.handle_standard_game_click(app, col, row);
            }
        }
    }

    /// Handle right mouse click  
    fn handle_mouse_right_click(&self, app: &mut App, _col: u16, _row: u16) {
        // Right-click in Blokus to rotate selected piece
        if matches!(app.game_wrapper, GameWrapper::Blokus(_)) && self.is_current_player_human(app) {
            app.blokus_rotate_piece();
        }
    }

    /// Handle keyboard input for games that weren't handled by components
    fn handle_keyboard_input(&self, app: &mut App, key: &KeyCode) -> EventResult {
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

    /// Handle Blokus-specific clicks during AI turn (only expand/collapse allowed)
    fn handle_blokus_mouse_click_ai_turn(&self, app: &mut App, col: u16, row: u16) {
        // Get layout areas - need to calculate from the render method's logic
        let terminal_size = Rect::new(0, 0, 120, 40); // Use a reasonable default size
        let (_, piece_area, _) = self.get_blokus_layout_areas(terminal_size);
        
        if col >= piece_area.x && col < piece_area.x + piece_area.width &&
           row >= piece_area.y && row < piece_area.y + piece_area.height {
            // Only handle expand/collapse clicks, not piece selection
            self.handle_blokus_piece_expand_collapse_only(app, col - piece_area.x, row - piece_area.y);
        }
    }

    /// Handle clicks on standard games (Gomoku, Connect4, Othello)
    fn handle_standard_game_click(&self, app: &mut App, col: u16, row: u16) {
        // Calculate board area - we need to estimate this based on terminal size
        // For now, use a simple calculation - in practice we'd get this from the layout
        let board_start_col = 1;
        let board_start_row = 1;
        
        if col >= board_start_col && row >= board_start_row {
            let board = app.game_wrapper.get_board();
            let board_height = board.len();
            let board_width = if board_height > 0 { board[0].len() } else { 0 };
            
            // Calculate which cell was clicked based on game type
            match &app.game_wrapper {
                GameWrapper::Connect4(_) => {
                    // Connect4: Each column is 4 characters wide
                    let cell_width = 4;
                    let board_col = ((col - board_start_col) / cell_width) as usize;
                    
                    if board_col < board_width {
                        // Update cursor position to this column
                        app.board_cursor.1 = board_col as u16;
                        // Update row to lowest available position
                        self.update_connect4_cursor_row(app);
                        self.make_move(app);
                    }
                }
                GameWrapper::Gomoku(_) | GameWrapper::Othello(_) => {
                    // Gomoku/Othello: Each cell is 2 characters wide
                    let cell_width = 2;
                    let cell_height = 1;
                    
                    let board_col = ((col - board_start_col) / cell_width) as usize;
                    let board_row = ((row - board_start_row) / cell_height) as usize;
                    
                    if board_row < board_height && board_col < board_width {
                        app.board_cursor = (board_row as u16, board_col as u16);
                        self.make_move(app);
                    }
                }
                _ => {}
            }
        }
    }

    /// Update Connect4 cursor to lowest available position in column  
    fn update_connect4_cursor_row(&self, app: &mut App) {
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

    /// Handle Blokus clicks
    fn handle_blokus_click(&self, app: &mut App, col: u16, row: u16) {
        // Get layout areas - we need terminal size for accurate calculations
        // For now, estimate based on common terminal sizes
        let terminal_size = Rect::new(0, 0, 120, 40); // This should come from the actual terminal size
        let (board_area, piece_area, _) = self.get_blokus_layout_areas(terminal_size);
        
        if col >= board_area.x && col < board_area.x + board_area.width &&
           row >= board_area.y && row < board_area.y + board_area.height {
            // Click on board area
            self.handle_blokus_board_click(app, col - board_area.x, row - board_area.y);
        } else if col >= piece_area.x && col < piece_area.x + piece_area.width &&
                  row >= piece_area.y && row < piece_area.y + piece_area.height {
            // Click on piece selection area
            self.handle_blokus_piece_selection_click(app, col - piece_area.x, row - piece_area.y, piece_area.width);
        }
    }

    /// Get Blokus layout areas (approximation - ideally would be calculated from actual render)
    fn get_blokus_layout_areas(&self, terminal_size: Rect) -> (Rect, Rect, Rect) {
        // This is a simplified version of the layout calculation
        // In practice, this should match exactly what the render method uses
        
        // Main game area (65% of height)
        let main_height = (terminal_size.height * 65) / 100;
        let main_area = Rect::new(terminal_size.x, terminal_size.y, terminal_size.width, main_height);
        
        // Split main area: 50% board, 35% pieces, 15% player info
        let board_width = (main_area.width * 50) / 100;
        let piece_width = (main_area.width * 35) / 100;
        let player_width = main_area.width - board_width - piece_width;
        
        let board_area = Rect::new(main_area.x, main_area.y, board_width, main_area.height);
        let piece_area = Rect::new(main_area.x + board_width, main_area.y, piece_width, main_area.height);
        let player_area = Rect::new(main_area.x + board_width + piece_width, main_area.y, player_width, main_area.height);
        
        (board_area, piece_area, player_area)
    }

    /// Handle clicks on Blokus board
    fn handle_blokus_board_click(&self, app: &mut App, col: u16, row: u16) {
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
                        self.make_move_with_move(app, player_move);
                        // Deselect piece after successful placement
                        app.blokus_ui_config.selected_piece_idx = None;
                    }
                }
            }
        }
    }

    /// Make a move with a specific move wrapper
    fn make_move_with_move(&self, app: &mut App, player_move: MoveWrapper) {
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

    /// Handle clicks in Blokus piece selection area (with full piece selection logic)
    fn handle_blokus_piece_selection_click(&self, app: &mut App, col: u16, row: u16, area_width: u16) {
        if let GameWrapper::Blokus(ref state) = app.game_wrapper {
            let current_player = app.game_wrapper.get_current_player();
            
            // Calculate scroll offset (same logic as main branch)
            let scroll_offset = if let Some(auto_scroll_pos) = app.calculate_piece_panel_auto_scroll_position() {
                auto_scroll_pos
            } else {
                app.blokus_ui_config.panel_scroll_offset
            };
            let absolute_row = row + scroll_offset as u16;
            
            // Get pieces for height calculations
            let pieces = crate::games::blokus::get_blokus_pieces();
            
            let mut content_row = 0u16;
            let pieces_per_row = 5;
            
            // Process each player section
            for player in 1..=4 {
                let is_current = player == current_player;
                let is_expanded = app.blokus_ui_config.players_expanded.get((player - 1) as usize).unwrap_or(&true);

                // Player header line
                if absolute_row == content_row {
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
                        
                        match self.try_select_piece_in_current_player_grid(
                            absolute_row, col, &mut content_row, pieces_per_row,
                            total_pieces_to_show, &pieces, &available_set, area_width
                        ) {
                            Some(selected_piece) => {
                                app.blokus_ui_config.select_piece(selected_piece);
                                return;
                            }
                            None => {
                                // Click was within current player area but not on a valid piece
                                return;
                            }
                        }
                    } else {
                        // OTHER PLAYERS: Simulate their content (same logic as main branch)
                        let visible_pieces = 10;
                        let total_pieces_to_show = visible_pieces.min(21);
                        
                        let mut other_player_content_rows = 0u16;
                        
                        // Process each chunk of pieces
                        for chunk_start in (0..total_pieces_to_show).step_by(pieces_per_row) {
                            let chunk_end = (chunk_start + pieces_per_row).min(total_pieces_to_show);
                            
                            // Calculate max height for this chunk
                            let mut max_height = 1;
                            for display_idx in chunk_start..chunk_end {
                                if display_idx < pieces.len() && !pieces[display_idx].transformations.is_empty() {
                                    let piece_shape = &pieces[display_idx].transformations[0];
                                    let piece_visual_lines = self.create_visual_piece_shape(piece_shape);
                                    max_height = max_height.max(piece_visual_lines.len());
                                }
                            }
                            
                            // Key/name line + shape lines + separator (if not last chunk)
                            other_player_content_rows += 1 + max_height as u16;
                            if chunk_start + pieces_per_row < total_pieces_to_show {
                                other_player_content_rows += 1;
                            }
                        }
                        
                        // Check if click is within this player's content area
                        if absolute_row >= content_row && absolute_row < content_row + other_player_content_rows {
                            return; // Click consumed by other player content
                        }
                        
                        content_row += other_player_content_rows;
                    }
                } else {
                    // Collapsed player - just the summary line
                    if absolute_row == content_row {
                        return; // Click on collapsed player summary
                    }
                    content_row += 1;
                }
                
                // Separator between players (empty line)
                if player < 4 {
                    content_row += 1;
                }
            }
        }
    }

    /// Try to select a piece in the current player's grid (from main branch logic)
    fn try_select_piece_in_current_player_grid(
        &self,
        absolute_row: u16,
        col: u16,
        content_row: &mut u16,
        pieces_per_row: usize,
        total_pieces_to_show: usize,
        pieces: &[crate::games::blokus::Piece],
        available_set: &std::collections::HashSet<usize>,
        area_width: u16,
    ) -> Option<usize> {
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
        
        // Process each row of pieces using direct coordinate mapping
        for chunk_start in (0..total_pieces_to_show).step_by(pieces_per_row) {
            let chunk_end = (chunk_start + pieces_per_row).min(total_pieces_to_show);
            
            // Calculate max height for this chunk - MUST match the rendering logic exactly
            let mut pieces_in_row_visual_lines = Vec::new();
            for display_idx in chunk_start..chunk_end {
                if display_idx < pieces.len() && !pieces[display_idx].transformations.is_empty() {
                    let piece_shape = &pieces[display_idx].transformations[0];
                    let piece_visual_lines = self.create_visual_piece_shape(piece_shape);
                    pieces_in_row_visual_lines.push(piece_visual_lines);
                }
            }
            
            // Calculate max_height exactly like the rendering code
            let max_height = pieces_in_row_visual_lines.iter()
                .map(|lines| lines.len())
                .max()
                .unwrap_or(1);
            
            // Calculate the row ranges for this chunk
            let chunk_start_row = *content_row;
            // Each chunk has: 1 name line + max_height shape lines + 1 separator line (if not last chunk)
            let has_separator = chunk_start + pieces_per_row < total_pieces_to_show;
            let chunk_total_rows = 1 + max_height + if has_separator { 1 } else { 0 };
            
            // Define the clickable area - be more generous to improve user experience
            // Include name line + ALL shape lines + one extra row for visual tolerance
            let clickable_start_row = chunk_start_row;
            let clickable_end_row = chunk_start_row + (1 + max_height + 1) as u16; // name + shape + 1 extra
            
            // Check if click is within this chunk's expanded clickable area
            if absolute_row >= clickable_start_row && absolute_row < clickable_end_row {
                // Click is within this chunk - check column position
                if col >= grid_content_start_col as u16 && col < (grid_content_start_col + content_width) as u16 {
                    let grid_col = col - grid_content_start_col as u16;
                    
                    // Calculate which piece column this click corresponds to
                    let pieces_in_chunk = chunk_end - chunk_start;
                    
                    // Use the exact same layout as the rendering:
                    // Each piece takes piece_width characters, followed by separator_width
                    // Grid structure: [piece0][sep][piece1][sep][piece2][sep][piece3][sep][piece4]
                    //                  0-6    7   8-14   15  16-22  23  24-30  31  32-38
                    
                    for piece_col in 0..pieces_in_chunk {
                        // Calculate the exact column range for this piece
                        let piece_start_col = piece_col * (piece_width + separator_width);
                        let piece_end_col = piece_start_col + piece_width;
                        
                        // Check if click is within this piece's column range
                        if grid_col >= piece_start_col as u16 && grid_col < piece_end_col as u16 {
                            let piece_idx = chunk_start + piece_col;
                            
                            if available_set.contains(&piece_idx) {
                                return Some(piece_idx);
                            } else {
                                // Piece exists but not available
                                return None;
                            }
                        }
                    }
                    
                    // Click was within the grid but on a vertical separator - return None
                    return None;
                }
                // Click was in chunk row range but outside the grid - return None
                return None;
            }
            
            // Update content_row to after this chunk (including separator if present)
            *content_row = chunk_start_row + chunk_total_rows as u16;
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
    fn create_visual_piece_shape(&self, piece_shape: &[(i32, i32)]) -> Vec<String> {
        if piece_shape.is_empty() {
            return vec!["▢".to_string()];
        }

        // Create a 2D visual representation
        let min_r = piece_shape.iter().map(|p| p.0).min().unwrap_or(0);
        let max_r = piece_shape.iter().map(|p| p.0).max().unwrap_or(0);
        let min_c = piece_shape.iter().map(|p| p.1).min().unwrap_or(0);
        let max_c = piece_shape.iter().map(|p| p.1).max().unwrap_or(0);

        let height = (max_r - min_r + 1) as usize;
        let width = (max_c - min_c + 1) as usize;

        // Create a grid to show the shape
        let mut grid = vec![vec![' '; width]; height];

        // Fill the grid with the piece shape
        for &(r, c) in piece_shape {
            let gr = (r - min_r) as usize;
            let gc = (c - min_c) as usize;
            grid[gr][gc] = '▢'; // Use empty square like ghost pieces
        }

        // Convert to vector of strings
        let mut result: Vec<String> = grid.iter()
            .map(|row| row.iter().collect::<String>())
            .collect();

        // Ensure minimum width for single character pieces
        if result.len() == 1 && result[0].trim().len() == 1 {
            result[0] = format!(" {} ", result[0].trim());
        }

        result
    }

    /// Handle expand/collapse clicks only (for AI turns)
    fn handle_blokus_piece_expand_collapse_only(&self, app: &mut App, col: u16, row: u16) {
        // Simplified version that only handles expand/collapse, not piece selection
        if let GameWrapper::Blokus(_) = app.game_wrapper {
            let scroll_offset = if let Some(auto_scroll_pos) = app.calculate_piece_panel_auto_scroll_position() {
                auto_scroll_pos
            } else {
                app.blokus_ui_config.panel_scroll_offset
            };
            let absolute_row = row + scroll_offset as u16;
            
            let mut content_row = 0u16;
            
            // Process each player section
            for player in 1..=4 {
                // Player header line
                if absolute_row == content_row {
                    // Check if clicking on expand/collapse area (first few columns)
                    if col <= 2 {
                        app.blokus_ui_config.toggle_player_expand((player - 1) as usize);
                    }
                    return;
                }
                content_row += 1;
                
                // Skip the rest of the player's content for simplicity
                let is_expanded = app.blokus_ui_config.players_expanded.get((player - 1) as usize).unwrap_or(&true);
                if *is_expanded {
                    // Estimate content height - this doesn't need to be perfect for expand/collapse
                    content_row += 10; // Rough estimate
                } else {
                    content_row += 1; // Summary line
                }
                
                // Separator between players
                if player < 4 {
                    content_row += 1;
                }
            }
        }
    }
}
