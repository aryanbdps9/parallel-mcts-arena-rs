//! # UI Widgets Module
//!
//! This module contains functions for drawing the different UI components (widgets)
//! on the screen, such as the game board, statistics, and menus.

use crate::app::{App, AppMode, GameStatus, Player};
use crate::game_wrapper::GameWrapper;
use crate::games::blokus::BlokusState;
use crate::tui::blokus_ui;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState};
use mcts::GameState;

pub fn render(app: &mut App, frame: &mut Frame) {
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(100)])
        .split(frame.size());

    match app.mode {
        AppMode::GameSelection => draw_game_selection_menu(frame, app, main_layout[0]),
        AppMode::Settings => draw_settings_menu(frame, app, main_layout[0]),
        AppMode::PlayerConfig => draw_player_config_menu(frame, app, main_layout[0]),
        AppMode::InGame | AppMode::GameOver => {
            draw_game_view(frame, app, main_layout[0])
        }
    }
}

fn draw_game_selection_menu(f: &mut Frame, app: &mut App, area: Rect) {
    let mut items: Vec<ListItem> = app
        .games
        .iter()
        .map(|(name, _)| ListItem::new(*name))
        .collect();

    items.push(ListItem::new("Settings"));
    items.push(ListItem::new("Quit"));

    let title = if app.ai_only {
        "Select a Game (AI-Only Mode)"
    } else {
        "Select a Game"
    };

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(Color::Yellow),
        )
        .highlight_symbol("> ");

    f.render_stateful_widget(list, area, &mut app.game_selection_state);
}

fn draw_settings_menu(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(3)])
        .split(area);

    let settings_items = vec![
        format!("Board Size: {}", app.settings_board_size),
        format!("Line Size: {}", app.settings_line_size),
        format!("AI Threads: {}", app.settings_ai_threads),
        format!("Max Nodes: {}", app.settings_max_nodes),
        format!("Exploration Constant: {:.2}", app.settings_exploration_constant),
        format!("Timeout (secs): {}", app.timeout_secs),
        format!("Stats Interval (secs): {}", app.stats_interval_secs),
        format!("AI Only Mode: {}", if app.ai_only { "Yes" } else { "No" }),
        format!("Shared Tree: {}", if app.shared_tree { "Yes" } else { "No" }),
        "".to_string(), // Separator
        "Back".to_string(),
    ];

    let items: Vec<ListItem> = settings_items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let style = if i == app.selected_settings_index {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            ListItem::new(item.as_str()).style(style)
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Settings"))
        .highlight_symbol("> ");

    f.render_widget(list, chunks[0]);

    // Instructions
    let instructions = Paragraph::new("Use Up/Down to navigate, Left/Right to adjust values, Enter to confirm, Esc to go back")
        .block(Block::default().borders(Borders::ALL).title("Instructions"));
    f.render_widget(instructions, chunks[1]);
}

fn draw_player_config_menu(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(3)])
        .split(area);

    // Player configuration list
    let mut items: Vec<ListItem> = app
        .player_options
        .iter()
        .enumerate()
        .map(|(i, (id, p_type))| {
            let type_str = match p_type {
                Player::Human => "Human",
                Player::AI => "AI",
            };
            let style = if i == app.selected_player_config_index {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            ListItem::new(format!("Player {}: {}", id, type_str)).style(style)
        })
        .collect();

    // Add "Start Game" option
    let start_style = if app.selected_player_config_index >= app.player_options.len() {
        Style::default().add_modifier(Modifier::REVERSED).fg(Color::Green)
    } else {
        Style::default().fg(Color::Green)
    };
    items.push(ListItem::new("Start Game").style(start_style));

    let title = if app.ai_only {
        format!("{} - Player Configuration (AI Only Mode)", app.get_selected_game_name())
    } else {
        format!("{} - Player Configuration", app.get_selected_game_name())
    };

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_symbol("> ");

    f.render_widget(list, chunks[0]);

    // Instructions
    let instructions_text = if app.ai_only {
        "AI Only Mode: All players will be set to AI automatically. Enter to start game, Esc to go back"
    } else {
        "Use Up/Down to navigate, Left/Right/Space to toggle player type, Enter to confirm, Esc to go back"
    };
    
    let instructions = Paragraph::new(instructions_text)
        .block(Block::default().borders(Borders::ALL).title("Instructions"));
    f.render_widget(instructions, chunks[1]);
}

fn draw_game_view(f: &mut Frame, app: &App, area: Rect) {
    if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
        draw_blokus_game_view(f, app, area);
    } else {
        draw_standard_game_view(f, app, area);
    }
}

fn draw_standard_game_view(f: &mut Frame, app: &App, area: Rect) {
    // Use the layout config to get the main areas
    let (board_area, instructions_area, stats_area) = app.layout_config.get_main_layout(area);

    // Draw the game board
    draw_board(f, app, board_area);
    
    // Draw game info/instructions
    draw_game_info(f, app, instructions_area);
    
    // Split the stats area for debug stats and move history
    let (debug_area, history_area) = app.layout_config.get_stats_layout(stats_area);

    // Draw debug statistics and move history
    draw_debug_stats(f, app, debug_area);
    draw_move_history(f, app, history_area);
}

fn draw_blokus_game_view(f: &mut Frame, app: &App, area: Rect) {
    // First split vertically to have the main game area and bottom info area
    let vertical_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(area);

    let main_game_area = vertical_chunks[0];
    let bottom_info_area = vertical_chunks[1];

    // Use Blokus-specific layout for the main game area
    let (board_area, piece_area, player_area) = app.layout_config.get_blokus_layout(main_game_area);

    // Draw the Blokus board with ghost pieces
    if let GameWrapper::Blokus(state) = &app.game_wrapper {
        // Get selected piece info from app state
        let selected_piece = if let Some((piece_idx, transformation_idx)) = app.blokus_ui_config.get_selected_piece_info() {
            Some((piece_idx, transformation_idx, app.board_cursor.0 as usize, app.board_cursor.1 as usize))
        } else {
            None
        };
        // Only show cursor for human turns
        let show_cursor = !app.is_current_player_ai();
        blokus_ui::draw_blokus_board(f, state, board_area, selected_piece, app.board_cursor, show_cursor);
    }

    // Draw piece selection panel
    blokus_ui::draw_blokus_piece_selection(f, app, piece_area);

    // Draw player status panel
    blokus_ui::draw_blokus_player_status(f, app, player_area);

    // Split the bottom area into instructions and stats/history
    let bottom_vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(bottom_info_area);

    let instructions_area = bottom_vertical[0];
    let stats_area = bottom_vertical[1];

    // Draw game info/instructions
    draw_game_info(f, app, instructions_area);

    // Split stats area horizontally for debug stats and move history
    let stats_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(stats_area);

    // Draw debug stats and move history
    draw_debug_stats(f, app, stats_chunks[0]);
    draw_move_history(f, app, stats_chunks[1]);
}

fn draw_debug_stats(f: &mut Frame, app: &App, area: Rect) {
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
        text.push(Line::from(format!("AI Status: Active")));
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
    let visible_height = chunks[0].height.saturating_sub(2) as usize; // Account for borders
    let max_scroll = content_height.saturating_sub(visible_height);
    let scroll_offset = (app.debug_scroll as usize).min(max_scroll);

    let visible_lines: Vec<Line> = text
        .into_iter()
        .skip(scroll_offset)
        .take(visible_height)
        .collect();

    let drag_indicator = if app.drag_state.is_dragging { "🔀" } else { "↔" };
    let title = format!("{} Debug Stats - {}%", 
        drag_indicator, 
        app.layout_config.stats_width_percent
    );

    let paragraph = Paragraph::new(visible_lines)
        .block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(paragraph, chunks[0]);

    // Render scrollbar if content is scrollable and we have space for it
    if max_scroll > 0 && chunks.len() > 1 && chunks[1].height > 2 {
        let mut scrollbar_state = ScrollbarState::default()
            .content_length(content_height)
            .viewport_content_length(visible_height)
            .position(scroll_offset);
            
        let scrollbar = Scrollbar::default()
            .orientation(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("↑"))
            .end_symbol(Some("↓"));
            
        f.render_stateful_widget(scrollbar, chunks[1], &mut scrollbar_state);
    }
}

fn draw_game_info(f: &mut Frame, app: &App, area: Rect) {
    let mut text = vec![
        Line::from(format!("Game: {}  |  Status: {:?}", app.get_selected_game_name(), app.game_status)),
    ];

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
                Span::styled("� AI: ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
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

    // Show basic statistics if available - compact format
    if let Some(stats) = &app.last_search_stats {
        text.push(Line::from(format!("Nodes: {} | Root Value: {:.3}", stats.total_nodes, stats.root_value)));
    }

    // Game-specific instructions - compact
    let instructions = match app.mode {
        AppMode::InGame => {
            if app.game_status == GameStatus::InProgress {
                match player_type {
                    Player::Human => "Arrows: move cursor | Enter/Space: make move | PgUp/PgDn: scroll",
                    Player::AI => "AI is thinking...",
                }
            } else {
                "Press 'r' to restart | Esc for menu"
            }
        }
        AppMode::GameOver => "Press 'r' to restart | Esc for menu",
        _ => "",
    };

    if !instructions.is_empty() {
        text.push(Line::from(instructions));
    }

    let paragraph = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title("Game Info"));
    f.render_widget(paragraph, area);
}

fn draw_move_history(f: &mut Frame, app: &App, area: Rect) {
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

    let items: Vec<ListItem> = app
        .move_history
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let player_symbol = match &app.game_wrapper {
                GameWrapper::Blokus(_) => format!("P{}", entry.player),
                _ => if entry.player == 1 { "X".to_string() } else { "O".to_string() },
            };
            let move_str = format!("{}. {}: {}", i + 1, player_symbol, entry.a_move);
            ListItem::new(move_str)
        })
        .collect();

    // Apply scrolling - show items starting from scroll offset
    let content_height = items.len();
    let visible_height = chunks[0].height.saturating_sub(2) as usize; // Account for borders
    let max_scroll = content_height.saturating_sub(visible_height);
    let scroll_offset = (app.history_scroll as usize).min(max_scroll);

    let visible_items: Vec<ListItem> = items
        .into_iter()
        .skip(scroll_offset)
        .take(visible_height)
        .collect();

    let drag_indicator = if app.drag_state.is_dragging { "🔀" } else { "↔" };
    let title = format!("{} Move History ({}) - {}%", 
        drag_indicator, 
        app.move_history.len(),
        100 - app.layout_config.stats_width_percent
    );

    let list = List::new(visible_items)
        .block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(list, chunks[0]);

    // Render scrollbar if content is scrollable and we have space for it
    if max_scroll > 0 && chunks.len() > 1 && chunks[1].height > 2 {
        let mut scrollbar_state = ScrollbarState::default()
            .content_length(content_height)
            .viewport_content_length(visible_height)
            .position(scroll_offset);
            
        let scrollbar = Scrollbar::default()
            .orientation(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("↑"))
            .end_symbol(Some("↓"));
            
        f.render_stateful_widget(scrollbar, chunks[1], &mut scrollbar_state);
    }
}

fn draw_board(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title("Game Board");
    let inner_area = block.inner(area);
    f.render_widget(block, area);

    match &app.game_wrapper {
        GameWrapper::Blokus(state) => draw_blokus_board(f, state, inner_area),
        _ => draw_standard_board(f, app, inner_area),
    }
}

fn draw_standard_board(f: &mut Frame, app: &App, area: Rect) {
    let board = app.game_wrapper.get_board();
    let board_height = board.len();
    let board_width = if board_height > 0 { board[0].len() } else { 0 };

    if board_height == 0 || board_width == 0 {
        let paragraph = Paragraph::new("No board to display");
        f.render_widget(paragraph, area);
        return;
    }

    // Calculate column width based on board size for optimal display
    // Since row height is 1, we need column width to be closer to 1 for square cells
    let col_width = match &app.game_wrapper {
        GameWrapper::Connect4(_) => 2, // Reduced for better aspect ratio
        GameWrapper::Othello(_) => 2,  // Reduced for better aspect ratio
        _ => 2, // Standard width for X/O
    };

    // Create row layout
    let row_constraints = vec![Constraint::Length(1); board_height];
    let board_area = Layout::default()
        .constraints(row_constraints)
        .split(area);

    for (r, row) in board.iter().enumerate() {
        // Create column layout for this row
        let col_constraints = vec![Constraint::Length(col_width); board_width];
        let row_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(col_constraints)
            .split(board_area[r]);

        for (c, &cell) in row.iter().enumerate() {
            let is_cursor = (r as u16, c as u16) == app.board_cursor;
            
            let (symbol, style) = match &app.game_wrapper {
                GameWrapper::Connect4(_) => {
                    match cell {
                        1 => ("🔴", Style::default().fg(Color::Red)),
                        -1 => ("🟡", Style::default().fg(Color::Yellow)),
                        _ => {
                            if is_cursor && !app.is_current_player_ai() {
                                ("▓", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                            } else {
                                ("·", Style::default().fg(Color::DarkGray))
                            }
                        }
                    }
                }
                GameWrapper::Othello(_) => {
                    match cell {
                        1 => ("⚫", Style::default().fg(Color::White)),
                        -1 => ("⚪", Style::default().fg(Color::White)),
                        _ => {
                            if is_cursor && !app.is_current_player_ai() {
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
                            if is_cursor && !app.is_current_player_ai() {
                                ("▓", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                            } else {
                                ("·", Style::default().fg(Color::DarkGray))
                            }
                        }
                    }
                }
            };

            let final_style = if is_cursor && cell != 0 && !app.is_current_player_ai() {  
                style.bg(Color::Yellow)
            } else {
                style
            };

            let paragraph = Paragraph::new(symbol)
                .style(final_style)
                .alignment(Alignment::Center);
            f.render_widget(paragraph, row_area[c]);
        }
    }
}

fn draw_blokus_board(f: &mut Frame, state: &BlokusState, area: Rect) {
    let board = state.get_board();
    let board_height = board.len();
    let board_width = if board_height > 0 { board[0].len() } else { 0 };

    if board_height == 0 || board_width == 0 {
        let paragraph = Paragraph::new("No board to display");
        f.render_widget(paragraph, area);
        return;
    }

    // Get last move positions for highlighting
    let last_move_positions: std::collections::HashSet<(usize, usize)> = state.get_last_move()
        .map(|coords| coords.into_iter().collect())
        .unwrap_or_default();

    // For Blokus, create a symmetrical grid with touching squares
    let mut board_lines = Vec::new();
    
    for (r, row) in board.iter().enumerate() {
        let mut line_spans = Vec::new();
        for (c, &cell) in row.iter().enumerate() {
            let is_last_move = last_move_positions.contains(&(r, c));
            
            let (symbol, style) = match cell {
                1 => {
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
                _ => ("░░", Style::default().fg(Color::DarkGray)), // Empty space
            };

            line_spans.push(Span::styled(symbol, style));
        }
        board_lines.push(Line::from(line_spans));
    }

    let paragraph = Paragraph::new(board_lines)
        .block(Block::default().borders(Borders::ALL).title("Blokus Board"));
    f.render_widget(paragraph, area);
}
