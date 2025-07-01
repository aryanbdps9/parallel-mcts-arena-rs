//! # UI Widgets Module
//!
//! This module contains functions for rendering the different UI components (widgets)
//! of the terminal user interface, including game boards, statistics panels, menus,
//! and game-specific overlays.
//!
//! ## Main Components
//! - **Game Selection Menu**: Main menu for choosing games and accessing settings
//! - **Settings Menu**: Configuration interface for game parameters and AI settings
//! - **Player Configuration**: Setup interface for human vs AI players
//! - **Game View**: In-game interface with board, stats, and controls
//! - **Move History**: Scrollable display of game moves with auto-scroll
//! - **Debug Statistics**: Real-time AI search statistics and performance metrics
//!
//! ## Game-Specific Views
//! - **Standard View**: Layout for 2-player games (Othello, Connect4, Gomoku)
//! - **Blokus View**: Specialized 4-player layout with piece selection panels

use crate::app::{App, AppMode, GameStatus, Player};
use crate::game_wrapper::GameWrapper;
use crate::games::blokus::BlokusState;
use crate::tui::blokus_ui;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState};
use mcts::GameState;

/// Main rendering function that dispatches to appropriate view based on application mode
///
/// This function serves as the entry point for all UI rendering, determining which
/// interface to display based on the current application state.
///
/// # Arguments
/// * `app` - Mutable reference to the application state
/// * `frame` - Ratatui frame for rendering widgets
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

/// Draws the main game selection menu
///
/// Displays a list of available games plus settings and quit options.
/// Shows different titles for AI-only mode vs normal mode.
///
/// # Arguments
/// * `f` - Ratatui frame for rendering
/// * `app` - Mutable application state (for list widget state)
/// * `area` - Screen area to render within
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

/// Draws the settings configuration menu
///
/// Displays all configurable game parameters including board size, AI settings,
/// and gameplay options. Shows current values and allows adjustment.
///
/// # Arguments
/// * `f` - Ratatui frame for rendering
/// * `app` - Application state containing settings values
/// * `area` - Screen area to render within
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

/// Draws the player configuration menu
///
/// Allows setting each player as Human or AI before starting a game.
/// In AI-only mode, shows informational message about automatic AI assignment.
///
/// # Arguments
/// * `f` - Ratatui frame for rendering
/// * `app` - Application state containing player configurations
/// * `area` - Screen area to render within
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

/// Dispatches to appropriate game view renderer
///
/// Determines whether to use the standard 2-player layout or the specialized
/// Blokus 4-player layout based on the current game type.
///
/// # Arguments
/// * `f` - Ratatui frame for rendering
/// * `app` - Application state containing game information
/// * `area` - Screen area to render within
fn draw_game_view(f: &mut Frame, app: &App, area: Rect) {
    if matches!(app.game_wrapper, GameWrapper::Blokus(_)) {
        draw_blokus_game_view(f, app, area);
    } else {
        draw_standard_game_view(f, app, area);
    }
}

/// Renders the standard game view for 2-player games
///
/// Uses a three-panel vertical layout: game board at top, game info in middle,
/// and horizontally split debug stats and move history at bottom.
/// Supports resizable panels via layout configuration.
///
/// # Arguments
/// * `f` - Ratatui frame for rendering
/// * `app` - Application state containing layout and game data
/// * `area` - Screen area to render within
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

/// Renders the specialized Blokus game view
///
/// Uses a custom layout with the game board on the left, piece selection panel
/// in the center, player status on the right, and game info/stats at the bottom.
/// Includes ghost piece preview and drag-and-drop piece placement.
///
/// # Arguments
/// * `f` - Ratatui frame for rendering
/// * `app` - Application state containing Blokus-specific UI state
/// * `area` - Screen area to render within
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

/// Draws the debug statistics panel
///
/// Displays real-time AI search statistics including node counts, evaluations,
/// and top move candidates. Shows scrollable content with draggable panel borders.
/// Updates automatically during AI thinking phases.
///
/// # Arguments
/// * `f` - Ratatui frame for rendering
/// * `app` - Application state containing MCTS statistics
/// * `area` - Screen area to render within
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

/// Draws the game information panel
///
/// Shows current game status, active player with color-coded indicators,
/// AI thinking status, basic statistics, and context-appropriate instructions.
/// Adapts display format based on game type (2-player vs 4-player).
///
/// # Arguments
/// * `f` - Ratatui frame for rendering
/// * `app` - Application state containing game and player information
/// * `area` - Screen area to render within
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

/// Draws the move history panel
///
/// Displays a scrollable, color-coded history of all moves made in the current game.
/// Uses different formatting for 2-player games (side-by-side pairs) vs 4-player
/// Blokus (4 moves per line). Supports auto-scroll to follow recent moves and
/// manual scrolling with preserved position.
///
/// # Arguments
/// * `f` - Ratatui frame for rendering
/// * `app` - Application state containing move history and scroll settings
/// * `area` - Screen area to render within
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
    };    // Group moves side-by-side based on game type
    let formatted_lines = match &app.game_wrapper {
        GameWrapper::Blokus(_) => {
            // For Blokus (4 players), group moves in sets of 4 per line
            format_blokus_moves_sidebyside_colored(&app.move_history, chunks[0].width.saturating_sub(2) as usize, app)
        },
        _ => {
            // For 2-player games, group moves in pairs per line
            format_two_player_moves_sidebyside_colored(&app.move_history, chunks[0].width.saturating_sub(2) as usize, app)
        }
    };
    
    // Calculate scrolling for text content using auto-scroll logic
    let content_height = formatted_lines.len();
    let visible_height = chunks[0].height.saturating_sub(2) as usize; // Account for borders
    let scroll_offset = app.get_history_scroll_offset(content_height, visible_height);
    
    // Take visible lines for display
    let visible_lines = formatted_lines
        .into_iter()
        .skip(scroll_offset)
        .take(visible_height)
        .collect::<Vec<Line>>();

    let drag_indicator = if app.drag_state.is_dragging { "🔀" } else { "↔" };
    let auto_scroll_indicator = if app.history_auto_scroll { "📜" } else { "📌" };
    let title = format!("{} {} Move History ({}) - {}%", 
        drag_indicator, 
        auto_scroll_indicator,
        app.move_history.len(),
        100 - app.layout_config.stats_width_percent
    );

    let paragraph = Paragraph::new(visible_lines)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(ratatui::widgets::Wrap { trim: true }); // Enable word wrap
    f.render_widget(paragraph, chunks[0]);

    // Render scrollbar if content is scrollable and we have space for it
    let max_scroll = content_height.saturating_sub(visible_height);
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

/// Formats move history for 2-player games with color-coded side-by-side display
///
/// Groups moves in pairs per line when space allows, with appropriate spacing
/// and color coding for each player. Falls back to separate lines for long moves.
///
/// # Arguments
/// * `move_history` - Slice of move history entries to format
/// * `max_width` - Maximum display width for text fitting
/// * `app` - Application state for color and symbol lookup
///
/// # Returns
/// Vector of styled text lines ready for display
fn format_two_player_moves_sidebyside_colored<'a>(move_history: &'a [crate::app::MoveHistoryEntry], max_width: usize, app: &'a App) -> Vec<Line<'a>> {
    use ratatui::prelude::*;
    let mut result = Vec::new();
    let mut moves_iter = move_history.iter().enumerate();
    
    while let Some((i, first_move)) = moves_iter.next() {
        let move_number = (i / 2) + 1;
        
        // Get player color and symbol
        let first_player_color = app.get_player_color(first_move.player);
        let first_player_symbol = app.get_player_symbol(first_move.player);
        
        // Format first player's move with color
        let first_player_spans = vec![
            Span::styled(format!("{}. ", move_number), Style::default().fg(Color::Gray)),
            Span::styled(first_player_symbol, Style::default()),
            Span::styled(format!(" {}", first_move.a_move), Style::default().fg(first_player_color).add_modifier(Modifier::BOLD)),
        ];
        
        // Check if there's a second move for this round
        if let Some((_, second_move)) = moves_iter.next() {
            let second_player_color = app.get_player_color(second_move.player);
            let second_player_symbol = app.get_player_symbol(second_move.player);
            
            // Calculate approximate text length for spacing
            let first_text_len = format!("{}. {} {}", move_number, first_player_symbol, first_move.a_move).len();
            let second_text_len = format!("{} {}", second_player_symbol, second_move.a_move).len();
            let combined_length = first_text_len + second_text_len + 3; // 3 spaces minimum
            
            if combined_length <= max_width {
                // Fit on one line with spacing
                let spacing = " ".repeat((max_width - first_text_len - second_text_len).max(3).min(10));
                let mut line_spans = first_player_spans;
                line_spans.push(Span::styled(spacing, Style::default()));
                line_spans.push(Span::styled(second_player_symbol, Style::default()));
                line_spans.push(Span::styled(format!(" {}", second_move.a_move), Style::default().fg(second_player_color).add_modifier(Modifier::BOLD)));
                result.push(Line::from(line_spans));
            } else {
                // Too long, put second move on new line with indentation
                result.push(Line::from(first_player_spans));
                let second_player_spans = vec![
                    Span::styled("    ", Style::default()),
                    Span::styled(second_player_symbol, Style::default()),
                    Span::styled(format!(" {}", second_move.a_move), Style::default().fg(second_player_color).add_modifier(Modifier::BOLD)),
                ];
                result.push(Line::from(second_player_spans));
            }
        } else {
            // Only first move exists
            result.push(Line::from(first_player_spans));
        }
    }
    
    result
}

/// Formats move history for 4-player Blokus with grouped rounds
///
/// Groups moves by rounds (4 moves per round) with round headers and
/// color-coded player moves. Handles text wrapping for long move descriptions.
///
/// # Arguments
/// * `move_history` - Slice of move history entries to format
/// * `max_width` - Maximum display width for text fitting
/// * `app` - Application state for color and symbol lookup
///
/// # Returns
/// Vector of styled text lines ready for display
fn format_blokus_moves_sidebyside_colored<'a>(move_history: &'a [crate::app::MoveHistoryEntry], max_width: usize, app: &'a App) -> Vec<Line<'a>> {
    use ratatui::prelude::*;
    let mut result = Vec::new();
    let mut moves_iter = move_history.chunks(4).enumerate();
    
    while let Some((round, round_moves)) = moves_iter.next() {
        let move_number = round + 1;
        result.push(Line::from(Span::styled(format!("Round {}:", move_number), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
        
        // Collect all moves for this round with their colors
        let mut move_spans = Vec::new();
        for (i, move_entry) in round_moves.iter().enumerate() {
            let player_color = app.get_player_color(move_entry.player);
            let player_symbol = app.get_player_symbol(move_entry.player);
            
            if i > 0 {
                move_spans.push(Span::styled(" | ", Style::default().fg(Color::Gray)));
            }
            move_spans.push(Span::styled(player_symbol, Style::default()));
            move_spans.push(Span::styled(format!(" {}", move_entry.a_move), Style::default().fg(player_color).add_modifier(Modifier::BOLD)));
        }
        
        // Check if the line fits
        let line_text = format!("  {}", round_moves.iter().map(|m| format!("{} {}", app.get_player_symbol(m.player), m.a_move)).collect::<Vec<_>>().join(" | "));
        if line_text.len() <= max_width {
            // Fit on one line
            let mut line_content = vec![Span::styled("  ", Style::default())];
            line_content.extend(move_spans);
            result.push(Line::from(line_content));
        } else {
            // Wrap moves - put 2 per line if possible
            let move_pairs: Vec<_> = round_moves.chunks(2).collect();
            for pair in move_pairs {
                let mut pair_spans = vec![Span::styled("  ", Style::default())];
                for (i, move_entry) in pair.iter().enumerate() {
                    let player_color = app.get_player_color(move_entry.player);
                    let player_symbol = app.get_player_symbol(move_entry.player);
                    
                    if i > 0 {
                        pair_spans.push(Span::styled(" | ", Style::default().fg(Color::Gray)));
                    }
                    pair_spans.push(Span::styled(player_symbol, Style::default()));
                    pair_spans.push(Span::styled(format!(" {}", move_entry.a_move), Style::default().fg(player_color).add_modifier(Modifier::BOLD)));
                }
                result.push(Line::from(pair_spans));
            }
        }
    }
    
    result
}

/// Dispatches board rendering to appropriate game-specific function
///
/// Creates a bordered frame and delegates to the correct board renderer
/// based on the current game type.
///
/// # Arguments
/// * `f` - Ratatui frame for rendering
/// * `app` - Application state containing game board data
/// * `area` - Screen area to render within
fn draw_board(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title("Game Board");
    let inner_area = block.inner(area);
    f.render_widget(block, area);

    match &app.game_wrapper {
        GameWrapper::Blokus(state) => draw_blokus_board(f, state, inner_area),
        _ => draw_standard_board(f, app, inner_area),
    }
}

/// Renders game boards for standard 2-player games
///
/// Handles display of Othello, Connect4, and Gomoku boards with appropriate
/// symbols and colors for each game type. Shows cursor position for human players
/// and highlights the last move made. Includes row and column labels for navigation.
///
/// # Arguments
/// * `f` - Ratatui frame for rendering
/// * `app` - Application state containing board and cursor data
/// * `area` - Screen area to render within
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
    let col_width = match &app.game_wrapper {
        GameWrapper::Connect4(_) => 2, // Reduced for better aspect ratio
        GameWrapper::Othello(_) => 2,  // Reduced for better aspect ratio
        _ => 2, // Standard width for X/O
    };

    // Determine if we need row labels (not for Connect4)
    let needs_row_labels = !matches!(app.game_wrapper, GameWrapper::Connect4(_));
    let row_label_width = if needs_row_labels { 2 } else { 0 };

    // Create layout with space for labels
    let mut layout_constraints = Vec::new();
    
    // Column header row
    layout_constraints.push(Constraint::Length(1));
    
    // Board rows
    for _ in 0..board_height {
        layout_constraints.push(Constraint::Length(1));
    }
    
    let rows_layout = Layout::default()
        .constraints(layout_constraints)
        .split(area);

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
                           !app.is_current_player_ai();
        
        let style = if is_cursor_col {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD).bg(Color::Blue)
        } else {
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
        };
        
        let paragraph = Paragraph::new(col_letter.to_string())
            .style(style)
            .alignment(Alignment::Center);
        f.render_widget(paragraph, col_label_area[col_start_idx + c]);
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
            f.render_widget(paragraph, cell_areas[0]);
        }

        // Draw board cells
        let cell_start_idx = if needs_row_labels { 1 } else { 0 };
        for (c, &cell) in row.iter().enumerate() {
            let is_cursor = matches!(app.game_wrapper, GameWrapper::Connect4(_)) == false && 
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
            f.render_widget(paragraph, cell_areas[cell_start_idx + c]);
        }
    }
}

/// Renders the Blokus game board with piece placement visualization
///
/// Displays the 20x20 Blokus board with colored squares for each player's pieces.
/// Highlights the most recent move and can show ghost pieces for preview during
/// human player turns.
///
/// # Arguments
/// * `f` - Ratatui frame for rendering
/// * `state` - Current Blokus game state
/// * `area` - Screen area to render within
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
