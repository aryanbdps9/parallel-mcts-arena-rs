//! # UI Widgets Module
//!
//! This module contains functions for drawing the different UI components (widgets)
//! on the screen, such as the game board, statistics, and menus.

use crate::app::{App, AppMode, GameStatus, Player};
use crate::game_wrapper::GameWrapper;
use crate::games::blokus::BlokusState;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
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

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Select a Game"))
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

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(format!("{} - Player Configuration", app.get_selected_game_name())))
        .highlight_symbol("> ");

    f.render_widget(list, chunks[0]);

    // Instructions
    let instructions = Paragraph::new("Use Up/Down to navigate, Left/Right/Space to toggle player type, Enter to confirm, Esc to go back")
        .block(Block::default().borders(Borders::ALL).title("Instructions"));
    f.render_widget(instructions, chunks[1]);
}

fn draw_game_view(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    draw_board(f, app, chunks[0]);
    draw_info(f, app, chunks[1]);
}

fn draw_info(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    draw_game_info(f, app, chunks[0]);
    draw_move_history(f, app, chunks[1]);
}

fn draw_game_info(f: &mut Frame, app: &App, area: Rect) {
    let mut text = vec![
        Line::from(format!("Game: {}", app.get_selected_game_name())),
        Line::from(format!("Status: {:?}", app.game_status)),
        Line::from(""),
    ];

    // Show current player
    let current_player = app.game_wrapper.get_current_player();
    let player_type = app.player_options
        .iter()
        .find(|(id, _)| *id == current_player)
        .map(|(_, p_type)| p_type)
        .unwrap_or(&Player::Human);

    let current_player_text = match app.game_wrapper {
        GameWrapper::Blokus(_) => format!("Current Player: Player {} ({:?})", current_player, player_type),
        _ => {
            let symbol = if current_player == 1 { "X" } else { "O" };
            format!("Current Player: {} ({:?})", symbol, player_type)
        }
    };
    text.push(Line::from(current_player_text));
    text.push(Line::from(""));

    // Show AI statistics if available
    if let Some(stats) = &app.last_search_stats {
        let mut sorted_children: Vec<_> = stats.children_stats.iter().collect();
        sorted_children.sort_by_key(|(_, (_, visits))| *visits);
        sorted_children.reverse();

        text.push(Line::from(format!("AI Nodes: {}", stats.total_nodes)));
        text.push(Line::from(format!("Root Visits: {}", stats.root_visits)));
        text.push(Line::from(format!("Root Value: {:.3}", stats.root_value)));
        text.push(Line::from(""));
        text.push(Line::from("Top Moves:"));
        for (m, (q, n)) in sorted_children.iter().take(5) {
            text.push(Line::from(format!("  {}: {:.3} ({})", m, q, n)));
        }
        text.push(Line::from(""));
    }

    // Game-specific instructions
    let instructions = match app.mode {
        AppMode::InGame => {
            if app.game_status == GameStatus::InProgress {
                match player_type {
                    Player::Human => "Use arrow keys to move cursor, Enter/Space to make move",
                    Player::AI => "AI is thinking...",
                }
            } else {
                "Press 'r' to restart, Esc for menu"
            }
        }
        AppMode::GameOver => "Press 'r' to restart, Esc for menu",
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
    let items: Vec<ListItem> = app
        .move_history
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let player_symbol = match &app.game_wrapper {
                GameWrapper::Blokus(_) => format!("P{}", entry.player),
                _ => if entry.player == 1 { "X".to_string() } else { "O".to_string() },
            };
            ListItem::new(format!("{}. {}: {}", i + 1, player_symbol, entry.a_move))
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Move History"));
    f.render_widget(list, area);
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

    // Create board display with cursor
    let mut board_lines = Vec::new();
    
    for (r, row) in board.iter().enumerate() {
        let mut line_spans = Vec::new();
        for (c, &cell) in row.iter().enumerate() {
            let is_cursor = (r as u16, c as u16) == app.board_cursor;
            
            let (symbol, style) = match cell {
                1 => ("X", Style::default().fg(Color::Red)),
                -1 => ("O", Style::default().fg(Color::Blue)),
                _ => {
                    if is_cursor {
                        ("▢", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                    } else {
                        ("·", Style::default().fg(Color::DarkGray))
                    }
                }
            };

            let final_style = if is_cursor && cell != 0 {
                style.bg(Color::Yellow)
            } else {
                style
            };

            line_spans.push(Span::styled(format!("{} ", symbol), final_style));
        }
        board_lines.push(Line::from(line_spans));
    }

    let paragraph = Paragraph::new(board_lines);
    f.render_widget(paragraph, area);
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

    // For Blokus, create a more compact display
    let mut board_lines = Vec::new();
    
    for row in board.iter() {
        let mut line_spans = Vec::new();
        for &cell in row.iter() {
            let (symbol, style) = match cell {
                1 => ("1", Style::default().fg(Color::Black).bg(Color::Red)),
                2 => ("2", Style::default().fg(Color::Black).bg(Color::Blue)),
                3 => ("3", Style::default().fg(Color::Black).bg(Color::Green)),
                4 => ("4", Style::default().fg(Color::Black).bg(Color::Yellow)),
                _ => ("·", Style::default().fg(Color::DarkGray)),
            };

            line_spans.push(Span::styled(symbol, style));
        }
        board_lines.push(Line::from(line_spans));
    }

    let paragraph = Paragraph::new(board_lines);
    f.render_widget(paragraph, area);
}
