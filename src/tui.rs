//! # Terminal User Interface Module
//!
//! This module provides the interactive terminal interface for the multi-game engine.
//! It uses Ratatui for rendering and Crossterm for input handling, creating a rich
//! terminal experience with mouse support, resizable panels, and real-time updates.
//!
//! ## Key Features
//! - **Multi-game UI**: Adaptive interface for all game types
//! - **Mouse Support**: Click to move, drag to resize panels
//! - **Keyboard Navigation**: Full keyboard support for all operations
//! - **Real-time Updates**: Live AI statistics and game state updates
//! - **Responsive Layout**: Automatically adjusts to terminal size
//! - **Scrollable Panels**: History and debug info with scroll support
//!
//! ## UI Layout
//! The interface is divided into three main areas:
//! 1. **Game Board**: Interactive game board with cursor and move highlighting
//! 2. **Instructions**: Game controls and current player information
//! 3. **Statistics & History**: Split between AI debug info and move history

use crate::{App, AppState, DragBoundary, PlayerType};
use crate::game_wrapper::{GameWrapper, MoveWrapper};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, MouseButton, MouseEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use mcts::GameState;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
};
use std::io;
use std::time::{Duration, Instant};

pub fn run_tui(app: &mut App) -> io::Result<()> {
    // setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // create app and run it
    let res = run_app(&mut terminal, app);

    // restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{err:?}");
    }

    Ok(())
}

fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> io::Result<()> {
    let mut last_key_event = Instant::now();
    let mut last_movement_key_event = Instant::now();
    
    loop {
        // Check for terminal size changes
        let terminal_size = terminal.size()?;
        if (terminal_size.width, terminal_size.height) != app.last_terminal_size {
            app.handle_window_resize(terminal_size.width, terminal_size.height);
        }
        
        terminal.draw(|f| ui(f, app))?;
        app.tick();

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) => {
                    // Always check for 'q' to quit, regardless of state and debounce
                    if key.code == KeyCode::Char('q') || key.code == KeyCode::Char('Q') {
                        return Ok(());
                    }
                    
                    // Check for Escape key as alternative quit method
                    if key.code == KeyCode::Esc {
                        return Ok(());
                    }
                    
                    // Apply aggressive debouncing for movement keys to prevent double-moves
                    let is_movement_key = matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Left | KeyCode::Right);
                    let movement_debounce = Duration::from_millis(200); // 200ms for movement keys
                    let general_debounce = Duration::from_millis(50);   // 50ms for other keys
                    
                    let should_process = if is_movement_key {
                        last_movement_key_event.elapsed() > movement_debounce
                    } else {
                        last_key_event.elapsed() > general_debounce
                    };
                    
                    if should_process {
                        match app.state {
                            AppState::Menu => match key.code {
                                KeyCode::Down => app.next(),
                                KeyCode::Up => app.previous(),
                                KeyCode::Enter => {
                                    if app.index == app.titles.len() - 1 { // Quit
                                        return Ok(());
                                    } else if app.index == app.titles.len() - 2 { // Settings
                                        app.state = AppState::Settings;
                                    } else {
                                        // When a game is selected, go to PlayerConfig first
                                        app.set_game(app.index);
                                        app.state = AppState::PlayerConfig;
                                    }
                                }
                                _ => {}
                            },
                            AppState::Settings => match key.code {
                                KeyCode::Down => app.settings_next(),
                                KeyCode::Up => app.settings_previous(),
                                KeyCode::Left => app.decrease_setting(),
                                KeyCode::Right => app.increase_setting(),
                                KeyCode::Enter => {
                                    if app.settings_index == app.settings_titles.len() - 1 { // Back to Menu
                                        app.state = AppState::Menu;
                                    }
                                }
                                KeyCode::Char('m') => app.state = AppState::Menu,
                                _ => {}
                            },
                            AppState::Playing => {
                                match key.code {
                                    KeyCode::Down => {
                                        if key.modifiers.contains(crossterm::event::KeyModifiers::SHIFT) {
                                            app.scroll_move_history_down();
                                        } else if !app.ai_only && app.game_type != "connect4" {
                                            app.move_cursor_down();
                                        }
                                    },
                                    KeyCode::Up => {
                                        if key.modifiers.contains(crossterm::event::KeyModifiers::SHIFT) {
                                            app.scroll_move_history_up();
                                        } else if !app.ai_only && app.game_type != "connect4" {
                                            app.move_cursor_up();
                                        }
                                    },
                                    KeyCode::Left => {
                                        if !app.ai_only {
                                            app.move_cursor_left();
                                        }
                                    },
                                    KeyCode::Right => {
                                        if !app.ai_only {
                                            app.move_cursor_right();
                                        }
                                    },
                                    KeyCode::Enter => {
                                        if !app.ai_only {
                                            if app.game_type == "blokus" {
                                                app.blokus_place_piece();
                                            } else {
                                                app.submit_move();
                                            }
                                        }
                                    },
                                    KeyCode::Char('r') => {
                                        if app.game_type == "blokus" {
                                            app.blokus_rotate_piece();
                                        }
                                    },
                                    KeyCode::Char('f') => {
                                        if app.game_type == "blokus" {
                                            app.blokus_flip_piece();
                                        }
                                    },
                                    KeyCode::Tab => {
                                        if app.game_type == "blokus" {
                                            app.blokus_cycle_pieces(true);
                                        }
                                    },
                                    KeyCode::BackTab => {
                                        if app.game_type == "blokus" {
                                            app.blokus_cycle_pieces(false);
                                        }
                                    },
                                    // Number keys for piece selection
                                    KeyCode::Char('1') => if app.game_type == "blokus" { app.blokus_select_piece(0); },
                                    KeyCode::Char('2') => if app.game_type == "blokus" { app.blokus_select_piece(1); },
                                    KeyCode::Char('3') => if app.game_type == "blokus" { app.blokus_select_piece(2); },
                                    KeyCode::Char('4') => if app.game_type == "blokus" { app.blokus_select_piece(3); },
                                    KeyCode::Char('5') => if app.game_type == "blokus" { app.blokus_select_piece(4); },
                                    KeyCode::Char('6') => if app.game_type == "blokus" { app.blokus_select_piece(5); },
                                    KeyCode::Char('7') => if app.game_type == "blokus" { app.blokus_select_piece(6); },
                                    KeyCode::Char('8') => if app.game_type == "blokus" { app.blokus_select_piece(7); },
                                    KeyCode::Char('9') => if app.game_type == "blokus" { app.blokus_select_piece(8); },
                                    KeyCode::Char('0') => if app.game_type == "blokus" { app.blokus_select_piece(9); },
                                    // Letter keys for pieces 11-36 (a-z)
                                    KeyCode::Char(c) if c >= 'a' && c <= 'z' && app.game_type == "blokus" => {
                                        let piece_idx = (c as u8 - b'a') as usize + 10;
                                        app.blokus_select_piece(piece_idx);
                                    },
                                    KeyCode::Char('m') => app.state = AppState::Menu,
                                    // Expand/collapse controls for Blokus
                                    KeyCode::Char('+') | KeyCode::Char('=') => {
                                        if app.game_type == "blokus" {
                                            app.blokus_expand_all_players();
                                        }
                                    },
                                    KeyCode::Char('-') | KeyCode::Char('_') => {
                                        if app.game_type == "blokus" {
                                            app.blokus_collapse_all_players();
                                        }
                                    },
                                    KeyCode::Char('e') | KeyCode::Char('E') => {
                                        if app.game_type == "blokus" {
                                            app.blokus_toggle_current_player_expand();
                                        }
                                    },
                                    // Panel scrolling (primary)
                                    KeyCode::Char('[') => {
                                        if app.game_type == "blokus" {
                                            app.blokus_scroll_panel_up();
                                        }
                                    },
                                    KeyCode::Char(']') => {
                                        if app.game_type == "blokus" {
                                            app.blokus_scroll_panel_down();
                                        }
                                    },
                                    // Per-player piece scrolling (secondary)
                                    KeyCode::Char('{') => {
                                        if app.game_type == "blokus" {
                                            app.blokus_scroll_pieces_up();
                                        }
                                    },
                                    KeyCode::Char('}') => {
                                        if app.game_type == "blokus" {
                                            app.blokus_scroll_pieces_down();
                                        }
                                    },
                                    KeyCode::PageUp => app.scroll_debug_up(),
                                    KeyCode::PageDown => app.scroll_debug_down(),
                                    _ => {}
                                }
                            }
                            AppState::GameOver => match key.code {
                                KeyCode::Char('r') => app.reset(),
                                KeyCode::Char('m') => app.state = AppState::Menu,
                                KeyCode::PageUp => app.scroll_debug_up(),
                                KeyCode::PageDown => app.scroll_debug_down(),
                                KeyCode::Down => {
                                    if key.modifiers.contains(crossterm::event::KeyModifiers::SHIFT) {
                                        app.scroll_move_history_down();
                                    }
                                },
                                KeyCode::Up => {
                                    if key.modifiers.contains(crossterm::event::KeyModifiers::SHIFT) {
                                        app.scroll_move_history_up();
                                    }
                                },
                                _ => {}
                            },
                            AppState::PlayerConfig => {
                                if let Event::Key(key) = event::read()? {
                                    match key.code {
                                        KeyCode::Esc => {
                                            app.state = AppState::Menu;
                                        }
                                        KeyCode::Up => {
                                            if app.player_config_index > 0 {
                                                app.player_config_index -= 1;
                                            }
                                        }
                                        KeyCode::Down => {
                                            // Allow navigating to Launch button (index = player_types.len())
                                            if app.player_config_index + 1 <= app.player_types.len() {
                                                app.player_config_index += 1;
                                            }
                                        }
                                        KeyCode::Left | KeyCode::Right | KeyCode::Char(' ') => {
                                            // Only toggle if we're on a player, not on Launch button
                                            if app.player_config_index < app.player_types.len() {
                                                app.toggle_player_type(app.player_config_index);
                                            }
                                        }
                                        KeyCode::Enter | KeyCode::Char('l') | KeyCode::Char('L') => {
                                            if app.player_config_index < app.player_types.len() {
                                                // If on a player slot, toggle the player type
                                                app.toggle_player_type(app.player_config_index);
                                            } else {
                                                // If on Launch button, start the game
                                                app.state = AppState::Playing;
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            },
                        }
                        // Update timing variables based on key type
                        if is_movement_key {
                            last_movement_key_event = Instant::now();
                        }
                        last_key_event = Instant::now();
                    }
                }
                Event::Mouse(mouse) => {
                    match mouse.kind {
                        MouseEventKind::Down(MouseButton::Left) => {
                            handle_mouse_click(app, mouse.column, mouse.row, terminal.size()?);
                        }
                        MouseEventKind::Drag(MouseButton::Left) => {
                            handle_mouse_drag(app, mouse.column, mouse.row, terminal.size()?);
                        }
                        MouseEventKind::Up(MouseButton::Left) => {
                            handle_mouse_release(app, mouse.column, mouse.row, terminal.size()?);
                        }
                        MouseEventKind::ScrollUp => {
                            handle_mouse_scroll(app, mouse.column, mouse.row, terminal.size()?, true);
                        }
                        MouseEventKind::ScrollDown => {
                            handle_mouse_scroll(app, mouse.column, mouse.row, terminal.size()?, false);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    // Update scroll bounds to ensure they're consistent with current terminal size
    app.update_move_history_scroll_bounds(f.size().height);
    
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(app.get_layout_constraints().as_ref())
        .split(f.size());

    match app.state {
        AppState::Menu => {
            let titles: Vec<ListItem> = app
                .titles
                .iter()
                .map(|t| ListItem::new(*t))
                .collect();

            let list = List::new(titles)
                .block(Block::default().title("Menu").borders(Borders::ALL))
                .highlight_style(Style::default().add_modifier(Modifier::BOLD))
                .highlight_symbol("> ");
            let mut list_state = ListState::default();
            list_state.select(Some(app.index));
            f.render_stateful_widget(list, main_chunks[0], &mut list_state);

            let instructions =
                Paragraph::new("Use arrow keys to navigate, Enter to select, or click with mouse. Press 'p' for Player Config. 'q' or Esc to quit.")
                    .block(Block::default().title("Instructions").borders(Borders::ALL));
            f.render_widget(instructions, main_chunks[1]);
        }
        AppState::Settings => {
            let settings_items: Vec<ListItem> = app
                .settings_titles
                .iter()
                .map(|t| ListItem::new(t.as_str()))
                .collect();

            let list = List::new(settings_items)
                .block(Block::default().title("Settings").borders(Borders::ALL))
                .highlight_style(Style::default().add_modifier(Modifier::BOLD))
                .highlight_symbol("> ");
            let mut list_state = ListState::default();
            list_state.select(Some(app.settings_index));
            f.render_stateful_widget(list, main_chunks[0], &mut list_state);

            let instructions = Paragraph::new(
                "Use arrow keys to navigate, Left/Right to change values, Enter to select 'Back to Menu', or 'm' for menu. 'q' or Esc to quit."
            )
            .block(Block::default().title("Instructions").borders(Borders::ALL));
            f.render_widget(instructions, main_chunks[1]);
        }
        AppState::Playing | AppState::GameOver => {
            if app.game_type == "blokus" {
                draw_blokus_ui(f, app, main_chunks[0]);
            } else {
                draw_board(f, app, main_chunks[0]);
            }

            let instructions_text = if !app.game.is_terminal() {
                if app.ai_only {
                    if app.is_ai_thinking() {
                        "AI vs AI mode - AI is thinking... Press 'm' for menu, 'q' or Esc to quit. PageUp/PageDown to scroll debug info, Shift+Up/Down to scroll move history. Drag boundaries to resize panes.".to_string()
                    } else {
                        "AI vs AI mode - Press 'm' for menu, 'q' or Esc to quit. PageUp/PageDown to scroll debug info, Shift+Up/Down to scroll move history. Drag boundaries to resize panes.".to_string()
                    }
                } else {
                    if app.is_ai_thinking() {
                        if app.game_type == "connect4" {
                            "AI is thinking... Please wait. 'm' for menu, 'q' or Esc to quit. PageUp/PageDown to scroll debug info, Shift+Up/Down to scroll move history. Drag boundaries to resize panes.".to_string()
                        } else if app.game_type == "blokus" {
                            "AI is thinking... Please wait. 'm' for menu, 'q' or Esc to quit. PageUp/PageDown to scroll debug info, Shift+Up/Down to scroll move history. Drag boundaries to resize panes.".to_string()
                        } else {
                            "AI is thinking... Please wait. 'm' for menu, 'q' or Esc to quit. PageUp/PageDown to scroll debug info, Shift+Up/Down to scroll move history. Drag boundaries to resize panes.".to_string()
                        }
                    } else {
                        if app.game_type == "connect4" {
                            "Left/Right arrows to select column, Enter to drop piece, or click column numbers. 'm' for menu, 'q' or Esc to quit. PageUp/PageDown to scroll debug info, Shift+Up/Down to scroll move history. Drag boundaries to resize panes.".to_string()
                        } else                        if app.game_type == "blokus" {
                            "Arrow keys to move cursor, 1-9,0 to select pieces, R to rotate, F to flip, Tab/Shift+Tab to cycle pieces, Enter to place. 'm' for menu, 'q' or Esc to quit. PageUp/PageDown to scroll debug info, Shift+Up/Down to scroll move history. Drag boundaries to resize panes.".to_string()
                        } else {
                            "Arrow keys to move, Enter to place, or click on board. 'm' for menu, 'q' or Esc to quit. PageUp/PageDown to scroll debug info, Shift+Up/Down to scroll move history. Drag boundaries to resize panes.".to_string()
                        }
                    }
                }
            } else {
                let (winner_text, winner_color) = if let Some(winner) = app.winner {
                    match app.game {
                        GameWrapper::Blokus(_) => {
                            // Blokus is a 4-player game
                            match winner {
                                1 => ("Player 1 wins!", Color::Red),
                                2 => ("Player 2 wins!", Color::Blue),
                                3 => ("Player 3 wins!", Color::Green),
                                4 => ("Player 4 wins!", Color::Yellow),
                                _ => ("Unknown player wins!", Color::White),
                            }
                        }
                        _ => {
                            // 2-player games (Gomoku, Connect4, Othello)
                            if winner == 1 {
                                ("Player X wins!", Color::Red)
                            } else {
                                ("Player O wins!", Color::Blue)
                            }
                        }
                    }
                } else {
                    ("It's a draw!", Color::Yellow)
                };
                
                // Create styled instruction text for game over
                let instruction_spans = vec![
                    Span::styled(winner_text, Style::default().fg(winner_color).add_modifier(Modifier::BOLD)),
                    Span::raw(" Press 'r' to play again, 'm' for menu, 'q' or Esc to quit. PageUp/PageDown to scroll debug info, Shift+Up/Down to scroll move history. Drag boundaries to resize panes."),
                ];
                
                let drag_indicator = if app.is_dragging { "🔀" } else { "↕" };
                let instructions_title = format!("{} Game Over ({}%|{}%|{}%)", 
                    drag_indicator,
                    app.board_height_percent, 
                    app.instructions_height_percent, 
                    app.stats_height_percent);
                
                let instructions = Paragraph::new(Line::from(instruction_spans))
                    .block(Block::default().title(instructions_title).borders(Borders::ALL));
                f.render_widget(instructions, main_chunks[1]);
                
                // Split the stats area horizontally for Debug Statistics and Move History
                let stats_area = main_chunks[2];
                let horizontal_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(app.stats_width_percent),
                        Constraint::Percentage(100 - app.stats_width_percent),
                    ])
                    .split(stats_area);
                
                draw_stats(f, app, horizontal_chunks[0]);
                draw_move_history(f, app, horizontal_chunks[1]);
                return;
            };

            let instructions_title = if app.state == AppState::Playing || app.state == AppState::GameOver {
                let drag_indicator = if app.is_dragging { "🔀" } else { "↕" };
                format!("{} Instructions ({}%|{}%|{}%)", 
                    drag_indicator,
                    app.board_height_percent, 
                    app.instructions_height_percent, 
                    app.stats_height_percent)
            } else {
                "Instructions".to_string()
            };

            let instructions = Paragraph::new(instructions_text)
                .block(Block::default().title(instructions_title).borders(Borders::ALL));
            f.render_widget(instructions, main_chunks[1]);
            
            // Split the stats area horizontally for Debug Statistics and Move History
            let stats_area = main_chunks[2];
            let horizontal_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(app.stats_width_percent),
                    Constraint::Percentage(100 - app.stats_width_percent),
                ])
                .split(stats_area);
            
            draw_stats(f, app, horizontal_chunks[0]);
            draw_move_history(f, app, horizontal_chunks[1]);
        }
        AppState::PlayerConfig => {
            draw_player_config_menu(f, app, f.size());
        }
    }
}

fn draw_player_config_menu(f: &mut Frame, app: &App, area: Rect) {
    let n = app.player_types.len();
    let mut items = Vec::with_capacity(n + 1); // +1 for Launch button
    
    // Add player type options
    for (i, pt) in app.player_types.iter().enumerate() {
        let label = format!("Player {}: {}", i + 1, match pt {
            PlayerType::Human => "Human",
            PlayerType::AI => "AI",
        });
        let style = if i == app.player_config_index {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        items.push(ListItem::new(label).style(style));
    }
    
    // Add Launch button
    let launch_label = "Launch Game";
    let launch_style = if app.player_config_index == n {
        Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Green)
    };
    items.push(ListItem::new(launch_label).style(launch_style));
    
    // Create title with game name
    let game_name = app.game_type.to_uppercase();
    let title = format!("{} - Player Configuration (Up/Down: Navigate, Space/Left/Right: Toggle, Enter: Select/Launch, Esc: Back)", game_name);
    
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(list, area);
}

fn draw_stats(f: &mut Frame, app: &App, area: Rect) {
    // Early return if area is too small
    if area.height < 3 || area.width < 10 {
        let paragraph = Paragraph::new("Area too small")
            .block(Block::default().title("Debug Statistics").borders(Borders::ALL));
        f.render_widget(paragraph, area);
        return;
    }

    let current_player = app.game.get_current_player();
    let (player_symbol, player_color) = match app.game {
        GameWrapper::Blokus(_) => {
            // Blokus is a 4-player game
            match current_player {
                1 => ("Player 1", Color::Red),
                2 => ("Player 2", Color::Blue),
                3 => ("Player 3", Color::Green),
                4 => ("Player 4", Color::Yellow),
                _ => ("Unknown", Color::White),
            }
        }
        _ => {
            // 2-player games (Gomoku, Connect4, Othello)
            if current_player == 1 {
                ("X", Color::Red)
            } else {
                ("O", Color::Blue)
            }
        }
    };

    let mut stats_lines = vec![
        Line::from(vec![
            Span::raw("Current Player: "),
            Span::styled(player_symbol, Style::default().fg(player_color).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(format!("AI State: {:?} | Threads: {}", app.ai_state, app.num_threads)),
    ];

    // Show AI thinking status and time remaining
    if app.is_ai_thinking() {
        stats_lines.push(Line::from(vec![
            Span::styled("🤔 AI is thinking...", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]));
        
        if let Some(time_remaining) = app.get_ai_time_remaining() {
            stats_lines.push(Line::from(format!("Time remaining: {:.1}s", time_remaining)));
        }
    } else {
        stats_lines.push(Line::from("AI ready"));
    }
    
    stats_lines.push(Line::from(""));

    // Show MCTS root value if available
    if let Some(root_value) = app.mcts_root_value {
        stats_lines.push(Line::from(format!("Root Value: {:.3}", root_value)));
        stats_lines.push(Line::from(""));
    }

    // Show grid-based statistics for Gomoku and Othello
    if matches!(app.game, GameWrapper::Gomoku(_) | GameWrapper::Othello(_)) {
        if let (Some(visits_grid), Some(values_grid), Some(wins_grid)) = (
            &app.mcts_visits_grid,
            &app.mcts_values_grid,
            &app.mcts_wins_grid,
        ) {
            let board_size = visits_grid.len();
            
            // Show grids for board sizes up to 20x20
            if board_size <= 20 {
                // Helper function to find top 5 positions in a grid
                let find_top_positions = |grid: &Vec<Vec<f64>>| -> Vec<(usize, usize)> {
                    let mut positions = Vec::new();
                    for r in 0..board_size {
                        for c in 0..board_size {
                            positions.push((r, c, grid[r][c]));
                        }
                    }
                    positions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                    positions.into_iter().take(5).map(|(r, c, _)| (r, c)).collect()
                };

                let find_top_visits = |grid: &Vec<Vec<i32>>| -> Vec<(usize, usize)> {
                    let mut positions = Vec::new();
                    for r in 0..board_size {
                        for c in 0..board_size {
                            positions.push((r, c, grid[r][c]));
                        }
                    }
                    positions.sort_by(|a, b| b.2.cmp(&a.2));
                    positions.into_iter().take(5).map(|(r, c, _)| (r, c)).collect()
                };

                let top_visits = find_top_visits(visits_grid);
                let top_values = find_top_positions(values_grid);
                let top_wins = find_top_positions(wins_grid);

                // Get current board state for mark display
                let current_board = app.game.get_board();
                
                // VISITS GRID
                stats_lines.push(Line::from(vec![
                    Span::styled("VISITS GRID (top 5 highlighted):", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                ]));
                
                // Add column headers
                let mut header_spans = vec![Span::raw("    ")]; // Space for row numbers
                for c in 0..board_size {
                    header_spans.push(Span::raw(format!("{:9}", c)));
                }
                stats_lines.push(Line::from(header_spans));
                
                for r in 0..board_size {
                    let mut line_spans = vec![Span::raw(format!("{:3} ", r))]; // Row number
                    for c in 0..board_size {
                        let visits = visits_grid[r][c];
                        let is_top = top_visits.contains(&(r, c));
                        
                        if visits > 0 {
                            let span = if is_top {
                                Span::styled(format!("{:9}", visits), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                            } else {
                                Span::styled(format!("{:9}", visits), Style::default().fg(Color::Green))
                            };
                            line_spans.push(span);
                        } else {
                            // Show actual board mark if position is occupied
                            let cell_content = match current_board[r][c] {
                                1 => "        X",
                                -1 => "        O", 
                                _ => "        .",
                            };
                            line_spans.push(Span::raw(cell_content));
                        }
                    }
                    stats_lines.push(Line::from(line_spans));
                }
                
                stats_lines.push(Line::from(""));
                
                // VALUES GRID (win rates)
                stats_lines.push(Line::from(vec![
                    Span::styled("VALUES GRID (win rates, top 5 highlighted):", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                ]));
                
                // Add column headers
                let mut header_spans = vec![Span::raw("    ")]; // Space for row numbers
                for c in 0..board_size {
                    header_spans.push(Span::raw(format!("{:9}", c)));
                }
                stats_lines.push(Line::from(header_spans));
                
                for r in 0..board_size {
                    let mut line_spans = vec![Span::raw(format!("{:3} ", r))]; // Row number
                    for c in 0..board_size {
                        let visits = visits_grid[r][c];
                        let value = values_grid[r][c];
                        let is_top = top_values.contains(&(r, c));
                        
                        if visits > 0 {
                            let base_color = if value > 0.6 {
                                Color::Green
                            } else if value > 0.4 {
                                Color::Yellow
                            } else {
                                Color::Red
                            };
                            
                            let span = if is_top {
                                Span::styled(format!("{:9.3}", value), Style::default().fg(base_color).add_modifier(Modifier::BOLD).add_modifier(Modifier::UNDERLINED))
                            } else {
                                Span::styled(format!("{:9.3}", value), Style::default().fg(base_color))
                            };
                            line_spans.push(span);
                        } else {
                            // Show actual board mark if position is occupied
                            let cell_content = match current_board[r][c] {
                                1 => "        X",
                                -1 => "        O", 
                                _ => "        .",
                            };
                            line_spans.push(Span::raw(cell_content));
                        }
                    }
                    stats_lines.push(Line::from(line_spans));
                }
                
                stats_lines.push(Line::from(""));
                
                // WINS GRID
                stats_lines.push(Line::from(vec![
                    Span::styled("WINS GRID (absolute wins, top 5 highlighted):", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                ]));
                
                // Add column headers
                let mut header_spans = vec![Span::raw("    ")]; // Space for row numbers
                for c in 0..board_size {
                    header_spans.push(Span::raw(format!("{:9}", c)));
                }
                stats_lines.push(Line::from(header_spans));
                
                for r in 0..board_size {
                    let mut line_spans = vec![Span::raw(format!("{:3} ", r))]; // Row number
                    for c in 0..board_size {
                        let visits = visits_grid[r][c];
                        let wins = wins_grid[r][c];
                        let is_top = top_wins.contains(&(r, c));
                        
                        if visits > 0 {
                            let span = if is_top {
                                Span::styled(format!("{:9.0}", wins), Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
                            } else {
                                Span::styled(format!("{:9.0}", wins), Style::default().fg(Color::Red))
                            };
                            line_spans.push(span);
                        } else {
                            // Show actual board mark if position is occupied
                            let cell_content = match current_board[r][c] {
                                1 => "        X",
                                -1 => "        O", 
                                _ => "        .",
                            };
                            line_spans.push(Span::raw(cell_content));
                        }
                    }
                    stats_lines.push(Line::from(line_spans));
                }
                
                stats_lines.push(Line::from(""));
                
                // Summary of top moves
                stats_lines.push(Line::from(vec![
                    Span::styled("TOP 5 MOVES SUMMARY:", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                ]));
                
                for (i, (r, c)) in top_visits.iter().enumerate() {
                    let visits = visits_grid[*r][*c];
                    let value = values_grid[*r][*c];
                    let wins = wins_grid[*r][*c];
                    
                    let value_color = if value > 0.6 {
                        Color::Green
                    } else if value > 0.4 {
                        Color::Yellow
                    } else {
                        Color::Red
                    };
                    
                    stats_lines.push(Line::from(vec![
                        Span::raw(format!("{}. ", i + 1)),
                        Span::styled(format!("({},{}) ", r, c), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                        Span::styled(format!("V:{} ", visits), Style::default().fg(Color::Green)),
                        Span::styled(format!("Rate:{:.3} ", value), Style::default().fg(value_color)),
                        Span::styled(format!("Wins:{:.0}", wins), Style::default().fg(Color::Red)),
                    ]));
                }
                
                stats_lines.push(Line::from(""));
            } else {
                stats_lines.push(Line::from("Board too large for grid display (max 20x20)"));
                stats_lines.push(Line::from(""));
            }
        } else {
            stats_lines.push(Line::from("Waiting for MCTS statistics..."));
            stats_lines.push(Line::from(""));
        }
    }

    // Show debug info if available
    if let Some(debug_info) = &app.mcts_debug_info {
        for line in debug_info.lines() {
            stats_lines.push(Line::from(line));
        }
    }
    
    // Calculate content height and scrolling bounds with strict enforcement
    let content_height = stats_lines.len();
    let visible_height = (area.height.saturating_sub(2) as usize).min(20); // Hard limit to prevent overflow
    let max_scroll = content_height.saturating_sub(visible_height);
    
    // Constrain scroll offset to valid range and update the app if needed
    let corrected_scroll_offset = app.debug_scroll_offset.min(max_scroll);
    
    // Note: We can't modify app here since we have an immutable reference,
    // but the UI will display the corrected offset
    let scroll_offset = corrected_scroll_offset;

    // Create the visible portion of content - ensure we don't exceed bounds
    // Be very strict about the number of lines we show
    let visible_lines: Vec<Line> = if content_height > visible_height && scroll_offset < content_height {
        let start_idx = scroll_offset;
        stats_lines.into_iter()
            .skip(start_idx)
            .take(visible_height) // Hard limit on visible lines
            .collect()
    } else {
        // If content fits in visible area or scroll is out of bounds, show from start
        stats_lines.into_iter()
            .take(visible_height) // Hard limit on visible lines
            .collect()
    };

    // Create the paragraph with scrollable content
    let drag_indicator = if app.is_dragging { "🔀" } else { "↕" };
    let title = if max_scroll > 0 {
        format!("{} Debug Statistics (scroll: {}/{}) - {}%", drag_indicator, scroll_offset, max_scroll, app.stats_height_percent)
    } else {
        format!("{} Debug Statistics - {}%", drag_indicator, app.stats_height_percent)
    };
    
    // Split area for content and scrollbar with minimum space check
    let chunks = if area.width > 5 { // Only show scrollbar if we have enough width
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(0), Constraint::Length(1)])
            .split(area)
    } else {
        // Use full area if too narrow for scrollbar
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(100)])
            .split(area)
    };
    
    let paragraph = Paragraph::new(visible_lines)
        .block(Block::default().title(title).borders(Borders::ALL));
    
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

fn draw_board(f: &mut Frame, app: &App, area: Rect) {
    // Draw the game board with automatic sizing to prevent large boards from taking too much screen space.
    // For large boards (e.g., Blokus 20x20), the display is made more compact to fit within reasonable bounds.
    
    let board = app.game.get_board();
    let last_move_coords = app.game.get_last_move().unwrap_or_default();
    let board_height = board.len();
    let board_width = if board_height > 0 { board[0].len() } else { 0 };
    let game_title = match app.game_type.as_str() {
        "gomoku" => "Gomoku",
        "connect4" => "Connect4",
        "blokus" => "Blokus",
        "othello" => "Othello",
        _ => "Game",
    };
    let block = Block::default().title(game_title).borders(Borders::ALL);
    f.render_widget(block, area);

    // Determine if we need row labels and column labels
    let show_row_labels = app.game_type != "connect4";
    let show_col_labels = true; // All games get column labels
    
    // Calculate space needed for labels
    let row_label_width = if show_row_labels { 3 } else { 0 }; // Space for row numbers like "  1"
    let col_label_height = if show_col_labels { 1 } else { 0 }; // Height for column labels
    
    // Create the main board layout with space for labels
    let board_with_labels_area = Layout::default()
        .margin(1)
        .constraints(if show_col_labels {
            vec![Constraint::Length(col_label_height), Constraint::Min(0)]
        } else {
            vec![Constraint::Min(0)]
        })
        .split(area);
    
    // Calculate column width based on board size and available space to keep board compact
    let max_board_width = (f.size().width * 2 / 3).max(20); // Don't take more than 2/3 screen width, minimum 20
    let col_width = if board_width > 0 {
        // Calculate optimal width per column
        let calculated_width = (max_board_width / board_width as u16).max(1);
        
        // Use different width based on board size for optimal display
        match board_width {
            1..=10 => calculated_width.min(5),    // Small boards: up to 5 chars per cell
            11..=15 => calculated_width.min(4),   // Medium boards: up to 4 chars per cell
            16..=25 => calculated_width.min(3),   // Large boards: up to 3 chars per cell
            _ => calculated_width.min(2).max(1)   // Very large boards: 1-2 chars per cell
        }
    } else {
        4
    };
    
    let content_area = if show_col_labels { board_with_labels_area[1] } else { board_with_labels_area[0] };
    
    // Draw column labels if needed
    if show_col_labels {
        let col_label_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints({
                let mut constraints = vec![];
                if show_row_labels {
                    constraints.push(Constraint::Length(row_label_width));
                }
                for _ in 0..board_width {
                    constraints.push(Constraint::Length(col_width));
                }
                constraints
            })
            .split(board_with_labels_area[0]);
        
        let start_idx = if show_row_labels { 1 } else { 0 };
        for c in 0..board_width {
            let col_label = if app.game_type == "connect4" {
                if col_width <= 2 && c >= 10 {
                    format!("{}", (c % 10)) // For very compact displays, show only last digit for 10+
                } else {
                    format!("{}", c + 1) // Connect4 uses 1-based column numbers
                }
            } else {
                if col_width <= 2 && c >= 10 {
                    format!("{}", (c % 10)) // For very compact displays, show only last digit for 10+
                } else {
                    format!("{}", c) // Other games use 0-based
                }
            };
            
            // Highlight the selected column for Connect4 when it's a human's turn
            let style = if app.game_type == "connect4" && c == app.cursor.1 && !app.is_current_player_ai() {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD).bg(Color::DarkGray)
            } else {
                Style::default().fg(Color::Gray)
            };
            
            let paragraph = Paragraph::new(col_label)
                .style(style)
                .alignment(Alignment::Center);
            f.render_widget(paragraph, col_label_area[start_idx + c]);
        }
    }
    
    // Use consistent row height of 1 to avoid gaps between consecutive rows
    let row_height = 1;
    
    // Create row layout with dynamic height
    let board_area = Layout::default()
        .constraints(vec![Constraint::Length(row_height); board_height])
        .split(content_area);

    for r in 0..board_height {
        let row_constraints = {
            let mut constraints = vec![];
            if show_row_labels {
                constraints.push(Constraint::Length(row_label_width));
            }
            for _ in 0..board_width {
                constraints.push(Constraint::Length(col_width));
            }
            constraints
        };
        
        let row_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(row_constraints)
            .split(board_area[r]);

        // Draw row label if needed
        if show_row_labels {
            let row_label = if row_height == 1 && r >= 10 {
                format!("{}", r % 10) // For very compact displays, show only last digit for 10+
            } else {
                format!("{:>2}", r)
            };
            let paragraph = Paragraph::new(row_label)
                .style(Style::default().fg(Color::Gray))
                .alignment(Alignment::Right);
            f.render_widget(paragraph, row_area[0]);
        }

        let start_idx = if show_row_labels { 1 } else { 0 };
        for c in 0..board_width {
            let player = board[r][c];
            
            // Different visual styles based on game type
            let (symbol, player_color, bg_color) = match app.game {
                GameWrapper::Gomoku(_) => {
                    // Keep Gomoku as is with X and O
                    match player {
                        1 => ("X", Color::Red, None),
                        -1 => ("O", Color::Blue, None),
                        _ => (".", Color::White, None),
                    }
                }
                GameWrapper::Othello(_) => {
                    // Use filled circles for Othello with circular borders
                    match player {
                        1 => ("⚫", Color::White, None), // Black player: black circle with white/light border
                        -1 => ("⚪", Color::White, None), // White player: white circle with dark border
                        _ => ("·", Color::DarkGray, None), // Empty positions with subtle dot
                    }
                }
                GameWrapper::Connect4(_) => {
                    // Use larger colored circles for Connect4
                    match player {
                        1 => ("🔴", Color::Red, None),
                        -1 => ("🟡", Color::Yellow, None),
                        _ => ("·", Color::DarkGray, None),
                    }
                }
                GameWrapper::Blokus(_) => {
                    // Use filled squares with player numbers in darker shade
                    match player {
                        1 => ("1", Color::Black, Some(Color::Red)), // Player 1: Black text on red background
                        2 => ("2", Color::Black, Some(Color::Blue)), // Player 2: Black text on blue background
                        3 => ("3", Color::Black, Some(Color::Green)), // Player 3: Black text on green background
                        4 => ("4", Color::Black, Some(Color::Yellow)), // Player 4: Black text on yellow background
                        _ => ("·", Color::DarkGray, None),
                    }
                }
            };

            let mut style = Style::default().fg(player_color);
            
            // Apply background color if specified (for Blokus squares)
            if let Some(bg) = bg_color {
                style = style.bg(bg);
            }

            // Highlight last move
            if last_move_coords.contains(&(r, c)) {
                style = style.bg(Color::Cyan);
            }

            // Show ghost piece for Blokus if piece preview is enabled
            let mut final_symbol = symbol;
            let mut ghost_piece_shown = false;
            if let GameWrapper::Blokus(_) = &app.game {
                if app.blokus_show_piece_preview {
                    if let Some(piece_idx) = app.blokus_selected_piece_idx {
                        // Get the selected piece and its current transformation
                        use crate::games::blokus::get_blokus_pieces;
                        let pieces = get_blokus_pieces();
                        if piece_idx < pieces.len() {
                            let piece = &pieces[piece_idx];
                            let transformation_idx = app.blokus_selected_transformation;
                            if transformation_idx < piece.transformations.len() {
                                let piece_shape = &piece.transformations[transformation_idx];
                                let preview_pos = app.cursor; // Use cursor position for immediate feedback
                                
                                // Check if current cell (r, c) is part of the ghost piece
                                for &(dr, dc) in piece_shape {
                                    let ghost_r = preview_pos.0 as i32 + dr;
                                    let ghost_c = preview_pos.1 as i32 + dc;
                                    
                                    if ghost_r == r as i32 && ghost_c == c as i32 {
                                        // This cell is part of the ghost piece
                                        let current_player = app.game.get_current_player();
                                        let ghost_color = match current_player {
                                            1 => Color::Red,
                                            2 => Color::Blue, 
                                            3 => Color::Green,
                                            4 => Color::Yellow,
                                            _ => Color::White,
                                        };
                                        
                                        // Only show ghost if the cell is empty
                                        if player == 0 {
                                            final_symbol = "▢"; // Hollow square for ghost piece
                                            style = Style::default().fg(ghost_color); // Use player color without DIM
                                            ghost_piece_shown = true;
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Show cursor (ghost icon) only when it's a human's turn to play AND no ghost piece is shown
            let show_cursor = (r, c) == app.cursor && !app.is_current_player_ai() && !ghost_piece_shown;
            let display_symbol = if show_cursor {
                // Check if the current cursor position represents a valid move
                let is_valid_move = match &app.game {
                    GameWrapper::Gomoku(_) => {
                        let move_wrapper = MoveWrapper::Gomoku(crate::games::gomoku::GomokuMove(r, c));
                        app.game.is_legal(&move_wrapper)
                    },
                    GameWrapper::Connect4(_) => {
                        let move_wrapper = MoveWrapper::Connect4(crate::games::connect4::Connect4Move(c));
                        app.game.is_legal(&move_wrapper)
                    },
                    GameWrapper::Othello(_) => {
                        let move_wrapper = MoveWrapper::Othello(crate::games::othello::OthelloMove(r, c));
                        app.game.is_legal(&move_wrapper)
                    },
                    GameWrapper::Blokus(_) => {
                        // For Blokus, we'll check if there's a valid piece placement at this position
                        // This is more complex, so for now we'll just show the cursor
                        true
                    },
                };
                
                if is_valid_move {
                    "▢" // Hollow square for valid moves
                } else {
                    "▢" // Hollow square for invalid moves (style will be grayed out)
                }
            } else {
                final_symbol // Use the final symbol (could be normal symbol or ghost piece)
            };

            // Apply grayed out style for invalid moves when showing cursor (but not when ghost piece is shown)
            if show_cursor && !ghost_piece_shown {
                // Check if the current cursor position represents a valid move
                let is_valid_move = match &app.game {
                    GameWrapper::Gomoku(_) => {
                        let move_wrapper = MoveWrapper::Gomoku(crate::games::gomoku::GomokuMove(r, c));
                        app.game.is_legal(&move_wrapper)
                    },
                    GameWrapper::Connect4(_) => {
                        let move_wrapper = MoveWrapper::Connect4(crate::games::connect4::Connect4Move(c));
                        app.game.is_legal(&move_wrapper)
                    },
                    GameWrapper::Othello(_) => {
                        let move_wrapper = MoveWrapper::Othello(crate::games::othello::OthelloMove(r, c));
                        app.game.is_legal(&move_wrapper)
                    },
                    GameWrapper::Blokus(_) => {
                        // For Blokus, we'll check if there's a valid piece placement at this position
                        // This is more complex, so for now we'll just show the cursor as valid
                        true
                    },
                };
                
                if is_valid_move {
                    // For Blokus, use the current player's color for the cursor
                    if let GameWrapper::Blokus(_) = &app.game {
                        let current_player = app.game.get_current_player();
                        let cursor_color = match current_player {
                            1 => Color::Red,
                            2 => Color::Blue,
                            3 => Color::Green,
                            4 => Color::Yellow,
                            _ => Color::White,
                        };
                        style = Style::default().fg(cursor_color);
                    } else {
                        style = Style::default().fg(Color::White); // White for other games
                    }
                } else {
                    style = Style::default().fg(Color::DarkGray); // Gray for invalid moves
                }
            }

            let paragraph = Paragraph::new(display_symbol)
                .style(style)
                .alignment(Alignment::Center);
            f.render_widget(paragraph, row_area[start_idx + c]);
        }
    }
}

fn draw_blokus_ui(f: &mut Frame, app: &App, area: Rect) {
    // Create a 3-column layout for Blokus: game board | piece selection (dynamic width) | player status
    let horizontal_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(40),                                   // Game board (expandable)
            Constraint::Length(app.blokus_piece_selection_width),  // Piece selection panel (dynamic width)
            Constraint::Length(20),                                // Player status panel
        ])
        .split(area);

    // Draw game board with ghost piece overlay
    draw_board(f, app, horizontal_chunks[0]);
    
    // Draw piece selection panel with scrollbar
    draw_blokus_piece_selection(f, app, horizontal_chunks[1]);
    
    // Draw player status panel (all 4 players)
    draw_blokus_player_status(f, app, horizontal_chunks[2]);
}

fn draw_blokus_player_status(f: &mut Frame, app: &App, area: Rect) {
    use crate::game_wrapper::GameWrapper;
    
    let block = Block::default().title("Players").borders(Borders::ALL);
    f.render_widget(block, area);
    
    let inner_area = Layout::default()
        .margin(1)
        .constraints([Constraint::Min(0)])
        .split(area)[0];
    
    if let GameWrapper::Blokus(blokus_state) = &app.game {
        let mut status_lines = Vec::new();
        let current_player = app.game.get_current_player();
        
        // Player colors for Blokus (consistent with other displays)
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
        
        // Add some spacing and instructions
        status_lines.push(Line::from(""));
        status_lines.push(Line::from(Span::styled("Controls:", Style::default().fg(Color::Gray))));
        status_lines.push(Line::from(Span::styled("1-9,0: Select", Style::default().fg(Color::Gray))));
        status_lines.push(Line::from(Span::styled("R: Rotate", Style::default().fg(Color::Gray))));
        status_lines.push(Line::from(Span::styled("F: Flip", Style::default().fg(Color::Gray))));
        status_lines.push(Line::from(Span::styled("Enter: Place", Style::default().fg(Color::Gray))));
        status_lines.push(Line::from(Span::styled("P: Pass", Style::default().fg(Color::Gray))));
        
        let paragraph = Paragraph::new(status_lines);
        f.render_widget(paragraph, inner_area);
    }
}

fn draw_blokus_piece_selection(f: &mut Frame, app: &App, area: Rect) {
    use crate::games::blokus::get_blokus_pieces;
    use crate::game_wrapper::GameWrapper;
    
    // Calculate area for content and scrollbar
    let chunks = if area.width > 5 { // Only show scrollbar if we have enough width
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(0), Constraint::Length(1)])
            .split(area)
    } else {
        // Use full area if too narrow for scrollbar
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(100)])
            .split(area)
    };
    
    let block = Block::default().title("Available Pieces (All Players)").borders(Borders::ALL);
    f.render_widget(block, area);
    
    let inner_area = Layout::default()
        .margin(1)
        .constraints([Constraint::Min(0)])
        .split(chunks[0])[0];
    
    if let GameWrapper::Blokus(blokus_state) = &app.game {
        let current_player = app.game.get_current_player();
        let pieces = get_blokus_pieces();
        let player_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
        let player_names = ["P1", "P2", "P3", "P4"];
        
        let mut all_lines = Vec::new();
        
        // Generate all content lines first (for all players)
        for player in 1..=4 {
            let available_pieces = blokus_state.get_available_pieces(player);
            let available_count = available_pieces.len();
            let color = player_colors[(player - 1) as usize];
            let is_current = player == current_player;
            let is_expanded = app.blokus_players_expanded.get((player - 1) as usize).unwrap_or(&true);
            
            // Convert available pieces to a set for quick lookup
            let available_set: std::collections::HashSet<usize> = available_pieces.iter().cloned().collect();
            
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
                let pieces_per_row = 5; // Show 5 pieces per row for better visibility
                let max_displayable_pieces: usize = if is_current { 21 } else { 10 }; // Show all 21 pieces for current player
                let total_pieces: usize = 21; // Total number of piece types in Blokus (0-20)
                
                // For current player, use per-player scrolling within the panel scrolling
                let (pieces_to_show, visible_range) = if is_current {
                    let scroll_offset = app.blokus_piece_selection_scroll.min(total_pieces.saturating_sub(max_displayable_pieces));
                    let pieces_to_show = (total_pieces - scroll_offset).min(max_displayable_pieces);
                    let visible_start = scroll_offset;
                    let visible_end = (scroll_offset + pieces_to_show).min(total_pieces);
                    (pieces_to_show, visible_start..visible_end)
                } else {
                    // For other players, show up to max_displayable_pieces from the beginning
                    let pieces_to_show = total_pieces.min(max_displayable_pieces);
                    (pieces_to_show, 0..pieces_to_show)
                };
                
                // Show pieces in rows of visual shapes
                for chunk_start in (0..pieces_to_show).step_by(pieces_per_row) {
                    let chunk_end = (chunk_start + pieces_per_row).min(pieces_to_show);
                    
                    // Collect all piece visuals for this row
                    let mut pieces_in_row = Vec::new();
                    
                    for display_idx in chunk_start..chunk_end {
                        let piece_idx = if is_current {
                            visible_range.start + display_idx
                        } else {
                            display_idx
                        };
                        
                        if piece_idx >= total_pieces {
                            break;
                        }
                        
                        let piece = &pieces[piece_idx];
                        let is_available = available_set.contains(&piece_idx);
                        let is_selected = is_current && app.blokus_selected_piece_idx == Some(piece_idx);
                        
                        // Create piece shape representation
                        let piece_shape = if !piece.transformations.is_empty() {
                            &piece.transformations[0]
                        } else {
                            continue;
                        };
                        
                        // For current player, adjust key labels based on scroll position
                        let key_label = if is_current {
                            let global_idx = piece_idx;
                            if global_idx < 9 { 
                                (global_idx + 1).to_string() 
                            } else if global_idx == 9 { 
                                "0".to_string() 
                            } else { 
                                ((b'a' + (global_idx - 10) as u8) as char).to_string()
                            }
                        } else {
                            // For other players, just use local display index
                            if display_idx < 9 { 
                                (display_idx + 1).to_string() 
                            } else if display_idx == 9 { 
                                "0".to_string() 
                            } else { 
                                ((b'a' + (display_idx - 10) as u8) as char).to_string()
                            }
                        };

                        // Create visual shape for this piece (now returns Vec<String>)
                        let piece_visual_lines = create_visual_piece_shape(piece_shape);
                        
                        // Show availability status and key label
                        let piece_name_text = if is_selected {
                            format!("[{}]", key_label)
                        } else if is_available {
                            format!(" {} ", key_label)  // Available piece
                        } else {
                            format!("({})", key_label)  // Used piece (parentheses indicate unavailable)
                        };
                        
                        let style = if is_selected {
                            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD).bg(Color::DarkGray)
                        } else if is_current && is_available {
                            Style::default().fg(Color::White)
                        } else if is_current && !is_available {
                            Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM)  // Grayed out for used pieces
                        } else if is_available {
                            Style::default().fg(color.clone())
                        } else {
                            Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM)  // Grayed out for used pieces
                        };
                        
                        pieces_in_row.push((piece_name_text, piece_visual_lines, style));
                    }
                    
                    if !pieces_in_row.is_empty() {
                        // Find the maximum height and width of pieces in this row
                        let max_height = pieces_in_row.iter()
                            .map(|(_, lines, _)| lines.len())
                            .max()
                            .unwrap_or(1);
                        
                        let max_width = pieces_in_row.iter()
                            .map(|(_, lines, _)| {
                                lines.iter().map(|line| line.chars().count()).max().unwrap_or(0)
                            })
                            .max()
                            .unwrap_or(8);
                        
                        // Use a consistent width that accommodates the widest piece with extra padding
                        let piece_width = max_width.max(6) + 1; // More compact for resizable panel
                        let spacing_between_pieces = 2; // Reduce spacing for resizable panel
                        
                        // First line: piece keys/names
                        let mut key_line_spans = Vec::new();
                        for (i, (piece_name, _, style)) in pieces_in_row.iter().enumerate() {
                            // Center the piece name within the allocated width
                            let padded_name = format!("{:^width$}", piece_name, width = piece_width);
                            key_line_spans.push(Span::styled(padded_name, *style));
                            if i < pieces_in_row.len() - 1 { // Don't add spacing after last piece
                                key_line_spans.push(Span::styled(" ".repeat(spacing_between_pieces), Style::default()));
                            }
                        }
                        all_lines.push(Line::from(key_line_spans));
                        
                        // Show each line of the pieces (true 2D representation)
                        for line_idx in 0..max_height {
                            let mut shape_line_spans = Vec::new();
                            for (i, (_, piece_visual_lines, style)) in pieces_in_row.iter().enumerate() {
                                let piece_line = if line_idx < piece_visual_lines.len() {
                                    // Center the piece shape within the allocated width
                                    format!("{:^width$}", piece_visual_lines[line_idx], width = piece_width)
                                } else {
                                    " ".repeat(piece_width) // Empty space for shorter pieces
                                };
                                shape_line_spans.push(Span::styled(piece_line, *style));
                                if i < pieces_in_row.len() - 1 { // Don't add spacing after last piece
                                    shape_line_spans.push(Span::styled(" ".repeat(spacing_between_pieces), Style::default()));
                                }
                            }
                            all_lines.push(Line::from(shape_line_spans));
                        }
                    }
                }
                
                // Show scroll hint for current player if there are many pieces
                if is_current && total_pieces > max_displayable_pieces {
                    let scroll_pos = app.blokus_piece_selection_scroll;
                    let showing_start = scroll_pos + 1;
                    let showing_end = (scroll_pos + pieces_to_show).min(total_pieces);
                    
                    all_lines.push(Line::from(vec![Span::styled(
                        format!("Showing pieces {}-{} of {} - Use [ ] to scroll pieces", 
                            showing_start, showing_end, total_pieces
                        ),
                        Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC)
                    )]));
                } else if !is_current && total_pieces > max_displayable_pieces {
                    all_lines.push(Line::from(vec![Span::styled(
                        format!("+{} more pieces", total_pieces - max_displayable_pieces),
                        Style::default().fg(Color::Gray)
                    )]));
                }
            } else if !*is_expanded {
                // Show a compact summary when collapsed
                let available_count = available_set.len();
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
            } else if *is_expanded {
                // Show when expanded but has no available pieces
                let available_count = available_set.len();
                if available_count == 0 {
                    all_lines.push(Line::from(Span::styled("  All pieces used", Style::default().fg(Color::Gray))));
                }
            }
            
            // Add separator line between players
            if player < 4 {
                all_lines.push(Line::from(""));
            }
        }
        
        // Add some controls at the bottom
        all_lines.push(Line::from(""));
        all_lines.push(Line::from(Span::styled("Controls:", Style::default().fg(Color::Gray))));
        all_lines.push(Line::from(Span::styled("1-9,0,a-u: Select available  (x): Used pieces  [ ]: Scroll", Style::default().fg(Color::Gray))));
        all_lines.push(Line::from(Span::styled("{ }: Scroll pieces  R: Rotate  F: Flip", Style::default().fg(Color::Gray))));
        all_lines.push(Line::from(Span::styled("E: Toggle expand  +/-: Expand/Collapse all", Style::default().fg(Color::Gray))));
        all_lines.push(Line::from(Span::styled("Drag walls ◀▶ to resize panel", Style::default().fg(Color::Cyan))));
        
        // Apply full panel scrolling
        let content_height = all_lines.len();
        let visible_height = inner_area.height as usize;
        let max_scroll = content_height.saturating_sub(visible_height);
        
        // Ensure scroll offset is within bounds
        let scroll_offset = app.blokus_panel_scroll_offset.min(max_scroll);
        
        // Create the visible portion of content
        let visible_lines: Vec<Line> = if content_height > visible_height && scroll_offset < content_height {
            all_lines.into_iter()
                .skip(scroll_offset)
                .take(visible_height)
                .collect()
        } else {
            all_lines.into_iter()
                .take(visible_height)
                .collect()
        };
        
        let paragraph = Paragraph::new(visible_lines);
        f.render_widget(paragraph, inner_area);
        
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
}



fn handle_mouse_click(app: &mut App, col: u16, row: u16, terminal_size: Rect) {
    // Check if the click is on a drag boundary first
    if let Some(boundary) = detect_boundary_click(app, col, row, terminal_size.width, terminal_size.height) {
        app.start_drag(boundary);
        return;
    }

    match app.state {
        AppState::Menu => {
            handle_menu_click(app, col, row, terminal_size);
        }
        AppState::Settings => {
            handle_settings_click(app, col, row, terminal_size);
        }
        AppState::Playing => {
            if !app.ai_only {
                if app.game_type == "blokus" {
                    handle_blokus_click(app, col, row, terminal_size);
                } else {
                    handle_board_click(app, col, row, terminal_size);
                }
            }
        }
        AppState::GameOver => {
            // Could add click handling for game over state if needed
        }
        AppState::PlayerConfig => {
            // Player configuration clicks handled in the event loop
        }
    }
}

fn handle_mouse_drag(app: &mut App, col: u16, row: u16, terminal_size: Rect) {
    if app.is_dragging {
        if let Some(boundary) = app.drag_boundary {
            match boundary {
                DragBoundary::StatsHistory | 
                DragBoundary::BlokusPieceSelectionLeft | 
                DragBoundary::BlokusPieceSelectionRight => {
                    app.handle_horizontal_drag(col, terminal_size.width);
                }
                _ => {
                    app.handle_drag(row, terminal_size.height);
                }
            }
        }
    }
}

fn handle_mouse_release(app: &mut App, _col: u16, _row: u16, _terminal_size: Rect) {
    if app.is_dragging {
        app.stop_drag();
    }
}

fn detect_boundary_click(app: &App, col: u16, row: u16, terminal_width: u16, terminal_height: u16) -> Option<DragBoundary> {
    // Only allow boundary dragging in Playing or GameOver states
    match app.state {
        AppState::Playing | AppState::GameOver => {
            // Check for Blokus piece selection panel boundaries first (only in Blokus mode)
            if app.game_type == "blokus" {
                // Calculate the approximate positions of the Blokus panel boundaries
                // New layout: game board | piece selection (dynamic width) | player status (20)
                let board_width = terminal_width.saturating_sub(app.blokus_piece_selection_width + 20).max(40);
                let piece_selection_width = app.blokus_piece_selection_width;
                let left_boundary = board_width; // Between board and piece selection
                let right_boundary = board_width + piece_selection_width; // Between piece selection and player status
                
                // Check if click is near the left boundary of piece selection panel
                if col.abs_diff(left_boundary) <= 2 {
                    return Some(DragBoundary::BlokusPieceSelectionLeft);
                }
                
                // Check if click is near the right boundary of piece selection panel
                if col.abs_diff(right_boundary) <= 2 {
                    return Some(DragBoundary::BlokusPieceSelectionRight);
                }
            }
            
            let (board_instructions_boundary, instructions_stats_boundary) = app.get_drag_area(terminal_height);
            
            // Check if click is near the board-instructions boundary (within 1 row)
            if row.abs_diff(board_instructions_boundary) <= 1 {
                return Some(DragBoundary::BoardInstructions);
            }
            
            // Check if click is near the instructions-stats boundary (within 1 row)
            if row.abs_diff(instructions_stats_boundary) <= 1 {
                return Some(DragBoundary::InstructionsStats);
            }
            
            // Check if click is in the stats area for horizontal dragging (stats/history boundary)
            if row > instructions_stats_boundary {
                // Calculate the boundary between debug stats and move history
                let stats_width_boundary = (terminal_width as f32 * app.stats_width_percent as f32 / 100.0) as u16;
                
                // Check if click is near the stats-history boundary (within 2 columns)
                if col.abs_diff(stats_width_boundary) <= 2 {
                    return Some(DragBoundary::StatsHistory);
                }
            }
        }
        _ => {}
    }
    None
}

fn handle_menu_click(app: &mut App, _col: u16, row: u16, terminal_size: Rect) {
    // Calculate the menu area based on the dynamic layout
    let main_area_height = (terminal_size.height as f32 * app.board_height_percent as f32 / 100.0) as u16;
    
    // Check if click is within the menu area
    if row < main_area_height {
        // Calculate which menu item was clicked
        // The menu starts at row 1 (border) and each item takes roughly 1 row
        let menu_start_row = 2; // Account for border and title
        if row >= menu_start_row {
            let clicked_item = (row - menu_start_row) as usize;
            if clicked_item < app.titles.len() {
                app.index = clicked_item;
                // Double-click simulation - if clicking the same item, select it
                if clicked_item == app.titles.len() - 1 {
                    // Quit was clicked
                    std::process::exit(0);
                } else if clicked_item == app.titles.len() - 2 {
                    // Settings was clicked
                    app.state = AppState::Settings;
                } else {
                    // A game was clicked
                    app.state = AppState::PlayerConfig;
                    app.set_game(app.index);
                }
            }
        }
    }
}

fn handle_blokus_click(app: &mut App, col: u16, row: u16, terminal_size: Rect) {
    // For Blokus, we have a 3-column layout: game board (min 40) | piece selection (dynamic) | player status (20)
    let board_width = terminal_size.width.saturating_sub(app.blokus_piece_selection_width + 20).max(40);
    let horizontal_chunks_widths = [board_width, app.blokus_piece_selection_width, 20u16];
    let mut accumulated_width = 0u16;
    
    // Determine which panel was clicked
    for (panel_idx, &panel_width) in horizontal_chunks_widths.iter().enumerate() {
        if col >= accumulated_width && col < accumulated_width + panel_width {
            match panel_idx {
                0 => {
                    // Game board panel - handle normal board clicks
                    handle_board_click(app, col, row, terminal_size);
                    return;
                }
                1 => {
                    // Piece selection panel - handle expand/collapse clicks
                    handle_blokus_piece_selection_click(app, col - accumulated_width, row);
                    return;
                }
                2 => {
                    // Player status panel - no interaction for now
                    return;
                }
                _ => break,
            }
        }
        accumulated_width += panel_width;
    }
}

fn handle_blokus_piece_selection_click(app: &mut App, col: u16, row: u16) {
    use crate::game_wrapper::GameWrapper;
    
    if let GameWrapper::Blokus(_) = &app.game {
        // Look for player header clicks to toggle expand/collapse
        // The headers are in the format "▼ ► P1 (N pieces) ◄" or "▶   P1 (N pieces)"
        
        // Since we can't easily detect exact row positions without recreating the layout logic,
        // we'll implement a simplified approach: detect clicks in the first few columns
        // which are likely to be on the expand/collapse indicators
        
        if col <= 5 { // Click on the expand/collapse indicator area
            // Estimate which player based on row position (rough approximation)
            // Each player section takes roughly 8-15 rows depending on expansion
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
}

fn handle_board_click(app: &mut App, col: u16, row: u16, terminal_size: Rect) {
    let board = app.game.get_board();
    let board_height = board.len();
    let board_width = if board_height > 0 { board[0].len() } else { 0 };
    
    // Calculate the game board area based on the dynamic layout
    let main_area_height = (terminal_size.height as f32 * app.board_height_percent as f32 / 100.0) as u16;
    
    // Check if click is within the board area
    if row < main_area_height {
        // Determine if we need row labels and column labels
        let show_row_labels = app.game_type != "connect4";
        let show_col_labels = true;
        
        // Calculate space needed for labels
        let row_label_width = if show_row_labels { 3 } else { 0 };
        let col_label_height = if show_col_labels { 1 } else { 0 };
        
        // Calculate column and row dimensions (same as draw_board)
        let max_board_width = (terminal_size.width * 2 / 3).max(20);
        let col_width = if board_width > 0 {
            let calculated_width = (max_board_width / board_width as u16).max(1);
            match board_width {
                1..=10 => calculated_width.min(5),
                11..=15 => calculated_width.min(4),
                16..=25 => calculated_width.min(3),
                _ => calculated_width.min(2).max(1)
            }
        } else {
            4
        };
        let row_height = 1; // Consistent with draw_board
        
        // The board area has borders and labels
        let board_start_col = 1 + row_label_width; // Border + row label space
        let board_start_row = 1 + col_label_height; // Border + column label space
        
        if col >= board_start_col && row >= board_start_row {
            let board_col = ((col - board_start_col) / col_width) as usize;
            let board_row = ((row - board_start_row) / row_height) as usize;
            
            if board_row < board_height && board_col < board_width {
                // For other games, exact cell positioning matters
                if board[board_row][board_col] == 0 {
                    app.cursor = (board_row, board_col);
                    app.submit_move();
                }
            }
        }
    }
}

fn handle_settings_click(app: &mut App, _col: u16, row: u16, terminal_size: Rect) {
    // Calculate the settings area based on the dynamic layout
    let main_area_height = (terminal_size.height as f32 * app.board_height_percent as f32 / 100.0) as u16;
    
    // Check if click is within the settings area
    if row < main_area_height {
        // Calculate which settings item was clicked
        // Account for the borders and spacing
        let settings_area_start = 1; // Top border
        
        if row >= settings_area_start {
            let clicked_index = (row - settings_area_start) as usize;
            if clicked_index < app.settings_titles.len() {
                app.settings_index = clicked_index;
                
                // If clicking on "Back to Menu", switch to menu
                if app.settings_index == app.settings_titles.len() - 1 {
                    app.state = AppState::Menu;
                }
            }
        }
    }
}

fn handle_mouse_scroll(app: &mut App, col: u16, row: u16, terminal_size: Rect, scroll_up: bool) {
    // Only handle scrolling when in Playing or GameOver state
    match app.state {
        AppState::Playing | AppState::GameOver => {
            // Special handling for Blokus piece selection scrolling
            if app.game_type == "blokus" {
                // Blokus has horizontal layout: game board | piece selection (dynamic width) | player status (20)
                let board_width = terminal_size.width.saturating_sub(app.blokus_piece_selection_width + 20).max(40);
                let piece_selection_start = board_width; // After game board
                let piece_selection_end = board_width + app.blokus_piece_selection_width; // Dynamic width
                
                if col >= piece_selection_start && col < piece_selection_end {
                    // Mouse is in the piece selection panel - use full panel scrolling
                    if scroll_up {
                        app.blokus_scroll_panel_up();
                    } else {
                        app.blokus_scroll_panel_down();
                    }
                    return;
                }
            }
            
            // Default scrolling for stats area (for non-Blokus games or other areas)
            let board_height = (terminal_size.height as f32 * app.board_height_percent as f32 / 100.0) as u16;
            let instructions_height = (terminal_size.height as f32 * app.instructions_height_percent as f32 / 100.0) as u16;
            let stats_area_start = board_height + instructions_height;
            
            // Check if the mouse is in the stats area
            if row >= stats_area_start {
                // Determine which horizontal section (debug stats vs move history)
                let stats_width_boundary = (terminal_size.width as f32 * app.stats_width_percent as f32 / 100.0) as u16;
                
                if col < stats_width_boundary {
                    // Mouse is in the debug stats area
                    if scroll_up {
                        app.scroll_debug_up();
                    } else {
                        app.scroll_debug_down();
                    }
                } else {
                    // Mouse is in the move history area
                    if scroll_up {
                        app.scroll_move_history_up();
                    } else {
                        app.scroll_move_history_down();
                    }
                }
            }
        }
        _ => {
            // No scrolling for other states
        }
    }
}

fn draw_move_history(f: &mut Frame, app: &App, area: Rect) {
    // Early return if area is too small
    if area.height < 3 || area.width < 10 {
        let paragraph = Paragraph::new("Area too small")
            .block(Block::default().title("Move History").borders(Borders::ALL));
        f.render_widget(paragraph, area);
        return;
    }

    let mut history_lines = vec![];
    
    // Add move history section
    if !app.move_history.is_empty() {
        // Group moves by move number for display
        let mut grouped_moves: std::collections::BTreeMap<u32, Vec<&crate::MoveHistoryEntry>> = std::collections::BTreeMap::new();
        for entry in &app.move_history {
            grouped_moves.entry(entry.move_number).or_insert_with(Vec::new).push(entry);
        }
        
        for (move_num, moves) in grouped_moves.iter() {
            let mut move_spans = vec![
                Span::styled(format!("{}. ", move_num), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            ];
            
            for (i, entry) in moves.iter().enumerate() {
                if i > 0 {
                    move_spans.push(Span::raw(" "));
                }
                
                let player_color = match entry.player {
                    1 => Color::Red,
                    -1 => Color::Blue,
                   
                    2 => Color::Green,
                    3 => Color::Yellow,
                    4 => Color::Cyan,
                    _ => Color::White,
                };
                
                let player_symbol = match app.game {
                    GameWrapper::Blokus(_) => format!("P{}", entry.player),
                    _ => if entry.player == 1 { "X".to_string() } else { "O".to_string() },
                };
                
                move_spans.push(Span::styled(
                    format!("{}:{}", player_symbol, entry.move_data),
                    Style::default().fg(player_color)
                ));
            }
            
            // Add timestamp for the move group (using the first move's timestamp)
            if let Some(first_move) = moves.first() {
                let timestamp = first_move.timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                move_spans.push(Span::styled(
                    format!(" ({:02}:{:02})", timestamp / 60 % 60, timestamp % 60),
                    Style::default().fg(Color::Gray)
                ));
            }
            
            history_lines.push(Line::from(move_spans));
        }
    } else {
        history_lines.push(Line::from("No moves yet"));
    }
    
    // Calculate content height and scrolling bounds
    let content_height = history_lines.len();
    let visible_height = (area.height.saturating_sub(2) as usize).min(20);
    let max_scroll = content_height.saturating_sub(visible_height);
    
    // Use move history scroll offset
    let scroll_offset = app.move_history_scroll_offset.min(max_scroll);

    // Create the visible portion of content with word wrapping
    let mut visible_lines: Vec<Line> = Vec::new();
    let content_width = area.width.saturating_sub(4) as usize; // Account for borders and padding
    
    for line in history_lines.iter().skip(scroll_offset).take(visible_height) {
        // Simple word wrapping - split long lines
        let line_text = line.spans.iter().map(|span| span.content.as_ref()).collect::<String>();
        if line_text.len() > content_width {
            // For now, just truncate long lines - proper word wrapping would be more complex
            let mut truncated_spans = line.spans.clone();
            let total_len_before_last = truncated_spans.iter().take(truncated_spans.len().saturating_sub(1))
                .map(|s| s.content.len()).sum::<usize>();
            
            if let Some(last_span) = truncated_spans.last_mut() {
                let available_for_last = content_width.saturating_sub(total_len_before_last);
                if available_for_last < last_span.content.len() {
                    last_span.content = format!("{}...", &last_span.content[..available_for_last.saturating_sub(3)]).into();
                }
            }
            visible_lines.push(Line::from(truncated_spans));
        } else {

            visible_lines.push(line.clone());
        }
    }

    // Add scrollbar information
    if max_scroll > 0 && content_height > visible_height {
        let showing_from = scroll_offset + 1;
        let showing_to = (scroll_offset + visible_height).min(content_height);
        visible_lines.push(Line::from(vec![
            Span::styled(
                format!("Moves {}-{} of {} (Shift+Up/Down)", showing_from, showing_to, content_height),
                Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC)
            ),
        ]));
    }

    // Create the paragraph with scrollable content
    let drag_indicator = if app.is_dragging { "🔀" } else { "↔" };
    let title = if max_scroll > 0 {
        format!("{} Move History (scroll: {}/{}) - {}%", drag_indicator, scroll_offset, max_scroll, 100 - app.stats_height_percent)
    } else {
        format!("{} Move History - {}%", drag_indicator, 100 - app.stats_height_percent)
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
    
    let paragraph = Paragraph::new(visible_lines)
        .block(Block::default().title(title).borders(Borders::ALL))
        .wrap(ratatui::widgets::Wrap { trim: true });
    
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

fn create_visual_piece_shape(piece_shape: &[(i32, i32)]) -> Vec<String> {
    if piece_shape.is_empty() {
        return vec!["▢".to_string()];
    }
    
    // Create a 2D visual representation using empty squares like the ghost moves
    let min_r = piece_shape.iter().map(|p| p.0).min().unwrap_or(0);
    let max_r = piece_shape.iter().map(|p| p.0).max().unwrap_or(0);
    let min_c = piece_shape.iter().map(|p| p.1).min().unwrap_or(0);
    let max_c = piece_shape.iter().map(|p| p.1).max().unwrap_or(0);
    
    let height = (max_r - min_r + 1) as usize;
    let width = (max_c - min_c + 1) as usize;
    
    // Create a grid to show the shape using empty squares
    let mut grid = vec![vec![' '; width]; height];
    
    // Fill the grid with the piece shape using empty squares
    for &(r, c) in piece_shape {
        let gr = (r - min_r) as usize;
        let gc = (c - min_c) as usize;
        grid[gr][gc] = '▢'; // Use empty square like ghost moves
    }
    
    // Convert to vector of strings - each string is a row, exactly like ghost piece
    let mut result: Vec<String> = grid.iter()
        .map(|row| row.iter().collect::<String>())
        .collect();
    
    // Ensure minimum width for single character pieces
    if result.len() == 1 && result[0].trim().len() == 1 {
        result[0] = format!(" {} ", result[0].trim());
    }
    
    result
}
