use crate::{App, AppState, DragBoundary};
use crate::game_wrapper::GameWrapper;
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
    let debounce_duration = Duration::from_millis(100); // 100ms debounce

    loop {
        // Check for terminal size changes
        let terminal_size = terminal.size()?;
        if (terminal_size.width, terminal_size.height) != app.last_terminal_size {
            app.handle_window_resize(terminal_size.width, terminal_size.height);
        }
        
        terminal.draw(|f| ui(f, app))?;
        app.tick();

        if event::poll(Duration::from_millis(100))? {
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
                    
                    if last_key_event.elapsed() > debounce_duration {
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
                                        app.state = AppState::Playing;
                                        app.set_game(app.index);
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
                                    KeyCode::Down => if !app.ai_only { app.move_cursor_down(); },
                                    KeyCode::Up => if !app.ai_only { app.move_cursor_up(); },
                                    KeyCode::Left => if !app.ai_only { app.move_cursor_left(); },
                                    KeyCode::Right => if !app.ai_only { app.move_cursor_right(); },
                                    KeyCode::Enter => if !app.ai_only { app.submit_move(); },
                                    KeyCode::Char('m') => app.state = AppState::Menu,
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
                                _ => {}
                            },
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
                Paragraph::new("Use arrow keys to navigate, Enter to select, or click with mouse. 'q' or Esc to quit.")
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
            draw_board(f, app, main_chunks[0]);

            let instructions_text = if !app.game.is_terminal() {
                if app.ai_only {
                    if app.is_ai_thinking() {
                        "AI vs AI mode - AI is thinking... Press 'm' for menu, 'q' or Esc to quit. PageUp/PageDown or scroll to navigate debug info. Drag boundaries to resize panes.".to_string()
                    } else {
                        "AI vs AI mode - Press 'm' for menu, 'q' or Esc to quit. PageUp/PageDown or scroll to navigate debug info. Drag boundaries to resize panes.".to_string()
                    }
                } else {
                    if app.is_ai_thinking() {
                        "AI is thinking... Please wait. 'm' for menu, 'q' or Esc to quit. PageUp/PageDown or scroll to navigate debug info. Drag boundaries to resize panes.".to_string()
                    } else {
                        "Arrow keys to move, Enter to place, or click on board. 'm' for menu, 'q' or Esc to quit. PageUp/PageDown or scroll to navigate debug info. Drag boundaries to resize panes.".to_string()
                    }
                }
            } else {
                let (winner_text, winner_color) = if let Some(winner) = app.winner {
                    if winner == 1 {
                        ("Player X wins!", Color::Red)
                    } else {
                        ("Player O wins!", Color::Blue)
                    }
                } else {
                    ("It's a draw!", Color::Yellow)
                };
                
                // Create styled instruction text for game over
                let instruction_spans = vec![
                    Span::styled(winner_text, Style::default().fg(winner_color).add_modifier(Modifier::BOLD)),
                    Span::raw(" Press 'r' to play again, 'm' for menu, 'q' or Esc to quit. PageUp/PageDown or scroll to navigate debug info. Drag boundaries to resize panes."),
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
                
                draw_stats(f, app, main_chunks[2]);
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
            
            draw_stats(f, app, main_chunks[2]);
        }
    }
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
    let (player_symbol, player_color) = if current_player == 1 {
        ("X", Color::Red)
    } else {
        ("O", Color::Blue)
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
    let board = app.game.get_board();
    let last_move_coords = app.game.get_last_move().unwrap_or_default();
    let board_size = board.len();
    let game_title = match app.game_type.as_str() {
        "gomoku" => "Gomoku",
        "connect4" => "Connect4",
        "blokus" => "Blokus",
        "othello" => "Othello",
        _ => "Game",
    };
    let block = Block::default().title(game_title).borders(Borders::ALL);
    f.render_widget(block, area);

    let board_area = Layout::default()
        .margin(1)
        .constraints(vec![Constraint::Length(2); board_size])
        .split(area);

    for r in 0..board_size {
        let row_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![Constraint::Length(4); board_size])
            .split(board_area[r]);

        for c in 0..board_size {
            let player = board[r][c];
            let (symbol, player_color) = match player {
                1 => ("X", Color::Red),
                -1 => ("O", Color::Blue),
                _ => (".", Color::White),
            };

            let mut style = Style::default().fg(player_color);

            if last_move_coords.contains(&(r, c)) {
                style = style.bg(Color::Cyan);
            }

            if (r, c) == app.cursor && !app.ai_only {
                style = style.bg(Color::Yellow).fg(Color::Black);
            }

            let paragraph = Paragraph::new(symbol)
                .style(style)
                .alignment(Alignment::Center);
            f.render_widget(paragraph, row_area[c]);
        }
    }
}

fn handle_mouse_click(app: &mut App, col: u16, row: u16, terminal_size: Rect) {
    // Check if the click is on a drag boundary first
    if let Some(boundary) = detect_boundary_click(app, row, terminal_size.height) {
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
                handle_board_click(app, col, row, terminal_size);
            }
        }
        AppState::GameOver => {
            // Could add click handling for game over state if needed
        }
    }
}

fn handle_mouse_drag(app: &mut App, _col: u16, row: u16, terminal_size: Rect) {
    if app.is_dragging {
        app.handle_drag(row, terminal_size.height);
    }
}

fn handle_mouse_release(app: &mut App, _col: u16, _row: u16, _terminal_size: Rect) {
    if app.is_dragging {
        app.stop_drag();
    }
}

fn detect_boundary_click(app: &App, row: u16, terminal_height: u16) -> Option<DragBoundary> {
    // Only allow boundary dragging in Playing or GameOver states
    match app.state {
        AppState::Playing | AppState::GameOver => {
            let (board_instructions_boundary, instructions_stats_boundary) = app.get_drag_area(terminal_height);
            
            // Check if click is near the board-instructions boundary (within 1 row)
            if row.abs_diff(board_instructions_boundary) <= 1 {
                return Some(DragBoundary::BoardInstructions);
            }
            
            // Check if click is near the instructions-stats boundary (within 1 row)
            if row.abs_diff(instructions_stats_boundary) <= 1 {
                return Some(DragBoundary::InstructionsStats);
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
                } else {
                    app.state = AppState::Playing;
                    app.set_game(app.index);
                }
            }
        }
    }
}

fn handle_board_click(app: &mut App, col: u16, row: u16, terminal_size: Rect) {
    let board_size = app.game.get_board().len();
    
    // Calculate the game board area based on the dynamic layout
    let main_area_height = (terminal_size.height as f32 * app.board_height_percent as f32 / 100.0) as u16;
    
    // Check if click is within the board area
    if row < main_area_height {
        // Calculate board position
        // The board area has a block border, then margin(1) creates the content area
        // So content starts at: border (1) + margin (1) = row 1 for columns, row 1 for rows
        let board_start_col = 1; // Border + margin
        let board_start_row = 1; // Border + margin
        
        if col >= board_start_col && row >= board_start_row {
            let board_col = ((col - board_start_col) / 4) as usize;
            // Each board cell occupies 2 terminal rows
            let board_row = ((row - board_start_row) / 2) as usize;
            
            if board_row < board_size && board_col < board_size {
                // Check if the position is valid (empty)
                if app.game.get_board()[board_row][board_col] == 0 {
                    // Update cursor position and make move
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

fn handle_mouse_scroll(app: &mut App, _col: u16, row: u16, terminal_size: Rect, scroll_up: bool) {
    // Only handle scrolling when in Playing or GameOver state
    match app.state {
        AppState::Playing | AppState::GameOver => {
            // Calculate the stats area position based on dynamic layout
            let board_height = (terminal_size.height as f32 * app.board_height_percent as f32 / 100.0) as u16;
            let instructions_height = (terminal_size.height as f32 * app.instructions_height_percent as f32 / 100.0) as u16;
            let stats_area_start = board_height + instructions_height;
            
            // Check if the mouse is in the stats area
            if row >= stats_area_start {
                if scroll_up {
                    app.scroll_debug_up();
                } else {
                    app.scroll_debug_down();
                }
            }
        }
        _ => {
            // No scrolling for other states
        }
    }
}
