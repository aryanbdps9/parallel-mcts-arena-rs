use crate::{App, AppState};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use mcts::GameState;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
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
        terminal.draw(|f| ui(f, app))?;
        app.tick();

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
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
                                if app.index == app.titles.len() - 1 {
                                    return Ok(());
                                } else {
                                    app.state = AppState::Playing;
                                    app.set_game(app.index);
                                }
                            }
                            _ => {}
                        },
                        AppState::Playing => {
                            match key.code {
                                KeyCode::Down => if !app.ai_only { app.move_cursor_down(); },
                                KeyCode::Up => if !app.ai_only { app.move_cursor_up(); },
                                KeyCode::Left => if !app.ai_only { app.move_cursor_left(); },
                                KeyCode::Right => if !app.ai_only { app.move_cursor_right(); },
                                KeyCode::Enter => if !app.ai_only { app.make_move(); },
                                KeyCode::Char('m') => app.state = AppState::Menu,
                                _ => {}
                            }
                        }
                        AppState::GameOver => match key.code {
                            KeyCode::Char('r') => app.reset(),
                            KeyCode::Char('m') => app.state = AppState::Menu,
                            _ => {}
                        },
                    }
                    last_key_event = Instant::now();
                }
            }
        }
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)].as_ref())
        .split(f.size());

    let game_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(90), Constraint::Percentage(10)].as_ref())
        .split(main_chunks[0]);

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
            f.render_stateful_widget(list, game_chunks[0], &mut list_state);

            let instructions =
                Paragraph::new("Use arrow keys to navigate, Enter to select, 'q' or Esc to quit.")
                    .block(Block::default().title("Instructions").borders(Borders::ALL));
            f.render_widget(instructions, game_chunks[1]);
        }
        AppState::Playing | AppState::GameOver => {
            draw_board(f, app, game_chunks[0]);
            draw_stats(f, app, main_chunks[1]);

            let instructions_text = if !app.game.is_terminal() {
                if app.ai_only {
                    "AI vs AI mode - Press 'm' for menu, 'q' or Esc to quit.".to_string()
                } else {
                    "Arrow keys to move, Enter to place a piece, 'm' for menu, 'q' or Esc to quit.".to_string()
                }
            } else {
                let winner_text = if let Some(winner) = app.winner {
                    if winner == 1 {
                        "Player X wins!"
                    } else {
                        "Player O wins!"
                    }
                } else {
                    "It's a draw!"
                };
                format!("{} Press 'r' to play again, 'm' for menu, 'q' or Esc to quit.", winner_text)
            };

            let instructions = Paragraph::new(instructions_text)
                .block(Block::default().title("Instructions").borders(Borders::ALL));
            f.render_widget(instructions, game_chunks[1]);
        }
    }
}

fn draw_stats(f: &mut Frame, app: &App, area: Rect) {
    let (root_wins, root_visits) = app.ai.get_root_stats();
    let root_value = if root_visits > 0 {
        root_wins as f64 / root_visits as f64 / 2.0
    } else {
        0.0
    };

    let mut stats_text = vec![
        Line::from(format!("Current Player: {}", if app.game.get_current_player() == 1 { "X" } else { "O" })),
        Line::from(format!("Root Visits: {}", root_visits)),
        Line::from(format!("Root Wins: {}", root_wins)),
        Line::from(format!("Root Value: {:.3}", root_value)),
        Line::from(format!("Threads: {}", app.num_threads)),
        Line::from(""),
        Line::from("Top 5 moves:"),
    ];

    let mut sorted_children: Vec<_> = app.ai.get_root_children_stats().into_iter().collect();
    sorted_children.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    
    for (mv, (wins, visits)) in sorted_children.iter().take(5) {
        let value = if *visits > 0 {
            *wins as f64 / *visits as f64 / 2.0
        } else {
            0.0
        };
        stats_text.push(Line::from(format!(
            "Move {:?} -> V: {}, W: {:.1}, Q: {:.3}",
            mv, visits, wins, value
        )));
    }
    
    stats_text.push(Line::from(""));
    stats_text.push(Line::from(app.ai.get_debug_info()));


    let paragraph = Paragraph::new(stats_text)
        .block(Block::default().title("Statistics").borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    f.render_widget(paragraph, area);
}

fn draw_board(f: &mut Frame, app: &App, area: Rect) {
    let board = app.game.get_board();
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
            let symbol = match player {
                1 => "X",
                -1 => "O",
                _ => ".",
            };

            let mut style = Style::default();
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
