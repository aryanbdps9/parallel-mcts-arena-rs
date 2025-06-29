use crate::app::App;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
};
use std::{io, time::Duration};

pub mod input;
pub mod widgets;

pub fn run(app: &mut App) -> io::Result<()> {
    let mut terminal = init_terminal()?;

    loop {
        if app.should_quit {
            app.shutdown();
            break;
        }

        app.update();

        terminal.draw(|f| widgets::render(app, f))?;

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => {
                    if key.kind == KeyEventKind::Press {
                        input::handle_key_press(app, key.code);
                    }
                }
                Event::Mouse(mouse) => {
                    input::handle_mouse_event(app, mouse.kind, mouse.column, mouse.row, terminal.size()?);
                }
                _ => {}
            }
        }
    }

    restore_terminal(&mut terminal)
}

fn init_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    execute!(
        handle,
        EnterAlternateScreen,
        EnableMouseCapture,
        crossterm::cursor::Hide
    )?;
    Terminal::new(CrosstermBackend::new(stdout))
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    terminal.show_cursor()?;
    disable_raw_mode()?;
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    execute!(
        handle,
        LeaveAlternateScreen,
        DisableMouseCapture,
        crossterm::cursor::Show
    )?;
    Ok(())
}
