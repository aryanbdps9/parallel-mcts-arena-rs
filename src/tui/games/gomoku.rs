use ratatui::style::{Color, Modifier, Style};

pub fn get_cell_style(cell: i32, is_cursor: bool) -> (&'static str, Style) {
    match cell {
        1 => ("X", Style::default().fg(Color::Red)),
        -1 => ("O", Style::default().fg(Color::Blue)),
        _ => {
            if is_cursor {
                (
                    "▓",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                ("·", Style::default().fg(Color::DarkGray))
            }
        }
    }
}
