use ratatui::style::{Color, Style};

pub fn get_cell_style(cell: i32, _is_cursor: bool) -> (&'static str, Style) {
    match cell {
        1 => ("🔴", Style::default().fg(Color::Red)),
        -1 => ("🟡", Style::default().fg(Color::Yellow)),
        _ => ("·", Style::default().fg(Color::DarkGray)),
    }
}
