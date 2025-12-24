use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{Paragraph, Widget},
};

#[derive(Debug, Clone)]
pub struct GenericGridConfig {
    pub cell_width: u16,
    pub show_row_labels: bool,
    pub show_col_labels: bool,
    pub col_label_height: u16,
    pub row_label_width: u16,
}

impl Default for GenericGridConfig {
    fn default() -> Self {
        Self {
            cell_width: 2,
            show_row_labels: true,
            show_col_labels: true,
            col_label_height: 1,
            row_label_width: 2,
        }
    }
}

pub struct GenericGrid<'a, F>
where
    F: Fn(usize, usize, i32, bool) -> Span<'a>,
{
    board: &'a Vec<Vec<i32>>,
    cursor: Option<(usize, usize)>,
    config: GenericGridConfig,
    cell_renderer: F,
    highlight_col_idx: Option<usize>, // For Connect4-style column highlighting
}

impl<'a, F> GenericGrid<'a, F>
where
    F: Fn(usize, usize, i32, bool) -> Span<'a>,
{
    pub fn new(board: &'a Vec<Vec<i32>>, cell_renderer: F) -> Self {
        Self {
            board,
            cursor: None,
            config: GenericGridConfig::default(),
            cell_renderer,
            highlight_col_idx: None,
        }
    }

    pub fn cursor(mut self, cursor: Option<(usize, usize)>) -> Self {
        self.cursor = cursor;
        self
    }

    pub fn config(mut self, config: GenericGridConfig) -> Self {
        self.config = config;
        self
    }

    pub fn highlight_col(mut self, col_idx: Option<usize>) -> Self {
        self.highlight_col_idx = col_idx;
        self
    }
}

impl<'a, F> Widget for GenericGrid<'a, F>
where
    F: Fn(usize, usize, i32, bool) -> Span<'a>,
{
    fn render(self, area: Rect, buf: &mut Buffer) {
        let board_height = self.board.len();
        let board_width = if board_height > 0 {
            self.board[0].len()
        } else {
            0
        };

        if board_height == 0 || board_width == 0 {
            Paragraph::new("No board to display").render(area, buf);
            return;
        }

        // Calculate layout constraints
        let mut layout_constraints = Vec::new();

        // Column header row
        if self.config.show_col_labels {
            layout_constraints.push(Constraint::Length(self.config.col_label_height));
        }

        // Board rows
        for _ in 0..board_height {
            layout_constraints.push(Constraint::Length(1));
        }

        let rows_layout = Layout::default()
            .constraints(layout_constraints)
            .split(area);

        let row_offset = if self.config.show_col_labels { 1 } else { 0 };

        // Draw column labels
        if self.config.show_col_labels {
            let mut col_constraints = Vec::new();
            if self.config.show_row_labels {
                col_constraints.push(Constraint::Length(self.config.row_label_width));
            }
            col_constraints.extend(vec![Constraint::Length(self.config.cell_width); board_width]);

            let col_label_area = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(col_constraints)
                .split(rows_layout[0]);

            let col_start_idx = if self.config.show_row_labels { 1 } else { 0 };
            for c in 0..board_width {
                let col_number = (c + 1).to_string();
                
                let is_highlighted = self.highlight_col_idx.map_or(false, |idx| idx == c);

                let style = if is_highlighted {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                        .bg(Color::Blue)
                } else {
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD)
                };

                Paragraph::new(col_number)
                    .style(style)
                    .alignment(Alignment::Center)
                    .render(col_label_area[col_start_idx + c], buf);
            }
        }

        // Draw board rows
        for (r, row) in self.board.iter().enumerate() {
            // Safety check for layout bounds
            if r + row_offset >= rows_layout.len() {
                break;
            }
            
            let row_area = rows_layout[r + row_offset];

            let mut row_constraints = Vec::new();
            if self.config.show_row_labels {
                row_constraints.push(Constraint::Length(self.config.row_label_width));
            }
            row_constraints.extend(vec![Constraint::Length(self.config.cell_width); board_width]);

            let cell_areas = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(row_constraints)
                .split(row_area);

            // Draw row label
            if self.config.show_row_labels {
                let row_number = (r + 1).to_string();
                Paragraph::new(row_number)
                    .style(
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    )
                    .alignment(Alignment::Center)
                    .render(cell_areas[0], buf);
            }

            // Draw cells
            let cell_start_idx = if self.config.show_row_labels { 1 } else { 0 };
            for (c, &cell_value) in row.iter().enumerate() {
                if cell_start_idx + c >= cell_areas.len() {
                    break;
                }

                let is_cursor = self.cursor.map_or(false, |(cr, cc)| cr == r && cc == c);
                let span = (self.cell_renderer)(r, c, cell_value, is_cursor);
                
                Paragraph::new(span)
                    .alignment(Alignment::Center)
                    .render(cell_areas[cell_start_idx + c], buf);
            }
        }
    }
}
