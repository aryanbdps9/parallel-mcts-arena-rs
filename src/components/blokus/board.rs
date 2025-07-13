//! Blokus board component.

use mcts::GameState;
use ratatui::{
    Frame,
    layout::Rect,
    style::Color,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};
use std::any::Any;
use std::collections::HashSet;

use crate::app::App;
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};
use crate::components::ui::UITheme;
use crate::game_wrapper::GameWrapper;
use crate::games::blokus::BlokusState;

/// Direction for cursor movement
#[derive(Debug, Clone, Copy)]
pub enum CursorDirection {
    Up,
    Down,
    Left,
    Right,
}

/// Component representing the Blokus game board
pub struct BlokusBoardComponent {
    id: ComponentId,
    area: Option<Rect>,
    theme: UITheme,
    cursor_position: (u16, u16),
    selected_piece: Option<(usize, usize)>, // (piece_idx, transformation_idx)
}

impl BlokusBoardComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            area: None,
            theme: UITheme::default(),
            cursor_position: (10, 10), // Start in center of board
            selected_piece: None,
        }
    }

    pub fn set_area(&mut self, area: Rect) {
        self.area = Some(area);
    }

    pub fn get_area(&self) -> Option<Rect> {
        self.area
    }

    /// Check if a point is within this component's area
    pub fn contains_point(&self, x: u16, y: u16) -> bool {
        if let Some(area) = self.area {
            x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height
        } else {
            false
        }
    }

    /// Handle board click to move cursor and make move
    pub fn handle_board_click(&mut self, app: &mut App, x: u16, y: u16) -> bool {
        if !self.contains_point(x, y) {
            return false;
        }

        let Some(area) = self.area else {
            return false;
        };

        // Calculate content area (inside borders)
        let content_area = Rect::new(
            area.x + 1,
            area.y + 1,
            area.width.saturating_sub(2),
            area.height.saturating_sub(2),
        );

        if x < content_area.x
            || x >= content_area.x + content_area.width
            || y < content_area.y
            || y >= content_area.y + content_area.height
        {
            return false;
        }

        // Calculate board position
        let rel_x = x - content_area.x;
        let rel_y = y - content_area.y;

        // Each cell is 2 characters wide in the rendering
        let cell_width = 2;
        let cell_height = 1;

        let board_col = (rel_x / cell_width) as usize;
        let board_row = (rel_y / cell_height) as usize;

        // Validate bounds
        if board_row < 20 && board_col < 20 {
            // Always update cursor position and ghost piece location
            self.cursor_position = (board_row as u16, board_col as u16);
            app.board_cursor = (board_row as u16, board_col as u16);

            // If player is human, attempt to make the move only if it's legal
            if !app.is_current_player_ai() {
                self.try_make_move(app);
            }

            return true;
        }

        false
    }

    /// Attempt to make a move at the current cursor position, only if legal
    fn try_make_move(&self, app: &mut App) {
        let (row, col) = (
            self.cursor_position.0 as usize,
            self.cursor_position.1 as usize,
        );

        // Get selected piece from app's blokus_ui_config
        let player_move = if let Some((piece_idx, transformation_idx)) =
            app.blokus_ui_config.get_selected_piece_info()
        {
            crate::game_wrapper::MoveWrapper::Blokus(crate::games::blokus::BlokusMove(
                piece_idx,
                transformation_idx,
                row,
                col,
            ))
        } else {
            // No piece selected, use pass move
            crate::game_wrapper::MoveWrapper::Blokus(crate::games::blokus::BlokusMove(
                usize::MAX,
                0,
                0,
                0,
            ))
        };

        // Only make the move if it's legal
        if app.game_wrapper.is_legal(&player_move) {
            let current_player = app.game_wrapper.get_current_player();
            app.move_history.push(crate::app::MoveHistoryEntry::new(
                current_player,
                player_move.clone(),
            ));
            app.on_move_added(); // Auto-scroll to bottom
            app.game_wrapper.make_move(&player_move);

            // Advance the AI worker's MCTS tree root to reflect the move that was just made
            app.ai_worker.advance_root(&player_move);

            // Clear selected piece if it becomes unavailable after move
            // TODO: Implement piece availability check in component

            // Check for game over
            if app.game_wrapper.is_terminal() {
                app.game_status = match app.game_wrapper.get_winner() {
                    Some(winner) => crate::app::GameStatus::Win(winner),
                    None => crate::app::GameStatus::Draw,
                };
                app.mode = crate::app::AppMode::GameOver;
            }
        }
        // If move is not legal, cursor position has already been updated above
        // so the ghost piece will still show at the clicked location
    }

    /// Render the Blokus board with ghost piece preview
    fn render_board_content(
        &self,
        frame: &mut Frame,
        area: Rect,
        state: &BlokusState,
        selected_piece: Option<(usize, usize, usize, usize)>,
        show_cursor: bool,
    ) -> ComponentResult<()> {
        let board = state.get_board();
        let current_player = state.get_current_player() as u8;

        // Calculate ghost piece positions if a piece is selected
        let mut ghost_positions = HashSet::new();
        if let Some((piece_idx, transformation_idx, ghost_row, ghost_col)) = selected_piece {
            if piece_idx < 21 {
                let pieces = crate::games::blokus::get_blokus_pieces();
                if let Some(piece_transformations) = pieces.get(piece_idx) {
                    if let Some(transformation) = piece_transformations
                        .transformations
                        .get(transformation_idx)
                    {
                        for &(dr, dc) in transformation {
                            let new_row = ghost_row as i32 + dr;
                            let new_col = ghost_col as i32 + dc;

                            if new_row >= 0 && new_row < 20 && new_col >= 0 && new_col < 20 {
                                ghost_positions.insert((new_row as usize, new_col as usize));
                            }
                        }
                    }
                }
            }
        }

        // Render the board
        for row in 0..std::cmp::min(board.len(), area.height as usize) {
            let mut line_spans = Vec::new();

            for col in 0..std::cmp::min(board[row].len(), (area.width / 2) as usize) {
                // Divide by 2 for double-char width
                let cell_value = board[row][col];
                let is_cursor = show_cursor
                    && self.cursor_position.0 == row as u16
                    && self.cursor_position.1 == col as u16;
                let is_ghost = ghost_positions.contains(&(row, col));

                // Check if this is a legal ghost position
                let is_legal_ghost = if is_ghost {
                    if let Some((piece_idx, transformation_idx, ghost_row, ghost_col)) =
                        selected_piece
                    {
                        use crate::games::blokus::BlokusMove;
                        let test_move =
                            BlokusMove(piece_idx, transformation_idx, ghost_row, ghost_col);
                        state.is_legal(&test_move)
                    } else {
                        false
                    }
                } else {
                    false
                };

                let (symbol, style) = if is_ghost {
                    // Ghost piece preview with legality check and current player color
                    self.theme
                        .blokus_ghost_style(is_legal_ghost, current_player)
                } else {
                    // Get last move positions for highlighting
                    let last_move_positions: std::collections::HashSet<(usize, usize)> = state
                        .get_last_move()
                        .map(|coords| coords.into_iter().collect())
                        .unwrap_or_default();
                    let is_last_move = last_move_positions.contains(&(row, col));

                    self.theme
                        .blokus_cell_style(cell_value as u8, is_last_move, row, col)
                };

                let final_style = if is_cursor && cell_value == 0 && show_cursor {
                    style.bg(self.theme.cursor_style().bg.unwrap_or(Color::Yellow))
                } else {
                    style
                };

                line_spans.push(Span::styled(symbol, final_style));
            }

            if !line_spans.is_empty() {
                let line_area = Rect::new(area.x, area.y + row as u16, area.width, 1);
                let paragraph = Paragraph::new(Line::from(line_spans));
                frame.render_widget(paragraph, line_area);
            }
        }

        Ok(())
    }

    pub fn get_cursor_position(&self) -> (u16, u16) {
        self.cursor_position
    }

    pub fn set_cursor_position(&mut self, row: u16, col: u16) {
        self.cursor_position = (row, col);
    }

    pub fn get_selected_piece(&self) -> Option<(usize, usize)> {
        self.selected_piece
    }

    pub fn set_selected_piece(&mut self, piece: Option<(usize, usize)>) {
        self.selected_piece = piece;
    }

    pub fn move_cursor(&mut self, app: &mut App, direction: CursorDirection) {
        let (row, col) = self.cursor_position;
        match direction {
            CursorDirection::Up => {
                if row > 0 {
                    self.cursor_position.0 = row - 1;
                    app.board_cursor.0 = row - 1;
                }
            }
            CursorDirection::Down => {
                if row < 19 {
                    self.cursor_position.0 = row + 1;
                    app.board_cursor.0 = row + 1;
                }
            }
            CursorDirection::Left => {
                if col > 0 {
                    self.cursor_position.1 = col - 1;
                    app.board_cursor.1 = col - 1;
                }
            }
            CursorDirection::Right => {
                if col < 19 {
                    self.cursor_position.1 = col + 1;
                    app.board_cursor.1 = col + 1;
                }
            }
        }
    }
}

impl Component for BlokusBoardComponent {
    fn id(&self) -> ComponentId {
        self.id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult {
        match event {
            ComponentEvent::Input(InputEvent::MouseClick { x, y, .. }) => {
                if self.handle_board_click(app, *x, *y) {
                    return Ok(true);
                }
            }
            ComponentEvent::Input(InputEvent::KeyPress(key)) => match key {
                crossterm::event::KeyCode::Up => {
                    self.move_cursor(app, CursorDirection::Up);
                    return Ok(true);
                }
                crossterm::event::KeyCode::Down => {
                    self.move_cursor(app, CursorDirection::Down);
                    return Ok(true);
                }
                crossterm::event::KeyCode::Left => {
                    self.move_cursor(app, CursorDirection::Left);
                    return Ok(true);
                }
                crossterm::event::KeyCode::Right => {
                    self.move_cursor(app, CursorDirection::Right);
                    return Ok(true);
                }
                crossterm::event::KeyCode::Enter | crossterm::event::KeyCode::Char(' ') => {
                    if !app.is_current_player_ai() {
                        self.try_make_move(app);
                        return Ok(true);
                    }
                }
                _ => {}
            },
            _ => {}
        }
        Ok(false)
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        self.set_area(area);

        // Draw border
        let block = Block::default().title("Blokus Board").borders(Borders::ALL);
        frame.render_widget(block, area);

        // Calculate content area
        let content_area = Rect::new(
            area.x + 1,
            area.y + 1,
            area.width.saturating_sub(2),
            area.height.saturating_sub(2),
        );

        if content_area.width == 0 || content_area.height == 0 {
            return Ok(());
        }

        // Render board content
        if let GameWrapper::Blokus(state) = &app.game_wrapper {
            // Get selected piece info from app's blokus_ui_config instead of component state
            let selected_piece = if let Some((piece_idx, transformation_idx)) =
                app.blokus_ui_config.get_selected_piece_info()
            {
                Some((
                    piece_idx,
                    transformation_idx,
                    app.board_cursor.0 as usize,
                    app.board_cursor.1 as usize,
                ))
            } else {
                None
            };

            // Only show cursor for human turns
            let show_cursor = !app.is_current_player_ai();

            self.render_board_content(frame, content_area, state, selected_piece, show_cursor)?;
        }

        Ok(())
    }
}
