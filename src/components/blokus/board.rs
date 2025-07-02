//! Blokus board component.

use ratatui::{
    layout::Rect,
    Frame,
    widgets::{Block, Borders, Paragraph},
    style::{Style, Color, Modifier},
    text::{Line, Span},
};
use std::collections::HashSet;
use std::any::Any;
use mcts::GameState;

use crate::app::App;
use crate::game_wrapper::GameWrapper;
use crate::games::blokus::BlokusState;
use crate::components::core::{Component, ComponentId, ComponentResult, EventResult};
use crate::components::events::{ComponentEvent, InputEvent};

/// Component representing the Blokus game board
pub struct BlokusBoardComponent {
    id: ComponentId,
    area: Option<Rect>,
}

impl BlokusBoardComponent {
    pub fn new() -> Self {
        Self {
            id: ComponentId::new(),
            area: None,
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
            x >= area.x && x < area.x + area.width &&
            y >= area.y && y < area.y + area.height
        } else {
            false
        }
    }

    /// Handle board click to move cursor and make move
    pub fn handle_board_click(&self, app: &mut App, x: u16, y: u16) -> bool {
        if !self.contains_point(x, y) {
            return false;
        }

        let Some(area) = self.area else { return false; };

        // Calculate content area (inside borders)
        let content_area = Rect::new(
            area.x + 1,
            area.y + 1,
            area.width.saturating_sub(2),
            area.height.saturating_sub(2),
        );

        if x < content_area.x || x >= content_area.x + content_area.width ||
           y < content_area.y || y >= content_area.y + content_area.height {
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
            // Update cursor position
            app.board_cursor = (board_row as u16, board_col as u16);

            // If player is human, make the move
            if !app.is_current_player_ai() {
                self.make_move(app);
            }

            return true;
        }

        false
    }

    /// Attempt to make a move at the current cursor position
    fn make_move(&self, app: &mut App) {
        let (row, col) = (app.board_cursor.0 as usize, app.board_cursor.1 as usize);
        
        // For Blokus, create a move from the selected piece and cursor position
        let player_move = if let Some((piece_idx, transformation_idx)) = app.blokus_ui_config.get_selected_piece_info() {
            crate::game_wrapper::MoveWrapper::Blokus(crate::games::blokus::BlokusMove(piece_idx, transformation_idx, row, col))
        } else {
            // No piece selected, use pass move
            crate::game_wrapper::MoveWrapper::Blokus(crate::games::blokus::BlokusMove(usize::MAX, 0, 0, 0))
        };

        if app.game_wrapper.is_legal(&player_move) {
            let current_player = app.game_wrapper.get_current_player();
            app.move_history.push(crate::app::MoveHistoryEntry::new(current_player, player_move.clone()));
            app.on_move_added(); // Auto-scroll to bottom
            app.game_wrapper.make_move(&player_move);
            
            // Advance the AI worker's MCTS tree root to reflect the move that was just made
            app.ai_worker.advance_root(&player_move);
            
            // Clear selected piece if it becomes unavailable after move
            app.clear_selected_piece_if_unavailable();
            
            // Check for game over
            if app.game_wrapper.is_terminal() {
                app.game_status = match app.game_wrapper.get_winner() {
                    Some(winner) => crate::app::GameStatus::Win(winner),
                    None => crate::app::GameStatus::Draw,
                };
                app.mode = crate::app::AppMode::GameOver;
            }
        }
    }

    /// Render the Blokus board with ghost piece preview
    fn render_board_content(&self, frame: &mut Frame, area: Rect, app: &App, state: &BlokusState, selected_piece: Option<(usize, usize, usize, usize)>, show_cursor: bool) -> ComponentResult<()> {
        let board = state.get_board();
        let player_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
        
        // Calculate ghost piece positions if a piece is selected
        let mut ghost_positions = HashSet::new();
        if let Some((piece_idx, transformation_idx, ghost_row, ghost_col)) = selected_piece {
            if piece_idx < 21 {
                let pieces = crate::games::blokus::get_blokus_pieces();
                if let Some(piece_transformations) = pieces.get(piece_idx) {
                    if let Some(transformation) = piece_transformations.transformations.get(transformation_idx) {
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
                let cell_value = board[row][col];
                let is_cursor = show_cursor && app.board_cursor.0 == row as u16 && app.board_cursor.1 == col as u16;
                let is_ghost = ghost_positions.contains(&(row, col));
                
                let (cell_char, style) = if is_ghost {
                    // Ghost piece preview
                    let current_player = app.game_wrapper.get_current_player();
                    let color = player_colors[(current_player - 1) as usize];
                    ('▓', Style::default().fg(color).add_modifier(Modifier::DIM))
                } else if cell_value == 0 {
                    // Empty cell
                    if is_cursor {
                        ('·', Style::default().fg(Color::White).bg(Color::DarkGray))
                    } else {
                        ('·', Style::default().fg(Color::DarkGray))
                    }
                } else {
                    // Occupied cell
                    let player = cell_value as usize;
                    let color = if player <= 4 {
                        player_colors[player - 1]
                    } else {
                        Color::White
                    };
                    
                    let style = if is_cursor {
                        Style::default().fg(color).bg(Color::DarkGray).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(color).add_modifier(Modifier::BOLD)
                    };
                    
                    ('█', style)
                };
                
                line_spans.push(Span::styled(format!("{} ", cell_char), style));
            }
            
            if !line_spans.is_empty() {
                let line_area = Rect::new(area.x, area.y + row as u16, area.width, 1);
                let paragraph = Paragraph::new(Line::from(line_spans));
                frame.render_widget(paragraph, line_area);
            }
        }

        Ok(())
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
            _ => {}
        }
        Ok(false)
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()> {
        self.set_area(area);

        // Draw border
        let block = Block::default()
            .title("Blokus Board")
            .borders(Borders::ALL);
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
            // Get selected piece info from app state
            let selected_piece = if let Some((piece_idx, transformation_idx)) = app.blokus_ui_config.get_selected_piece_info() {
                Some((piece_idx, transformation_idx, app.board_cursor.0 as usize, app.board_cursor.1 as usize))
            } else {
                None
            };
            
            // Only show cursor for human turns
            let show_cursor = !app.is_current_player_ai();
            
            self.render_board_content(frame, content_area, app, state, selected_piece, show_cursor)?;
        }

        Ok(())
    }
}
