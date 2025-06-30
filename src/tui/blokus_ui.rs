//! # Blokus UI Module
//!
//! This module provides specialized UI components for the Blokus game,
//! including piece selection, ghost piece previews, and player status.

use crate::app::App;
use crate::game_wrapper::GameWrapper;
use crate::games::blokus::{get_blokus_pieces, BlokusState};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState};
use mcts::GameState;
use std::collections::HashSet;

/// Configuration for Blokus UI display
pub struct BlokusUIConfig {
    /// Whether each player's piece list is expanded
    pub players_expanded: [bool; 4],
    /// Currently selected piece index
    pub selected_piece_idx: Option<usize>,
    /// Current transformation index for the selected piece
    pub selected_transformation_idx: usize,
    /// Scroll offset for piece selection within current player
    pub piece_selection_scroll: usize,
    /// Scroll offset for the entire piece panel
    pub panel_scroll_offset: usize,
}

impl Default for BlokusUIConfig {
    fn default() -> Self {
        Self {
            players_expanded: [true, true, true, true],
            selected_piece_idx: None,
            selected_transformation_idx: 0,
            piece_selection_scroll: 0,
            panel_scroll_offset: 0,
        }
    }
}

impl BlokusUIConfig {
    /// Toggle expansion state for a player
    pub fn toggle_player_expand(&mut self, player_idx: usize) {
        if player_idx < 4 {
            self.players_expanded[player_idx] = !self.players_expanded[player_idx];
        }
    }

    /// Expand all players
    pub fn expand_all(&mut self) {
        self.players_expanded = [true, true, true, true];
    }

    /// Collapse all players
    pub fn collapse_all(&mut self) {
        self.players_expanded = [false, false, false, false];
    }

    /// Scroll panel up
    pub fn scroll_panel_up(&mut self) {
        self.panel_scroll_offset = self.panel_scroll_offset.saturating_sub(1);
    }

    /// Scroll panel down
    pub fn scroll_panel_down(&mut self) {
        self.panel_scroll_offset = self.panel_scroll_offset.saturating_add(1);
    }

    /// Scroll pieces up for current player
    pub fn scroll_pieces_up(&mut self) {
        self.piece_selection_scroll = self.piece_selection_scroll.saturating_sub(1);
    }

    /// Scroll pieces down for current player
    pub fn scroll_pieces_down(&mut self) {
        self.piece_selection_scroll = self.piece_selection_scroll.saturating_add(1);
    }

    /// Select a piece by index
    pub fn select_piece(&mut self, piece_idx: usize) {
        self.selected_piece_idx = Some(piece_idx);
        self.selected_transformation_idx = 0; // Reset transformation when selecting new piece
    }

    /// Rotate the selected piece (cycle through transformations)
    pub fn rotate_piece(&mut self, total_transformations: usize) {
        if total_transformations > 0 {
            self.selected_transformation_idx = (self.selected_transformation_idx + 1) % total_transformations;
        }
    }

    /// Get the currently selected piece and transformation
    pub fn get_selected_piece_info(&self) -> Option<(usize, usize)> {
        self.selected_piece_idx.map(|piece_idx| (piece_idx, self.selected_transformation_idx))
    }
}

/// Draw the Blokus board with ghost pieces
pub fn draw_blokus_board(f: &mut Frame, state: &BlokusState, area: Rect, selected_piece: Option<(usize, usize, usize, usize)>, cursor_pos: (u16, u16), show_cursor: bool) {
    let board = state.get_board();
    let board_height = board.len();
    let board_width = if board_height > 0 { board[0].len() } else { 0 };

    if board_height == 0 || board_width == 0 {
        let paragraph = Paragraph::new("No board to display");
        f.render_widget(paragraph, area);
        return;
    }

    let block = Block::default().borders(Borders::ALL).title("Blokus Board");
    let inner_area = block.inner(area);
    f.render_widget(block, area);

    // Create board display with ghost pieces
    let mut board_lines = Vec::new();
    
    // Get ghost piece positions if a piece is selected
    let ghost_positions = if let Some((piece_id, transformation, row, col)) = selected_piece {
        get_ghost_piece_positions(state, piece_id, transformation, row, col)
    } else {
        HashSet::new()
    };
    
    // Get last move positions for highlighting
    let last_move_positions: HashSet<(usize, usize)> = state.get_last_move()
        .map(|coords| coords.into_iter().collect())
        .unwrap_or_default();
    
    for (r, row) in board.iter().enumerate() {
        let mut line_spans = Vec::new();
        for (c, &cell) in row.iter().enumerate() {
            let is_cursor = (r as u16, c as u16) == cursor_pos;
            let is_ghost = ghost_positions.contains(&(r, c));
            let is_last_move = last_move_positions.contains(&(r, c));
            
            let (symbol, style) = if is_ghost {
                // Check if this ghost position would be legal
                let is_legal = if let Some((piece_id, transformation, cursor_row, cursor_col)) = selected_piece {
                    use crate::games::blokus::BlokusMove;
                    let test_move = BlokusMove(piece_id, transformation, cursor_row, cursor_col);
                    state.is_legal(&test_move)
                } else {
                    false
                };
                
                if is_legal {
                    // Legal ghost piece preview (cyan)
                    ("▓▓", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                } else {
                    // Illegal ghost piece preview (red)
                    ("▓▓", Style::default().fg(Color::Red).add_modifier(Modifier::DIM))
                }
            } else {
                match cell {
                    1 => {
                        let color = if is_last_move { Color::LightRed } else { Color::Red };
                        ("██", Style::default().fg(color).add_modifier(if is_last_move { Modifier::BOLD } else { Modifier::empty() }))
                    }
                    2 => {
                        let color = if is_last_move { Color::LightBlue } else { Color::Blue };
                        ("██", Style::default().fg(color).add_modifier(if is_last_move { Modifier::BOLD } else { Modifier::empty() }))
                    }
                    3 => {
                        let color = if is_last_move { Color::LightGreen } else { Color::Green };
                        ("██", Style::default().fg(color).add_modifier(if is_last_move { Modifier::BOLD } else { Modifier::empty() }))
                    }
                    4 => {
                        let color = if is_last_move { Color::LightYellow } else { Color::Yellow };
                        ("██", Style::default().fg(color).add_modifier(if is_last_move { Modifier::BOLD } else { Modifier::empty() }))
                    }
                    _ => ("┼┼", Style::default().fg(Color::DarkGray)), // Empty space with grid
                }
            };

            let final_style = if is_cursor && cell == 0 && show_cursor {
                style.bg(Color::Yellow)
            } else {
                style
            };

            line_spans.push(Span::styled(symbol, final_style));
        }
        board_lines.push(Line::from(line_spans));
    }

    let paragraph = Paragraph::new(board_lines);
    f.render_widget(paragraph, inner_area);
}

/// Draw the piece selection panel for all players
pub fn draw_blokus_piece_selection(f: &mut Frame, app: &App, area: Rect) {
    if let GameWrapper::Blokus(blokus_state) = &app.game_wrapper {
        // Calculate area for content and scrollbar
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

        let block = Block::default()
            .title("Available Pieces (All Players)")
            .borders(Borders::ALL);
        f.render_widget(block, area);

        let inner_area = Layout::default()
            .margin(1)
            .constraints([Constraint::Min(0)])
            .split(chunks[0])[0];

        let current_player = app.game_wrapper.get_current_player();
        let pieces = get_blokus_pieces();
        let player_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
        let player_names = ["P1", "P2", "P3", "P4"];

        let mut all_lines = Vec::new();

        // Generate content for all players
        for player in 1..=4 {
            let available_pieces = blokus_state.get_available_pieces(player);
            let available_count = available_pieces.len();
            let color = player_colors[(player - 1) as usize];
            let is_current = player == current_player;
            let is_expanded = app.blokus_ui_config.players_expanded.get((player - 1) as usize).unwrap_or(&true);

            // Convert available pieces to a set for quick lookup
            let _available_set: HashSet<usize> = available_pieces.iter().cloned().collect();

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
                let pieces_per_row = 5;
                let visible_pieces = if is_current { 21 } else { 10 };
                
                // Convert available pieces to a set for quick lookup
                let available_set: HashSet<usize> = available_pieces.iter().cloned().collect();
                
                // Show all pieces (0-20), graying out unavailable ones
                let total_pieces_to_show = if is_current { 21 } else { visible_pieces.min(21) };
                
                // Show pieces in rows
                for chunk_start in (0..total_pieces_to_show).step_by(pieces_per_row) {
                    let chunk_end = (chunk_start + pieces_per_row).min(total_pieces_to_show);
                    
                    let mut pieces_in_row = Vec::new();
                    for display_idx in chunk_start..chunk_end {
                        let piece_idx = display_idx; // Show pieces 0-20 in order
                        let piece = &pieces[piece_idx];
                        let is_available = available_set.contains(&piece_idx);
                        let is_selected = is_current && app.blokus_ui_config.selected_piece_idx == Some(piece_idx);
                        
                        // Create piece shape representation
                        let piece_shape = if !piece.transformations.is_empty() {
                            &piece.transformations[0]
                        } else {
                            continue;
                        };
                        
                        let key_label = if display_idx < 9 {
                            (display_idx + 1).to_string()
                        } else if display_idx == 9 {
                            "0".to_string()
                        } else {
                            ((b'a' + (display_idx - 10) as u8) as char).to_string()
                        };
                        
                        // Create visual shape for this piece
                        let piece_visual_lines = create_visual_piece_shape(piece_shape);
                        
                        let piece_name_text = if is_selected {
                            format!("[{}]", key_label)
                        } else {
                            format!(" {} ", key_label)
                        };
                        
                        // Determine style based on availability and selection
                        let style = if !is_available {
                            // Grayed out for used pieces
                            Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM)
                        } else if is_selected {
                            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD).bg(Color::DarkGray)
                        } else if is_current {
                            Style::default().fg(Color::White)
                        } else {
                            Style::default().fg(color)
                        };
                        
                        pieces_in_row.push((piece_name_text, piece_visual_lines, style));
                    }
                    
                    if !pieces_in_row.is_empty() {
                        // Find max height and width for alignment
                        let max_height = pieces_in_row.iter()
                            .map(|(_, lines, _)| lines.len())
                            .max()
                            .unwrap_or(1);
                        let piece_width = 8;
                        
                        // First line: piece keys/names
                        let mut key_line_spans = Vec::new();
                        for (i, (piece_name, _, style)) in pieces_in_row.iter().enumerate() {
                            let padded_name = format!("{:^width$}", piece_name, width = piece_width);
                            key_line_spans.push(Span::styled(padded_name, *style));
                            if i < pieces_in_row.len() - 1 {
                                key_line_spans.push(Span::styled("  ", Style::default()));
                            }
                        }
                        all_lines.push(Line::from(key_line_spans));
                        
                        // Show each line of the pieces
                        for line_idx in 0..max_height {
                            let mut shape_line_spans = Vec::new();
                            for (i, (_, piece_visual_lines, style)) in pieces_in_row.iter().enumerate() {
                                let piece_line = if line_idx < piece_visual_lines.len() {
                                    format!("{:^width$}", piece_visual_lines[line_idx], width = piece_width)
                                } else {
                                    " ".repeat(piece_width)
                                };
                                shape_line_spans.push(Span::styled(piece_line, *style));
                                if i < pieces_in_row.len() - 1 {
                                    shape_line_spans.push(Span::styled("  ", Style::default()));
                                }
                            }
                            all_lines.push(Line::from(shape_line_spans));
                        }
                    }
                }
                
                // No "more pieces" line needed since we show all pieces for current player
                // or a fixed number for other players
            } else {
                // Show compact summary when collapsed
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
            }
            
            // Add separator between players
            if player < 4 {
                all_lines.push(Line::from(""));
            }
        }

        // Add controls at the bottom
        all_lines.push(Line::from(""));
        all_lines.push(Line::from(Span::styled("Controls:", Style::default().fg(Color::Gray))));
        all_lines.push(Line::from(Span::styled("1-9,0,a-k: Select piece  R: Rotate  X: Flip", Style::default().fg(Color::Gray))));
        all_lines.push(Line::from(Span::styled("Z: Toggle expand  +/-: Expand/Collapse all", Style::default().fg(Color::Gray))));
        all_lines.push(Line::from(Span::styled("Drag walls ◀▶ to resize panel", Style::default().fg(Color::Cyan))));

        // Apply scrolling
        let content_height = all_lines.len();
        let visible_height = inner_area.height as usize;
        let max_scroll = content_height.saturating_sub(visible_height);
        let scroll_offset = app.blokus_ui_config.panel_scroll_offset.min(max_scroll);

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

        // Render scrollbar if needed
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

/// Draw the player status panel
pub fn draw_blokus_player_status(f: &mut Frame, app: &App, area: Rect) {
    if let GameWrapper::Blokus(blokus_state) = &app.game_wrapper {
        let block = Block::default().title("Players").borders(Borders::ALL);
        f.render_widget(block, area);
        
        let inner_area = Layout::default()
            .margin(1)
            .constraints([Constraint::Min(0)])
            .split(area)[0];

        let mut status_lines = Vec::new();
        let current_player = app.game_wrapper.get_current_player();
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

        // Add controls
        status_lines.push(Line::from(""));
        status_lines.push(Line::from(Span::styled("Controls:", Style::default().fg(Color::Gray))));
        status_lines.push(Line::from(Span::styled("1-9,0,a-k: Select", Style::default().fg(Color::Gray))));
        status_lines.push(Line::from(Span::styled("R: Rotate  X: Flip", Style::default().fg(Color::Gray))));
        status_lines.push(Line::from(Span::styled("Enter: Place", Style::default().fg(Color::Gray))));
        status_lines.push(Line::from(Span::styled("P: Pass", Style::default().fg(Color::Gray))));

        let paragraph = Paragraph::new(status_lines);
        f.render_widget(paragraph, inner_area);
    }
}

/// Draw debug info for selected piece (temporary for debugging)
pub fn draw_debug_selected_piece(f: &mut Frame, app: &App, area: Rect) {
    let selected_info = if let Some((piece_idx, transformation_idx)) = app.blokus_ui_config.get_selected_piece_info() {
        format!("Selected: Piece {} Transform {}", piece_idx, transformation_idx)
    } else {
        "No piece selected".to_string()
    };
    
    let debug_text = format!("DEBUG: {} | Cursor: {:?}", selected_info, app.board_cursor);
    let paragraph = Paragraph::new(debug_text)
        .block(Block::default().borders(Borders::ALL).title("Debug Info"));
    f.render_widget(paragraph, area);
}

/// Draw debug info for ghost piece troubleshooting
pub fn draw_ghost_debug_info(f: &mut Frame, app: &App, area: Rect) {
    let mut debug_lines = Vec::new();
    
    // Show selected piece info
    if let Some((piece_idx, transformation_idx)) = app.blokus_ui_config.get_selected_piece_info() {
        debug_lines.push(format!("Selected: Piece {} Transform {}", piece_idx, transformation_idx));
    } else {
        debug_lines.push("No piece selected".to_string());
    }
    
    // Show cursor position
    debug_lines.push(format!("Cursor: ({}, {})", app.board_cursor.0, app.board_cursor.1));
    
    // Show current player and available pieces
    if let GameWrapper::Blokus(state) = &app.game_wrapper {
        let current_player = state.get_current_player();
        let available_pieces = state.get_available_pieces(current_player);
        debug_lines.push(format!("Player: {} | Available: {:?}", current_player, &available_pieces[..std::cmp::min(available_pieces.len(), 5)]));
    }
    
    debug_lines.push("".to_string());
    debug_lines.push("Instructions:".to_string());
    debug_lines.push("1-9: Select piece".to_string());
    debug_lines.push("R: Rotate piece".to_string());
    debug_lines.push("Arrows: Move cursor".to_string());
    debug_lines.push("Enter: Place piece".to_string());
    
    let debug_text = debug_lines.join("\n");
    let paragraph = Paragraph::new(debug_text)
        .block(Block::default().borders(Borders::ALL).title("Debug - Ghost Piece"));
    f.render_widget(paragraph, area);
}

/// Create a visual representation of a piece shape
fn create_visual_piece_shape(piece_shape: &[(i32, i32)]) -> Vec<String> {
    if piece_shape.is_empty() {
        return vec!["▢".to_string()];
    }

    // Create a 2D visual representation
    let min_r = piece_shape.iter().map(|p| p.0).min().unwrap_or(0);
    let max_r = piece_shape.iter().map(|p| p.0).max().unwrap_or(0);
    let min_c = piece_shape.iter().map(|p| p.1).min().unwrap_or(0);
    let max_c = piece_shape.iter().map(|p| p.1).max().unwrap_or(0);

    let height = (max_r - min_r + 1) as usize;
    let width = (max_c - min_c + 1) as usize;

    // Create a grid to show the shape
    let mut grid = vec![vec![' '; width]; height];

    // Fill the grid with the piece shape
    for &(r, c) in piece_shape {
        let gr = (r - min_r) as usize;
        let gc = (c - min_c) as usize;
        grid[gr][gc] = '▢'; // Use empty square like ghost pieces
    }

    // Convert to vector of strings
    let mut result: Vec<String> = grid.iter()
        .map(|row| row.iter().collect::<String>())
        .collect();

    // Ensure minimum width for single character pieces
    if result.len() == 1 && result[0].trim().len() == 1 {
        result[0] = format!(" {} ", result[0].trim());
    }

    result
}

/// Get ghost piece positions for preview
fn get_ghost_piece_positions(state: &BlokusState, piece_id: usize, transformation: usize, row: usize, col: usize) -> HashSet<(usize, usize)> {
    let mut positions = HashSet::new();
    
    // Get the current player's available pieces
    let current_player = state.get_current_player();
    let available_pieces = state.get_available_pieces(current_player);
    
    // Check if this piece is still available for the current player
    if available_pieces.contains(&piece_id) {
        // Get the piece and its transformation
        let pieces = get_blokus_pieces();
        if let Some(piece) = pieces.iter().find(|p| p.id == piece_id) {
            if transformation < piece.transformations.len() {
                let shape = &piece.transformations[transformation];
                let board = state.get_board();
                let board_height = board.len();
                let board_width = if board_height > 0 { board[0].len() } else { 0 };
                
                // First, check if ALL blocks of the piece would be within bounds
                let mut all_positions_valid = true;
                let mut valid_positions = Vec::new();
                
                for &(dr, dc) in shape {
                    let board_r = row as i32 + dr;
                    let board_c = col as i32 + dc;
                    
                    // Check if this position is within board bounds
                    if board_r >= 0 && board_r < board_height as i32 && board_c >= 0 && board_c < board_width as i32 {
                        let board_r = board_r as usize;
                        let board_c = board_c as usize;
                        valid_positions.push((board_r, board_c));
                    } else {
                        // If any block is out of bounds, don't show the ghost piece at all
                        all_positions_valid = false;
                        break;
                    }
                }
                
                // Only show ghost piece if ALL blocks are within bounds
                if all_positions_valid {
                    for (board_r, board_c) in valid_positions {
                        // Only show ghost piece on empty squares
                        if board[board_r][board_c] == 0 {
                            positions.insert((board_r, board_c));
                        }
                    }
                }
            }
        }
    }
    
    positions
}
