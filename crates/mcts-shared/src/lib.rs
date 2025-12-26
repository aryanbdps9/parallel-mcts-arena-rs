#![no_std]

/// Checks for a win in a grid-based game (Connect4, Gomoku, etc.)
/// 
/// # Arguments
/// * `board` - The board data as a flat slice
/// * `width` - Board width
/// * `height` - Board height
/// * `player` - The player ID to check for (e.g., 1 or -1)
/// * `line_size` - Number of consecutive pieces needed to win
pub fn check_line_win(board: &[i32], width: usize, height: usize, player: i32, line_size: usize) -> bool {
    check_line_win_with_offset(board, 0, width as i32, height as i32, player, line_size as i32)
}

/// Checks for a win in a grid-based game with an offset into the board slice
pub fn check_line_win_with_offset(board: &[i32], offset: usize, width: i32, height: i32, player: i32, line_size: i32) -> bool {
    let w = width;
    let h = height;
    
    // Helper to get cell value
    let get_cell = |x: i32, y: i32| -> i32 {
        if x < 0 || y < 0 || x >= w || y >= h {
            return 0;
        }
        let idx = offset + (y * w + x) as usize;
        if idx < board.len() {
            board[idx]
        } else {
            0
        }
    };
    
    // Horizontal
    for y in 0..h {
        for x in 0..=(w - line_size) {
            let mut match_len = 0;
            for k in 0..line_size {
                if get_cell(x + k, y) == player {
                    match_len += 1;
                } else {
                    break;
                }
            }
            if match_len == line_size {
                return true;
            }
        }
    }
    
    // Vertical
    for x in 0..w {
        for y in 0..=(h - line_size) {
            let mut match_len = 0;
            for k in 0..line_size {
                if get_cell(x, y + k) == player {
                    match_len += 1;
                } else {
                    break;
                }
            }
            if match_len == line_size {
                return true;
            }
        }
    }
    
    // Diagonal (TL-BR)
    for y in 0..=(h - line_size) {
        for x in 0..=(w - line_size) {
            let mut match_len = 0;
            for k in 0..line_size {
                if get_cell(x + k, y + k) == player {
                    match_len += 1;
                } else {
                    break;
                }
            }
            if match_len == line_size {
                return true;
            }
        }
    }
    
    // Diagonal (TR-BL)
    for y in 0..=(h - line_size) {
        for x in (line_size - 1)..w {
            let mut match_len = 0;
            for k in 0..line_size {
                if get_cell(x - k, y + k) == player {
                    match_len += 1;
                } else {
                    break;
                }
            }
            if match_len == line_size {
                return true;
            }
        }
    }
    
    false
}
