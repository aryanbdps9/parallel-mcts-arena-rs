// WGSL does not support #include. Manually paste shared code here if needed.

// Check N-in-a-row win condition
fn check_line_win(board_idx: u32, player: i32, line_size: i32) -> bool {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    
    // Horizontal
    for (var y = 0; y < h; y++) {
        for (var x = 0; x <= w - line_size; x++) {
            var match_len = 0;
            for (var k = 0; k < line_size; k++) {
                if (get_cell(board_idx, x + k, y) == player) { match_len++; } else { break; }
            }
            if (match_len == line_size) { return true; }
        }
    }
    
    // Vertical
    for (var x = 0; x < w; x++) {
        for (var y = 0; y <= h - line_size; y++) {
            var match_len = 0;
            for (var k = 0; k < line_size; k++) {
                if (get_cell(board_idx, x, y + k) == player) { match_len++; } else { break; }
            }
            if (match_len == line_size) { return true; }
        }
    }
    
    // Diagonal (TL-BR)
    for (var y = 0; y <= h - line_size; y++) {
        for (var x = 0; x <= w - line_size; x++) {
            var match_len = 0;
            for (var k = 0; k < line_size; k++) {
                if (get_cell(board_idx, x + k, y + k) == player) { match_len++; } else { break; }
            }
            if (match_len == line_size) { return true; }
        }
    }

    // Diagonal (TR-BL)
    for (var y = 0; y <= h - line_size; y++) {
        for (var x = line_size - 1; x < w; x++) {
            var match_len = 0;
            for (var k = 0; k < line_size; k++) {
                if (get_cell(board_idx, x - k, y + k) == player) { match_len++; } else { break; }
            }
            if (match_len == line_size) { return true; }
        }
    }
    
    return false;
}

fn count_pattern(board_idx: u32, player: i32, length: i32) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    var count = 0;
    
    // Horizontal patterns
    for (var y = 0; y < h; y++) {
        for (var x = 0; x <= w - length; x++) {
            var match_count = 0;
            var empty_count = 0;
            for (var k = 0; k < length; k++) {
                let cell = get_cell(board_idx, x + k, y);
                if (cell == player) { match_count++; }
                else if (cell == 0) { empty_count++; }
                else { break; }
            }
            if (match_count > 0 && match_count + empty_count == length) { count += 1; }
        }
    }
    
    // Vertical patterns
    for (var x = 0; x < w; x++) {
        for (var y = 0; y <= h - length; y++) {
            var match_count = 0;
            var empty_count = 0;
            for (var k = 0; k < length; k++) {
                let cell = get_cell(board_idx, x, y + k);
                if (cell == player) { match_count++; }
                else if (cell == 0) { empty_count++; }
                else { break; }
            }
            if (match_count > 0 && match_count + empty_count == length) { count += 1; }
        }
    }
    
    // Diagonal (TL-BR)
    for (var y = 0; y <= h - length; y++) {
        for (var x = 0; x <= w - length; x++) {
            var match_count = 0;
            var empty_count = 0;
            for (var k = 0; k < length; k++) {
                let cell = get_cell(board_idx, x + k, y + k);
                if (cell == player) { match_count++; }
                else if (cell == 0) { empty_count++; }
                else { break; }
            }
            if (match_count > 0 && match_count + empty_count == length) { count += 1; }
        }
    }
    
    // Diagonal (TR-BL)
    for (var y = 0; y <= h - length; y++) {
        for (var x = length - 1; x < w; x++) {
            var match_count = 0;
            var empty_count = 0;
            for (var k = 0; k < length; k++) {
                let cell = get_cell(board_idx, x - k, y + k);
                if (cell == player) { match_count++; }
                else if (cell == 0) { empty_count++; }
                else { break; }
            }
            if (match_count > 0 && match_count + empty_count == length) { count += 1; }
        }
    }
    
    return count;
}

// Check if a cell is playable in Connect4 (piece can be placed there)
fn is_playable_c4(board_idx: u32, x: i32, y: i32) -> bool {
    let h = i32(params.board_height);
    // Cell must be empty
    if (get_cell(board_idx, x, y) != 0) { return false; }
    // Bottom row is always playable if empty
    if (y == h - 1) { return true; }
    // Otherwise, cell below must be filled
    return get_cell(board_idx, x, y + 1) != 0;
}

// Count immediate winning threats (N-1 in a row with playable winning cell) for Connect4
fn count_immediate_threats_c4(board_idx: u32, player: i32, line_size: i32) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    let need = line_size - 1; // e.g., 3 for Connect4
    var threats = 0;
    
    // Horizontal threats: look for (need) pieces with a playable gap
    for (var y = 0; y < h; y++) {
        for (var x = 0; x <= w - line_size; x++) {
            var match_count = 0;
            var empty_x = -1;
            var blocked = false;
            for (var k = 0; k < line_size; k++) {
                let cell = get_cell(board_idx, x + k, y);
                if (cell == player) { match_count++; }
                else if (cell == 0) {
                    if (empty_x >= 0) { blocked = true; break; } // More than one empty = not a threat
                    empty_x = x + k;
                }
                else { blocked = true; break; } // Opponent piece
            }
            if (!blocked && match_count == need && empty_x >= 0) {
                if (is_playable_c4(board_idx, empty_x, y)) {
                    threats++;
                }
            }
        }
    }
    
    // Vertical threats
    for (var x = 0; x < w; x++) {
        for (var y = 0; y <= h - line_size; y++) {
            var match_count = 0;
            var empty_y = -1;
            var blocked = false;
            for (var k = 0; k < line_size; k++) {
                let cell = get_cell(board_idx, x, y + k);
                if (cell == player) { match_count++; }
                else if (cell == 0) {
                    if (empty_y >= 0) { blocked = true; break; }
                    empty_y = y + k;
                }
                else { blocked = true; break; }
            }
            if (!blocked && match_count == need && empty_y >= 0) {
                if (is_playable_c4(board_idx, x, empty_y)) {
                    threats++;
                }
            }
        }
    }
    
    // Diagonal (TL-BR) threats
    for (var y = 0; y <= h - line_size; y++) {
        for (var x = 0; x <= w - line_size; x++) {
            var match_count = 0;
            var empty_k = -1;
            var blocked = false;
            for (var k = 0; k < line_size; k++) {
                let cell = get_cell(board_idx, x + k, y + k);
                if (cell == player) { match_count++; }
                else if (cell == 0) {
                    if (empty_k >= 0) { blocked = true; break; }
                    empty_k = k;
                }
                else { blocked = true; break; }
            }
            if (!blocked && match_count == need && empty_k >= 0) {
                if (is_playable_c4(board_idx, x + empty_k, y + empty_k)) {
                    threats++;
                }
            }
        }
    }
    
    // Diagonal (TR-BL) threats
    for (var y = 0; y <= h - line_size; y++) {
        for (var x = line_size - 1; x < w; x++) {
            var match_count = 0;
            var empty_k = -1;
            var blocked = false;
            for (var k = 0; k < line_size; k++) {
                let cell = get_cell(board_idx, x - k, y + k);
                if (cell == player) { match_count++; }
                else if (cell == 0) {
                    if (empty_k >= 0) { blocked = true; break; }
                    empty_k = k;
                }
                else { blocked = true; break; }
            }
            if (!blocked && match_count == need && empty_k >= 0) {
                if (is_playable_c4(board_idx, x - empty_k, y + empty_k)) {
                    threats++;
                }
            }
        }
    }
    
    return threats;
}

fn gomoku_random_rollout(idx: u32, current_player: i32, line_size: i32, game_type: u32) -> f32 {
    rng_state = params.seed + idx * 719393u;
    
    var sim_board: array<i32, 400>; 
    let board_size = params.board_width * params.board_height;
    let safe_board_size = min(board_size, 400u);
    
    for (var i = 0u; i < safe_board_size; i++) {
        sim_board[i] = get_cell(idx, i32(i % params.board_width), i32(i / params.board_width));
    }
    
    var sim_player = current_player;
    let max_moves = i32(safe_board_size);
    var moves_made = 0;
    let win_count = line_size;
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    
    let is_connect4 = (game_type == GAME_CONNECT4);
    
    loop {
        if (moves_made >= max_moves) { break; }
        
        var move_idx = 0u;
        var found_move = false;
        
        if (is_connect4) {
            var valid_cols = 0u;
            for (var c = 0u; c < params.board_width; c++) {
                if (sim_board[c] == 0) { valid_cols++; }
            }
            
            if (valid_cols == 0u) { break; }
            
            let pick = rand_range(0u, valid_cols);
            var current_valid = 0u;
            var chosen_col = 0u;
            
            for (var c = 0u; c < params.board_width; c++) {
                if (sim_board[c] == 0) {
                    if (current_valid == pick) {
                        chosen_col = c;
                        break;
                    }
                    current_valid++;
                }
            }
            
            for (var r = i32(params.board_height) - 1; r >= 0; r--) {
                let check_idx = u32(r) * params.board_width + chosen_col;
                if (sim_board[check_idx] == 0) {
                    move_idx = check_idx;
                    found_move = true;
                    break;
                }
            }
        } else {
            var empty_count = 0u;
            for (var i = 0u; i < safe_board_size; i++) {
                if (sim_board[i] == 0) { empty_count++; }
            }
            
            if (empty_count == 0u) { break; }
            
            let pick = rand_range(0u, empty_count);
            var current_empty = 0u;
            
            for (var i = 0u; i < safe_board_size; i++) {
                if (sim_board[i] == 0) {
                    if (current_empty == pick) {
                        move_idx = i;
                        found_move = true;
                        break;
                    }
                    current_empty++;
                }
            }
        }
        
        if (!found_move) { break; }
        
        sim_board[move_idx] = sim_player;
        
        var won = false;
        let row = i32(move_idx) / w;
        let col = i32(move_idx) % w;
        
        // Horizontal
        var count = 1;
        var x = col - 1;
        while (x >= 0) {
            let check_idx = row * w + x;
            if (sim_board[check_idx] == sim_player) { count++; } else { break; }
            x--;
        }
        x = col + 1;
        while (x < w) {
            let check_idx = row * w + x;
            if (sim_board[check_idx] == sim_player) { count++; } else { break; }
            x++;
        }
        if (count >= win_count) { won = true; }
        
        if (!won) {
            // Vertical
            count = 1;
            var y = row - 1;
            while (y >= 0) {
                let check_idx = y * w + col;
                if (sim_board[check_idx] == sim_player) { count++; } else { break; }
                y--;
            }
            y = row + 1;
            while (y < h) {
                let check_idx = y * w + col;
                if (sim_board[check_idx] == sim_player) { count++; } else { break; }
                y++;
            }
            if (count >= win_count) { won = true; }
        }
        
        if (!won) {
            // Diagonal TL-BR
            count = 1;
            var dx = -1;
            var dy = -1;
            var cx = col + dx;
            var cy = row + dy;
            while (cx >= 0 && cx < w && cy >= 0 && cy < h) {
                let check_idx = cy * w + cx;
                if (sim_board[check_idx] == sim_player) { count++; } else { break; }
                cx += dx;
                cy += dy;
            }
            dx = 1;
            dy = 1;
            cx = col + dx;
            cy = row + dy;
            while (cx >= 0 && cx < w && cy >= 0 && cy < h) {
                let check_idx = cy * w + cx;
                if (sim_board[check_idx] == sim_player) { count++; } else { break; }
                cx += dx;
                cy += dy;
            }
            if (count >= win_count) { won = true; }
        }
        
        if (!won) {
            // Diagonal TR-BL
            count = 1;
            var dx = 1;
            var dy = -1;
            var cx = col + dx;
            var cy = row + dy;
            while (cx >= 0 && cx < w && cy >= 0 && cy < h) {
                let check_idx = cy * w + cx;
                if (sim_board[check_idx] == sim_player) { count++; } else { break; }
                cx += dx;
                cy += dy;
            }
            dx = -1;
            dy = 1;
            cx = col + dx;
            cy = row + dy;
            while (cx >= 0 && cx < w && cy >= 0 && cy < h) {
                let check_idx = cy * w + cx;
                if (sim_board[check_idx] == sim_player) { count++; } else { break; }
                cx += dx;
                cy += dy;
            }
            if (count >= win_count) { won = true; }
        }
        
        if (won) {
            if (sim_player == current_player) { return 4000.0; } else { return -4000.0; }
        }
        
        sim_player = -sim_player;
        moves_made++;
    }
    
    return 0.0;
}

fn evaluate_grid_game_common(idx: u32, game_type: u32) {
    if (params.board_width > 32u || params.board_height > 32u) { return; }
    
    let current_player = 1;
    let line_size = get_line_size();
    let is_connect4 = (game_type == GAME_CONNECT4);
    
    rng_state = params.seed + idx * 719393u;
    
    // Check for immediate wins/losses
    if (check_line_win(idx, current_player, line_size)) {
        results[idx].score = 4000.0;
        return;
    }
    if (check_line_win(idx, -current_player, line_size)) {
        results[idx].score = -4000.0;
        return;
    }
    
    if (params.use_heuristic != 0u) {
        // For Connect4, use gravity-aware threat detection
        if (is_connect4) {
            // Count immediate threats (can win on next move)
            let player_immediate = count_immediate_threats_c4(idx, current_player, line_size);
            let opp_immediate = count_immediate_threats_c4(idx, -current_player, line_size);
            
            // If player has immediate winning threat, it's nearly a win
            if (player_immediate > 0) {
                results[idx].score = 2000.0 + f32(player_immediate) * 100.0;
                return;
            }
            
            // If opponent has immediate winning threat (and it's their turn to play in
            // the position we're evaluating - which it would be after our move),
            // this is very bad - we either need to block or we lose
            if (opp_immediate > 0) {
                // Multiple threats = almost certainly lost
                if (opp_immediate >= 2) {
                    results[idx].score = -1500.0;
                    return;
                }
                // Single threat = must block, slight disadvantage
                results[idx].score = -500.0;
                return;
            }
        }
        
        // General pattern-based evaluation (for non-immediate situations)
        let player_near_wins = count_pattern(idx, current_player, line_size);
        let player_threats = count_pattern(idx, current_player, line_size - 1);
        let player_builds = count_pattern(idx, current_player, line_size - 2);
        
        let opp_near_wins = count_pattern(idx, -current_player, line_size);
        let opp_threats = count_pattern(idx, -current_player, line_size - 1);
        let opp_builds = count_pattern(idx, -current_player, line_size - 2);
        
        // Improved weights: prioritize defensive awareness
        // Near-wins (one step from winning) are very valuable
        // Threats (two steps) matter more than before
        let player_score = f32(player_near_wins) * 200.0 + f32(player_threats) * 30.0 + f32(player_builds) * 3.0;
        let opp_score = f32(opp_near_wins) * 250.0 + f32(opp_threats) * 40.0 + f32(opp_builds) * 3.0;
        
        results[idx].score = player_score - opp_score;
    } else {
        results[idx].score = gomoku_random_rollout(idx, current_player, line_size, game_type);
    }
}
