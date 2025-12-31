
// BEGIN inlined common.wgsl (required for grid_common.wgsl)
struct SimulationParams {
    board_width: u32,
    board_height: u32,
    current_player: i32,
    use_heuristic: u32,
    seed: u32,
}

struct SimulationResult {
    score: f32,
}

@group(0) @binding(0) var<storage, read_write> boards: array<i32>;
@group(0) @binding(1) var<storage, read_write> results: array<SimulationResult>;
@group(0) @binding(2) var<uniform> params: SimulationParams;

// PCG Random Number Generator
var<private> rng_state: u32;

// Game type constants
const GAME_GOMOKU: u32 = 0u;
const GAME_CONNECT4: u32 = 1u;
const GAME_OTHELLO: u32 = 2u;
const GAME_BLOKUS: u32 = 3u;
const GAME_HIVE: u32 = 4u;

// Extract game parameters from encoded current_player field
fn get_line_size() -> i32 {
    let encoded = params.current_player;
    let line_size = (encoded >> 8) & 0xFF;
    if (line_size > 0) {
        return line_size;
    }
    return 5; // Default for Gomoku
}

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Improved random function using xorshift-based LCG
fn rand() -> f32 {
    // Use a hybrid approach: PCG for mixing, then xorshift for the sequence
    rng_state ^= rng_state << 13u;
    rng_state ^= rng_state >> 17u;
    rng_state ^= rng_state << 5u;
    // Mix in PCG for better quality
    rng_state = pcg_hash(rng_state);
    return f32(rng_state) / 4294967296.0;
}

fn rand_range(min: u32, max: u32) -> u32 {
    return min + u32(rand() * f32(max - min));
}

fn get_cell(board_idx: u32, x: i32, y: i32) -> i32 {
    if (x < 0 || x >= i32(params.board_width) || y < 0 || y >= i32(params.board_height)) {
        return 0; // Out of bounds treated as empty
    }
    let idx = board_idx * params.board_width * params.board_height + u32(y) * params.board_width + u32(x);
    return boards[idx];
}
// END inlined common.wgsl

// BEGIN inlined grid_common.wgsl
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

// ...existing code from grid_common.wgsl (all helper functions, including evaluate_grid_game_common) inlined here...

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
fn is_playable_c4(board_idx: u32, x: i32, y: i32) -> bool {
    let h = i32(params.board_height);
    if (get_cell(board_idx, x, y) != 0) { return false; }
    if (y == h - 1) { return true; }
    return get_cell(board_idx, x, y + 1) != 0;
}
fn count_immediate_threats_c4(board_idx: u32, player: i32, line_size: i32) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    let need = line_size - 1;
    var threats = 0;
    for (var y = 0; y < h; y++) {
        for (var x = 0; x <= w - line_size; x++) {
            var match_count = 0;
            var empty_x = -1;
            var blocked = false;
            for (var k = 0; k < line_size; k++) {
                let cell = get_cell(board_idx, x + k, y);
                if (cell == player) { match_count++; }
                else if (cell == 0) {
                    if (empty_x >= 0) { blocked = true; break; }
                    empty_x = x + k;
                }
                else { blocked = true; break; }
            }
            if (!blocked && match_count == need && empty_x >= 0) {
                if (is_playable_c4(board_idx, empty_x, y)) {
                    threats++;
                }
            }
        }
    }
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
    if (check_line_win(idx, current_player, line_size)) {
        results[idx].score = 4000.0;
        return;
    }
    if (check_line_win(idx, -current_player, line_size)) {
        results[idx].score = -4000.0;
        return;
    }
    if (params.use_heuristic != 0u) {
        if (is_connect4) {
            let player_immediate = count_immediate_threats_c4(idx, current_player, line_size);
            let opp_immediate = count_immediate_threats_c4(idx, -current_player, line_size);
            if (player_immediate > 0) {
                results[idx].score = 2000.0 + f32(player_immediate) * 100.0;
                return;
            }
            if (opp_immediate > 0) {
                if (opp_immediate >= 2) {
                    results[idx].score = -1500.0;
                    return;
                }
                results[idx].score = -500.0;
                return;
            }
        }
        let player_near_wins = count_pattern(idx, current_player, line_size);
        let player_threats = count_pattern(idx, current_player, line_size - 1);
        let player_builds = count_pattern(idx, current_player, line_size - 2);
        let opp_near_wins = count_pattern(idx, -current_player, line_size);
        let opp_threats = count_pattern(idx, -current_player, line_size - 1);
        let opp_builds = count_pattern(idx, -current_player, line_size - 2);
        let player_score = f32(player_near_wins) * 200.0 + f32(player_threats) * 30.0 + f32(player_builds) * 3.0;
        let opp_score = f32(opp_near_wins) * 250.0 + f32(opp_threats) * 40.0 + f32(opp_builds) * 3.0;
        results[idx].score = player_score - opp_score;
    } else {
        results[idx].score = gomoku_random_rollout(idx, current_player, line_size, game_type);
    }
}
// END inlined grid_common.wgsl

@compute @workgroup_size(64)
fn evaluate_connect4(@builtin(global_invocation_id) global_id: vec3<u32>) {
    evaluate_grid_game_common(global_id.x, GAME_CONNECT4);
}
