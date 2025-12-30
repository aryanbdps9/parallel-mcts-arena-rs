#include "grid_common.wgsl"



fn othello_count_flips_dir(board: ptr<function, array<i32, 64>>, x: i32, y: i32, player: i32, d: i32) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    var dx = 0;
    var dy = 0;
    switch d {
        case 0: { dx = 0; dy = -1; }
        case 1: { dx = 1; dy = -1; }
        case 2: { dx = 1; dy = 0; }
        case 3: { dx = 1; dy = 1; }
        case 4: { dx = 0; dy = 1; }
        case 5: { dx = -1; dy = 1; }
        case 6: { dx = -1; dy = 0; }
        case 7: { dx = -1; dy = -1; }
        default: { dx = 0; dy = 0; }
    }
    let opponent = -player;
    var cx = x + dx;
    var cy = y + dy;
    var count = 0;
    while (cx >= 0 && cx < w && cy >= 0 && cy < h) {
        let cell = (*board)[cy * w + cx];
        if (cell == opponent) {
            count++;
            cx += dx;
            cy += dy;
        } else if (cell == player && count > 0) {
            return count;
        } else {
            return 0;
        }
    }
    return 0;
}

fn othello_is_valid_move(board: ptr<function, array<i32, 64>>, x: i32, y: i32, player: i32) -> bool {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    if (x < 0 || x >= w || y < 0 || y >= h) { return false; }
    if ((*board)[y * w + x] != 0) { return false; }
    for (var d = 0; d < 8; d++) {
        if (othello_count_flips_dir(board, x, y, player, d) > 0) { return true; }
    }
    return false;
}

fn othello_make_move(board: ptr<function, array<i32, 64>>, x: i32, y: i32, player: i32) -> array<i32, 64> {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    var new_board = *board;
    new_board[y * w + x] = player;
    for (var d = 0; d < 8; d++) {
        let flip_count = othello_count_flips_dir(&new_board, x, y, player, d);
        if (flip_count > 0) {
            var dx = 0;
            var dy = 0;
            switch d {
                case 0: { dx = 0; dy = -1; }
                case 1: { dx = 1; dy = -1; }
                case 2: { dx = 1; dy = 0; }
                case 3: { dx = 1; dy = 1; }
                case 4: { dx = 0; dy = 1; }
                case 5: { dx = -1; dy = 1; }
                case 6: { dx = -1; dy = 0; }
                case 7: { dx = -1; dy = -1; }
                default: { dx = 0; dy = 0; }
            }
            var cx = x + dx;
            var cy = y + dy;
            for (var i = 0; i < flip_count; i++) {
                new_board[cy * w + cx] = player;
                cx += dx;
                cy += dy;
            }
        }
    }
    return new_board;
}

fn othello_count_valid_moves(board: ptr<function, array<i32, 64>>, player: i32) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    var count = 0;
    for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
            if (othello_is_valid_move(board, x, y, player)) { count++; }
        }
    }
    return count;
}

fn othello_get_move_weight(x: i32, y: i32) -> f32 {
    // Uniform random - all moves equally weighted
    return 1.0;
}


fn othello_random_rollout(board_idx: u32, current_player: i32) -> f32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    let board_size = params.board_width * params.board_height;
    var sim_board: array<i32, 64>;
    let safe_board_size = min(board_size, 64u);
    for (var i = 0u; i < safe_board_size; i++) {
        sim_board[i] = boards[board_idx * board_size + i];
    }
    var sim_player = current_player;
    var consecutive_passes = 0;
    var moves_made = 0;
    let max_moves = 64;
    while (consecutive_passes < 2 && moves_made < max_moves) {
        var total_weight = 0.0;
        var valid_count = 0;
        for (var y = 0; y < h; y++) {
            for (var x = 0; x < w; x++) {
                if (othello_is_valid_move(&sim_board, x, y, sim_player)) {
                    valid_count++;
                    if (params.use_heuristic > 0u) {
                        total_weight += othello_get_move_weight(x, y);
                    } else {
                        total_weight += 1.0;
                    }
                }
            }
        }
        if (valid_count == 0) {
            consecutive_passes++;
            sim_player = -sim_player;
            continue;
        }
        consecutive_passes = 0;
        var threshold = rand() * total_weight;
        var picked_x = -1;
        var picked_y = -1;
        for (var y = 0; y < h; y++) {
            for (var x = 0; x < w; x++) {
                if (othello_is_valid_move(&sim_board, x, y, sim_player)) {
                    let weight = select(1.0, othello_get_move_weight(x, y), params.use_heuristic > 0u);
                    if (threshold < weight) {
                        picked_x = x;
                        picked_y = y;
                        y = h;
                        break;
                    }
                    threshold -= weight;
                }
            }
        }
        if (picked_x == -1) {
            for (var y = 0; y < h; y++) {
                for (var x = 0; x < w; x++) {
                    if (othello_is_valid_move(&sim_board, x, y, sim_player)) {
                        picked_x = x;
                        picked_y = y;
                        y = h;
                        break;
                    }
                }
            }
        }
        sim_board = othello_make_move(&sim_board, picked_x, picked_y, sim_player);
        sim_player = -sim_player;
        moves_made++;
    }
    var player_count = 0;
    var opp_count = 0;
    for (var i = 0; i < 64; i++) {
        if (i32(i) >= w * h) { break; }
        if (sim_board[i] == current_player) { player_count++; }
        else if (sim_board[i] == -current_player) { opp_count++; }
    }
    if (player_count > opp_count) { return 4000.0; }
    else if (opp_count > player_count) { return -4000.0; }
    else { return 0.0; }
}

@compute @workgroup_size(64)
fn evaluate_othello(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let current_player = 1;
    rng_state = params.seed + idx * 719393u;
    results[idx].score = othello_random_rollout(idx, current_player);
}
