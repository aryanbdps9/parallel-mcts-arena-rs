use std::sync::OnceLock;

fn source(name: &str) -> Option<&'static str> {
    match name {
        "puct.wgsl" => Some(PUCT_WGSL),
        "common.wgsl" => Some(COMMON_WGSL),
        "grid_common.wgsl" => Some(GRID_COMMON_WGSL),
        "gomoku.wgsl" => Some(GOMOKU_WGSL),
        "othello.wgsl" => Some(OTHELLO_WGSL),
        "blokus.wgsl" => Some(BLOKUS_WGSL),
        "hive.wgsl" => Some(HIVE_WGSL),
        _ => None,
    }
}

fn expand_includes(name: &str) -> String {
    fn expand_inner(name: &str, stack: &mut Vec<String>) -> String {
        if stack.iter().any(|s| s == name) {
            panic!("WGSL include cycle detected: {} -> {}", stack.join(" -> "), name);
        }
        let src = source(name).unwrap_or_else(|| panic!("Unknown embedded WGSL file: {name}"));
        stack.push(name.to_string());

        let mut out = String::with_capacity(src.len());
        for line in src.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("#include \"") {
                if let Some(include_name) = rest.strip_suffix('"') {
                    out.push_str(&expand_inner(include_name, stack));
                    out.push('\n');
                    continue;
                }
            }
            out.push_str(line);
            out.push('\n');
        }

        stack.pop();
        out
    }

    expand_inner(name, &mut Vec::new())
}

static GOMOKU_EXPANDED: OnceLock<String> = OnceLock::new();
static OTHELLO_EXPANDED: OnceLock<String> = OnceLock::new();
static BLOKUS_EXPANDED: OnceLock<String> = OnceLock::new();
static HIVE_EXPANDED: OnceLock<String> = OnceLock::new();

pub fn puct_wgsl() -> &'static str {
    PUCT_WGSL
}

pub fn gomoku_wgsl() -> &'static str {
    GOMOKU_EXPANDED
        .get_or_init(|| expand_includes("gomoku.wgsl"))
        .as_str()
}

pub fn othello_wgsl() -> &'static str {
    OTHELLO_EXPANDED
        .get_or_init(|| expand_includes("othello.wgsl"))
        .as_str()
}

pub fn blokus_wgsl() -> &'static str {
    BLOKUS_EXPANDED
        .get_or_init(|| expand_includes("blokus.wgsl"))
        .as_str()
}

pub fn hive_wgsl() -> &'static str {
    HIVE_EXPANDED
        .get_or_init(|| expand_includes("hive.wgsl"))
        .as_str()
}

const PUCT_WGSL: &str = r##"// Node data structure for PUCT calculation
struct NodeData {
    visits: i32,          // Number of visits to this node
    wins: i32,            // Accumulated wins (scaled by 2: 2=win, 1=draw, 0=loss)
    virtual_losses: i32,  // Virtual losses for parallel coordination
    parent_visits: i32,   // Parent node's visit count
    prior_prob: f32,      // Prior probability (uniform = 1/num_children)
    exploration: f32,     // Exploration parameter (C_puct)
    _padding: vec2<f32>,  // Padding for alignment
}

// Result of PUCT calculation
struct PuctResult {
    puct_score: f32,      // Calculated PUCT score
    q_value: f32,         // Q value (exploitation term)
    exploration_term: f32, // Exploration term
    node_index: u32,      // Original index for sorting
}

@group(0) @binding(0) var<storage, read> nodes: array<NodeData>;
@group(0) @binding(1) var<storage, read_write> results: array<PuctResult>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = num_nodes, y-w = reserved

@compute @workgroup_size(256)
fn compute_puct(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_nodes = params.x;
    
    if (idx >= num_nodes) {
        return;
    }
    
    let node = nodes[idx];
    let visits = node.visits;
    let virtual_losses = node.virtual_losses;
    let effective_visits = visits + virtual_losses;
    
    var q_value: f32 = 0.0;
    var exploration_term: f32;
    var puct_score: f32;
    
    let parent_visits_sqrt = sqrt(f32(node.parent_visits));
    
    if (effective_visits == 0) {
        // Unvisited nodes: high exploration bonus
        exploration_term = node.exploration * node.prior_prob * parent_visits_sqrt;
        q_value = 0.0;
        puct_score = exploration_term;
    } else {
        // Visited nodes: balance exploitation and exploration
        let effective_visits_f = f32(effective_visits);
        
        if (visits > 0) {
            // Q value: win rate from this node's perspective
            q_value = (f32(node.wins) / f32(visits)) / 2.0;
        }
        
        // Exploration term with virtual loss penalty
        exploration_term = node.exploration * node.prior_prob * parent_visits_sqrt / (1.0 + effective_visits_f);
        puct_score = q_value + exploration_term;
    }
    
    // Store result
    results[idx].puct_score = puct_score;
    results[idx].q_value = q_value;
    results[idx].exploration_term = exploration_term;
    results[idx].node_index = idx;
}
"##;

const COMMON_WGSL: &str = r##"struct SimulationParams {
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
    let line_size_bits = (bitcast<u32>(encoded) >> 8u) & 0xFFu;
    let line_size = i32(line_size_bits);
    if (line_size > 0) {
        return line_size;
    }
    return 5; // Default for Gomoku
}

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand() -> f32 {
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
"##;

const GRID_COMMON_WGSL: &str = r##"#include "common.wgsl"

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
        let player_near_wins = count_pattern(idx, current_player, line_size);
        let player_threats = count_pattern(idx, current_player, line_size - 1);
        let player_builds = count_pattern(idx, current_player, line_size - 2);
        
        let opp_near_wins = count_pattern(idx, -current_player, line_size);
        let opp_threats = count_pattern(idx, -current_player, line_size - 1);
        let opp_builds = count_pattern(idx, -current_player, line_size - 2);
        
        let player_score = f32(player_near_wins) * 100.0 + f32(player_threats) * 10.0 + f32(player_builds) * 1.0;
        let opp_score = f32(opp_near_wins) * 100.0 + f32(opp_threats) * 10.0 + f32(opp_builds) * 1.0;
        
        results[idx].score = player_score - opp_score;
    } else {
        results[idx].score = gomoku_random_rollout(idx, current_player, line_size, game_type);
    }
}
"##;

const GOMOKU_WGSL: &str = r##"#include "grid_common.wgsl"

@compute @workgroup_size(64)
fn evaluate_gomoku(@builtin(global_invocation_id) global_id: vec3<u32>) {
    evaluate_grid_game_common(global_id.x, GAME_GOMOKU);
}
"##;

const OTHELLO_WGSL: &str = r##"#include "grid_common.wgsl"

fn othello_dir(d: i32) -> vec2<i32> {
    // Naga (via wgpu) rejects dynamic indexing into const arrays in some backends.
    // Use an explicit switch to keep shader validation happy.
    switch (d) {
        case 0: { return vec2<i32>(0, -1); }
        case 1: { return vec2<i32>(1, -1); }
        case 2: { return vec2<i32>(1, 0); }
        case 3: { return vec2<i32>(1, 1); }
        case 4: { return vec2<i32>(0, 1); }
        case 5: { return vec2<i32>(-1, 1); }
        case 6: { return vec2<i32>(-1, 0); }
        case 7: { return vec2<i32>(-1, -1); }
        default: { return vec2<i32>(0, 0); }
    }
}

fn othello_count_flips_dir(board: ptr<function, array<i32, 64>>, x: i32, y: i32, player: i32, d: i32) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    let dir = othello_dir(d);
    let dx = dir.x;
    let dy = dir.y;
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

fn othello_make_move(board: ptr<function, array<i32, 64>>, x: i32, y: i32, player: i32) {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    
    (*board)[y * w + x] = player;
    
    for (var d = 0; d < 8; d++) {
        let flip_count = othello_count_flips_dir(board, x, y, player, d);
        if (flip_count > 0) {
            let dir = othello_dir(d);
            let dx = dir.x;
            let dy = dir.y;
            var cx = x + dx;
            var cy = y + dy;
            for (var i = 0; i < flip_count; i++) {
                (*board)[cy * w + cx] = player;
                cx += dx;
                cy += dy;
            }
        }
    }
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

fn othello_get_nth_valid_move(board: ptr<function, array<i32, 64>>, player: i32, n: i32) -> vec2<i32> {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    var count = 0;
    
    for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
            if (othello_is_valid_move(board, x, y, player)) {
                if (count == n) { return vec2<i32>(x, y); }
                count++;
            }
        }
    }
    return vec2<i32>(-1, -1);
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
        let valid_count = othello_count_valid_moves(&sim_board, sim_player);
        
        if (valid_count == 0) {
            consecutive_passes++;
            sim_player = -sim_player;
            continue;
        }
        
        consecutive_passes = 0;
        let pick = i32(rand() * f32(valid_count));
        let move_pos = othello_get_nth_valid_move(&sim_board, sim_player, pick);
        
        othello_make_move(&sim_board, move_pos.x, move_pos.y, sim_player);
        
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
"##;

const BLOKUS_WGSL: &str = r##"#include "grid_common.wgsl"

var<private> BLOKUS_PIECES: array<u32, 168> = array<u32, 168>(
    0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u,
    0x00000003u, 0x00000021u, 0x00000003u, 0x00000021u, 0x00000003u, 0x00000021u, 0x00000003u, 0x00000021u,
    0x00000007u, 0x00000421u, 0x00000007u, 0x00000421u, 0x00000007u, 0x00000421u, 0x00000007u, 0x00000421u,
    0x00000061u, 0x00000023u, 0x00000043u, 0x00000062u, 0x00000062u, 0x00000061u, 0x00000023u, 0x00000043u,
    0x0000000Fu, 0x00008421u, 0x0000000Fu, 0x00008421u, 0x0000000Fu, 0x00008421u, 0x0000000Fu, 0x00008421u,
    0x00000C21u, 0x00000027u, 0x00000843u, 0x000000E4u, 0x00000C42u, 0x000000E1u, 0x00000423u, 0x00000087u,
    0x00000063u, 0x00000063u, 0x00000063u, 0x00000063u, 0x00000063u, 0x00000063u, 0x00000063u, 0x00000063u,
    0x000000C3u, 0x00000462u, 0x000000C3u, 0x00000462u, 0x00000066u, 0x00000861u, 0x00000066u, 0x00000861u,
    0x00000047u, 0x00000862u, 0x000000E2u, 0x00000461u, 0x00000047u, 0x00000862u, 0x000000E2u, 0x00000461u,
    0x0000001Fu, 0x00108421u, 0x0000001Fu, 0x00108421u, 0x0000001Fu, 0x00108421u, 0x0000001Fu, 0x00108421u,
    0x00018421u, 0x0000002Fu, 0x00010843u, 0x000001E8u, 0x00018842u, 0x000001E1u, 0x00008423u, 0x0000010Fu,
    0x00000463u, 0x000000C7u, 0x00000C62u, 0x000000E3u, 0x00000863u, 0x000000E6u, 0x00000C61u, 0x00000067u,
    0x00000866u, 0x000010E2u, 0x00000CC2u, 0x000008E1u, 0x000008C3u, 0x000008E4u, 0x00001862u, 0x000004E2u,
    0x00000847u, 0x000010E4u, 0x00001C42u, 0x000004E1u, 0x00000847u, 0x000010E4u, 0x00001C42u, 0x000004E1u,
    0x000000E5u, 0x00000C23u, 0x000000A7u, 0x00000C43u, 0x000000E5u, 0x00000C23u, 0x000000A7u, 0x00000C43u,
    0x00001C21u, 0x00000427u, 0x00001087u, 0x00001C84u, 0x00001C84u, 0x00001C21u, 0x00000427u, 0x00001087u,
    0x00001861u, 0x00000466u, 0x000010C3u, 0x00000CC4u, 0x00000CC4u, 0x00001861u, 0x00000466u, 0x000010C3u,
    0x000008E2u, 0x000008E2u, 0x000008E2u, 0x000008E2u, 0x000008E2u, 0x000008E2u, 0x000008E2u, 0x000008E2u,
    0x00008461u, 0x0000008Fu, 0x00010C42u, 0x000001E2u, 0x00010862u, 0x000001E4u, 0x00008C21u, 0x0000004Fu,
    0x00001843u, 0x000004E4u, 0x00001843u, 0x000004E4u, 0x00000C46u, 0x000010E1u, 0x00000C46u, 0x000010E1u,
    0x00008423u, 0x0000010Fu, 0x00018842u, 0x000001E1u, 0x00010843u, 0x000001E8u, 0x00018421u, 0x0000002Fu
);

fn blokus_random_rollout(board_idx: u32, start_player: i32) -> f32 {
    let state_row_idx = board_idx * 420u + 400u;
    
    var cur_player = start_player;
    var consecutive_passes = 0;
    
    var p1_pieces = u32(boards[state_row_idx]);
    var p2_pieces = u32(boards[state_row_idx + 1u]);
    var p3_pieces = u32(boards[state_row_idx + 2u]);
    var p4_pieces = u32(boards[state_row_idx + 3u]);
    var first_move_flags = u32(boards[state_row_idx + 4u]);
    
    for (var turn = 0; turn < 100; turn++) {
        if (consecutive_passes >= 4) { break; }
        
        var my_pieces: u32;
        if (cur_player == 1) { my_pieces = p1_pieces; }
        else if (cur_player == 2) { my_pieces = p2_pieces; }
        else if (cur_player == 3) { my_pieces = p3_pieces; }
        else { my_pieces = p4_pieces; }
        
        if (my_pieces == 0u) {
            consecutive_passes++;
            cur_player = (cur_player % 4) + 1;
            continue;
        }
        
        var move_found = false;
        for (var attempt = 0; attempt < 20; attempt++) {
            let p_idx = rand_range(0u, 21u);
            if ((my_pieces & (1u << p_idx)) == 0u) { continue; }
            
            let pos_x = i32(rand_range(0u, 20u));
            let pos_y = i32(rand_range(0u, 20u));
            
            let start_var = rand_range(0u, 8u);
            for (var v = 0u; v < 8u; v++) {
                let var_idx = (start_var + v) % 8u;
                let piece_mask = BLOKUS_PIECES[p_idx * 8u + var_idx];
                
                var valid = true;
                var touches_corner = false;
                
                for (var i = 0u; i < 25u; i++) {
                    if ((piece_mask & (1u << i)) != 0u) {
                        let r = i32(i) / 5;
                        let c = i32(i) % 5;
                        let bx = pos_x + c;
                        let by = pos_y + r;
                        
                        if (bx >= 20 || by >= 20) { valid = false; break; }
                        
                        let cell = get_cell(board_idx, bx, by);
                        if (cell != 0) { valid = false; break; }
                        
                        if (get_cell(board_idx, bx + 1, by) == cur_player) { valid = false; break; }
                        if (get_cell(board_idx, bx - 1, by) == cur_player) { valid = false; break; }
                        if (get_cell(board_idx, bx, by + 1) == cur_player) { valid = false; break; }
                        if (get_cell(board_idx, bx, by - 1) == cur_player) { valid = false; break; }
                        
                        if (get_cell(board_idx, bx + 1, by + 1) == cur_player) { touches_corner = true; }
                        if (get_cell(board_idx, bx - 1, by - 1) == cur_player) { touches_corner = true; }
                        if (get_cell(board_idx, bx + 1, by - 1) == cur_player) { touches_corner = true; }
                        if (get_cell(board_idx, bx - 1, by + 1) == cur_player) { touches_corner = true; }
                        
                        let p_idx_0 = (cur_player - 1);
                        if (((first_move_flags >> u32(p_idx_0)) & 1u) != 0u) {
                            if (cur_player == 1 && bx == 0 && by == 0) { touches_corner = true; }
                            else if (cur_player == 2 && bx == 19 && by == 0) { touches_corner = true; }
                            else if (cur_player == 3 && bx == 19 && by == 19) { touches_corner = true; }
                            else if (cur_player == 4 && bx == 0 && by == 19) { touches_corner = true; }
                        }
                    }
                }
                
                if (valid && touches_corner) {
                    for (var i = 0u; i < 25u; i++) {
                        if ((piece_mask & (1u << i)) != 0u) {
                            let r = i32(i) / 5;
                            let c = i32(i) % 5;
                            let bx = pos_x + c;
                            let by = pos_y + r;
                            let idx = board_idx * 420u + u32(by) * 20u + u32(bx);
                            boards[idx] = cur_player;
                        }
                    }
                    
                    if (cur_player == 1) { p1_pieces &= ~(1u << p_idx); }
                    else if (cur_player == 2) { p2_pieces &= ~(1u << p_idx); }
                    else if (cur_player == 3) { p3_pieces &= ~(1u << p_idx); }
                    else { p4_pieces &= ~(1u << p_idx); }
                    
                    let p_idx_0 = (cur_player - 1);
                    if (((first_move_flags >> u32(p_idx_0)) & 1u) != 0u) {
                        first_move_flags &= ~(1u << u32(p_idx_0));
                    }
                    
                    move_found = true;
                    consecutive_passes = 0;
                    break;
                }
            }
            if (move_found) { break; }
        }
        
        if (!move_found) {
            consecutive_passes++;
        }
        
        cur_player = (cur_player % 4) + 1;
    }
    
    var scores = vec4<i32>(0, 0, 0, 0);
    for (var i = 0u; i < 400u; i++) {
        let cell = boards[board_idx * 420u + i];
        if (cell >= 1 && cell <= 4) {
            scores[cell - 1] += 1;
        }
    }
    
    let my_score = scores[start_player - 1];
    var max_opp_score = -1;
    for (var i = 0; i < 4; i++) {
        if (i != (start_player - 1)) {
            max_opp_score = max(max_opp_score, scores[i]);
        }
    }
    
    if (my_score > max_opp_score) { return 4000.0; }
    else if (my_score < max_opp_score) { return -4000.0; }
    else { return 0.0; }
}

@compute @workgroup_size(64)
fn evaluate_blokus(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let current_player = 1;
    rng_state = params.seed + idx * 719393u;
    results[idx].score = blokus_random_rollout(idx, current_player);
}
"##;

const HIVE_WGSL: &str = r##"#include "common.wgsl"

// Hive Constants
const HIVE_QUEEN: i32 = 0;
const HIVE_BEETLE: i32 = 1;
const HIVE_SPIDER: i32 = 2;
const HIVE_GRASSHOPPER: i32 = 3;
const HIVE_ANT: i32 = 4;

// State offsets in the extra row
const OFF_P1_HAND: u32 = 0u;
const OFF_P2_HAND: u32 = 5u;
const OFF_TURN: u32 = 10u;
const OFF_P1_PLACED: u32 = 11u;
const OFF_P2_PLACED: u32 = 12u;
const OFF_P1_QUEEN: u32 = 13u;
const OFF_P2_QUEEN: u32 = 14u;
const OFF_P1_QUEEN_Q: u32 = 15u;
const OFF_P1_QUEEN_R: u32 = 16u;
const OFF_P2_QUEEN_Q: u32 = 17u;
const OFF_P2_QUEEN_R: u32 = 18u;

fn hive_get_stride() -> u32 {
    return params.board_width * params.board_height;
}

fn hive_get_cell(board_idx: u32, q: i32, r: i32) -> i32 {
    let w = 32;
    let h = 32;
    let offset_q = 16;
    let offset_r = 16;
    let aq = q + offset_q;
    let ar = r + offset_r;
    
    if (aq < 0 || aq >= w || ar < 0 || ar >= h) { return 0; }
    
    let idx = board_idx * hive_get_stride() + u32(ar) * params.board_width + u32(aq);
    return boards[idx];
}

fn hive_set_cell(board_idx: u32, q: i32, r: i32, val: i32) {
    let w = 32;
    let h = 32;
    let offset_q = 16;
    let offset_r = 16;
    let aq = q + offset_q;
    let ar = r + offset_r;
    
    if (aq < 0 || aq >= w || ar < 0 || ar >= h) { return; }
    
    let idx = board_idx * hive_get_stride() + u32(ar) * params.board_width + u32(aq);
    boards[idx] = val;
}

fn hive_get_state(board_idx: u32, offset: u32) -> i32 {
    let stride = hive_get_stride();
    // State is in the last row (row 32)
    let row_idx = board_idx * stride + 32u * params.board_width + offset;
    return boards[row_idx];
}

fn hive_set_state(board_idx: u32, offset: u32, val: i32) {
    let stride = hive_get_stride();
    let row_idx = board_idx * stride + 32u * params.board_width + offset;
    boards[row_idx] = val;
}

fn hive_decode(val: i32) -> vec3<i32> {
    // Use u32 bit operations; some backends reject shifting i32 by abstract-int literals.
    let u = bitcast<u32>(val);
    let count = i32((u >> 16u) & 0xFFu);
    let player = i32((u >> 8u) & 0xFFu);
    let piece_type = i32(u & 0xFFu);
    return vec3<i32>(count, player, piece_type);
}

fn hive_encode(count: i32, player: i32, piece_type: i32) -> i32 {
    let u = (u32(count) << 16u) | (u32(player) << 8u) | u32(piece_type);
    return bitcast<i32>(u);
}

fn hive_get_neighbor(q: i32, r: i32, dir: i32) -> vec2<i32> {
    // Avoid dynamic indexing into literal arrays; some backends restrict it.
    switch (dir) {
        case 0: { return vec2<i32>(q + 1, r); }
        case 1: { return vec2<i32>(q - 1, r); }
        case 2: { return vec2<i32>(q, r + 1); }
        case 3: { return vec2<i32>(q, r - 1); }
        case 4: { return vec2<i32>(q + 1, r - 1); }
        case 5: { return vec2<i32>(q - 1, r + 1); }
        default: { return vec2<i32>(q, r); }
    }
}

fn hive_check_win(board_idx: u32) -> f32 {
    var p1_queen_surrounded = false;
    var p2_queen_surrounded = false;
    
    let p1_q = hive_get_state(board_idx, OFF_P1_QUEEN_Q);
    let p1_r = hive_get_state(board_idx, OFF_P1_QUEEN_R);
    let p2_q = hive_get_state(board_idx, OFF_P2_QUEEN_Q);
    let p2_r = hive_get_state(board_idx, OFF_P2_QUEEN_R);
    
    if (p1_q != -100) {
        var neighbor_count = 0;
        for (var i = 0; i < 6; i++) {
            let n = hive_get_neighbor(p1_q - 16, p1_r - 16, i);
            let n_val = hive_get_cell(board_idx, n.x, n.y);
            if (n_val != 0) { neighbor_count++; }
        }
        if (neighbor_count == 6) { p1_queen_surrounded = true; }
    }
    
    if (p2_q != -100) {
        var neighbor_count = 0;
        for (var i = 0; i < 6; i++) {
            let n = hive_get_neighbor(p2_q - 16, p2_r - 16, i);
            let n_val = hive_get_cell(board_idx, n.x, n.y);
            if (n_val != 0) { neighbor_count++; }
        }
        if (neighbor_count == 6) { p2_queen_surrounded = true; }
    }
    
    if (p1_queen_surrounded && p2_queen_surrounded) { return 0.0; }
    if (p1_queen_surrounded) { return -1.0; }
    if (p2_queen_surrounded) { return 1.0; }
    
    return 2.0;
}

fn hive_can_slide(board_idx: u32, from_q: i32, from_r: i32, to_q: i32, to_r: i32) -> bool {
    var occupied_count = 0;
    
    for (var i = 0; i < 6; i++) {
        let n = hive_get_neighbor(from_q, from_r, i);
        let nq = n.x;
        let nr = n.y;
        
        let dq = nq - to_q;
        let dr = nr - to_r;
        
        var is_neighbor_of_to = false;
        if ((dq == 1 && dr == 0) || (dq == -1 && dr == 0) || 
            (dq == 0 && dr == 1) || (dq == 0 && dr == -1) || 
            (dq == 1 && dr == -1) || (dq == -1 && dr == 1)) {
            is_neighbor_of_to = true;
        }
        
        if (is_neighbor_of_to) {
            if (hive_get_cell(board_idx, nq, nr) != 0) {
                occupied_count++;
            }
        }
    }
    
    return occupied_count < 2;
}

fn hive_is_connected_excluding(board_idx: u32, ex_q: i32, ex_r: i32) -> bool {
    let p1_placed = hive_get_state(board_idx, OFF_P1_PLACED);
    let p2_placed = hive_get_state(board_idx, OFF_P2_PLACED);
    let total_pieces = p1_placed + p2_placed;
    
    if (total_pieces <= 1) { return true; }
    
    var start_q = -100;
    var start_r = -100;
    
    // Optimization: Try to find a start node from neighbors of the excluded piece
    for (var i = 0; i < 6; i++) {
        let n = hive_get_neighbor(ex_q, ex_r, i);
        if (hive_get_cell(board_idx, n.x, n.y) != 0) {
            start_q = n.x;
            start_r = n.y;
            break;
        }
    }
    
    // Fallback: Scan board if no neighbor found (should be rare for connected hives)
    if (start_q == -100) {
        for (var r = 0; r < 32; r++) {
            for (var q = 0; q < 32; q++) {
                let rq = q - 16;
                let rr = r - 16;
                if (rq == ex_q && rr == ex_r) { continue; }
                
                if (hive_get_cell(board_idx, rq, rr) != 0) {
                    start_q = rq;
                    start_r = rr;
                    break;
                }
            }
            if (start_q != -100) { break; }
        }
    }
    
    if (start_q == -100) { return true; } // Should not happen if total_pieces > 1
    
    var visited: array<u32, 32>;
    var queue_q: array<i32, 32>;
    var queue_r: array<i32, 32>;
    var head = 0;
    var tail = 0;
    
    queue_q[tail] = start_q;
    queue_r[tail] = start_r;
    tail = (tail + 1) % 32;
    
    let start_idx = (start_r + 16) * 32 + (start_q + 16);
    visited[start_idx / 32] |= (1u << u32(start_idx % 32));
    
    var visited_count = 1;
    
    while (head != tail) {
        let curr_q = queue_q[head];
        let curr_r = queue_r[head];
        head = (head + 1) % 32;
        
        for (var i = 0; i < 6; i++) {
            let n = hive_get_neighbor(curr_q, curr_r, i);
            let nq = n.x;
            let nr = n.y;
            
            if (nq == ex_q && nr == ex_r) { continue; }
            
            if (hive_get_cell(board_idx, nq, nr) != 0) {
                let n_idx = (nr + 16) * 32 + (nq + 16);
                let word_idx = n_idx / 32;
                let bit_mask = (1u << u32(n_idx % 32));
                
                if ((visited[word_idx] & bit_mask) == 0u) {
                    visited[word_idx] |= bit_mask;
                    visited_count++;
                    
                    queue_q[tail] = nq;
                    queue_r[tail] = nr;
                    tail = (tail + 1) % 32;
                }
            }
        }
    }
    
    // We expect to visit total_pieces - 1 (excluding the one we removed)
    // However, if the piece we removed was part of a stack, total_pieces is still the same count of occupied cells?
    // Wait, total_pieces is count of pieces placed. But board cells might have stacks.
    // hive_is_connected_excluding checks connectivity of *stacks*.
    // If we remove a piece from a stack of height > 1, the cell is still occupied, so connectivity is trivially true.
    // But this function is called when stack_height == 1.
    // So the cell becomes empty.
    // The number of occupied cells should be (occupied_cells_before - 1).
    // But we don't track occupied_cells count, only pieces_placed.
    // pieces_placed includes pieces under stacks.
    // So visited_count comparison is tricky.
    
    // Let's re-scan to count expected pieces? No, that defeats optimization.
    // Actually, we can just check if we visited all *reachable* pieces.
    // But we don't know how many there are without scanning.
    // Optimization: We can count pieces AS WE SCAN in the fallback loop.
    // But we want to avoid the scan.
    
    // Let's stick to the scan for counting total *occupied cells* (not pieces placed) for now, 
    // but optimize the start node finding.
    // Or better: The BFS visits connected component. If there are any pieces NOT visited, then it's disconnected.
    // So we need to know the total number of occupied cells.
    
    // Revert to scanning for total count, but use the neighbor optimization for start node.
    // Scanning 1024 ints is fast enough if we don't do complex logic.
    
    var occupied_cells = 0;
    for (var r = 0; r < 32; r++) {
        for (var q = 0; q < 32; q++) {
            let rq = q - 16;
            let rr = r - 16;
            if (rq == ex_q && rr == ex_r) { continue; }
            if (hive_get_cell(board_idx, rq, rr) != 0) { occupied_cells++; }
        }
    }
    
    return visited_count == occupied_cells;
}

fn hive_try_place_random(board_idx: u32, player: i32) -> bool {
    let hand_offset = select(OFF_P2_HAND, OFF_P1_HAND, player == 1);
    var available_types = 0u;
    var type_map = array<i32, 5>(0, 0, 0, 0, 0);
    var count = 0;
    
    let placed_offset = select(OFF_P2_PLACED, OFF_P1_PLACED, player == 1);
    let queen_offset = select(OFF_P2_QUEEN, OFF_P1_QUEEN, player == 1);
    
    let pieces_placed = hive_get_state(board_idx, placed_offset);
    let queen_placed = hive_get_state(board_idx, queen_offset);
    
    if (pieces_placed == 3 && queen_placed == 0) {
        if (hive_get_state(board_idx, hand_offset + u32(HIVE_QUEEN)) > 0) {
            type_map[0] = HIVE_QUEEN;
            count = 1;
        } else { return false; }
    } else {
        for (var i = 0; i < 5; i++) {
            if (hive_get_state(board_idx, hand_offset + u32(i)) > 0) {
                type_map[count] = i;
                count++;
            }
        }
    }
    
    if (count == 0) { return false; }
    
    let type_idx = rand_range(0u, u32(count));
    let piece_type = type_map[type_idx];
    let turn = hive_get_state(board_idx, OFF_TURN);
    
    if (turn == 1) {
        hive_set_cell(board_idx, 0, 0, hive_encode(1, player, piece_type));
        hive_set_state(board_idx, hand_offset + u32(piece_type), hive_get_state(board_idx, hand_offset + u32(piece_type)) - 1);
        hive_set_state(board_idx, placed_offset, pieces_placed + 1);
        if (piece_type == HIVE_QUEEN) {
            hive_set_state(board_idx, queen_offset, 1);
            hive_set_state(board_idx, select(OFF_P2_QUEEN_Q, OFF_P1_QUEEN_Q, player == 1), 0);
            hive_set_state(board_idx, select(OFF_P2_QUEEN_R, OFF_P1_QUEEN_R, player == 1), 0);
        }
        return true;
    }
    
    for (var attempt = 0; attempt < 20; attempt++) {
        let q = i32(rand_range(0u, 32u)) - 16;
        let r = i32(rand_range(0u, 32u)) - 16;
        
        if (hive_get_cell(board_idx, q, r) != 0) { continue; }
        
        var has_own_neighbor = false;
        var has_opp_neighbor = false;
        
        for (var i = 0; i < 6; i++) {
            let n = hive_get_neighbor(q, r, i);
            let val = hive_get_cell(board_idx, n.x, n.y);
            if (val != 0) {
                let decoded = hive_decode(val);
                if (decoded.y == player) { has_own_neighbor = true; }
                else { has_opp_neighbor = true; }
            }
        }
        
        if (turn == 2) {
            if (has_opp_neighbor) {
                 hive_set_cell(board_idx, q, r, hive_encode(1, player, piece_type));
                 hive_set_state(board_idx, hand_offset + u32(piece_type), hive_get_state(board_idx, hand_offset + u32(piece_type)) - 1);
                 hive_set_state(board_idx, placed_offset, pieces_placed + 1);
                 if (piece_type == HIVE_QUEEN) {
                     hive_set_state(board_idx, queen_offset, 1);
                     hive_set_state(board_idx, select(OFF_P2_QUEEN_Q, OFF_P1_QUEEN_Q, player == 1), q + 16);
                     hive_set_state(board_idx, select(OFF_P2_QUEEN_R, OFF_P1_QUEEN_R, player == 1), r + 16);
                 }
                 return true;
            }
        } else {
            if (has_own_neighbor && !has_opp_neighbor) {
                 hive_set_cell(board_idx, q, r, hive_encode(1, player, piece_type));
                 hive_set_state(board_idx, hand_offset + u32(piece_type), hive_get_state(board_idx, hand_offset + u32(piece_type)) - 1);
                 hive_set_state(board_idx, placed_offset, pieces_placed + 1);
                 if (piece_type == HIVE_QUEEN) {
                     hive_set_state(board_idx, queen_offset, 1);
                     hive_set_state(board_idx, select(OFF_P2_QUEEN_Q, OFF_P1_QUEEN_Q, player == 1), q + 16);
                     hive_set_state(board_idx, select(OFF_P2_QUEEN_R, OFF_P1_QUEEN_R, player == 1), r + 16);
                 }
                 return true;
            }
        }
    }
    return false;
}

fn hive_try_move_random(board_idx: u32, player: i32) -> bool {
    let queen_offset = select(OFF_P2_QUEEN, OFF_P1_QUEEN, player == 1);
    let queen_placed = hive_get_state(board_idx, queen_offset);
    if (queen_placed == 0) { return false; }
    
    var my_pieces_q: array<i32, 32>;
    var my_pieces_r: array<i32, 32>;
    var count = 0;
    
    for (var r = 0; r < 32; r++) {
        for (var q = 0; q < 32; q++) {
            let val = hive_get_cell(board_idx, q - 16, r - 16);
            if (val != 0) {
                let decoded = hive_decode(val);
                if (decoded.y == player) {
                    my_pieces_q[count] = q - 16;
                    my_pieces_r[count] = r - 16;
                    count++;
                    if (count >= 32) { break; }
                }
            }
        }
        if (count >= 32) { break; }
    }
    
    if (count == 0) { return false; }
    
    for (var attempt = 0; attempt < 10; attempt++) {
        let idx = rand_range(0u, u32(count));
        let q = my_pieces_q[idx];
        let r = my_pieces_r[idx];
        
        let val = hive_get_cell(board_idx, q, r);
        let decoded = hive_decode(val);
        let piece_type = decoded.z;
        let stack_height = decoded.x;
        
        if (stack_height == 1) {
            if (!hive_is_connected_excluding(board_idx, q, r)) { continue; }
        }
        
        var moved = false;
        
        if (piece_type == HIVE_QUEEN) {
            let dir = rand_range(0u, 6u);
            let n = hive_get_neighbor(q, r, i32(dir));
            if (hive_get_cell(board_idx, n.x, n.y) == 0 && hive_can_slide(board_idx, q, r, n.x, n.y)) {
                hive_set_cell(board_idx, n.x, n.y, hive_encode(1, player, piece_type));
                hive_set_cell(board_idx, q, r, 0);
                hive_set_state(board_idx, select(OFF_P2_QUEEN_Q, OFF_P1_QUEEN_Q, player == 1), n.x + 16);
                hive_set_state(board_idx, select(OFF_P2_QUEEN_R, OFF_P1_QUEEN_R, player == 1), n.y + 16);
                moved = true;
            }
        } else if (piece_type == HIVE_BEETLE) {
            let dir = rand_range(0u, 6u);
            let n = hive_get_neighbor(q, r, i32(dir));
            let target_val = hive_get_cell(board_idx, n.x, n.y);
            let target_height = hive_decode(target_val).x;
            
            if (target_val == 0) {
                if (stack_height > 1 || hive_can_slide(board_idx, q, r, n.x, n.y)) {
                    hive_set_cell(board_idx, n.x, n.y, hive_encode(1, player, piece_type));
                    hive_set_cell(board_idx, q, r, 0);
                    moved = true;
                }
            } else {
                // Climb
                hive_set_cell(board_idx, n.x, n.y, hive_encode(target_height + 1, player, piece_type));
                hive_set_cell(board_idx, q, r, hive_encode(stack_height - 1, player, piece_type));
                moved = true;
            }
        } else if (piece_type == HIVE_SPIDER) {
             var curr_q = q;
             var curr_r = r;
             var visited_q = array<i32, 4>(q, 0, 0, 0);
             var visited_r = array<i32, 4>(r, 0, 0, 0);
             var valid_path = true;
             for (var step = 0; step < 3; step++) {
                 var found_step = false;
                 let start_dir = rand_range(0u, 6u);
                 for (var d = 0u; d < 6u; d++) {
                     let dir = (start_dir + d) % 6u;
                     let n = hive_get_neighbor(curr_q, curr_r, i32(dir));
                     if (hive_get_cell(board_idx, n.x, n.y) != 0) { continue; }
                     if (!hive_can_slide(board_idx, curr_q, curr_r, n.x, n.y)) { continue; }
                     var is_visited = false;
                     for (var v = 0; v <= step; v++) {
                         if (visited_q[v] == n.x && visited_r[v] == n.y) { is_visited = true; break; }
                     }
                     if (is_visited) { continue; }
                     curr_q = n.x;
                     curr_r = n.y;
                     visited_q[step + 1] = n.x;
                     visited_r[step + 1] = n.y;
                     found_step = true;
                     break;
                 }
                 if (!found_step) { valid_path = false; break; }
             }
             if (valid_path) {
                 hive_set_cell(board_idx, curr_q, curr_r, hive_encode(1, player, piece_type));
                 hive_set_cell(board_idx, q, r, 0);
                 moved = true;
             }
        } else if (piece_type == HIVE_GRASSHOPPER) {
            let dir = rand_range(0u, 6u);
            let n = hive_get_neighbor(q, r, i32(dir));
            if (hive_get_cell(board_idx, n.x, n.y) != 0) {
                var jump_q = n.x;
                var jump_r = n.y;
                while (hive_get_cell(board_idx, jump_q, jump_r) != 0) {
                    let next = hive_get_neighbor(jump_q, jump_r, i32(dir));
                    jump_q = next.x;
                    jump_r = next.y;
                }
                hive_set_cell(board_idx, jump_q, jump_r, hive_encode(1, player, piece_type));
                hive_set_cell(board_idx, q, r, 0);
                moved = true;
            }
        } else if (piece_type == HIVE_ANT) {
            var curr_q = q;
            var curr_r = r;
            let steps = rand_range(1u, 10u);
            for (var s = 0u; s < steps; s++) {
                 let start_dir = rand_range(0u, 6u);
                 for (var d = 0u; d < 6u; d++) {
                     let dir = (start_dir + d) % 6u;
                     let n = hive_get_neighbor(curr_q, curr_r, i32(dir));
                     if (hive_get_cell(board_idx, n.x, n.y) == 0 && hive_can_slide(board_idx, curr_q, curr_r, n.x, n.y)) {
                         curr_q = n.x;
                         curr_r = n.y;
                         break;
                     }
                 }
            }
            if (curr_q != q || curr_r != r) {
                hive_set_cell(board_idx, curr_q, curr_r, hive_encode(1, player, piece_type));
                hive_set_cell(board_idx, q, r, 0);
                moved = true;
            }
        }
        
        if (moved) { return true; }
    }
    return false;
}

fn hive_random_rollout(board_idx: u32, start_player: i32) -> f32 {
    var cur_player = start_player;
    
    for (var step = 0; step < 60; step++) {
        let status = hive_check_win(board_idx);
        if (status != 2.0) {
            if (start_player == 1) { return status * 4000.0; }
            else { return -status * 4000.0; }
        }
        
        var moved = false;
        let queen_offset = select(OFF_P2_QUEEN, OFF_P1_QUEEN, cur_player == 1);
        let placed_offset = select(OFF_P2_PLACED, OFF_P1_PLACED, cur_player == 1);
        
        let queen_placed = hive_get_state(board_idx, queen_offset);
        let pieces_placed = hive_get_state(board_idx, placed_offset);
        
        var must_place_queen = (pieces_placed == 3 && queen_placed == 0);
        
        if (must_place_queen) {
            moved = hive_try_place_random(board_idx, cur_player);
        } else {
            if (rand() < 0.5) {
                moved = hive_try_place_random(board_idx, cur_player);
                if (!moved) { moved = hive_try_move_random(board_idx, cur_player); }
            } else {
                moved = hive_try_move_random(board_idx, cur_player);
                if (!moved) { moved = hive_try_place_random(board_idx, cur_player); }
            }
        }
        
        let turn = hive_get_state(board_idx, OFF_TURN);
        hive_set_state(board_idx, OFF_TURN, turn + 1);
        
        if (cur_player == 1) { cur_player = 2; } else { cur_player = 1; }
    }
    
    return 0.0;
}

@compute @workgroup_size(64)
fn evaluate_hive(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let current_player = 1;
    rng_state = params.seed + idx * 719393u;
    results[idx].score = hive_random_rollout(idx, current_player);
}
"##;
