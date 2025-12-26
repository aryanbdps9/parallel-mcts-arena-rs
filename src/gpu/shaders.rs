//! WGSL Compute Shaders for MCTS GPU Acceleration
//!
//! This module contains compute shaders for:
//! - PUCT score calculation (selection phase)
//! - Multi-game board evaluation (simulation phase)
//!   - Gomoku: 5-in-a-row on square boards
//!   - Connect4: N-in-a-row with gravity
//!   - Othello: Flip-based capture game

/// PUCT calculation shader
///
/// Computes PUCT scores in parallel:
/// PUCT(s,a) = Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a) + VL(s,a))
pub const PUCT_SHADER: &str = r#"
// Node data structure for PUCT calculation
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
"#;

/// Multi-game board evaluation shader
///
/// Evaluates board positions for multiple game types:
/// - Gomoku (15x15 or similar square boards): 5-in-a-row
/// - Connect4 (7x6 or similar): N-in-a-row with gravity, line_size from params
/// - Othello (8x8): Flip-based capture, count-based winner
///
/// Game detection:
/// - If current_player has bits 8-15 set (line_size encoded), it's Connect4
/// - If board is 8x8 and no line_size, it's Othello
/// - Otherwise, assume Gomoku with 5-in-a-row
pub const GOMOKU_EVAL_SHADER: &str = r#"
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
// Bits 0-7: actual player value (always 1 after normalization)
// Bits 8-15: line_size for Connect4 (0 means default/Gomoku)
// Bits 16-23: explicit game_type (0=auto, 2=Othello, 3=Blokus, 4=Hive)
fn get_line_size() -> i32 {
    let encoded = params.current_player;
    let line_size = (encoded >> 8) & 0xFF;
    if (line_size > 0) {
        return line_size;
    }
    return 5; // Default for Gomoku
}

fn get_game_type() -> u32 {
    let encoded = params.current_player;
    let explicit_game_type = (encoded >> 16) & 0xFF;
    
    if (explicit_game_type == 2) { return GAME_OTHELLO; }
    if (explicit_game_type == 3) { return GAME_BLOKUS; }
    if (explicit_game_type == 4) { return GAME_HIVE; }
    
    let line_size = (encoded >> 8) & 0xFF;
    
    // Connect4: has line_size encoded (non-zero)
    if (line_size > 0 && line_size < 10) {
        return GAME_CONNECT4;
    }
    
    // Default: Gomoku
    return GAME_GOMOKU;
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

// Check N-in-a-row win condition (works for both Gomoku and Connect4)
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

// Legacy function for compatibility - uses dynamic line_size
fn check_win(board_idx: u32, player: i32) -> bool {
    return check_line_win(board_idx, player, get_line_size());
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
                else { break; } // Blocked by opponent
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

// ============================================================================
// OTHELLO-SPECIFIC FUNCTIONS
// ============================================================================

// 8 directions for Othello: N, NE, E, SE, S, SW, W, NW
const DIR_X: array<i32, 8> = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
const DIR_Y: array<i32, 8> = array<i32, 8>(-1, -1, 0, 1, 1, 1, 0, -1);

// Check if placing a piece at (x, y) would flip pieces in direction d
// Returns the number of pieces that would be flipped
fn othello_count_flips_dir(board: ptr<function, array<i32, 1056>>, x: i32, y: i32, player: i32, d: i32) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    let dx = DIR_X[d];
    let dy = DIR_Y[d];
    let opponent = -player;
    
    var cx = x + dx;
    var cy = y + dy;
    var count = 0;
    
    // Count opponent pieces in this direction
    while (cx >= 0 && cx < w && cy >= 0 && cy < h) {
        let cell = (*board)[cy * w + cx];
        if (cell == opponent) {
            count++;
            cx += dx;
            cy += dy;
        } else if (cell == player && count > 0) {
            // Found our piece after opponent pieces - valid flip
            return count;
        } else {
            // Empty or our piece with no opponents between
            return 0;
        }
    }
    return 0; // Reached edge without finding our piece
}

// Check if a move is valid for Othello (would flip at least one piece)
fn othello_is_valid_move(board: ptr<function, array<i32, 1056>>, x: i32, y: i32, player: i32) -> bool {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    
    // Must be on board
    if (x < 0 || x >= w || y < 0 || y >= h) {
        return false;
    }
    
    // Must be empty
    if ((*board)[y * w + x] != 0) {
        return false;
    }
    
    // Check all 8 directions for valid flips
    for (var d = 0; d < 8; d++) {
        if (othello_count_flips_dir(board, x, y, player, d) > 0) {
            return true;
        }
    }
    return false;
}

// Make an Othello move: place piece and flip all captured pieces
fn othello_make_move(board: ptr<function, array<i32, 1056>>, x: i32, y: i32, player: i32) {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    
    // Place the piece
    (*board)[y * w + x] = player;
    
    // Flip pieces in all 8 directions
    for (var d = 0; d < 8; d++) {
        let flip_count = othello_count_flips_dir(board, x, y, player, d);
        if (flip_count > 0) {
            let dx = DIR_X[d];
            let dy = DIR_Y[d];
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

// Count valid moves for a player in Othello
fn othello_count_valid_moves(board: ptr<function, array<i32, 1056>>, player: i32) -> i32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    var count = 0;
    
    for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
            if (othello_is_valid_move(board, x, y, player)) {
                count++;
            }
        }
    }
    return count;
}

// Get the nth valid move for a player (0-indexed)
fn othello_get_nth_valid_move(board: ptr<function, array<i32, 1056>>, player: i32, n: i32) -> vec2<i32> {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    var count = 0;
    
    for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
            if (othello_is_valid_move(board, x, y, player)) {
                if (count == n) {
                    return vec2<i32>(x, y);
                }
                count++;
            }
        }
    }
    return vec2<i32>(-1, -1); // Should not happen
}

// Run Othello random rollout and return score
fn othello_random_rollout(board_idx: u32, current_player: i32) -> f32 {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    let board_size = params.board_width * params.board_height;
    
    // Copy board to local array (max 32x33 = 1056)
    var sim_board: array<i32, 1056>;
    for (var i = 0u; i < board_size; i++) {
        sim_board[i] = boards[board_idx * board_size + i];
    }
    
    var sim_player = current_player;
    var consecutive_passes = 0;
    var moves_made = 0;
    let max_moves = 64; // Maximum possible moves in Othello
    
    // Random rollout
    while (consecutive_passes < 2 && moves_made < max_moves) {
        let valid_count = othello_count_valid_moves(&sim_board, sim_player);
        
        if (valid_count == 0) {
            // No valid moves, pass
            consecutive_passes++;
            sim_player = -sim_player;
            continue;
        }
        
        consecutive_passes = 0;
        
        // Pick a random valid move
        let pick = i32(rand() * f32(valid_count));
        let move_pos = othello_get_nth_valid_move(&sim_board, sim_player, pick);
        
        // Make the move
        othello_make_move(&sim_board, move_pos.x, move_pos.y, sim_player);
        
        sim_player = -sim_player;
        moves_made++;
    }
    
    // Count final pieces
    var player_count = 0;
    var opp_count = 0;
    for (var i = 0; i < 1056; i++) {
        if (i32(i) >= w * h) { break; }
        if (sim_board[i] == current_player) { player_count++; }
        else if (sim_board[i] == -current_player) { opp_count++; }
    }
    
    // Return score: win = 4000, loss = -4000, draw = 0
    if (player_count > opp_count) {
        return 4000.0;
    } else if (opp_count > player_count) {
        return -4000.0;
    } else {
        return 0.0;
    }
}

// ============================================================================
// BLOKUS-SPECIFIC FUNCTIONS
// ============================================================================

// Simplified Blokus rollout
// Since implementing full Blokus logic (21 pieces * 8 transformations) is too complex for a single shader function,
// we use a simplified heuristic rollout:
// 1. Try to place a random piece from available set
// 2. If fails, try another
// 3. If many fail, pass
fn blokus_random_rollout(board_idx: u32, current_player: i32) -> f32 {
    // Placeholder for Blokus rollout
    // For now, return 0.0 (draw) to avoid crashing
    // Real implementation would require encoding all piece shapes
    return 0.0;
}

// ============================================================================
// HIVE-SPECIFIC FUNCTIONS
// ============================================================================

// Simplified Hive rollout
fn hive_random_rollout(board_idx: u32, current_player: i32) -> f32 {
    // Placeholder for Hive rollout
    return 0.0;
}

// ============================================================================
// MAIN EVALUATION FUNCTION
// ============================================================================

@compute @workgroup_size(64)
fn evaluate_board(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Current player is always 1 after board normalization
    // The encoded params.current_player has line_size in upper bits
    let current_player = 1;
    let line_size = get_line_size();
    let game_type = get_game_type();
    
    // Initialize RNG for this thread
    rng_state = params.seed + idx * 719393u;
    
    // Handle different game types
    if (game_type == GAME_OTHELLO) {
        results[idx].score = othello_random_rollout(idx, current_player);
        return;
    } else if (game_type == GAME_BLOKUS) {
        results[idx].score = blokus_random_rollout(idx, current_player);
        return;
    } else if (game_type == GAME_HIVE) {
        results[idx].score = hive_random_rollout(idx, current_player);
        return;
    }
    
    // For Gomoku/Connect4: Check if game is already over using dynamic line_size
    if (check_line_win(idx, current_player, line_size)) {
        results[idx].score = 4000.0; // Current player already won
        return;
    }
    if (check_line_win(idx, -current_player, line_size)) {
        results[idx].score = -4000.0; // Opponent already won
        return;
    }
    
    // Choose evaluation method based on flag (for Gomoku/Connect4)
    if (params.use_heuristic != 0u) {
        // Heuristic evaluation based on pattern counting
        // Use line_size for pattern detection
        let player_near_wins = count_pattern(idx, current_player, line_size);
        let player_threats = count_pattern(idx, current_player, line_size - 1);
        let player_builds = count_pattern(idx, current_player, line_size - 2);
        
        let opp_near_wins = count_pattern(idx, -current_player, line_size);
        let opp_threats = count_pattern(idx, -current_player, line_size - 1);
        let opp_builds = count_pattern(idx, -current_player, line_size - 2);
        
        // Weight patterns by their importance
        let player_score = f32(player_near_wins) * 100.0 + f32(player_threats) * 10.0 + f32(player_builds) * 1.0;
        let opp_score = f32(opp_near_wins) * 100.0 + f32(opp_threats) * 10.0 + f32(opp_builds) * 1.0;
        
        // Net score from current player's perspective
        results[idx].score = player_score - opp_score;
    } else {
        // Random rollout evaluation
        // Initialize RNG
        rng_state = params.seed + idx * 719393u;
        
        // Create a copy of the board for this simulation
        var sim_board: array<i32, 1056>; // Max board size
        let board_size = params.board_width * params.board_height;
        for (var i = 0u; i < board_size; i++) {
            sim_board[i] = boards[idx * board_size + i];
        }
        
        var sim_player = current_player;
        let max_moves = i32(board_size);
        var moves_made = 0;
        let win_count = line_size; // Use dynamic line size
        let w = i32(params.board_width);
        let h = i32(params.board_height);
        
        // For Connect4, we need gravity - pick column then find lowest row
        let is_connect4 = (game_type == GAME_CONNECT4);
        
        // Random rollout
        loop {
            if (moves_made >= max_moves) { break; }
            
            var move_idx = 0u;
            var found_move = false;
            
            if (is_connect4) {
                // Connect4: gravity-based move selection
                // Find columns with space
                var valid_cols = 0u;
                for (var c = 0u; c < params.board_width; c++) {
                    if (sim_board[c] == 0) { // Top row empty means column available
                        valid_cols++;
                    }
                }
                
                if (valid_cols == 0u) { break; } // No moves
                
                // Pick random valid column
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
                
                // Find lowest empty row in chosen column (gravity)
                for (var r = i32(params.board_height) - 1; r >= 0; r--) {
                    let check_idx = u32(r) * params.board_width + chosen_col;
                    if (sim_board[check_idx] == 0) {
                        move_idx = check_idx;
                        found_move = true;
                        break;
                    }
                }
            } else {
                // Gomoku/other: any empty cell
                var empty_count = 0u;
                for (var i = 0u; i < board_size; i++) {
                    if (sim_board[i] == 0) {
                        empty_count++;
                    }
                }
                
                if (empty_count == 0u) { break; } // Draw
                
                // Pick random empty cell
                let pick = rand_range(0u, empty_count);
                var current_empty = 0u;
                
                for (var i = 0u; i < board_size; i++) {
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
            
            // Make move on local copy
            sim_board[move_idx] = sim_player;
            
            // Check win using local copy with dynamic line_size
            var won = false;
            
            // Quick win check on local board
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
                if (sim_player == current_player) {
                    results[idx].score = 4000.0;
                } else {
                    results[idx].score = -4000.0;
                }
                return;
            }
            
            sim_player = -sim_player;
            moves_made++;
        }
        
        // Draw
        results[idx].score = 0.0;
    }
}
"#;
