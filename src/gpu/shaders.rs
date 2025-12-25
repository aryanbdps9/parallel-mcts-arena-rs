//! WGSL Compute Shaders for MCTS GPU Acceleration
//!
//! This module contains the compute shader source code for GPU-accelerated MCTS operations.

/// WGSL shader for batch PUCT (Predictor + Upper Confidence bounds applied to Trees) calculation
///
/// This shader computes PUCT scores for multiple nodes in parallel, which is the core
/// of the selection phase in MCTS. The formula implemented is:
///
/// PUCT(s,a) = Q(s,a) + C_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a) + VL(s,a))
///
/// Where:
/// - Q(s,a) = win rate for action a from state s
/// - C_puct = exploration parameter
/// - P(s,a) = prior probability of action a
/// - N(s) = visit count of parent state
/// - N(s,a) = visit count of child state
/// - VL(s,a) = virtual losses applied to prevent thread contention
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

/// WGSL shader for batch node expansion preprocessing
///
/// This shader prepares data for node expansion by computing which nodes
/// should be expanded based on depth, visit count, and tree capacity.
pub const EXPANSION_SHADER: &str = r#"
// Node metadata for expansion decision
struct ExpansionInput {
    depth: u32,           // Node depth in tree
    visits: i32,          // Current visit count
    is_leaf: u32,         // 1 if node has no children, 0 otherwise
    is_terminal: u32,     // 1 if game state is terminal, 0 otherwise
}

// Expansion decision output
struct ExpansionOutput {
    should_expand: u32,   // 1 if should expand, 0 otherwise
    expansion_priority: f32, // Priority for expansion (higher = more urgent)
    node_index: u32,      // Original index
    _padding: u32,        // Alignment padding
}

@group(0) @binding(0) var<storage, read> inputs: array<ExpansionInput>;
@group(0) @binding(1) var<storage, read_write> outputs: array<ExpansionOutput>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = num_nodes, y = max_nodes, z = current_nodes, w = seed

// Simple pseudo-random function for probabilistic expansion
fn rand(seed: u32, idx: u32) -> f32 {
    let x = seed ^ (idx * 1103515245u + 12345u);
    let y = x ^ (x >> 16u);
    let z = y * 0x85ebca6bu;
    let w = z ^ (z >> 13u);
    let v = w * 0xc2b2ae35u;
    let result = v ^ (v >> 16u);
    return f32(result) / f32(0xFFFFFFFFu);
}

@compute @workgroup_size(256)
fn compute_expansion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_nodes = params.x;
    
    if (idx >= num_nodes) {
        return;
    }
    
    let input = inputs[idx];
    var output: ExpansionOutput;
    output.node_index = idx;
    output._padding = 0u;
    
    // Check basic conditions
    let tree_capacity_available = params.z < params.y;
    let is_expandable = input.is_leaf == 1u && input.is_terminal == 0u;
    
    if (!tree_capacity_available || !is_expandable) {
        output.should_expand = 0u;
        output.expansion_priority = 0.0;
        outputs[idx] = output;
        return;
    }
    
    // Root node (depth 0) always expands
    if (input.depth == 0u) {
        output.should_expand = 1u;
        output.expansion_priority = 1000.0; // Highest priority
        outputs[idx] = output;
        return;
    }
    
    // Probabilistic expansion based on depth and visits
    let depth_factor = 1.0 / (1.0 + f32(input.depth) * 0.5);
    let visit_factor = sqrt(f32(input.visits)) / 10.0;
    let expansion_probability = min(depth_factor + visit_factor, 1.0);
    
    let random_value = rand(params.w, idx);
    
    if (random_value < expansion_probability) {
        output.should_expand = 1u;
        output.expansion_priority = expansion_probability * f32(input.visits + 1);
    } else {
        output.should_expand = 0u;
        output.expansion_priority = 0.0;
    }
    
    outputs[idx] = output;
}
"#;

/// WGSL shader for batch backpropagation preparation
///
/// This shader prepares reward calculations for backpropagation phase,
/// computing the appropriate reward values based on game outcome.
pub const BACKPROP_SHADER: &str = r#"
// Path node data for backpropagation
struct PathNode {
    player_who_moved: i32, // Player who made the move leading to this node
    winner: i32,           // Game winner (-1 for no winner/draw, 0+ for player index)
    is_draw: u32,          // 1 if game ended in draw, 0 otherwise
    _padding: u32,         // Alignment padding
}

// Backpropagation update data
struct BackpropUpdate {
    visit_delta: i32,      // How much to add to visits (always 1)
    reward: i32,           // Reward to add: 2=win, 1=draw, 0=loss
    node_index: u32,       // Index of the node to update
    _padding: u32,         // Alignment padding
}

@group(0) @binding(0) var<storage, read> paths: array<PathNode>;
@group(0) @binding(1) var<storage, read_write> updates: array<BackpropUpdate>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = num_paths

@compute @workgroup_size(256)
fn compute_backprop(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_paths = params.x;
    
    if (idx >= num_paths) {
        return;
    }
    
    let path = paths[idx];
    var update: BackpropUpdate;
    update.visit_delta = 1;
    update.node_index = idx;
    update._padding = 0u;
    
    // Calculate reward based on game outcome
    if (path.is_draw == 1u) {
        update.reward = 1; // Draw
    } else if (path.winner == path.player_who_moved) {
        update.reward = 2; // Win
    } else if (path.winner >= 0) {
        update.reward = 0; // Loss (another player won)
    } else {
        update.reward = 1; // No winner yet (shouldn't happen in terminal states)
    }
    
    updates[idx] = update;
}
"#;

/// WGSL shader for finding the maximum PUCT score (reduction)
///
/// This shader performs a parallel reduction to find the node(s) with
/// the maximum PUCT score, which is used for move selection.
pub const MAX_REDUCTION_SHADER: &str = r#"
struct PuctResult {
    puct_score: f32,
    q_value: f32,
    exploration_term: f32,
    node_index: u32,
}

@group(0) @binding(0) var<storage, read> input: array<PuctResult>;
@group(0) @binding(1) var<storage, read_write> output: array<PuctResult>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = num_elements

var<workgroup> shared_data: array<PuctResult, 256>;

@compute @workgroup_size(256)
fn reduce_max(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let num_elements = params.x;
    
    // Load data into shared memory
    if (gid < num_elements) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid].puct_score = -1e38; // Very negative value
        shared_data[tid].node_index = 0u;
    }
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            if (shared_data[tid + stride].puct_score > shared_data[tid].puct_score) {
                shared_data[tid] = shared_data[tid + stride];
            }
        }
        workgroupBarrier();
    }
    
    // Write result from first thread of each workgroup
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
"#;

/// WGSL shader for Gomoku heuristic evaluation
///
/// This shader evaluates a Gomoku board state by counting patterns (lines of stones).
/// It is used as a replacement for random rollouts in MCTS.
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

fn check_win(board_idx: u32, player: i32) -> bool {
    let w = i32(params.board_width);
    let h = i32(params.board_height);
    
    // Horizontal
    for (var y = 0; y < h; y++) {
        for (var x = 0; x <= w - 5; x++) {
            var match_len = 0;
            for (var k = 0; k < 5; k++) {
                if (get_cell(board_idx, x + k, y) == player) { match_len++; } else { break; }
            }
            if (match_len == 5) { return true; }
        }
    }
    
    // Vertical
    for (var x = 0; x < w; x++) {
        for (var y = 0; y <= h - 5; y++) {
            var match_len = 0;
            for (var k = 0; k < 5; k++) {
                if (get_cell(board_idx, x, y + k) == player) { match_len++; } else { break; }
            }
            if (match_len == 5) { return true; }
        }
    }
    
    // Diagonal (TL-BR)
    for (var y = 0; y <= h - 5; y++) {
        for (var x = 0; x <= w - 5; x++) {
            var match_len = 0;
            for (var k = 0; k < 5; k++) {
                if (get_cell(board_idx, x + k, y + k) == player) { match_len++; } else { break; }
            }
            if (match_len == 5) { return true; }
        }
    }

    // Diagonal (TR-BL)
    for (var y = 0; y <= h - 5; y++) {
        for (var x = 4; x < w; x++) {
            var match_len = 0;
            for (var k = 0; k < 5; k++) {
                if (get_cell(board_idx, x - k, y + k) == player) { match_len++; } else { break; }
            }
            if (match_len == 5) { return true; }
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

@compute @workgroup_size(64)
fn evaluate_board(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    let current_player = params.current_player;
    
    // Check if game is already over
    if (check_win(idx, current_player)) {
        results[idx].score = 4000.0; // Current player already won
        return;
    }
    if (check_win(idx, -current_player)) {
        results[idx].score = -4000.0; // Opponent already won
        return;
    }
    
    // Choose evaluation method based on flag
    if (params.use_heuristic != 0u) {
        // Heuristic evaluation based on pattern counting
        // Count threatening patterns for both players
        let player_fours = count_pattern(idx, current_player, 5);
        let player_threes = count_pattern(idx, current_player, 4);
        let player_twos = count_pattern(idx, current_player, 3);
        
        let opp_fours = count_pattern(idx, -current_player, 5);
        let opp_threes = count_pattern(idx, -current_player, 4);
        let opp_twos = count_pattern(idx, -current_player, 3);
        
        // Weight patterns by their importance
        // Four-in-a-row (one away from winning) is very valuable
        // Three-in-a-row is moderately valuable
        // Two-in-a-row is slightly valuable
        let player_score = f32(player_fours) * 100.0 + f32(player_threes) * 10.0 + f32(player_twos) * 1.0;
        let opp_score = f32(opp_fours) * 100.0 + f32(opp_threes) * 10.0 + f32(opp_twos) * 1.0;
        
        // Net score from current player's perspective
        results[idx].score = player_score - opp_score;
    } else {
        // Random rollout evaluation
        // Initialize RNG
        rng_state = params.seed + idx * 719393u;
        
        // Create a copy of the board for this simulation
        var sim_board: array<i32, 225>; // Max 15x15 board
        let board_size = params.board_width * params.board_height;
        for (var i = 0u; i < board_size; i++) {
            sim_board[i] = boards[idx * board_size + i];
        }
        
        var sim_player = current_player;
        let max_moves = i32(board_size);
        var moves_made = 0;
        
        // Random rollout
        loop {
            if (moves_made >= max_moves) { break; }
            
            // Find all empty cells
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
            var move_idx = 0u;
            
            for (var i = 0u; i < board_size; i++) {
                if (sim_board[i] == 0) {
                    if (current_empty == pick) {
                        move_idx = i;
                        break;
                    }
                    current_empty++;
                }
            }
            
            // Make move on local copy
            sim_board[move_idx] = sim_player;
            
            // Check win using local copy
            var won = false;
            let w = i32(params.board_width);
            let h = i32(params.board_height);
            
            // Quick win check on local board
            let row = i32(move_idx) / w;
            let col = i32(move_idx) % w;
            
            // Horizontal
            var count = 1;
            var x = col - 1;
            while (x >= 0 && x < w) {
                let check_idx = row * w + x;
                if (sim_board[check_idx] == sim_player) { count++; } else { break; }
                x--;
            }
            x = col + 1;
            while (x >= 0 && x < w) {
                let check_idx = row * w + x;
                if (sim_board[check_idx] == sim_player) { count++; } else { break; }
                x++;
            }
            if (count >= 5) { won = true; }
            
            if (!won) {
                // Vertical
                count = 1;
                var y = row - 1;
                while (y >= 0 && y < h) {
                    let check_idx = y * w + col;
                    if (sim_board[check_idx] == sim_player) { count++; } else { break; }
                    y--;
                }
                y = row + 1;
                while (y >= 0 && y < h) {
                    let check_idx = y * w + col;
                    if (sim_board[check_idx] == sim_player) { count++; } else { break; }
                    y++;
                }
                if (count >= 5) { won = true; }
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
                if (count >= 5) { won = true; }
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
                if (count >= 5) { won = true; }
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
