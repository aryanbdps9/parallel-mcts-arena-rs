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
