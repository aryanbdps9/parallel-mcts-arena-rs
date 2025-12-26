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
