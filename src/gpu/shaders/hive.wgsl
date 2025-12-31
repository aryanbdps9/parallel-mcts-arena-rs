
// BEGIN inlined common.wgsl
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
    let count = (val >> 16) & 0xFF;
    let player = (val >> 8) & 0xFF;
    let piece_type = val & 0xFF;
    return vec3<i32>(count, player, piece_type);
}

fn hive_encode(count: i32, player: i32, piece_type: i32) -> i32 {
    return (count << 16) | (player << 8) | piece_type;
}

fn hive_get_neighbor(q: i32, r: i32, dir: i32) -> vec2<i32> {
    let neighbors_q = array<i32, 6>(1, -1, 0, 0, 1, -1);
    let neighbors_r = array<i32, 6>(0, 0, 1, -1, -1, 1);
    return vec2<i32>(q + neighbors_q[dir], r + neighbors_r[dir]);
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
            let target_height = (target_val >> 16) & 0xFF;
            
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
