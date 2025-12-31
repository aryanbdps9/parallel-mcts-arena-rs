// END inlined RNG

// BEGIN inlined get_cell from common.wgsl
fn get_cell(board_idx: u32, x: i32, y: i32) -> i32 {
    if (x < 0 || x >= i32(params.board_width) || y < 0 || y >= i32(params.board_height)) {
        return 0; // Out of bounds treated as empty
    }
    let idx = board_idx * params.board_width * params.board_height + u32(y) * params.board_width + u32(x);
    return boards[idx];
}
// END inlined get_cell
// END inlined common.wgsl

// BEGIN inlined RNG from common.wgsl
var<private> rng_state: u32;

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand() -> f32 {
    rng_state ^= rng_state << 13u;
    rng_state ^= rng_state >> 17u;
    rng_state ^= rng_state << 5u;
    rng_state = pcg_hash(rng_state);
    return f32(rng_state) / 4294967296.0;
}

fn rand_range(min: u32, max: u32) -> u32 {
    return min + u32(rand() * f32(max - min));
}
// END inlined RNG

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
// END inlined common.wgsl

const BLOKUS_PIECES: array<u32, 168> = array<u32, 168>(
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
