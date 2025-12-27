#![no_std]
#![feature(asm_experimental_arch)]

use spirv_std::glam::{UVec3, UVec4};
use spirv_std::spirv;

// Avoid core::cmp::{min,max} in shaders: they pull in Ordering (repr(i8)),
// which requires Int8 capability in SPIR-V.

// NOTE: This crate is compiled to SPIR-V via rust-gpu and then translated to WGSL at build time.
// The runtime consumes the generated WGSL to avoid requiring SPIR-V passthrough support.

#[derive(Copy, Clone)]
#[repr(C)]
pub struct SimulationParams {
    pub board_width: u32,
    pub board_height: u32,
    pub current_player: i32,
    pub use_heuristic: u32,
    pub seed: u32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct SimulationResult {
    pub score: f32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct NodeData {
    pub visits: i32,
    pub wins: i32,
    pub virtual_losses: i32,
    pub parent_visits: i32,
    pub prior_prob: f32,
    pub exploration: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct PuctResult {
    pub puct_score: f32,
    pub q_value: f32,
    pub exploration_term: f32,
    pub node_index: u32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct PuctParams {
    // Pack params into a single 16-byte vector to satisfy Uniform std140-style alignment rules.
    // x = num_elements, y/z/w reserved.
    pub packed: UVec4,
}

#[inline(always)]
fn load_i32(slice: &[i32], idx: usize) -> i32 {
    if idx < slice.len() {
        slice[idx]
    } else {
        0
    }
}

#[inline(always)]
fn store_i32(slice: &mut [i32], idx: usize, val: i32) {
    if idx < slice.len() {
        slice[idx] = val;
    }
}

// Game type constants
const GAME_GOMOKU: u32 = 0;
const GAME_CONNECT4: u32 = 1;
const GAME_OTHELLO: u32 = 2;
const GAME_BLOKUS: u32 = 3;
const GAME_HIVE: u32 = 4;

// PCG Random Number Generator
struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn pcg_hash(&mut self) -> u32 {
        let state = self.state.wrapping_mul(747796405).wrapping_add(2891336453);
        self.state = state;
        let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
        (word >> 22) ^ word
    }

    fn rand(&mut self) -> f32 {
        let hash = self.pcg_hash();
        hash as f32 / 4294967296.0
    }

    fn rand_range(&mut self, min: u32, max: u32) -> u32 {
        min + (self.rand() * (max - min) as f32) as u32
    }
}

fn get_line_size(params: &SimulationParams) -> i32 {
    let encoded = params.current_player;
    let line_size = (encoded >> 8) & 0xFF;
    if line_size > 0 {
        line_size
    } else {
        5 // Default for Gomoku
    }
}

fn get_cell(boards: &[i32], params: &SimulationParams, board_idx: u32, x: i32, y: i32) -> i32 {
    let w = params.board_width as i32;
    let h = params.board_height as i32;

    if x < 0 || x >= w || y < 0 || y >= h {
        return 0;
    }

    let board_size = (params.board_width * params.board_height) as u32;
    let idx = board_idx
        .wrapping_mul(board_size)
        .wrapping_add((y as u32).wrapping_mul(params.board_width))
        .wrapping_add(x as u32);
    load_i32(boards, idx as usize)
}

fn set_cell(boards: &mut [i32], params: &SimulationParams, board_idx: u32, x: i32, y: i32, val: i32) {
    let w = params.board_width as i32;
    let h = params.board_height as i32;

    if x < 0 || x >= w || y < 0 || y >= h {
        return;
    }

    let board_size = (params.board_width * params.board_height) as u32;
    let idx = board_idx
        .wrapping_mul(board_size)
        .wrapping_add((y as u32).wrapping_mul(params.board_width))
        .wrapping_add(x as u32);
    store_i32(boards, idx as usize, val);
}

fn check_line_win(boards: &[i32], params: &SimulationParams, board_idx: u32, player: i32, line_size: i32) -> bool {
    let w = params.board_width as i32;
    let h = params.board_height as i32;

    // Horizontal
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x <= w - line_size {
            let mut match_len = 0;
            let mut k = 0;
            while k < line_size {
                if get_cell(boards, params, board_idx, x + k, y) == player {
                    match_len += 1;
                } else {
                    break;
                }
                k += 1;
            }
            if match_len == line_size {
                return true;
            }
            x += 1;
        }
        y += 1;
    }

    // Vertical
    let mut x = 0;
    while x < w {
        let mut y2 = 0;
        while y2 <= h - line_size {
            let mut match_len = 0;
            let mut k = 0;
            while k < line_size {
                if get_cell(boards, params, board_idx, x, y2 + k) == player {
                    match_len += 1;
                } else {
                    break;
                }
                k += 1;
            }
            if match_len == line_size {
                return true;
            }
            y2 += 1;
        }
        x += 1;
    }

    // Diagonal (TL-BR)
    let mut y3 = 0;
    while y3 <= h - line_size {
        let mut x3 = 0;
        while x3 <= w - line_size {
            let mut match_len = 0;
            let mut k = 0;
            while k < line_size {
                if get_cell(boards, params, board_idx, x3 + k, y3 + k) == player {
                    match_len += 1;
                } else {
                    break;
                }
                k += 1;
            }
            if match_len == line_size {
                return true;
            }
            x3 += 1;
        }
        y3 += 1;
    }

    // Diagonal (TR-BL)
    let mut y4 = 0;
    while y4 <= h - line_size {
        let mut x4 = line_size - 1;
        while x4 < w {
            let mut match_len = 0;
            let mut k = 0;
            while k < line_size {
                if get_cell(boards, params, board_idx, x4 - k, y4 + k) == player {
                    match_len += 1;
                } else {
                    break;
                }
                k += 1;
            }
            if match_len == line_size {
                return true;
            }
            x4 += 1;
        }
        y4 += 1;
    }

    false
}

fn count_pattern(boards: &[i32], params: &SimulationParams, board_idx: u32, player: i32, length: i32) -> i32 {
    let w = params.board_width as i32;
    let h = params.board_height as i32;
    let mut count_total = 0;

    // Horizontal
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x <= w - length {
            let mut match_count = 0;
            let mut empty_count = 0;
            let mut k = 0;
            while k < length {
                let cell = get_cell(boards, params, board_idx, x + k, y);
                if cell == player {
                    match_count += 1;
                } else if cell == 0 {
                    empty_count += 1;
                } else {
                    break;
                }
                k += 1;
            }
            if match_count > 0 && match_count + empty_count == length {
                count_total += 1;
            }
            x += 1;
        }
        y += 1;
    }

    // Vertical
    let mut x2 = 0;
    while x2 < w {
        let mut y2 = 0;
        while y2 <= h - length {
            let mut match_count = 0;
            let mut empty_count = 0;
            let mut k = 0;
            while k < length {
                let cell = get_cell(boards, params, board_idx, x2, y2 + k);
                if cell == player {
                    match_count += 1;
                } else if cell == 0 {
                    empty_count += 1;
                } else {
                    break;
                }
                k += 1;
            }
            if match_count > 0 && match_count + empty_count == length {
                count_total += 1;
            }
            y2 += 1;
        }
        x2 += 1;
    }

    // Diagonal (TL-BR)
    let mut y3 = 0;
    while y3 <= h - length {
        let mut x3 = 0;
        while x3 <= w - length {
            let mut match_count = 0;
            let mut empty_count = 0;
            let mut k = 0;
            while k < length {
                let cell = get_cell(boards, params, board_idx, x3 + k, y3 + k);
                if cell == player {
                    match_count += 1;
                } else if cell == 0 {
                    empty_count += 1;
                } else {
                    break;
                }
                k += 1;
            }
            if match_count > 0 && match_count + empty_count == length {
                count_total += 1;
            }
            x3 += 1;
        }
        y3 += 1;
    }

    // Diagonal (TR-BL)
    let mut y4 = 0;
    while y4 <= h - length {
        let mut x4 = length - 1;
        while x4 < w {
            let mut match_count = 0;
            let mut empty_count = 0;
            let mut k = 0;
            while k < length {
                let cell = get_cell(boards, params, board_idx, x4 - k, y4 + k);
                if cell == player {
                    match_count += 1;
                } else if cell == 0 {
                    empty_count += 1;
                } else {
                    break;
                }
                k += 1;
            }
            if match_count > 0 && match_count + empty_count == length {
                count_total += 1;
            }
            x4 += 1;
        }
        y4 += 1;
    }

    count_total
}

fn gomoku_random_rollout(
    boards: &mut [i32],
    params: &SimulationParams,
    idx: u32,
    current_player: i32,
    line_size: i32,
    game_type: u32,
    last_move_idx: i32,
) -> f32 {
    // Mirror the main-branch WGSL approach: copy the board into a local array and
    // do the rollout entirely on that local array. This avoids repeated storage-buffer
    // accesses (very slow on some DX12 GPUs).
    let mut rng = Rng::new(params.seed.wrapping_add(idx.wrapping_mul(719_393)));

    let board_size = params.board_width * params.board_height;
    let safe_board_size = if board_size < 400 { board_size } else { 400 };

    let w_u = params.board_width;
    let h_u = params.board_height;
    let w = w_u as i32;
    let h = h_u as i32;
    let win_count = line_size;

    let is_connect4 = if game_type == GAME_CONNECT4 { 1u32 } else { 0u32 };

    // NOTE: Avoid `[0i32; 400]` initialization here. rust-gpu occasionally lowers
    // large array zero-inits into invalid SPIR-V (u32 0 stored into i32 slots).
    //
    let mut sim_board: [i32; 400] = unsafe { core::mem::MaybeUninit::uninit().assume_init() };
    let mut i = 0u32;
    while i < safe_board_size {
        let x_u = i % w_u;
        let y_u = i / w_u;
        // For i < safe_board_size, (x_u, y_u) is always within the board.
        sim_board[i as usize] = get_cell(boards, params, idx, x_u as i32, y_u as i32);
        i += 1;
    }

    let current_player = current_player;

    #[inline(always)]
    fn check_move_win_local(board: &[i32; 400], w: i32, h: i32, win_count: i32, move_idx: u32, player: i32) -> bool {
        if move_idx >= 400 {
            return false;
        }

        let row = (move_idx / (w as u32)) as i32;
        let col = (move_idx % (w as u32)) as i32;

        // Horizontal
        let mut count = 1;
        let mut x = col - 1;
        let mut steps = 0;
        while x >= 0 && steps < win_count - 1 {
            let idx_lin = (row * w + x) as usize;
            if idx_lin < 400 && board[idx_lin] == player {
                count += 1;
            } else {
                break;
            }
            x -= 1;
            steps += 1;
        }
        x = col + 1;
        steps = 0;
        while x < w && steps < win_count - 1 {
            let idx_lin = (row * w + x) as usize;
            if idx_lin < 400 && board[idx_lin] == player {
                count += 1;
            } else {
                break;
            }
            x += 1;
            steps += 1;
        }
        if count >= win_count {
            return true;
        }

        // Vertical
        count = 1;
        let mut y = row - 1;
        steps = 0;
        while y >= 0 && steps < win_count - 1 {
            let idx_lin = (y * w + col) as usize;
            if idx_lin < 400 && board[idx_lin] == player {
                count += 1;
            } else {
                break;
            }
            y -= 1;
            steps += 1;
        }
        y = row + 1;
        steps = 0;
        while y < h && steps < win_count - 1 {
            let idx_lin = (y * w + col) as usize;
            if idx_lin < 400 && board[idx_lin] == player {
                count += 1;
            } else {
                break;
            }
            y += 1;
            steps += 1;
        }
        if count >= win_count {
            return true;
        }

        // Diagonal TL-BR
        count = 1;
        let mut cx = col - 1;
        let mut cy = row - 1;
        steps = 0;
        while cx >= 0 && cx < w && cy >= 0 && cy < h && steps < win_count - 1 {
            let idx_lin = (cy * w + cx) as usize;
            if idx_lin < 400 && board[idx_lin] == player {
                count += 1;
            } else {
                break;
            }
            cx -= 1;
            cy -= 1;
            steps += 1;
        }
        cx = col + 1;
        cy = row + 1;
        steps = 0;
        while cx >= 0 && cx < w && cy >= 0 && cy < h && steps < win_count - 1 {
            let idx_lin = (cy * w + cx) as usize;
            if idx_lin < 400 && board[idx_lin] == player {
                count += 1;
            } else {
                break;
            }
            cx += 1;
            cy += 1;
            steps += 1;
        }
        if count >= win_count {
            return true;
        }

        // Diagonal TR-BL
        count = 1;
        cx = col + 1;
        cy = row - 1;
        steps = 0;
        while cx >= 0 && cx < w && cy >= 0 && cy < h && steps < win_count - 1 {
            let idx_lin = (cy * w + cx) as usize;
            if idx_lin < 400 && board[idx_lin] == player {
                count += 1;
            } else {
                break;
            }
            cx += 1;
            cy -= 1;
            steps += 1;
        }
        cx = col - 1;
        cy = row + 1;
        steps = 0;
        while cx >= 0 && cx < w && cy >= 0 && cy < h && steps < win_count - 1 {
            let idx_lin = (cy * w + cx) as usize;
            if idx_lin < 400 && board[idx_lin] == player {
                count += 1;
            } else {
                break;
            }
            cx -= 1;
            cy += 1;
            steps += 1;
        }
        count >= win_count
    }

    // Mirror CPU behavior: terminal detection is based on the last move only.
    // This avoids full-board scans and matches GomokuState::get_winner().
    if last_move_idx >= 0 {
        let lm = last_move_idx as u32;
        if lm < safe_board_size {
            let p = sim_board[lm as usize];
            if p != 0 {
                if check_move_win_local(&sim_board, w, h, win_count, lm, p) {
                    return if p == current_player { 4000.0 } else { -4000.0 };
                }
            }
        }
    } else {
        // Fallback: if we don't have last_move_idx, do a rare full-board scan.
        // This should mostly only happen for the initial empty board.
        #[inline(always)]
        fn sim_get(board: &[i32; 400], w: i32, h: i32, x: i32, y: i32) -> i32 {
            if x < 0 || x >= w || y < 0 || y >= h {
                return 0;
            }
            let idx = (y * w + x) as usize;
            if idx < 400 { board[idx] } else { 0 }
        }

        #[inline(always)]
        fn check_line_win_local(board: &[i32; 400], w: i32, h: i32, player: i32, line_size: i32) -> bool {
            // Horizontal
            let mut y = 0;
            while y < h {
                let mut x = 0;
                while x <= w - line_size {
                    let mut k = 0;
                    while k < line_size {
                        if sim_get(board, w, h, x + k, y) != player {
                            break;
                        }
                        k += 1;
                    }
                    if k == line_size {
                        return true;
                    }
                    x += 1;
                }
                y += 1;
            }

            // Vertical
            let mut x = 0;
            while x < w {
                let mut y2 = 0;
                while y2 <= h - line_size {
                    let mut k = 0;
                    while k < line_size {
                        if sim_get(board, w, h, x, y2 + k) != player {
                            break;
                        }
                        k += 1;
                    }
                    if k == line_size {
                        return true;
                    }
                    y2 += 1;
                }
                x += 1;
            }

            // Diagonal (TL-BR)
            let mut y3 = 0;
            while y3 <= h - line_size {
                let mut x3 = 0;
                while x3 <= w - line_size {
                    let mut k = 0;
                    while k < line_size {
                        if sim_get(board, w, h, x3 + k, y3 + k) != player {
                            break;
                        }
                        k += 1;
                    }
                    if k == line_size {
                        return true;
                    }
                    x3 += 1;
                }
                y3 += 1;
            }

            // Diagonal (TR-BL)
            let mut y4 = 0;
            while y4 <= h - line_size {
                let mut x4 = line_size - 1;
                while x4 < w {
                    let mut k = 0;
                    while k < line_size {
                        if sim_get(board, w, h, x4 - k, y4 + k) != player {
                            break;
                        }
                        k += 1;
                    }
                    if k == line_size {
                        return true;
                    }
                    x4 += 1;
                }
                y4 += 1;
            }

            false
        }

        if check_line_win_local(&sim_board, w, h, current_player, line_size) {
            return 4000.0;
        }
        if check_line_win_local(&sim_board, w, h, -current_player, line_size) {
            return -4000.0;
        }
    }

    // For Gomoku-style rollouts, build a compact list of empty cells once and
    // then pick via swap-remove. This avoids scanning the entire board every ply.
    let mut empty_positions: [u32; 400] = unsafe { core::mem::MaybeUninit::uninit().assume_init() };
    let mut empty_count = 0u32;
    if is_connect4 == 0 {
        let mut j = 0u32;
        while j < safe_board_size {
            if sim_board[j as usize] == 0 {
                empty_positions[empty_count as usize] = j;
                empty_count += 1;
            }
            j += 1;
        }
    }

    let mut sim_player = current_player;
    let max_moves = safe_board_size as i32;
    let mut moves_made = 0;

    loop {
        if moves_made >= max_moves {
            break;
        }

        let mut move_idx = 0u32;
        let mut found_move = 0u32;

        if is_connect4 != 0 {
            // Choose a random valid column based on the top cell.
            let mut valid_cols = 0u32;
            let mut c = 0u32;
            while c < w_u {
                if sim_board[c as usize] == 0 {
                    valid_cols += 1;
                }
                c += 1;
            }

            if valid_cols == 0 {
                break;
            }

            let pick = rng.rand_range(0, valid_cols);
            let mut current_valid = 0u32;
            let mut chosen_col = 0u32;

            let mut c2 = 0u32;
            while c2 < w_u {
                if sim_board[c2 as usize] == 0 {
                    if current_valid == pick {
                        chosen_col = c2;
                        break;
                    }
                    current_valid += 1;
                }
                c2 += 1;
            }

            let mut r = h_u as i32 - 1;
            while r >= 0 {
                let idx_lin = (r as u32) * w_u + chosen_col;
                if idx_lin < 400 && sim_board[idx_lin as usize] == 0 {
                    move_idx = idx_lin;
                    found_move = 1;
                    break;
                }
                r -= 1;
            }
        } else {
            // Choose a random remaining empty cell and remove it from the list.
            if empty_count == 0 {
                break;
            }
            let pick = rng.rand_range(0, empty_count);
            move_idx = empty_positions[pick as usize];
            found_move = 1;

            // swap-remove
            empty_count -= 1;
            empty_positions[pick as usize] = empty_positions[empty_count as usize];
        }

        if found_move == 0 {
            break;
        }

        if move_idx >= 400 {
            break;
        }

        sim_board[move_idx as usize] = sim_player;

        if check_move_win_local(&sim_board, w, h, win_count, move_idx, sim_player) {
            return if sim_player == current_player { 4000.0 } else { -4000.0 };
        }

        sim_player = -sim_player;
        moves_made += 1;
    }

    0.0
}

fn evaluate_grid_game_common(
    boards: &mut [i32],
    results: &mut [SimulationResult],
    params: &SimulationParams,
    idx: u32,
    game_type: u32,
    last_move_idx: i32,
) {
    if params.board_width > 32 || params.board_height > 32 {
        return;
    }

    let current_player = 1;
    let line_size = get_line_size(params);

    // For non-heuristic mode we do terminal detection inside the local-board rollout
    // to avoid expensive global-buffer scans.
    if params.use_heuristic != 0 {
        if check_line_win(boards, params, idx, current_player, line_size) {
            results[idx as usize].score = 4000.0;
            return;
        }

        if check_line_win(boards, params, idx, -current_player, line_size) {
            results[idx as usize].score = -4000.0;
            return;
        }
    }

    if params.use_heuristic != 0 {
        let player_near_wins = count_pattern(boards, params, idx, current_player, line_size);
        let player_threats = count_pattern(boards, params, idx, current_player, line_size - 1);
        let player_builds = count_pattern(boards, params, idx, current_player, line_size - 2);

        let opp_near_wins = count_pattern(boards, params, idx, -current_player, line_size);
        let opp_threats = count_pattern(boards, params, idx, -current_player, line_size - 1);
        let opp_builds = count_pattern(boards, params, idx, -current_player, line_size - 2);

        let player_score = (player_near_wins as f32) * 100.0
            + (player_threats as f32) * 10.0
            + (player_builds as f32) * 1.0;
        let opp_score = (opp_near_wins as f32) * 100.0
            + (opp_threats as f32) * 10.0
            + (opp_builds as f32) * 1.0;
        results[idx as usize].score = player_score - opp_score;
    } else {
        results[idx as usize].score = gomoku_random_rollout(boards, params, idx, current_player, line_size, game_type, last_move_idx);
    }
}

// ----- Othello -----

fn othello_dir(d: i32) -> (i32, i32) {
    match d {
        0 => (0, -1),
        1 => (1, -1),
        2 => (1, 0),
        3 => (1, 1),
        4 => (0, 1),
        5 => (-1, 1),
        6 => (-1, 0),
        7 => (-1, -1),
        _ => (0, 0),
    }
}

fn othello_count_flips_dir(board: &[i32; 64], params: &SimulationParams, x: i32, y: i32, player: i32, d: i32) -> i32 {
    let w = params.board_width as i32;
    let h = params.board_height as i32;
    let (dx, dy) = othello_dir(d);
    let opponent = -player;

    let mut cx = x + dx;
    let mut cy = y + dy;
    let mut count = 0;

    while cx >= 0 && cx < w && cy >= 0 && cy < h {
        let cell = board[(cy * w + cx) as usize];
        if cell == opponent {
            count += 1;
            cx += dx;
            cy += dy;
        } else if cell == player && count > 0 {
            return count;
        } else {
            return 0;
        }
    }
    0
}

fn othello_is_valid_move(board: &[i32; 64], params: &SimulationParams, x: i32, y: i32, player: i32) -> bool {
    let w = params.board_width as i32;
    let h = params.board_height as i32;
    if x < 0 || x >= w || y < 0 || y >= h {
        return false;
    }
    if board[(y * w + x) as usize] != 0 {
        return false;
    }
    let mut d = 0;
    while d < 8 {
        if othello_count_flips_dir(board, params, x, y, player, d) > 0 {
            return true;
        }
        d += 1;
    }
    false
}

fn othello_make_move(board: &mut [i32; 64], params: &SimulationParams, x: i32, y: i32, player: i32) {
    let w = params.board_width as i32;
    board[(y * w + x) as usize] = player;

    let mut d = 0;
    while d < 8 {
        let flip_count = othello_count_flips_dir(board, params, x, y, player, d);
        if flip_count > 0 {
            let (dx, dy) = othello_dir(d);
            let mut cx = x + dx;
            let mut cy = y + dy;
            let mut i = 0;
            while i < flip_count {
                board[(cy * w + cx) as usize] = player;
                cx += dx;
                cy += dy;
                i += 1;
            }
        }
        d += 1;
    }
}

fn othello_count_valid_moves(board: &[i32; 64], params: &SimulationParams, player: i32) -> i32 {
    let w = params.board_width as i32;
    let h = params.board_height as i32;
    let mut count = 0;
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            if othello_is_valid_move(board, params, x, y, player) {
                count += 1;
            }
            x += 1;
        }
        y += 1;
    }
    count
}

fn othello_get_nth_valid_move(board: &[i32; 64], params: &SimulationParams, player: i32, n: i32) -> (i32, i32) {
    let w = params.board_width as i32;
    let h = params.board_height as i32;
    let mut count = 0;
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            if othello_is_valid_move(board, params, x, y, player) {
                if count == n {
                    return (x, y);
                }
                count += 1;
            }
            x += 1;
        }
        y += 1;
    }
    (-1, -1)
}

fn othello_random_rollout(boards: &mut [i32], params: &SimulationParams, board_idx: u32, current_player: i32) -> f32 {
    let board_size = params.board_width * params.board_height;
    let safe_board_size = if board_size < 64 { board_size } else { 64 };
    let mut player_count = 0;
    let mut opp_count = 0;

    let mut i = 0u32;
    while i < safe_board_size {
        let idx_u = (board_idx * board_size + i) as usize;
        let cell = load_i32(boards, idx_u);
        if cell == current_player {
            player_count += 1;
        } else if cell == -current_player {
            opp_count += 1;
        }
        i += 1;
    }

    if player_count > opp_count {
        4000.0
    } else if opp_count > player_count {
        -4000.0
    } else {
        0.0
    }
}

// ----- Blokus -----

const BLOKUS_PIECES: [u32; 168] = [
    0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001,
    0x00000003, 0x00000021, 0x00000003, 0x00000021, 0x00000003, 0x00000021, 0x00000003, 0x00000021,
    0x00000007, 0x00000421, 0x00000007, 0x00000421, 0x00000007, 0x00000421, 0x00000007, 0x00000421,
    0x00000061, 0x00000023, 0x00000043, 0x00000062, 0x00000062, 0x00000061, 0x00000023, 0x00000043,
    0x0000000F, 0x00008421, 0x0000000F, 0x00008421, 0x0000000F, 0x00008421, 0x0000000F, 0x00008421,
    0x00000C21, 0x00000027, 0x00000843, 0x000000E4, 0x00000C42, 0x000000E1, 0x00000423, 0x00000087,
    0x00000063, 0x00000063, 0x00000063, 0x00000063, 0x00000063, 0x00000063, 0x00000063, 0x00000063,
    0x000000C3, 0x00000462, 0x000000C3, 0x00000462, 0x00000066, 0x00000861, 0x00000066, 0x00000861,
    0x00000047, 0x00000862, 0x000000E2, 0x00000461, 0x00000047, 0x00000862, 0x000000E2, 0x00000461,
    0x0000001F, 0x00108421, 0x0000001F, 0x00108421, 0x0000001F, 0x00108421, 0x0000001F, 0x00108421,
    0x00018421, 0x0000002F, 0x00010843, 0x000001E8, 0x00018842, 0x000001E1, 0x00008423, 0x0000010F,
    0x00000463, 0x000000C7, 0x00000C62, 0x000000E3, 0x00000863, 0x000000E6, 0x00000C61, 0x00000067,
    0x00000866, 0x000010E2, 0x00000CC2, 0x000008E1, 0x000008C3, 0x000008E4, 0x00001862, 0x000004E2,
    0x00000847, 0x000010E4, 0x00001C42, 0x000004E1, 0x00000847, 0x000010E4, 0x00001C42, 0x000004E1,
    0x000000E5, 0x00000C23, 0x000000A7, 0x00000C43, 0x000000E5, 0x00000C23, 0x000000A7, 0x00000C43,
    0x00001C21, 0x00000427, 0x00001087, 0x00001C84, 0x00001C84, 0x00001C21, 0x00000427, 0x00001087,
    0x00001861, 0x00000466, 0x000010C3, 0x00000CC4, 0x00000CC4, 0x00001861, 0x00000466, 0x000010C3,
    0x000008E2, 0x000008E2, 0x000008E2, 0x000008E2, 0x000008E2, 0x000008E2, 0x000008E2, 0x000008E2,
    0x00008461, 0x0000008F, 0x00010C42, 0x000001E2, 0x00010862, 0x000001E4, 0x00008C21, 0x0000004F,
    0x00001843, 0x000004E4, 0x00001843, 0x000004E4, 0x00000C46, 0x000010E1, 0x00000C46, 0x000010E1,
    0x00008423, 0x0000010F, 0x00018842, 0x000001E1, 0x00010843, 0x000001E8, 0x00018421, 0x0000002F,
];

fn blokus_random_rollout(boards: &mut [i32], params: &SimulationParams, board_idx: u32, start_player: i32) -> f32 {
    let state_row_idx = board_idx * 420 + 400;
    let mut cur_player = start_player;
    let mut consecutive_passes = 0;

    let mut p1_pieces = load_i32(boards, state_row_idx as usize) as u32;
    let mut p2_pieces = load_i32(boards, (state_row_idx + 1) as usize) as u32;
    let mut p3_pieces = load_i32(boards, (state_row_idx + 2) as usize) as u32;
    let mut p4_pieces = load_i32(boards, (state_row_idx + 3) as usize) as u32;
    let mut first_move_flags = load_i32(boards, (state_row_idx + 4) as usize) as u32;

    let mut rng = Rng::new(params.seed.wrapping_add(board_idx.wrapping_mul(719_393)));

    let mut turn = 0;
    while turn < 100 {
        if consecutive_passes >= 4 {
            break;
        }

        let my_pieces = match cur_player {
            1 => p1_pieces,
            2 => p2_pieces,
            3 => p3_pieces,
            _ => p4_pieces,
        };

        if my_pieces == 0 {
            consecutive_passes += 1;
            cur_player = (cur_player % 4) + 1;
            turn += 1;
            continue;
        }

        let mut move_found = 0u32;
        let mut attempt = 0;
        while attempt < 20 {
            let p_idx = rng.rand_range(0, 21);
            if (my_pieces & (1u32 << p_idx)) == 0 {
                attempt += 1;
                continue;
            }

            let pos_x = rng.rand_range(0, 20) as i32;
            let pos_y = rng.rand_range(0, 20) as i32;
            let start_var = rng.rand_range(0, 8);

            let mut v = 0u32;
            while v < 8 {
                let var_idx = (start_var + v) % 8;
                let piece_mask = BLOKUS_PIECES[(p_idx * 8 + var_idx) as usize];

                let mut valid = 1u32;
                let mut touches_corner = 0u32;

                let mut i = 0u32;
                while i < 25 {
                    if (piece_mask & (1u32 << i)) != 0 {
                        let r = (i as i32) / 5;
                        let c = (i as i32) % 5;
                        let bx = pos_x + c;
                        let by = pos_y + r;

                        if bx >= 20 || by >= 20 {
                            valid = 0;
                            break;
                        }

                        let cell = get_cell(boards, params, board_idx, bx, by);
                        if cell != 0 {
                            valid = 0;
                            break;
                        }

                        if get_cell(boards, params, board_idx, bx + 1, by) == cur_player {
                            valid = 0;
                            break;
                        }
                        if get_cell(boards, params, board_idx, bx - 1, by) == cur_player {
                            valid = 0;
                            break;
                        }
                        if get_cell(boards, params, board_idx, bx, by + 1) == cur_player {
                            valid = 0;
                            break;
                        }
                        if get_cell(boards, params, board_idx, bx, by - 1) == cur_player {
                            valid = 0;
                            break;
                        }

                        if get_cell(boards, params, board_idx, bx + 1, by + 1) == cur_player {
                            touches_corner = 1;
                        }
                        if get_cell(boards, params, board_idx, bx - 1, by - 1) == cur_player {
                            touches_corner = 1;
                        }
                        if get_cell(boards, params, board_idx, bx + 1, by - 1) == cur_player {
                            touches_corner = 1;
                        }
                        if get_cell(boards, params, board_idx, bx - 1, by + 1) == cur_player {
                            touches_corner = 1;
                        }

                        let p_idx_0 = cur_player - 1;
                        if ((first_move_flags >> (p_idx_0 as u32)) & 1) != 0 {
                            if cur_player == 1 && bx == 0 && by == 0 {
                                touches_corner = 1;
                            } else if cur_player == 2 && bx == 19 && by == 0 {
                                touches_corner = 1;
                            } else if cur_player == 3 && bx == 19 && by == 19 {
                                touches_corner = 1;
                            } else if cur_player == 4 && bx == 0 && by == 19 {
                                touches_corner = 1;
                            }
                        }
                    }
                    i += 1;
                }

                if valid != 0 && touches_corner != 0 {
                    let mut i2 = 0u32;
                    while i2 < 25 {
                        if (piece_mask & (1u32 << i2)) != 0 {
                            let r = (i2 as i32) / 5;
                            let c = (i2 as i32) % 5;
                            let bx = pos_x + c;
                            let by = pos_y + r;
                            let idx = board_idx * 420 + (by as u32) * 20 + (bx as u32);
                            store_i32(boards, idx as usize, cur_player);
                        }
                        i2 += 1;
                    }

                    match cur_player {
                        1 => p1_pieces &= !(1u32 << p_idx),
                        2 => p2_pieces &= !(1u32 << p_idx),
                        3 => p3_pieces &= !(1u32 << p_idx),
                        _ => p4_pieces &= !(1u32 << p_idx),
                    }

                    let p_idx_0 = cur_player - 1;
                    if ((first_move_flags >> (p_idx_0 as u32)) & 1) != 0 {
                        first_move_flags &= !(1u32 << (p_idx_0 as u32));
                    }

                    move_found = 1;
                    consecutive_passes = 0;
                    break;
                }

                v += 1;
            }

            if move_found != 0 {
                break;
            }
            attempt += 1;
        }

        if move_found == 0 {
            consecutive_passes += 1;
        }

        cur_player = (cur_player % 4) + 1;
        turn += 1;
    }

    let mut s1 = 0i32;
    let mut s2 = 0i32;
    let mut s3 = 0i32;
    let mut s4 = 0i32;

    let mut i = 0u32;
    while i < 400 {
        let cell = load_i32(boards, (board_idx * 420 + i) as usize);
        if cell == 1 {
            s1 += 1;
        } else if cell == 2 {
            s2 += 1;
        } else if cell == 3 {
            s3 += 1;
        } else if cell == 4 {
            s4 += 1;
        }
        i += 1;
    }

    let my_score = match start_player {
        1 => s1,
        2 => s2,
        3 => s3,
        _ => s4,
    };

    let mut max_opp_score = -1i32;
    if start_player != 1 && s1 > max_opp_score {
        max_opp_score = s1;
    }
    if start_player != 2 && s2 > max_opp_score {
        max_opp_score = s2;
    }
    if start_player != 3 && s3 > max_opp_score {
        max_opp_score = s3;
    }
    if start_player != 4 && s4 > max_opp_score {
        max_opp_score = s4;
    }

    if my_score > max_opp_score {
        4000.0
    } else if my_score < max_opp_score {
        -4000.0
    } else {
        0.0
    }
}

// ----- Hive -----

const HIVE_QUEEN: i32 = 0;
const HIVE_BEETLE: i32 = 1;
const HIVE_SPIDER: i32 = 2;
const HIVE_GRASSHOPPER: i32 = 3;
const HIVE_ANT: i32 = 4;

const OFF_P1_HAND: u32 = 0;
const OFF_P2_HAND: u32 = 5;
const OFF_TURN: u32 = 10;
const OFF_P1_PLACED: u32 = 11;
const OFF_P2_PLACED: u32 = 12;
const OFF_P1_QUEEN: u32 = 13;
const OFF_P2_QUEEN: u32 = 14;
const OFF_P1_QUEEN_Q: u32 = 15;
const OFF_P1_QUEEN_R: u32 = 16;
const OFF_P2_QUEEN_Q: u32 = 17;
const OFF_P2_QUEEN_R: u32 = 18;

fn hive_stride(params: &SimulationParams) -> u32 {
    params.board_width * params.board_height
}

fn hive_cell_index(params: &SimulationParams, board_idx: u32, q: i32, r: i32) -> usize {
    let w = 32i32;
    let h = 32i32;
    let offset_q = 16i32;
    let offset_r = 16i32;
    let aq = q + offset_q;
    let ar = r + offset_r;
    if aq < 0 || aq >= w || ar < 0 || ar >= h {
        return 0usize;
    }
    let idx = board_idx * hive_stride(params) + (ar as u32) * params.board_width + (aq as u32);
    idx as usize
}

fn hive_get_cell(boards: &[i32], params: &SimulationParams, board_idx: u32, q: i32, r: i32) -> i32 {
    let w = 32i32;
    let h = 32i32;
    let aq = q + 16;
    let ar = r + 16;
    if aq < 0 || aq >= w || ar < 0 || ar >= h {
        return 0;
    }
    let idx = hive_cell_index(params, board_idx, q, r);
    load_i32(boards, idx)
}

fn hive_set_cell(boards: &mut [i32], params: &SimulationParams, board_idx: u32, q: i32, r: i32, val: i32) {
    let w = 32i32;
    let h = 32i32;
    let aq = q + 16;
    let ar = r + 16;
    if aq < 0 || aq >= w || ar < 0 || ar >= h {
        return;
    }
    let idx = hive_cell_index(params, board_idx, q, r);
    store_i32(boards, idx, val);
}

fn hive_state_index(params: &SimulationParams, board_idx: u32, offset: u32) -> usize {
    let stride = hive_stride(params);
    (board_idx * stride + 32 * params.board_width + offset) as usize
}

fn hive_get_state(boards: &[i32], params: &SimulationParams, board_idx: u32, offset: u32) -> i32 {
    load_i32(boards, hive_state_index(params, board_idx, offset))
}

fn hive_set_state(boards: &mut [i32], params: &SimulationParams, board_idx: u32, offset: u32, val: i32) {
    let idx = hive_state_index(params, board_idx, offset);
    store_i32(boards, idx, val);
}

fn hive_decode(val: i32) -> (i32, i32, i32) {
    let u = val as u32;
    let count = ((u >> 16) & 0xFF) as i32;
    let player = ((u >> 8) & 0xFF) as i32;
    let piece_type = (u & 0xFF) as i32;
    (count, player, piece_type)
}

fn hive_encode(count: i32, player: i32, piece_type: i32) -> i32 {
    let u = ((count as u32) << 16) | ((player as u32) << 8) | (piece_type as u32);
    u as i32
}

fn hive_neighbor(q: i32, r: i32, dir: i32) -> (i32, i32) {
    match dir {
        0 => (q + 1, r),
        1 => (q - 1, r),
        2 => (q, r + 1),
        3 => (q, r - 1),
        4 => (q + 1, r - 1),
        5 => (q - 1, r + 1),
        _ => (q, r),
    }
}

fn hive_check_win(boards: &[i32], params: &SimulationParams, board_idx: u32) -> f32 {
    let mut p1_surrounded = 0u32;
    let mut p2_surrounded = 0u32;

    let p1_q = hive_get_state(boards, params, board_idx, OFF_P1_QUEEN_Q);
    let p1_r = hive_get_state(boards, params, board_idx, OFF_P1_QUEEN_R);
    let p2_q = hive_get_state(boards, params, board_idx, OFF_P2_QUEEN_Q);
    let p2_r = hive_get_state(boards, params, board_idx, OFF_P2_QUEEN_R);

    if p1_q != -100 {
        let mut neighbor_count = 0;
        let mut i = 0;
        while i < 6 {
            let (nq, nr) = hive_neighbor(p1_q - 16, p1_r - 16, i);
            if hive_get_cell(boards, params, board_idx, nq, nr) != 0 {
                neighbor_count += 1;
            }
            i += 1;
        }
        if neighbor_count == 6 {
            p1_surrounded = 1;
        }
    }

    if p2_q != -100 {
        let mut neighbor_count = 0;
        let mut i = 0;
        while i < 6 {
            let (nq, nr) = hive_neighbor(p2_q - 16, p2_r - 16, i);
            if hive_get_cell(boards, params, board_idx, nq, nr) != 0 {
                neighbor_count += 1;
            }
            i += 1;
        }
        if neighbor_count == 6 {
            p2_surrounded = 1;
        }
    }

    if p1_surrounded != 0 && p2_surrounded != 0 {
        return 0.0;
    }
    if p1_surrounded != 0 {
        return -1.0;
    }
    if p2_surrounded != 0 {
        return 1.0;
    }
    2.0
}

fn hive_can_slide(boards: &[i32], params: &SimulationParams, board_idx: u32, from_q: i32, from_r: i32, to_q: i32, to_r: i32) -> bool {
    let mut occupied_count = 0;
    let mut i = 0;
    while i < 6 {
        let (nq, nr) = hive_neighbor(from_q, from_r, i);
        let dq = nq - to_q;
        let dr = nr - to_r;
        if ((dq == 1 && dr == 0)
            || (dq == -1 && dr == 0)
            || (dq == 0 && dr == 1)
            || (dq == 0 && dr == -1)
            || (dq == 1 && dr == -1)
            || (dq == -1 && dr == 1))
            && hive_get_cell(boards, params, board_idx, nq, nr) != 0
        {
            occupied_count += 1;
        }
        i += 1;
    }
    occupied_count < 2
}

fn hive_is_connected_excluding(boards: &[i32], params: &SimulationParams, board_idx: u32, ex_q: i32, ex_r: i32) -> bool {
    let p1_placed = hive_get_state(boards, params, board_idx, OFF_P1_PLACED);
    let p2_placed = hive_get_state(boards, params, board_idx, OFF_P2_PLACED);
    let total_pieces = p1_placed + p2_placed;
    if total_pieces <= 1 {
        return true;
    }

    let mut start_q = -100;
    let mut start_r = -100;
    let mut i = 0;
    while i < 6 {
        let (nq, nr) = hive_neighbor(ex_q, ex_r, i);
        if hive_get_cell(boards, params, board_idx, nq, nr) != 0 {
            start_q = nq;
            start_r = nr;
            break;
        }
        i += 1;
    }

    if start_q == -100 {
        let mut r = 0;
        while r < 32 {
            let mut q = 0;
            while q < 32 {
                let rq = q - 16;
                let rr = r - 16;
                if rq == ex_q && rr == ex_r {
                    q += 1;
                    continue;
                }
                if hive_get_cell(boards, params, board_idx, rq, rr) != 0 {
                    start_q = rq;
                    start_r = rr;
                    break;
                }
                q += 1;
            }
            if start_q != -100 {
                break;
            }
            r += 1;
        }
    }

    if start_q == -100 {
        return true;
    }

    // Flood fill using a 1024-bit visited bitset (32 words), avoiding queue arrays.
    let mut visited = [0u32; 32];

    let start_idx = (start_r + 16) * 32 + (start_q + 16);
    let start_word = (start_idx / 32) as usize;
    let start_bit = 1u32 << ((start_idx % 32) as u32);
    visited[start_word] |= start_bit;
    let mut visited_count = 1;

    let mut changed = 1u32;
    while changed != 0 {
        changed = 0;
        let mut r = 0;
        while r < 32 {
            let mut q = 0;
            while q < 32 {
                let rq = q - 16;
                let rr = r - 16;
                if rq == ex_q && rr == ex_r {
                    q += 1;
                    continue;
                }

                if hive_get_cell(boards, params, board_idx, rq, rr) == 0 {
                    q += 1;
                    continue;
                }

                let idx = (rr + 16) * 32 + (rq + 16);
                let word_idx = (idx / 32) as usize;
                let bit_mask = 1u32 << ((idx % 32) as u32);
                if (visited[word_idx] & bit_mask) != 0 {
                    q += 1;
                    continue;
                }

                let mut adjacent = 0u32;
                let mut d = 0;
                while d < 6 {
                    let (nq, nr) = hive_neighbor(rq, rr, d);
                    if nq == ex_q && nr == ex_r {
                        d += 1;
                        continue;
                    }
                    if nq < -16 || nq >= 16 || nr < -16 || nr >= 16 {
                        d += 1;
                        continue;
                    }
                    let n_idx = (nr + 16) * 32 + (nq + 16);
                    let n_word = (n_idx / 32) as usize;
                    let n_bit = 1u32 << ((n_idx % 32) as u32);
                    if (visited[n_word] & n_bit) != 0 {
                        adjacent = 1;
                        break;
                    }
                    d += 1;
                }

                if adjacent != 0 {
                    visited[word_idx] |= bit_mask;
                    visited_count += 1;
                    changed = 1;
                }

                q += 1;
            }
            r += 1;
        }
    }

    let mut occupied_cells = 0;
    let mut r = 0;
    while r < 32 {
        let mut q = 0;
        while q < 32 {
            let rq = q - 16;
            let rr = r - 16;
            if rq == ex_q && rr == ex_r {
                q += 1;
                continue;
            }
            if hive_get_cell(boards, params, board_idx, rq, rr) != 0 {
                occupied_cells += 1;
            }
            q += 1;
        }
        r += 1;
    }

    visited_count == occupied_cells
}

fn hive_try_place_random(boards: &mut [i32], params: &SimulationParams, board_idx: u32, player: i32, rng: &mut Rng) -> bool {
    let hand_offset = if player == 1 { OFF_P1_HAND } else { OFF_P2_HAND };
    let placed_offset = if player == 1 { OFF_P1_PLACED } else { OFF_P2_PLACED };
    let queen_offset = if player == 1 { OFF_P1_QUEEN } else { OFF_P2_QUEEN };

    let pieces_placed = hive_get_state(boards, params, board_idx, placed_offset);
    let queen_placed = hive_get_state(boards, params, board_idx, queen_offset);

    let piece_type = if pieces_placed == 3 && queen_placed == 0 {
        if hive_get_state(boards, params, board_idx, hand_offset + (HIVE_QUEEN as u32)) > 0 {
            HIVE_QUEEN
        } else {
            return false;
        }
    } else {
        let mut attempt = 0;
        let mut chosen = 0i32;
        let mut found = 0u32;
        while attempt < 10 {
            let p = rng.rand_range(0, 5) as i32;
            if hive_get_state(boards, params, board_idx, hand_offset + (p as u32)) > 0 {
                chosen = p;
                found = 1;
                break;
            }
            attempt += 1;
        }
        if found == 0 {
            return false;
        }
        chosen
    };
    let turn = hive_get_state(boards, params, board_idx, OFF_TURN);

    if turn == 1 {
        hive_set_cell(boards, params, board_idx, 0, 0, hive_encode(1, player, piece_type));
        let left = hive_get_state(boards, params, board_idx, hand_offset + (piece_type as u32)) - 1;
        hive_set_state(boards, params, board_idx, hand_offset + (piece_type as u32), left);
        hive_set_state(boards, params, board_idx, placed_offset, pieces_placed + 1);
        if piece_type == HIVE_QUEEN {
            hive_set_state(boards, params, board_idx, queen_offset, 1);
            hive_set_state(boards, params, board_idx, if player == 1 { OFF_P1_QUEEN_Q } else { OFF_P2_QUEEN_Q }, 0);
            hive_set_state(boards, params, board_idx, if player == 1 { OFF_P1_QUEEN_R } else { OFF_P2_QUEEN_R }, 0);
        }
        return true;
    }

    let mut attempt = 0;
    while attempt < 20 {
        let q = rng.rand_range(0, 32) as i32 - 16;
        let r = rng.rand_range(0, 32) as i32 - 16;
        if hive_get_cell(boards, params, board_idx, q, r) != 0 {
            attempt += 1;
            continue;
        }

        let mut has_own_neighbor = 0u32;
        let mut has_opp_neighbor = 0u32;
        let mut i = 0;
        while i < 6 {
            let (nq, nr) = hive_neighbor(q, r, i);
            let val = hive_get_cell(boards, params, board_idx, nq, nr);
            if val != 0 {
                let (_, p, _) = hive_decode(val);
                if p == player {
                    has_own_neighbor = 1;
                } else {
                    has_opp_neighbor = 1;
                }
            }
            i += 1;
        }

        if turn == 2 {
            if has_opp_neighbor != 0 {
                hive_set_cell(boards, params, board_idx, q, r, hive_encode(1, player, piece_type));
                let left = hive_get_state(boards, params, board_idx, hand_offset + (piece_type as u32)) - 1;
                hive_set_state(boards, params, board_idx, hand_offset + (piece_type as u32), left);
                hive_set_state(boards, params, board_idx, placed_offset, pieces_placed + 1);
                if piece_type == HIVE_QUEEN {
                    hive_set_state(boards, params, board_idx, queen_offset, 1);
                    hive_set_state(boards, params, board_idx, if player == 1 { OFF_P1_QUEEN_Q } else { OFF_P2_QUEEN_Q }, q + 16);
                    hive_set_state(boards, params, board_idx, if player == 1 { OFF_P1_QUEEN_R } else { OFF_P2_QUEEN_R }, r + 16);
                }
                return true;
            }
        } else if has_own_neighbor != 0 && has_opp_neighbor == 0 {
            hive_set_cell(boards, params, board_idx, q, r, hive_encode(1, player, piece_type));
            let left = hive_get_state(boards, params, board_idx, hand_offset + (piece_type as u32)) - 1;
            hive_set_state(boards, params, board_idx, hand_offset + (piece_type as u32), left);
            hive_set_state(boards, params, board_idx, placed_offset, pieces_placed + 1);
            if piece_type == HIVE_QUEEN {
                hive_set_state(boards, params, board_idx, queen_offset, 1);
                hive_set_state(boards, params, board_idx, if player == 1 { OFF_P1_QUEEN_Q } else { OFF_P2_QUEEN_Q }, q + 16);
                hive_set_state(boards, params, board_idx, if player == 1 { OFF_P1_QUEEN_R } else { OFF_P2_QUEEN_R }, r + 16);
            }
            return true;
        }

        attempt += 1;
    }

    false
}

fn hive_try_move_random(boards: &mut [i32], params: &SimulationParams, board_idx: u32, player: i32, rng: &mut Rng) -> bool {
    let queen_offset = if player == 1 { OFF_P1_QUEEN } else { OFF_P2_QUEEN };
    let queen_placed = hive_get_state(boards, params, board_idx, queen_offset);
    if queen_placed == 0 {
        return false;
    }

    let mut attempt = 0;
    while attempt < 20 {
        let q0 = rng.rand_range(0, 32) as i32 - 16;
        let r0 = rng.rand_range(0, 32) as i32 - 16;
        let val = hive_get_cell(boards, params, board_idx, q0, r0);
        if val == 0 {
            attempt += 1;
            continue;
        }

        let (stack_height, p, piece_type) = hive_decode(val);
        if p != player {
            attempt += 1;
            continue;
        }

        if stack_height == 1 {
            if !hive_is_connected_excluding(boards, params, board_idx, q0, r0) {
                attempt += 1;
                continue;
            }
        }

        let mut moved = 0u32;

        if piece_type == HIVE_QUEEN {
            let dir = rng.rand_range(0, 6) as i32;
            let (nq, nr) = hive_neighbor(q0, r0, dir);
            if hive_get_cell(boards, params, board_idx, nq, nr) == 0 && hive_can_slide(boards, params, board_idx, q0, r0, nq, nr) {
                hive_set_cell(boards, params, board_idx, nq, nr, hive_encode(1, player, piece_type));
                hive_set_cell(boards, params, board_idx, q0, r0, 0);
                hive_set_state(boards, params, board_idx, if player == 1 { OFF_P1_QUEEN_Q } else { OFF_P2_QUEEN_Q }, nq + 16);
                hive_set_state(boards, params, board_idx, if player == 1 { OFF_P1_QUEEN_R } else { OFF_P2_QUEEN_R }, nr + 16);
                moved = 1;
            }
        } else if piece_type == HIVE_BEETLE {
            let dir = rng.rand_range(0, 6) as i32;
            let (nq, nr) = hive_neighbor(q0, r0, dir);
            let target_val = hive_get_cell(boards, params, board_idx, nq, nr);
            let (target_height, _, _) = hive_decode(target_val);
            if target_val == 0 {
                if stack_height > 1 || hive_can_slide(boards, params, board_idx, q0, r0, nq, nr) {
                    hive_set_cell(boards, params, board_idx, nq, nr, hive_encode(1, player, piece_type));
                    hive_set_cell(boards, params, board_idx, q0, r0, 0);
                    moved = 1;
                }
            } else {
                hive_set_cell(boards, params, board_idx, nq, nr, hive_encode(target_height + 1, player, piece_type));
                hive_set_cell(boards, params, board_idx, q0, r0, hive_encode(stack_height - 1, player, piece_type));
                moved = 1;
            }
        } else if piece_type == HIVE_SPIDER {
            let mut curr_q = q0;
            let mut curr_r = r0;
            let v0q = q0;
            let v0r = r0;
            let mut v1q = 100i32;
            let mut v1r = 100i32;
            let mut v2q = 100i32;
            let mut v2r = 100i32;
            let mut v3q = 100i32;
            let mut v3r = 100i32;
            let mut valid_path = 1u32;
            let mut step = 0;
            while step < 3 {
                let mut found_step = 0u32;
                let start_dir = rng.rand_range(0, 6);
                let mut d = 0u32;
                while d < 6 {
                    let dir = ((start_dir + d) % 6) as i32;
                    let (nq, nr) = hive_neighbor(curr_q, curr_r, dir);
                    if hive_get_cell(boards, params, board_idx, nq, nr) != 0 {
                        d += 1;
                        continue;
                    }
                    if !hive_can_slide(boards, params, board_idx, curr_q, curr_r, nq, nr) {
                        d += 1;
                        continue;
                    }
                    let mut is_visited = 0u32;
                    if v0q == nq && v0r == nr {
                        is_visited = 1;
                    }
                    if is_visited == 0 && step >= 1 {
                        if v1q == nq && v1r == nr {
                            is_visited = 1;
                        }
                    }
                    if is_visited == 0 && step >= 2 {
                        if v2q == nq && v2r == nr {
                            is_visited = 1;
                        }
                    }
                    if is_visited != 0 {
                        d += 1;
                        continue;
                    }
                    curr_q = nq;
                    curr_r = nr;
                    if step == 0 {
                        v1q = nq;
                        v1r = nr;
                    } else if step == 1 {
                        v2q = nq;
                        v2r = nr;
                    } else {
                        v3q = nq;
                        v3r = nr;
                    }
                    found_step = 1;
                    break;
                }
                if found_step == 0 {
                    valid_path = 0;
                    break;
                }
                step += 1;
            }
            if valid_path != 0 {
                hive_set_cell(boards, params, board_idx, curr_q, curr_r, hive_encode(1, player, piece_type));
                hive_set_cell(boards, params, board_idx, q0, r0, 0);
                moved = 1;
            }
        } else if piece_type == HIVE_GRASSHOPPER {
            let dir = rng.rand_range(0, 6) as i32;
            let (nq, nr) = hive_neighbor(q0, r0, dir);
            if hive_get_cell(boards, params, board_idx, nq, nr) != 0 {
                let mut jump_q = nq;
                let mut jump_r = nr;
                while hive_get_cell(boards, params, board_idx, jump_q, jump_r) != 0 {
                    let (nnq, nnr) = hive_neighbor(jump_q, jump_r, dir);
                    jump_q = nnq;
                    jump_r = nnr;
                }
                hive_set_cell(boards, params, board_idx, jump_q, jump_r, hive_encode(1, player, piece_type));
                hive_set_cell(boards, params, board_idx, q0, r0, 0);
                moved = 1;
            }
        } else if piece_type == HIVE_ANT {
            let mut curr_q = q0;
            let mut curr_r = r0;
            let steps = rng.rand_range(1, 10);
            let mut s = 0u32;
            while s < steps {
                let start_dir = rng.rand_range(0, 6);
                let mut d = 0u32;
                while d < 6 {
                    let dir = ((start_dir + d) % 6) as i32;
                    let (nq, nr) = hive_neighbor(curr_q, curr_r, dir);
                    if hive_get_cell(boards, params, board_idx, nq, nr) == 0
                        && hive_can_slide(boards, params, board_idx, curr_q, curr_r, nq, nr)
                    {
                        curr_q = nq;
                        curr_r = nr;
                        break;
                    }
                    d += 1;
                }
                s += 1;
            }
            if curr_q != q0 || curr_r != r0 {
                hive_set_cell(boards, params, board_idx, curr_q, curr_r, hive_encode(1, player, piece_type));
                hive_set_cell(boards, params, board_idx, q0, r0, 0);
                moved = 1;
            }
        }

        if moved != 0 {
            return true;
        }
        attempt += 1;
    }

    false
}

fn hive_random_rollout(boards: &mut [i32], params: &SimulationParams, board_idx: u32, start_player: i32) -> f32 {
    let mut rng = Rng::new(params.seed.wrapping_add(board_idx.wrapping_mul(719_393)));
    let mut cur_player = start_player;

    let mut step = 0;
    while step < 60 {
        let status = hive_check_win(boards, params, board_idx);
        if status != 2.0 {
            return if start_player == 1 { status * 4000.0 } else { -status * 4000.0 };
        }

        let queen_offset = if cur_player == 1 { OFF_P1_QUEEN } else { OFF_P2_QUEEN };
        let placed_offset = if cur_player == 1 { OFF_P1_PLACED } else { OFF_P2_PLACED };
        let queen_placed = hive_get_state(boards, params, board_idx, queen_offset);
        let pieces_placed = hive_get_state(boards, params, board_idx, placed_offset);

        let mut moved = 0u32;
        if pieces_placed == 3 && queen_placed == 0 {
            moved = if hive_try_place_random(boards, params, board_idx, cur_player, &mut rng) { 1 } else { 0 };
        } else if rng.rand() < 0.5 {
            moved = if hive_try_place_random(boards, params, board_idx, cur_player, &mut rng) { 1 } else { 0 };
            if moved == 0 {
                moved = if hive_try_move_random(boards, params, board_idx, cur_player, &mut rng) { 1 } else { 0 };
            }
        } else {
            moved = if hive_try_move_random(boards, params, board_idx, cur_player, &mut rng) { 1 } else { 0 };
            if moved == 0 {
                moved = if hive_try_place_random(boards, params, board_idx, cur_player, &mut rng) { 1 } else { 0 };
            }
        }

        let turn = hive_get_state(boards, params, board_idx, OFF_TURN);
        hive_set_state(boards, params, board_idx, OFF_TURN, turn + 1);

        if cur_player == 1 {
            cur_player = 2;
        } else {
            cur_player = 1;
        }

        // moved flag intentionally unused; mirrors WGSL behavior
        let _ = moved;

        step += 1;
    }

    0.0
}

// ----- Entry points -----

#[spirv(compute(threads(256)))]
pub fn compute_puct(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] nodes: &[NodeData],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] results: &mut [PuctResult],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] params: &PuctParams,
) {
    let idx = global_id.x as usize;
    let num_nodes = params.packed.x as usize;
    if idx >= num_nodes || idx >= results.len() || idx >= nodes.len() {
        return;
    }

    let node = nodes[idx];
    let effective_visits = node.visits + node.virtual_losses;
    let parent_visits_sqrt = libm::sqrtf(node.parent_visits as f32);

    let (mut q_value, mut exploration_term) = (0.0f32, 0.0f32);
    if effective_visits == 0 {
        exploration_term = node.exploration * node.prior_prob * parent_visits_sqrt;
        q_value = 0.0;
    } else {
        if node.visits > 0 {
            q_value = (node.wins as f32 / node.visits as f32) / 2.0;
        }
        exploration_term = node.exploration * node.prior_prob * parent_visits_sqrt / (1.0 + effective_visits as f32);
    }

    results[idx].puct_score = q_value + exploration_term;
    results[idx].q_value = q_value;
    results[idx].exploration_term = exploration_term;
    results[idx].node_index = idx as u32;
}

#[spirv(compute(threads(64)))]
pub fn evaluate_connect4(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] boards: &mut [i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] results: &mut [SimulationResult],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] params: &SimulationParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] _last_moves: &[i32],
) {
    let idx = global_id.x as usize;
    if idx >= results.len() {
        return;
    }

    let board_size = (params.board_width * params.board_height) as usize;
    let board_start = idx * board_size;
    
    // Simple validation to prevent out of bounds
    if board_start + board_size > boards.len() {
        results[idx].score = 0.0;
        return;
    }

    let current_player = 1;
    let _line_size = get_line_size(params);
    let mut rng = Rng::new(params.seed.wrapping_add((idx as u32).wrapping_mul(719393)));

    // Simple heuristic scoring instead of full rollout to avoid complex memory access
    let mut score = 0.0f32;
    
    // Count pieces for each player
    let mut player_count = 0;
    let mut opponent_count = 0;
    
    let mut i = 0usize;
    while i < board_size {
        if board_start + i < boards.len() {
            let cell = boards[board_start + i];
            if cell == current_player {
                player_count += 1;
            } else if cell == -current_player {
                opponent_count += 1;
            }
        }
        i += 1;
    }
    
    // Simple heuristic: more pieces = better position
    score = (player_count as f32 - opponent_count as f32) * 0.1;
    
    // Add some randomness
    score += (rng.rand() - 0.5) * 0.2;
    
    results[idx].score = score;
}

#[spirv(compute(threads(64)))]
pub fn evaluate_gomoku(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] boards: &mut [i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] results: &mut [SimulationResult],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] params: &SimulationParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] last_moves: &[i32],
) {
    let idx = global_id.x as usize;
    if idx >= results.len() {
        return;
    }

    let last_move_idx = if idx < last_moves.len() { last_moves[idx] } else { -1 };
    evaluate_grid_game_common(boards, results, params, idx as u32, GAME_GOMOKU, last_move_idx);
}

#[spirv(compute(threads(64)))]
pub fn evaluate_othello(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] boards: &mut [i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] results: &mut [SimulationResult],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] params: &SimulationParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] _last_moves: &[i32],
) {
    let idx = global_id.x as usize;
    if idx >= results.len() {
        return;
    }
    let current_player = 1;
    results[idx].score = othello_random_rollout(boards, params, idx as u32, current_player);
}

#[spirv(compute(threads(64)))]
pub fn evaluate_blokus(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] boards: &mut [i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] results: &mut [SimulationResult],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] params: &SimulationParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] _last_moves: &[i32],
) {
    let idx = global_id.x as usize;
    if idx >= results.len() {
        return;
    }
    let current_player = 1;
    results[idx].score = blokus_random_rollout(boards, params, idx as u32, current_player);
}

#[spirv(compute(threads(64)))]
pub fn evaluate_hive(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] boards: &mut [i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] results: &mut [SimulationResult],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] params: &SimulationParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] _last_moves: &[i32],
) {
    let idx = global_id.x as usize;
    if idx >= results.len() {
        return;
    }
    let current_player = 1;
    results[idx].score = hive_random_rollout(boards, params, idx as u32, current_player);
}
