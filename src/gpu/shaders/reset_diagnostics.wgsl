// Shader to reset diagnostics buffer
// This is needed because queue.write_buffer() doesn't reliably reset atomicMax values

struct Diagnostics {
    selection_terminal: atomic<u32>,
    selection_no_children: atomic<u32>,
    selection_invalid_child: atomic<u32>,
    selection_path_cap: atomic<u32>,
    expansion_attempts: atomic<u32>,
    expansion_success: atomic<u32>,
    expansion_locked: atomic<u32>,
    exp_lock_rollout: atomic<u32>,
    exp_lock_sibling: atomic<u32>,
    exp_lock_retry: atomic<u32>,
    expansion_terminal: atomic<u32>,
    alloc_failures: atomic<u32>,
    nodes_freed: atomic<u32>,
    rollouts: atomic<u32>,
    root_board_hash: atomic<u32>,
    init_nodes_count: atomic<u32>,
    total_children_gen: atomic<u32>,
    prune_work_claimed: atomic<u32>,
    prune_push_attempts: atomic<u32>,
    max_temp_boost: atomic<u32>,
    random_rollout_wins: atomic<u32>,
    random_rollout_draws: atomic<u32>,
    random_rollout_losses: atomic<u32>,
    random_rollout_from_p1: atomic<u32>,
    random_rollout_from_p2: atomic<u32>,
    random_rollout_p1_wins_raw: atomic<u32>,
    random_rollout_p1_losses_raw: atomic<u32>,
    random_rollout_p1_score_sum: atomic<u32>,
    random_rollout_min_tid: atomic<u32>,
    random_rollout_max_tid: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> diagnostics: Diagnostics;

@compute @workgroup_size(1)
fn main() {
    // Reset all diagnostic counters to 0
    atomicStore(&diagnostics.selection_terminal, 0u);
    atomicStore(&diagnostics.selection_no_children, 0u);
    atomicStore(&diagnostics.selection_invalid_child, 0u);
    atomicStore(&diagnostics.selection_path_cap, 0u);
    atomicStore(&diagnostics.expansion_attempts, 0u);
    atomicStore(&diagnostics.expansion_success, 0u);
    atomicStore(&diagnostics.expansion_locked, 0u);
    atomicStore(&diagnostics.exp_lock_rollout, 0u);
    atomicStore(&diagnostics.exp_lock_sibling, 0u);
    atomicStore(&diagnostics.exp_lock_retry, 0u);
    atomicStore(&diagnostics.expansion_terminal, 0u);
    atomicStore(&diagnostics.alloc_failures, 0u);
    atomicStore(&diagnostics.nodes_freed, 0u);
    atomicStore(&diagnostics.rollouts, 0u);
    // Don't reset root_board_hash - it's not a counter
    atomicStore(&diagnostics.init_nodes_count, 0u);
    atomicStore(&diagnostics.total_children_gen, 0u);
    atomicStore(&diagnostics.prune_work_claimed, 0u);
    atomicStore(&diagnostics.prune_push_attempts, 0u);
    atomicStore(&diagnostics.max_temp_boost, 0u);
    atomicStore(&diagnostics.random_rollout_wins, 0u);
    atomicStore(&diagnostics.random_rollout_draws, 0u);
    atomicStore(&diagnostics.random_rollout_losses, 0u);
    atomicStore(&diagnostics.random_rollout_from_p1, 0u);
    atomicStore(&diagnostics.random_rollout_from_p2, 0u);
    atomicStore(&diagnostics.random_rollout_p1_wins_raw, 0u);
    atomicStore(&diagnostics.random_rollout_p1_losses_raw, 0u);
    atomicStore(&diagnostics.random_rollout_p1_score_sum, 0u);
    atomicStore(&diagnostics.random_rollout_min_tid, 0xFFFFFFFFu);
    atomicStore(&diagnostics.random_rollout_max_tid, 0u);
}
