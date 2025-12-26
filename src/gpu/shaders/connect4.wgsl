#include "grid_common.wgsl"

@compute @workgroup_size(64)
fn evaluate_connect4(@builtin(global_invocation_id) global_id: vec3<u32>) {
    evaluate_grid_game_common(global_id.x, GAME_CONNECT4);
}
