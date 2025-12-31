// Minimal urgent event emission kernel for diagnostics
struct UrgentEvent {
    timestamp: u32,
    event_type: u32,
    _pad: u32,
    payload: array<u32, 255>,
};

@group(0) @binding(0) var<storage, read_write> urgent_event_buffer: array<UrgentEvent, 256>;
@group(0) @binding(1) var<storage, read_write> urgent_event_write_head: atomic<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u) {
        let idx = atomicAdd(&urgent_event_write_head, 1u) % 256u;
        urgent_event_buffer[idx].timestamp = 12345678u;
        urgent_event_buffer[idx].event_type = 42u;
        urgent_event_buffer[idx]._pad = 0u;
        for (var i = 0u; i < 255u; i = i + 1u) {
            urgent_event_buffer[idx].payload[i] = i;
        }
    }
}
