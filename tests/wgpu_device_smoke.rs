// Minimal wgpu device creation test for CI/headless debugging
#[test]
fn test_wgpu_device_smoke() {
    use wgpu::Instance;
    use wgpu::Backends;
    use pollster::block_on;
    eprintln!("[DIAG] wgpu_device_smoke: before instance");
    let instance = Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    eprintln!("[DIAG] wgpu_device_smoke: after instance");
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).expect("No suitable GPU adapters found on the system!");
    eprintln!("[DIAG] wgpu_device_smoke: after adapter");
    let _device = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: Default::default(),
    }, None)).expect("Failed to create device");
    eprintln!("[DIAG] wgpu_device_smoke: after device");
}
