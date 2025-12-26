//! Build script for parallel-mcts-arena
//!
//! This build script handles platform-specific linking requirements for various backends
//! and validates WGSL shaders at compile time.

use std::env;
use std::fs;
use std::path::Path;
use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() {
    // Emit rerun-if-changed for feature flags
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=crates/mcts-shaders/Cargo.toml");
    println!("cargo::rerun-if-changed=crates/mcts-shaders/src");
    println!("cargo::rerun-if-changed=crates/mcts-shared/src");

    // Build SPIR-V shaders (rust-gpu)
    // We then translate the full module to WGSL for wgpu compatibility.
    let result = SpirvBuilder::new("crates/mcts-shaders", "spirv-unknown-spv1.5")
        .print_metadata(MetadataPrintout::None)
        .build()
        .expect("Failed to build SPIR-V shaders");

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    // Convert rust-gpu SPIR-V -> WGSL for all kernels
    let spv_path = result.module.unwrap_single();
    let spv_bytes = fs::read(&spv_path).expect("Failed to read rust-gpu SPIR-V module");

    // Also ship the validated SPIR-V alongside the generated WGSL.
    fs::write(out_path.join("mcts_shaders.spv"), &spv_bytes)
        .expect("Failed to write mcts_shaders.spv to OUT_DIR");

    // Naga's SPIR-V parsing/validation and WGSL emission can be stack-hungry on Windows.
    // Run it on a thread with a larger stack to avoid build-script stack overflow.
    let module_wgsl = std::thread::Builder::new()
        .name("naga_spv_to_wgsl".to_string())
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            let mut wgsl = spirv_to_wgsl(&spv_bytes).expect("Failed to convert rust-gpu SPIR-V -> WGSL");
            normalize_generated_entry_points(&mut wgsl);
            validate_generated_wgsl("mcts_shaders.wgsl (generated)", &wgsl);
            wgsl
        })
        .expect("Failed to spawn naga conversion thread")
        .join()
        .expect("naga conversion thread panicked");
    fs::write(out_path.join("mcts_shaders.wgsl"), module_wgsl)
        .expect("Failed to write generated mcts_shaders.wgsl to OUT_DIR");
}

fn spirv_to_wgsl(spv_bytes: &[u8]) -> Result<String, String> {
    use naga::front::spv;

    let options = spv::Options {
        adjust_coordinate_space: false,
        strict_capabilities: false,
        block_ctx_dump_prefix: None,
    };

    let module = spv::parse_u8_slice(spv_bytes, &options)
        .map_err(|e| format!("SPIR-V parse failed: {e:?}"))?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator
        .validate(&module)
        .map_err(|e| format!("SPIR-V validation failed: {e:?}"))?;

    let wgsl = naga::back::wgsl::write_string(
        &module,
        &info,
        naga::back::wgsl::WriterFlags::empty(),
    )
    .map_err(|e| format!("WGSL write failed: {e:?}"))?;

    Ok(wgsl)
}

fn normalize_generated_entry_points(wgsl: &mut String) {
    // Naga may append an underscore to exported entry points to avoid collisions.
    // We normalize any entry-point function `name_` -> `name` so the runtime can
    // consistently refer to stable entry point names.
    //
    // We only rewrite the function that immediately follows a stage attribute.
    let mut out = String::with_capacity(wgsl.len());
    let mut last_line_was_stage_attribute = false;

    for line in wgsl.lines() {
        let trimmed = line.trim_start();
        let is_stage_attr = trimmed.starts_with("@compute")
            || trimmed.starts_with("@vertex")
            || trimmed.starts_with("@fragment");

        if last_line_was_stage_attribute {
            if let Some(after_fn) = trimmed.strip_prefix("fn ") {
                if let Some(paren_idx) = after_fn.find('(') {
                    let (name, rest) = after_fn.split_at(paren_idx);
                    if let Some(name_no_underscore) = name.strip_suffix('_') {
                        let indent_len = line.len().saturating_sub(trimmed.len());
                        out.push_str(&line[..indent_len]);
                        out.push_str("fn ");
                        out.push_str(name_no_underscore);
                        out.push_str(rest);
                        out.push('\n');
                        last_line_was_stage_attribute = false;
                        continue;
                    }
                }
            }
        }

        out.push_str(line);
        out.push('\n');
        last_line_was_stage_attribute = is_stage_attr;
    }

    *wgsl = out;
}

fn validate_generated_wgsl(name: &str, source: &str) {
    let mut parser = naga::front::wgsl::Frontend::new();
    let module = match parser.parse(source) {
        Ok(module) => module,
        Err(e) => {
            let msg = e.emit_to_string(source);
            panic!("Failed to parse shader '{}': {}", name, msg);
        }
    };

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    if let Err(e) = validator.validate(&module) {
        panic!("Failed to validate shader '{}': {:?}", name, e);
    }
}
