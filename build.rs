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
    println!("cargo::rerun-if-changed=src/gpu/shaders");

    // Build SPIR-V shaders
    // This will compile the mcts-shaders crate to SPIR-V and place it in OUT_DIR
    let result = SpirvBuilder::new("crates/mcts-shaders", "spirv-unknown-spv1.5")
        .print_metadata(MetadataPrintout::None)
        .build()
        .expect("Failed to build SPIR-V shaders");

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("mcts_shaders.spv");
    fs::copy(result.module.unwrap_single(), dest_path).expect("Failed to copy SPIR-V module");

    let shader_dir = Path::new("src/gpu/shaders");
    if !shader_dir.exists() {
        return;
    }
    
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    // List of shaders to process
    let shaders = [
        "puct.wgsl",
        "common.wgsl",
        "grid_common.wgsl",
        "gomoku.wgsl",
        "connect4.wgsl",
        "othello.wgsl",
        "blokus.wgsl",
        "hive.wgsl",
    ];

    for shader in shaders {
        let (resolved, source_map) = resolve_shader_source(shader_dir, shader);
        
        // Validate
        validate_shader(shader, &resolved, &source_map);
        
        // Write to OUT_DIR
        let dest_path = out_path.join(shader);
        fs::write(&dest_path, &resolved).expect("Failed to write shader to OUT_DIR");
    }
}

fn resolve_shader_source(dir: &Path, filename: &str) -> (String, Vec<(String, usize)>) {
    let content = fs::read_to_string(dir.join(filename))
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", filename, e));
    
    let mut full_source = String::new();
    let mut source_map = Vec::new();

    for (i, line) in content.lines().enumerate() {
        let current_line_num = i + 1;
        if let Some(included_file) = line.trim().strip_prefix("#include \"").and_then(|s| s.strip_suffix("\"")) {
             let (inc_source, inc_map) = resolve_shader_source(dir, included_file);
             full_source.push_str(&inc_source);
             source_map.extend(inc_map);
             
             // Add a newline to replace the #include line
             full_source.push('\n');
             source_map.push((filename.to_string(), current_line_num));
        } else {
            full_source.push_str(line);
            full_source.push('\n');
            source_map.push((filename.to_string(), current_line_num));
        }
    }
    (full_source, source_map)
}

fn validate_shader(name: &str, source: &str, source_map: &[(String, usize)]) {
    let mut parser = naga::front::wgsl::Frontend::new();
    let module = match parser.parse(source) {
        Ok(module) => module,
        Err(e) => {
            let msg = e.emit_to_string(source);
            if let Some(loc) = e.location(source) {
                let line_index = loc.line_number as usize - 1;
                if let Some((orig_file, orig_line)) = source_map.get(line_index) {
                    panic!("Failed to parse shader '{}':\n  --> {}:{}\n\n{}", name, orig_file, orig_line, msg);
                }
            }
            panic!("Failed to parse shader '{}': {}", name, msg);
        }
    };

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    if let Err(e) = validator.validate(&module) {
        let mut orig_location = String::new();
        if let Some((span, _)) = e.spans().next() {
             if let Some(range) = span.to_range() {
                 let start = range.start;
                 if start < source.len() {
                     let line_number = source[..start].matches('\n').count() + 1;
                     if let Some((orig_file, orig_line)) = source_map.get(line_number - 1) {
                         orig_location = format!("\n  --> {}:{}", orig_file, orig_line);
                     }
                 }
             }
        }
        panic!("Failed to validate shader '{}': {:?}{}", name, e, orig_location);
    }
}
