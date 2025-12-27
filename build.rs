//! Build script for parallel-mcts-arena
//!
//! This build script handles platform-specific linking requirements for various backends
//! and validates WGSL shaders at compile time.

use std::env;
use std::fs;
use std::path::Path;
use std::collections::HashMap;
use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() {
    // Emit rerun-if-changed for feature flags
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=crates/mcts-shaders/Cargo.toml");
    println!("cargo::rerun-if-changed=crates/mcts-shaders/src");
    println!("cargo::rerun-if-changed=crates/mcts-shared/src");
    println!("cargo::rerun-if-env-changed=MCTS_SKIP_SHADER_BUILD");

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    // Dev escape hatch: skip rust-gpu shader rebuild when iterating on host-side code.
    // This is only safe if OUT_DIR already has the generated artifacts.
    if env::var_os("MCTS_SKIP_SHADER_BUILD").is_some() {
        let wgsl_path = out_path.join("mcts_shaders.wgsl");
        let spv_path = out_path.join("mcts_shaders.spv");
        if wgsl_path.is_file() && spv_path.is_file() {
            println!(
                "cargo:warning=Skipping rust-gpu shader rebuild (MCTS_SKIP_SHADER_BUILD=1); using cached artifacts in OUT_DIR"
            );
            return;
        }
        println!(
            "cargo:warning=MCTS_SKIP_SHADER_BUILD=1 set, but cached shader artifacts not found; rebuilding shaders"
        );
    }

    // Build SPIR-V shaders (rust-gpu)
    // We then translate the full module to WGSL for wgpu compatibility.
    let result = SpirvBuilder::new("crates/mcts-shaders", "spirv-unknown-spv1.5")
        .print_metadata(MetadataPrintout::None)
        .build()
        .expect("Failed to build SPIR-V shaders");

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
            normalize_struct_zero_member_accesses(&mut wgsl);
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

fn normalize_struct_zero_member_accesses(wgsl: &mut String) {
    // DX12 (FXC) can fail to compile HLSL emitted from WGSL that uses zero-value struct
    // constructors followed by member access, e.g. `type_35().member_1`.
    // This is valid WGSL, but can trigger invalid/unsupported patterns in the downstream
    // WGSL->HLSL path.
    //
    // Since the WGSL is generated, we do a small, mechanical rewrite:
    //   `<StructName>().<member>` -> `<MemberType>()`
    // using member types parsed from `struct` declarations in the generated module.

    let struct_member_types = parse_struct_member_types(wgsl);
    if struct_member_types.is_empty() {
        return;
    }

    let bytes = wgsl.as_bytes();
    let mut out = String::with_capacity(wgsl.len());
    let mut i = 0usize;

    while let Some(rel) = wgsl[i..].find("().") {
        let paren_pos = i + rel;
        let mut struct_start = paren_pos;
        while struct_start > 0 && is_ident_byte(bytes[struct_start - 1]) {
            struct_start -= 1;
        }

        let member_start = paren_pos + 3;
        if struct_start < paren_pos
            && member_start < wgsl.len()
            && is_ident_start_byte(bytes[struct_start])
            && is_ident_start_byte(bytes[member_start])
        {
            let mut member_end = member_start;
            while member_end < wgsl.len() && is_ident_byte(bytes[member_end]) {
                member_end += 1;
            }

            let struct_name = &wgsl[struct_start..paren_pos];
            let member_name = &wgsl[member_start..member_end];
            if let Some(member_map) = struct_member_types.get(struct_name) {
                if let Some(member_ty) = member_map.get(member_name) {
                    out.push_str(&wgsl[i..struct_start]);
                    out.push_str(member_ty);
                    out.push_str("()");
                    i = member_end;
                    continue;
                }
            }
        }

        // No rewrite applied; copy through the delimiter and continue.
        out.push_str(&wgsl[i..member_start]);
        i = member_start;
    }

    out.push_str(&wgsl[i..]);
    *wgsl = out;
}

fn parse_struct_member_types(wgsl: &str) -> HashMap<String, HashMap<String, String>> {
    let mut structs: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut current_struct: Option<String> = None;

    for line in wgsl.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(struct_name) = parse_struct_decl_line(trimmed) {
            current_struct = Some(struct_name.clone());
            structs.entry(struct_name).or_default();
            continue;
        }

        if trimmed.starts_with('}') {
            current_struct = None;
            continue;
        }

        if let Some(struct_name) = current_struct.as_ref() {
            if let Some((member, ty)) = parse_struct_member_line(trimmed) {
                if let Some(member_map) = structs.get_mut(struct_name) {
                    member_map.insert(member, ty);
                }
            }
        }
    }

    structs
}

fn parse_struct_decl_line(trimmed: &str) -> Option<String> {
    // Matches e.g. `struct type_35 {`
    let rest = trimmed.strip_prefix("struct ")?;
    let mut name_end = 0usize;
    for (idx, ch) in rest.char_indices() {
        if ch == '{' || ch.is_whitespace() {
            break;
        }
        name_end = idx + ch.len_utf8();
    }
    if name_end == 0 {
        return None;
    }
    let name = &rest[..name_end];
    if !trimmed.contains('{') {
        return None;
    }
    Some(name.to_string())
}

fn parse_struct_member_line(trimmed: &str) -> Option<(String, String)> {
    // Matches e.g. `member_1: u32,`
    let (lhs, rhs) = trimmed.split_once(':')?;
    let member = lhs.trim();
    if member.is_empty() {
        return None;
    }
    let rhs = rhs.trim();
    let ty = rhs
        .trim_end_matches(',')
        .trim_end();
    if ty.is_empty() {
        return None;
    }
    Some((member.to_string(), ty.to_string()))
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_ident_start_byte(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}
