# Rust-GPU Shader Investigation Report

**Date:** December 2024  
**Branch:** `feature/unifiedCode`  
**Status:** ⚠️ Experimental - Not production ready

## Executive Summary

This branch attempted to replace hand-written WGSL shaders with Rust code compiled via [rust-gpu](https://github.com/EmbarkStudios/rust-gpu). While the approach works functionally, it suffers from a **5.8x GPU performance regression** compared to hand-written shaders due to fundamental SPIR-V limitations.

**Recommendation:** Keep the `main` branch with hand-written WGSL shaders for production use.

---

## Performance Comparison

| Metric | `main` (Hand-written WGSL) | `feature/unifiedCode` (rust-gpu) |
|--------|---------------------------|----------------------------------|
| GPU Root Visits (5s) | 22,285 | 4,015 |
| CPU Root Visits (5s) | ~110,000 | ~110,000 |
| Shader Size (Gomoku) | ~200 lines | 13,468 lines |
| Total WGSL Size | ~47 KB | ~3.5 MB |
| GPU vs CPU Ratio | 20% | 3.6% |

**The rust-gpu approach is 5.8x slower on GPU.**

---

## Root Cause: Mandatory Pointer Inlining

### The Problem

SPIR-V (the intermediate representation rust-gpu compiles to) has strict rules about pointer types. From rust-gpu's source code:

> "This algorithm is not intended to be an optimization, it is rather for **legalization**. Specifically, SPIR-V disallows things like a `StorageClass::Function` pointer to a `StorageClass::Input` pointer."

**Any function that takes a pointer/reference parameter MUST be inlined** because SPIR-V doesn't support passing arbitrary pointers between functions.

### Example

```rust
// This function WILL be inlined at every call site
fn check_win(board: &[i32; 400], player: i32) -> bool {
    // ... 50 lines of logic
}

fn rollout() {
    let mut sim_board: [i32; 400] = [0; 400];
    
    // Each of these calls duplicates the entire check_win function body
    if check_win(&sim_board, 1) { ... }  // +50 lines
    if check_win(&sim_board, -1) { ... } // +50 lines
    // ... repeated for horizontal, vertical, diagonal checks
}
```

The resulting WGSL has **every function body duplicated** at each call site, leading to:
- 100x code bloat
- GPU register pressure
- Instruction cache thrashing
- Longer shader compilation times

### Why `#[inline(never)]` Doesn't Help

rust-gpu explicitly ignores `#[inline(never)]` when inlining is required for SPIR-V legality:

```rust
// From rust-gpu's linker/inline.rs
sess.warn(format!(
    "`#[inline(never)]` function `{}` needs to be inlined \
     because it has illegal argument or return types",
    get_name(&names, f)
));
```

---

## Technical Architecture

### Build Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         build.rs                                 │
├─────────────────────────────────────────────────────────────────┤
│  1. spirv-builder compiles crates/mcts-shaders → SPIR-V         │
│  2. spirv-opt runs optimization passes                          │
│  3. naga translates SPIR-V → WGSL                               │
│  4. WGSL embedded in binary via include_str!                    │
└─────────────────────────────────────────────────────────────────┘
```

### Crate Structure

```
crates/
├── mcts-shaders/       # Rust shader code (compiled to SPIR-V)
│   ├── Cargo.toml      # spirv-std dependency
│   └── src/lib.rs      # All shader entry points
└── mcts-shared/        # Shared types between CPU and GPU
    └── src/lib.rs      # GameType enum, etc.
```

### Entry Points

| Function | Purpose | Generated Size |
|----------|---------|----------------|
| `compute_puct` | PUCT score calculation | 62 lines ✓ |
| `evaluate_connect4` | Connect4 rollouts | 126 lines ✓ |
| `evaluate_gomoku` | Gomoku rollouts | 13,468 lines ❌ |
| `evaluate_othello` | Othello rollouts | 114 lines ✓ |
| `evaluate_blokus` | Blokus rollouts | 2,124 lines ⚠️ |
| `evaluate_hive` | Hive rollouts | 15,966 lines ❌ |

---

## What Works Well

1. **Simple shaders** (PUCT, Othello) generate reasonable code
2. **Type safety** - Rust's type system catches errors at compile time
3. **Code sharing** - `mcts-shared` crate enables CPU/GPU type sharing
4. **Build integration** - `spirv-builder` integrates cleanly with Cargo

## What Doesn't Work

1. **Complex rollout simulations** generate massive code due to inlining
2. **Local array manipulation** requires pointer parameters → forced inlining
3. **No way to prevent inlining** - it's required for SPIR-V legality
4. **Performance is unacceptable** for production use

---

## Attempted Mitigations

### 1. Moving Nested Functions to Module Level
**Result:** ❌ No improvement  
Functions still get inlined due to `&[i32; 400]` parameters.

### 2. Using `#[inline(never)]`
**Result:** ❌ Ignored by rust-gpu  
Legalization requirements override the attribute.

### 3. Reducing Function Call Sites
**Result:** ⚠️ Partial improvement  
Manually inlining code reduces call sites but defeats the purpose of using Rust.

---

## How Hand-Written WGSL Avoids This

The `main` branch's hand-written shaders work because:

```wgsl
fn gomoku_random_rollout(idx: u32, ...) -> f32 {
    // Local array declared directly - no function parameters
    var sim_board: array<i32, 400>;
    
    // Win-checking logic INLINED directly here
    // No function calls with array references
    var count = 1;
    var x = col - 1;
    while (x >= 0) {
        if (sim_board[row * w + x] == player) { count++; }
        // ...
    }
}
```

Key differences:
- No helper functions with array parameters
- All array access is direct, not through function calls
- Repetitive but efficient

---

## Future Options

### Option 1: Keep `main` Branch (Recommended)
- Best performance
- Proven working
- Maintain WGSL shaders separately

### Option 2: Hybrid Approach
- Use rust-gpu for simple shaders (PUCT, Othello)
- Hand-write complex shaders (Gomoku, Hive)
- Complex build system

### Option 3: Wait for Improvements
- SPIR-V may get better function call support
- rust-gpu team is aware of the issue
- No timeline

### Option 4: Rewrite Without Pointer Parameters
- Restructure all code to avoid reference parameters
- Use global mutable state instead
- Defeats purpose of using Rust

---

## Files Modified in This Branch

| File | Purpose |
|------|---------|
| `build.rs` | Added spirv-builder integration |
| `Cargo.toml` | Added shader crate dependencies |
| `crates/mcts-shaders/` | New: Rust shader code |
| `crates/mcts-shared/` | New: Shared CPU/GPU types |
| `src/gpu/shaders.rs` | Modified to load generated WGSL |

---

## Reproducing the Investigation

```powershell
# Build the shaders
cargo build --release --bin benchmark

# Find generated WGSL
Get-ChildItem -Path "target/release/build" -Recurse -Filter "mcts_shaders.wgsl"

# Check line counts
(Get-Content "path/to/mcts_shaders.wgsl").Count

# Run benchmark (compare with main branch)
./target/release/benchmark.exe --duration 5 --gpu-use-heuristic false
```

---

## References

- [rust-gpu Repository](https://github.com/EmbarkStudios/rust-gpu)
- [rust-gpu Inliner Source](https://github.com/EmbarkStudios/rust-gpu/blob/main/crates/rustc_codegen_spirv/src/linker/inline.rs)
- [SPIR-V Specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
- [naga (SPIR-V → WGSL translator)](https://github.com/gfx-rs/naga)

---

## Conclusion

The rust-gpu approach is **technically functional but not performant** for this use case. The fundamental issue is SPIR-V's pointer legality requirements forcing aggressive inlining, which cannot be worked around without significant architectural changes that would defeat the purpose of using Rust.

**Keep using hand-written WGSL shaders on `main` for production.**

This branch serves as documentation of the investigation and a starting point if rust-gpu's inlining behavior improves in the future.
