//! Build script for parallel-mcts-arena
//!
//! This build script handles platform-specific linking requirements for various backends.

fn main() {
    // Emit rerun-if-changed for feature flags
    println!("cargo::rerun-if-changed=build.rs");
}
