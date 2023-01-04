//! This build script emits the openblas linking directive if requested
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "cblas")]
    println!("cargo:rustc-link-lib=dylib=cblas");
}
