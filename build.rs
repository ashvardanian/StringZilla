fn main() {
    cc::Build::new()
        .file("c/lib.c")
        .include("include")
        .warnings(false)
        .flag_if_supported("-std=c99")
        .flag_if_supported("-fcolor-diagnostics")
        .flag_if_supported("-fPIC")
        .compile("stringzilla");

    println!("cargo:rerun-if-changed=c/lib.c");
    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=include/stringzilla/stringzilla.h");
}
