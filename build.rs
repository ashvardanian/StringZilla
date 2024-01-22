fn main() {
    cc::Build::new()
        .file("c/lib.c")
        .include("include")
        .flag_if_supported("-std=c99")
        .flag_if_supported("-fcolor-diagnostics")
        .flag_if_supported("-Wno-unknown-pragmas")
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-cast-function-type")
        .flag_if_supported("-Wno-incompatible-function-pointer-types")
        .flag_if_supported("-Wno-incompatible-pointer-types")
        .flag_if_supported("-Wno-discarded-qualifiers")
        .flag_if_supported("-fPIC")
        .compile("stringzilla");

    println!("cargo:rerun-if-changed=c/lib.c");
    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=include/stringzilla/stringzilla.h");
}
