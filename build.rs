use std::env;
use std::path::PathBuf;

fn main() {
    cc::Build::new()
        .include("./stringzilla/")
        .include("./rust/")
        .file("./rust/rust-bindings.c")
        .compile("libstaticstringzilla");

    let bindings = bindgen::Builder::default()
        .header("rust/rust-bindings.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate_inline_functions(true)
        .clang_args(["-I./stringzilla/"])
        .generate()
        .unwrap();

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs")).unwrap();
}