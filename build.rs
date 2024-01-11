use std::env;
use std::path::PathBuf;
use std::process::Command;

fn link_static_fns(out_dir_path: PathBuf) {
    let obj_path = out_dir_path.join("extern.o");

    let clang_output = Command::new("clang")
        .arg("-O")
        .arg("-c")
        .arg("-o")
        .arg(&obj_path)
        .arg(env::temp_dir().join("bindgen").join("extern.c"))
        .arg("-include")
        .arg("stringzilla/stringzilla.h")
        .output();
    if let Err(e) = clang_output {
        match e.kind() {
            std::io::ErrorKind::NotFound => panic!("command `clang` was not found in the system"),
            _ => panic!("{}", e),
        }
    }

    let clang_output = clang_output.ok().unwrap();
    if !clang_output.status.success() {
        panic!(
            "Could not compile object file:\n{}",
            String::from_utf8_lossy(&clang_output.stderr)
        );
    }

    #[cfg(not(target_os = "windows"))]
    let lib_output = Command::new("ar")
        .arg("rcs")
        .arg(out_dir_path.join("libextern.a"))
        .arg(obj_path)
        .output();
    #[cfg(target_os = "windows")]
    let lib_output = Command::new("lib").arg(&obj_path).output();
    if let Err(e) = lib_output {
        match e.kind() {
            std::io::ErrorKind::NotFound => {
                if !cfg!(target_os = "windows") {
                    panic!("command `ar` was not found in the system")
                } else {
                    panic!("command `lib` was not found in the system")
                }
            }
            _ => panic!("{}", e),
        }
    }

    let lib_output = lib_output.ok().unwrap();
    if !lib_output.status.success() {
        panic!(
            "Could not emit library file:\n{}",
            String::from_utf8_lossy(&lib_output.stderr)
        );
    }

    println!(
        "cargo:rustc-link-search=native={}",
        out_dir_path.to_string_lossy()
    );
    println!("cargo:rustc-link-lib=static=extern");
}

fn main() {
    let root_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    #[cfg(not(debug_assertions))]
    env::set_var(
        "RUSTFLAGS",
        "-Clinker-plugin-lto -Clinker=clang -Clink-arg=-fuse-ld=lld",
    );
    println!("RUSTFLAGS: {}", env::var("RUSTFLAGS").unwrap_or_default());

    let bindings = bindgen::Builder::default()
        .header(format!(
            "{}/stringzilla/stringzilla.h",
            root_path.to_str().unwrap()
        ))
        .wrap_static_fns(true)
        .generate_inline_functions(true)
        .generate()
        .unwrap();

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .unwrap();
    link_static_fns(out_path);
}
