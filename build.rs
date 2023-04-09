extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    //println!("cargo:rustc-link-search=/path/to/lib");

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.

    //let dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    //#[cfg(windows)]
    // println!("cargo:rustc-link-search=native={}/libs/win_avx2", dir);
    // println!("cargo:rustc-link-lib=static=rwkv");

    // #[cfg(unix)]
    // println!("cargo:rustc-link-lib=bz2");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=rwkv-cpp/rwkv.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("rwkv-cpp/rwkv.h")
        .dynamic_link_require_all(true)
        .dynamic_library_name("rwkv")
        //.enable_cxx_namespaces()
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .clang_arg("-stdlib=libc++")
        .allowlist_function("rwkv_init_from_file")
        .allowlist_function("rwkv_eval")
        .allowlist_function("rwkv_free")
        .allowlist_function("rwkv_get_state_buffer_element_count")
        .allowlist_function("rwkv_get_logits_buffer_element_count")
        .allowlist_function("rwkv_get_system_info_string")
        .allowlist_function("rwkv_quantize_model_file")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
