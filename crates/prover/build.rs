// use cudart_sys::{cuda_lib_path, cuda_path};
use std::path::PathBuf;
fn main() {
    #[cfg(target_os = "macos")]
    std::process::exit(0);
    let dst = cmake::Config::new("src/stwo_cuda/cuda")
        .profile("Release")
        .build_arg("--jobs=8")
        .build();
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=stwo_cuda");
    let cuda_lib_path: PathBuf = PathBuf::from("/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    println!("cargo:rustc-link-lib=cudart");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=stdc++");
}