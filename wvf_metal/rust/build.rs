use std::fs;
use std::path::Path;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .file("rust/src/vkfft_bridge.cpp")
            .include("rust/third_party/vkFFT")
            .include("rust/third_party/metal-cpp")
            .define("VKFFT_BACKEND", "5")
            .flag_if_supported("-w")
            .flag_if_supported("-Wno-deprecated-declarations")
            .flag_if_supported("-Wno-c++17-extensions")
            .flag_if_supported("-Wno-unused-parameter")
            .compile("wvf_vkfft_bridge");

        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    println!("cargo:rerun-if-changed=rust/src/vkfft_bridge.cpp");
    rerun_headers(Path::new("rust/third_party/vkFFT"));
    rerun_headers(Path::new("rust/third_party/metal-cpp"));
}

fn rerun_headers(path: &Path) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rerun_headers(&path);
            continue;
        }
        if matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("h" | "hpp")
        ) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
