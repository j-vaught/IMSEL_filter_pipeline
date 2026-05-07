use std::fs;
use std::path::Path;
use std::path::PathBuf;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    println!("cargo:rustc-check-cfg=cfg(wvf_has_vkfft)");
    for key in [
        "WVF_CUDA_HOME",
        "CUDA_HOME",
        "CUDA_PATH",
        "WVF_CUDA_HOST_CXX",
        "CUDAHOSTCXX",
        "CXX",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
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

        println!("cargo:rustc-cfg=wvf_has_vkfft");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=Metal");
    } else if target_os == "linux" {
        if let Some(cuda) = detect_cuda_toolkit() {
            if let Some(host_compiler) = detect_cuda_host_compiler() {
                let nvcc = cuda
                    .nvcc
                    .clone()
                    .unwrap_or_else(|| cuda.root.join("bin").join("nvcc"));
                if let Some(bin_dir) = nvcc.parent() {
                    prepend_env_path("PATH", bin_dir);
                }
                let nvvm_bin_dir = cuda.root.join("nvvm").join("bin");
                if nvvm_bin_dir.exists() {
                    prepend_env_path("PATH", &nvvm_bin_dir);
                }
                prepend_env_path("LD_LIBRARY_PATH", &cuda.lib_dir);
                let linker_dir = prepare_cuda_link_dir(&cuda);
                let mut build = cc::Build::new();
                build
                    .cuda(true)
                    .cudart("shared")
                    .cpp(true)
                    .no_default_flags(true)
                    .warnings(false)
                    .compiler(nvcc)
                    .std("c++17")
                    .file("rust/src/vkfft_cuda_bridge.cu")
                    .include("rust/third_party/vkFFT")
                    .include(&cuda.include)
                    .define("VKFFT_BACKEND", "1")
                    .flag("-allow-unsupported-compiler")
                    .flag("-O2")
                    .flag("-Xcompiler=-fPIC")
                    .flag("-w");
                build.flag(format!("-ccbin={}", host_compiler.display()));
                build.compile("wvf_vkfft_bridge");

                println!("cargo:rustc-cfg=wvf_has_vkfft");
                println!("cargo:rustc-link-search=native={}", linker_dir.display());
                if linker_dir != cuda.lib_dir {
                    println!("cargo:rustc-link-search=native={}", cuda.lib_dir.display());
                }
                println!("cargo:rustc-link-lib=dylib=cuda");
                println!("cargo:rustc-link-lib=dylib=cudart");
                println!("cargo:rustc-link-lib=dylib=nvrtc");
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda.lib_dir.display());
            } else {
                println!(
                    "cargo:warning=No supported CUDA host C++ compiler found. Building fast_wvf without VkFFT GPU support on Linux. Set WVF_CUDA_HOST_CXX to a compatible compiler to enable it."
                );
                println!("cargo:rerun-if-env-changed=PATH");
            }
        } else {
            println!(
                "cargo:warning=CUDA toolkit not found. Building fast_wvf without VkFFT GPU support on Linux."
            );
        }
    }

    println!("cargo:rerun-if-changed=rust/src/vkfft_bridge.cpp");
    println!("cargo:rerun-if-changed=rust/src/vkfft_cuda_bridge.cu");
    rerun_headers(Path::new("rust/third_party/vkFFT"));
    rerun_headers(Path::new("rust/third_party/metal-cpp"));
}

fn prepare_cuda_link_dir(cuda: &CudaToolkit) -> PathBuf {
    let mut needs_shim = false;
    for library in ["libcudart.so", "libnvrtc.so"] {
        if !cuda.lib_dir.join(library).exists() {
            needs_shim = true;
            break;
        }
    }
    if !needs_shim {
        return cuda.lib_dir.clone();
    }

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let link_dir = out_dir.join("cuda-link");
    let _ = fs::create_dir_all(&link_dir);

    for library in ["libcudart.so", "libnvrtc.so"] {
        let target = find_versioned_cuda_library(&cuda.lib_dir, library)
            .unwrap_or_else(|| panic!("missing required CUDA runtime library for {library}"));
        let symlink_path = link_dir.join(library);
        if symlink_path.exists() {
            let _ = fs::remove_file(&symlink_path);
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &symlink_path)
            .unwrap_or_else(|err| panic!("failed to create CUDA link shim for {library}: {err}"));
    }

    link_dir
}

fn find_versioned_cuda_library(dir: &Path, base_name: &str) -> Option<PathBuf> {
    let prefix = format!("{base_name}.");
    let mut candidates = Vec::new();
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name == base_name || file_name.starts_with(&prefix) {
            candidates.push(path);
        }
    }
    candidates.sort();
    if let Some(exact) = candidates
        .iter()
        .find(|path| path.file_name().and_then(|name| name.to_str()) == Some(base_name))
    {
        return Some(exact.clone());
    }
    candidates.into_iter().next()
}

fn prepend_env_path(name: &str, prefix: &Path) {
    let mut values = vec![prefix.to_path_buf()];
    if let Some(existing) = std::env::var_os(name) {
        values.extend(std::env::split_paths(&existing));
    }
    if let Ok(joined) = std::env::join_paths(values) {
        std::env::set_var(name, joined);
    }
}

fn detect_cuda_host_compiler() -> Option<PathBuf> {
    for key in ["WVF_CUDA_HOST_CXX", "CUDAHOSTCXX", "CXX"] {
        if let Some(raw) = std::env::var_os(key) {
            let path = PathBuf::from(raw);
            if path.exists() {
                return Some(path);
            }
        }
    }

    for candidate in ["/usr/bin/g++-12", "/usr/bin/g++-11", "/usr/bin/g++-10"] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }

    for candidate in ["g++-12", "x86_64-linux-gnu-g++-12", "g++-11", "g++-10"] {
        if let Some(path) = find_executable_in_path(candidate) {
            return Some(path);
        }
    }
    None
}

fn find_executable_in_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for entry in std::env::split_paths(&path) {
        let candidate = entry.join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

#[derive(Clone, Debug)]
struct CudaToolkit {
    root: PathBuf,
    include: PathBuf,
    lib_dir: PathBuf,
    nvcc: Option<PathBuf>,
}

fn detect_cuda_toolkit() -> Option<CudaToolkit> {
    let mut candidates = Vec::new();
    for key in ["WVF_CUDA_HOME", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(path) = std::env::var_os(key) {
            candidates.push(PathBuf::from(path));
        }
    }
    candidates.push(PathBuf::from("/usr/local/cuda"));
    candidates.push(PathBuf::from(
        "/usr/local/MATLAB/R2024b/sys/cuda/glnxa64/cuda",
    ));
    candidates.push(PathBuf::from(
        "/usr/local/MATLAB/R2025a/sys/cuda/glnxa64/cuda",
    ));

    for root in candidates {
        let include = root.join("include");
        let nvcc = root.join("bin").join("nvcc");
        let lib64 = root.join("lib64");
        let targets_lib = root.join("targets").join("x86_64-linux").join("lib");
        if include.join("cuda_runtime.h").exists() {
            if lib64.join("libcudart.so").exists() && lib64.join("libnvrtc.so").exists() {
                return Some(CudaToolkit {
                    root,
                    include,
                    lib_dir: lib64,
                    nvcc: nvcc.exists().then_some(nvcc),
                });
            }
            if targets_lib.join("libcudart.so").exists() && targets_lib.join("libnvrtc.so").exists()
            {
                return Some(CudaToolkit {
                    root,
                    include,
                    lib_dir: targets_lib,
                    nvcc: nvcc.exists().then_some(nvcc),
                });
            }
        }
    }

    for matlab_root in [
        PathBuf::from("/usr/local/MATLAB/R2024b"),
        PathBuf::from("/usr/local/MATLAB/R2025a"),
    ] {
        let root = matlab_root
            .join("sys")
            .join("cuda")
            .join("glnxa64")
            .join("cuda");
        let include = root.join("include");
        let nvcc = root.join("bin").join("nvcc");
        let lib_dir = matlab_root.join("bin").join("glnxa64");
        if include.join("cuda_runtime.h").exists()
            && lib_dir.join("libcudart.so.12").exists()
            && lib_dir.join("libnvrtc.so.12").exists()
        {
            return Some(CudaToolkit {
                root,
                include,
                lib_dir,
                nvcc: nvcc.exists().then_some(nvcc),
            });
        }
    }

    None
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
