fn main() {
    println!("cargo:rerun-if-changed=src/canonical_minimizers_simd.cpp");
    println!("cargo:rerun-if-changed=src/canonical_minimizers_scalar.cpp");
    println!("cargo:rerun-if-changed=src/canonical_minimizers.hpp");
    println!("cargo:rerun-if-changed=src/packed_seq.hpp");
    println!("cargo:rerun-if-env-changed=CPP_PGO");
    println!("cargo:rerun-if-env-changed=CPP_PGO_DIR");

    // Check for PGO mode
    let pgo_phase = std::env::var("CPP_PGO").ok();
    let pgo_dir = std::env::var("CPP_PGO_DIR")
        .unwrap_or_else(|_| "/tmp/cpp-pgo-data".to_string());

    // Compile the C++ implementation with AVX2 instructions
    // Both SIMD and scalar files are compiled together
    // Use Clang if available (LLVM backend like Rust for better optimization parity)
    // Set CC_GCC=1 to force GCC even when Clang is available
    let force_gcc = std::env::var("CC_GCC").is_ok();
    let use_clang = !force_gcc && (std::env::var("CC_CLANG").is_ok() || std::path::Path::new("/usr/bin/clang++").exists());

    let mut cc = cc::Build::new();

    if use_clang {
        cc.compiler("clang++");
        println!("cargo:warning=Using Clang for C++ compilation");
    }

    cc.cpp(true)
        .file("src/canonical_minimizers_simd.cpp")
        .file("src/canonical_minimizers_scalar.cpp")
        .flag_if_supported("-march=native")  // CPU-specific optimizations (like Rust's target-cpu=native)
        .flag_if_supported("-O3")
        .flag_if_supported("-std=c++17")  // For structured bindings
        .flag_if_supported("-Wno-array-bounds")  // Suppress array bounds warnings
        .flag_if_supported("-Wno-unused-variable")  // Suppress unused variable warnings
        .flag_if_supported("-Wno-ignored-attributes")  // Suppress ignored attributes warnings
        .flag_if_supported("-Wno-narrowing");  // Suppress narrowing conversion warnings

    // Clang-specific optimizations for better inlining (similar to LLVM's rustc backend)
    if use_clang {
        cc.flag_if_supported("-flto=thin");  // Link-time optimization
        cc.flag_if_supported("-fno-omit-frame-pointer");
    }

    // Add PGO flags based on phase
    match pgo_phase.as_deref() {
        Some("generate") => {
            println!("cargo:warning=Building C++ with PGO instrumentation (profile-generate)");
            cc.flag(&format!("-fprofile-generate={}", pgo_dir));
            // Link gcov library for profile instrumentation
            println!("cargo:rustc-link-lib=gcov");
        }
        Some("use") => {
            println!("cargo:warning=Building C++ with PGO optimization (profile-use)");
            cc.flag(&format!("-fprofile-use={}", pgo_dir));
            cc.flag("-fprofile-correction");  // Handle slightly mismatched profiles
        }
        _ => {}
    }

    cc.compile("canonical_minimizers_simd");
}