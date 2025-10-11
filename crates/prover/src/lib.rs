#![allow(incomplete_features)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    feature(stdarch_x86_avx512)
)]
#![feature(
    array_chunks,
    array_try_from_fn,
    array_windows,
    assert_matches,
    exact_size_is_empty,
    int_roundings,
    iter_array_chunks,
    portable_simd,
    slice_ptr_get
)]
pub mod constraint_framework;
pub mod core;
pub mod examples;

// TODO(Ohad): find a better way to test poseidon and depracate `math.rs`.
pub mod math;
pub mod stwo_cuda;

#[cfg(feature = "tracing")]
pub mod tracing;

// When running benchmarks we want to avoid noisy stdout produced by ad-hoc
// println! calls sprinkled in example/utility code. Use the macro below
// instead of println! where such output is optional. It becomes a no-op when
// the environment variable STWO_QUIET is set.
#[doc(hidden)]
#[inline]
pub fn __bench_print(args: std::fmt::Arguments<'_>) {
    if !should_quiet_bench() {
        // Defer to the standard println formatting.
        println!("{}", args);
    }
}

#[inline]
pub fn should_quiet_bench() -> bool {
    // Silence optional prints when the user asks for quiet bench runs.
    // We use an env var so README examples can toggle this without
    // rebuilding with a dedicated Cargo feature.
    std::env::var_os("STWO_QUIET").is_some()
}

#[macro_export]
macro_rules! bench_println {
    ($($arg:tt)*) => ({
        $crate::__bench_print(format_args!($($arg)*));
    })
}
