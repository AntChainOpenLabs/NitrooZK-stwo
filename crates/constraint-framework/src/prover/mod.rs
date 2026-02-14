mod assert;
mod component_prover;
mod cpu_domain;
mod logup;
pub mod relation_tracker;
mod simd_domain;

pub use assert::{assert_constraints_on_polys, assert_constraints_on_trace, AssertEvaluator};
// Re-export for CUDA testing
pub use assert::assert_constraints_on_polys_cuda;
pub use cpu_domain::CpuDomainEvaluator;
pub use logup::{FractionWriter, LogupColGenerator, LogupTraceGenerator};
pub use simd_domain::SimdDomainEvaluator;
