use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use stwo_prover::core::pcs::PcsConfig;
use stwo_prover::examples::poseidon::prove_poseidon;
use stwo_prover::examples::utils::get_env_var;

pub fn simd_poseidon(c: &mut Criterion) {
    let log_n_instances = get_env_var("LOG_N_INSTANCES", 18u32);
    let mut group = c.benchmark_group("poseidon2");
    group.throughput(Throughput::Elements(1u64 << log_n_instances));
    group.bench_function(format!("poseidon2 2^{} instances", log_n_instances), |b| {
        b.iter(|| prove_poseidon(log_n_instances, PcsConfig::default()));
    });
}

criterion_group!(
    name = bit_rev;
    config = Criterion::default().sample_size(10);
    targets = simd_poseidon);
criterion_main!(bit_rev);
