use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use stwo_prover::core::pcs::PcsConfig;
use stwo_prover::examples::poseidon::prove_poseidon;
use stwo_prover::examples::utils::get_env_var;

pub fn simd_poseidon_blake2s(c: &mut Criterion) {
    let log_n_instances = get_env_var("LOG_N_INSTANCES", 18u32);
    let mut group = c.benchmark_group("poseidon2_simd_blake2s");
    group.throughput(Throughput::Elements(1u64 << log_n_instances));
    group.bench_function(format!("poseidon2_simd_blake2s 2^{} instances", log_n_instances), |b| {
        b.iter(|| prove_poseidon(log_n_instances, PcsConfig::default()));
    });
}

criterion_group!(
    name = poseidon_simd_blake2s_group;
    config = Criterion::default().sample_size(10);
    targets = simd_poseidon_blake2s);
criterion_main!(poseidon_simd_blake2s_group);
