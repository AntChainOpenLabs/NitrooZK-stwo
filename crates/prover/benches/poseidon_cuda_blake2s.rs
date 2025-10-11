use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use stwo_prover::core::pcs::PcsConfig;
use stwo_prover::examples::poseidon::cuda_prove_poseidon;
use stwo_prover::examples::utils::get_env_var;

pub fn bench_cuda_poseidon_blake2s(c: &mut Criterion) {
    let log_n_instances = get_env_var("LOG_N_INSTANCES", 22u32);
    let mut group = c.benchmark_group("poseidon2_cuda_blake2s");
    group.throughput(Throughput::Elements(1u64 << log_n_instances));
    group.bench_function(format!("poseidon2_cuda_blake2s 2^{} instances", log_n_instances), |b| {
        b.iter(|| cuda_prove_poseidon(log_n_instances, PcsConfig::default()));
    });
}

criterion_group!(
    name = cuda_poseidon_blake2s_group;
    config = Criterion::default().sample_size(10);
    targets = bench_cuda_poseidon_blake2s
);
criterion_main!(cuda_poseidon_blake2s_group);
