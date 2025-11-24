use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use stwo::core::pcs::PcsConfig;
use stwo_examples::utils::get_env_var;
use stwo_examples::wide_fibonacci::cuda_prove_wide_fibonacci_poseidon;

const FIB_SEQUENCE_LENGTH: usize = 100;

pub fn bench_wide_fibonacci_cuda_poseidon(c: &mut Criterion) {
    let log_n_instances = get_env_var("LOG_N_INSTANCES", 20u32);
    let mut group = c.benchmark_group("wide_fibonacci_cuda_poseidon");
    group.throughput(Throughput::Elements(1u64 << log_n_instances));

    group.bench_function(format!("wide_fibonacci_cuda_poseidon 2^{} instances", log_n_instances), |b| {
        b.iter(|| {
            let _ = cuda_prove_wide_fibonacci_poseidon::<FIB_SEQUENCE_LENGTH>(
                log_n_instances,
                PcsConfig::default(),
            );
        })
    });
}

criterion_group!(
    name = wide_fibonacci_poseidon_cuda_group;
    config = Criterion::default().sample_size(10);
    targets = bench_wide_fibonacci_cuda_poseidon
);
criterion_main!(wide_fibonacci_poseidon_cuda_group);
