use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use stwo::core::pcs::PcsConfig;
use stwo_examples::wide_fibonacci::cuda_prove_wide_fibonacci;
use stwo_examples::utils::get_env_var;

const FIB_SEQUENCE_LENGTH: usize = 100;

pub fn bench_wide_fibonacci_cuda_blake2s(c: &mut Criterion) {
    let log_n_instances = get_env_var("LOG_N_INSTANCES", 20u32);
    let mut group = c.benchmark_group("wide_fibonacci_cuda_blake2s");
    group.throughput(Throughput::Elements(1u64 << log_n_instances));
    group.bench_function(format!("wide_fibonacci_cuda_blake2s 2^{} instances", log_n_instances), |b| {
        b.iter(|| cuda_prove_wide_fibonacci::<FIB_SEQUENCE_LENGTH>(log_n_instances, PcsConfig::default()));
    });
}

criterion_group!(
    name = wide_fibonacci_cuda_blake2s_group;
    config = Criterion::default().sample_size(10);
    targets = bench_wide_fibonacci_cuda_blake2s
);
criterion_main!(wide_fibonacci_cuda_blake2s_group);
