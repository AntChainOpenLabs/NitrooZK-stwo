use itertools::Itertools;

use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, TraceLocationAllocator, fnv1a_eval_id_gen};
use stwo::prover::backend::simd::m31::PackedBaseField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::{Col, Column};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::core::ColumnVec;
use stwo::core::pcs::PcsConfig;
use stwo::core::channel::{Blake2sChannel, Poseidon252Channel};
use stwo::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::vcs::poseidon252_merkle::Poseidon252MerkleChannel;
use stwo::core::proof::StarkProof;
use stwo::prover::{prove, CommitmentSchemeProver};
use num_traits::Zero;

pub type WideFibonacciComponent<const N: usize> = FrameworkComponent<WideFibonacciEval<N>>;

pub struct FibInput {
    a: PackedBaseField,
    b: PackedBaseField,
}

/// A component that enforces the Fibonacci sequence.
/// Each row contains a seperate Fibonacci sequence of length `N`.
#[derive(Clone)]
#[repr(C)]
pub struct WideFibonacciEval<const N: usize> {
    pub eval_id: u32,
    pub log_n_rows: u32,
}
impl<const N: usize> FrameworkEval for WideFibonacciEval<N> {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let mut a = eval.next_trace_mask();
        let mut b = eval.next_trace_mask();
        for _ in 2..N {
            let c = eval.next_trace_mask();
            eval.add_constraint(c.clone() - (a.square() + b.square()));
            a = b;
            b = c;
        }
        eval
    }
}

pub fn generate_trace<const N: usize>(
    log_size: u32,
    inputs: &[FibInput],
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let mut trace = (0..N)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();
    for (vec_index, input) in inputs.iter().enumerate() {
        let mut a = input.a;
        let mut b = input.b;
        trace[0].data[vec_index] = a;
        trace[1].data[vec_index] = b;
        trace.iter_mut().skip(2).for_each(|col| {
            (a, b) = (b, a.square() + b.square());
            col.data[vec_index] = b;
        });
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::stwo_cuda::bindings;
use num_traits::One;
pub fn generate_cuda_trace<const N_COLUMNS: usize>(
    log_n_instances: u32,
) -> ColumnVec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
    assert!(log_n_instances >= LOG_N_LANES);
    let inputs = (0..(1 << (log_n_instances - LOG_N_LANES)))
        .map(|i| FibInput {
            a: PackedBaseField::one(),
            b: PackedBaseField::from_array(std::array::from_fn(|j| {
                BaseField::from_u32_unchecked((i * 16 + j) as u32)
            })),
        })
        .collect_vec();

    let trace = (0..N_COLUMNS)
        .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << log_n_instances))
        .collect_vec();

    let input_a: Vec<_> = inputs.iter()
        .flat_map(|input| input.a.to_array().to_vec())
        .collect();
    let input_b: Vec<_> = inputs.iter()
        .flat_map(|input| input.b.to_array().to_vec())
        .collect();

    let input_a_dev = BaseFieldVec::from_vec(input_a);
    let input_b_dev = BaseFieldVec::from_vec(input_b);

    let traces_vec = trace
        .iter()
        .map(|column_evaluations| column_evaluations.device_ptr)
        .collect_vec();

    unsafe {
        bindings::generate_wide_fibonacci_trace(
            input_a_dev.device_ptr,
            input_b_dev.device_ptr,
            1 << log_n_instances,
            traces_vec.as_ptr(),
            N_COLUMNS as u32,
            N_COLUMNS as u32,
        );
    }

    let domain = CanonicCoset::new(log_n_instances).circle_domain();
    let traces = trace
        .into_iter()
        .map(|eval| CircleEvaluation::new(domain, eval))
        .collect();
    traces
}

pub fn cuda_prove_wide_fibonacci<const N: usize>(
    log_n_instances: u32,
    config: PcsConfig,
) -> (WideFibonacciComponent<N>, StarkProof<Blake2sMerkleHasher>) {
    // Precompute twiddles.
    let twiddles = CudaBackend::precompute_twiddles(
        CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol.
    let prover_channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<CudaBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    // Preprocessed trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([]);
    tree_builder.commit(prover_channel);

    // Trace.
    let trace = generate_cuda_trace::<N>(log_n_instances);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(prover_channel);

    // Prove constraints.
    let component = WideFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        WideFibonacciEval::<N> {
            eval_id: fnv1a_eval_id_gen("fibonacci_example"),
            log_n_rows: log_n_instances,
        },
        SecureField::zero(),
    );

    let proof = prove::<CudaBackend, Blake2sMerkleChannel>(
        &[&component],
        prover_channel,
        commitment_scheme,
    ).unwrap();

    (component, proof)
}

pub fn simd_prove_wide_fibonacci<const N: usize>(
    log_n_instances: u32,
    config: PcsConfig,
) -> (WideFibonacciComponent<N>, StarkProof<Blake2sMerkleHasher>) {
    use stwo::prover::backend::simd::m31::LOG_N_LANES;

    let inputs = if log_n_instances < LOG_N_LANES {
        let n_instances = 1 << log_n_instances;
        vec![FibInput {
            a: PackedBaseField::from_array(std::array::from_fn(|j| {
                if j < n_instances {
                    BaseField::one()
                } else {
                    BaseField::zero()
                }
            })),
            b: PackedBaseField::from_array(std::array::from_fn(|j| {
                if j < n_instances {
                    BaseField::from_u32_unchecked(j as u32)
                } else {
                    BaseField::zero()
                }
            })),
        }]
    } else {
        (0..(1 << (log_n_instances - LOG_N_LANES)))
            .map(|i| FibInput {
                a: PackedBaseField::one(),
                b: PackedBaseField::from_array(std::array::from_fn(|j| {
                    BaseField::from_u32_unchecked((i * 16 + j) as u32)
                })),
            })
            .collect_vec()
    };

    // Precompute twiddles.
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol.
    let prover_channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    // Preprocessed trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([]);
    tree_builder.commit(prover_channel);

    // Trace.
    let trace = generate_trace::<N>(log_n_instances, &inputs);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(prover_channel);

    // Prove constraints.
    let component = WideFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        WideFibonacciEval::<N> {
            eval_id: fnv1a_eval_id_gen("fibonacci_example"),
            log_n_rows: log_n_instances,
        },
        SecureField::zero(),
    );

    let proof = prove::<SimdBackend, Blake2sMerkleChannel>(
        &[&component],
        prover_channel,
        commitment_scheme,
    ).unwrap();

    (component, proof)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn simd_prove_wide_fibonacci_poseidon<const N: usize>(
    log_n_instances: u32,
    config: PcsConfig,
) -> (WideFibonacciComponent<N>, StarkProof<stwo::core::vcs::poseidon252_merkle::Poseidon252MerkleHasher>) {
    use stwo::core::channel::Poseidon252Channel;
    use stwo::core::vcs::poseidon252_merkle::Poseidon252MerkleChannel;

    let inputs = if log_n_instances < LOG_N_LANES {
        let n_instances = 1 << log_n_instances;
        vec![FibInput {
            a: PackedBaseField::from_array(std::array::from_fn(|j| {
                if j < n_instances { BaseField::one() } else { BaseField::zero() }
            })),
            b: PackedBaseField::from_array(std::array::from_fn(|j| {
                if j < n_instances { BaseField::from_u32_unchecked(j as u32) } else { BaseField::zero() }
            })),
        }]
    } else {
        (0..(1 << (log_n_instances - LOG_N_LANES)))
            .map(|i| FibInput {
                a: PackedBaseField::one(),
                b: PackedBaseField::from_array(std::array::from_fn(|j| {
                    BaseField::from_u32_unchecked((i * 16 + j) as u32)
                })),
            })
            .collect_vec()
    };

    // Precompute twiddles.
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol.
    let prover_channel = &mut Poseidon252Channel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Poseidon252MerkleChannel>::new(config, &twiddles);

    // Preprocessed trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([]);
    tree_builder.commit(prover_channel);

    // Trace.
    let trace = generate_trace::<N>(log_n_instances, &inputs);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(prover_channel);

    // Prove constraints.
    let component = WideFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        WideFibonacciEval::<N> {
            eval_id: fnv1a_eval_id_gen("fibonacci_example"),
            log_n_rows: log_n_instances,
        },
        SecureField::zero(),
    );

    let proof = prove::<SimdBackend, Poseidon252MerkleChannel>(
        &[&component],
        prover_channel,
        commitment_scheme,
    ).unwrap();

    (component, proof)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn cuda_prove_wide_fibonacci_poseidon<const N: usize>(
    log_n_instances: u32,
    config: PcsConfig,
) -> (WideFibonacciComponent<N>, StarkProof<stwo::core::vcs::poseidon252_merkle::Poseidon252MerkleHasher>) {
    // Precompute twiddles.
    let twiddles = CudaBackend::precompute_twiddles(
        CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol.
    let prover_channel = &mut Poseidon252Channel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<CudaBackend, Poseidon252MerkleChannel>::new(config, &twiddles);

    // Preprocessed trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([]);
    tree_builder.commit(prover_channel);

    // Trace.
    let trace = generate_cuda_trace::<N>(log_n_instances);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(prover_channel);

    // Prove constraints.
    let component = WideFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        WideFibonacciEval::<N> {
            eval_id: fnv1a_eval_id_gen("fibonacci_example"),
            log_n_rows: log_n_instances,
        },
        SecureField::zero(),
    );

    let proof = prove::<CudaBackend, Poseidon252MerkleChannel>(
        &[&component],
        prover_channel,
        commitment_scheme,
    ).unwrap();

    (component, proof)
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};

    use super::WideFibonacciEval;
    use stwo_constraint_framework::{
        assert_constraints_on_polys, AssertEvaluator, FrameworkEval,
    };
    use stwo::core::air::Component;
    use stwo::prover::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::Column;
    use stwo::core::channel::Blake2sChannel;
    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::channel::Poseidon252Channel;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::prover::poly::circle::CircleEvaluation;
    use stwo::prover::poly::BitReversedOrder;
    use stwo::core::verifier::verify;
    use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::vcs::poseidon252_merkle::Poseidon252MerkleChannel;
    use stwo::core::ColumnVec;
    use crate::wide_fibonacci::{generate_trace, FibInput};

    const FIB_SEQUENCE_LENGTH: usize = 100;

    fn generate_test_trace(
        log_n_instances: u32,
    ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        if log_n_instances < LOG_N_LANES {
            let n_instances = 1 << log_n_instances;
            let inputs = vec![FibInput {
                a: PackedBaseField::from_array(std::array::from_fn(|j| {
                    if j < n_instances {
                        BaseField::one()
                    } else {
                        BaseField::zero()
                    }
                })),
                b: PackedBaseField::from_array(std::array::from_fn(|j| {
                    if j < n_instances {
                        BaseField::from_u32_unchecked((j) as u32)
                    } else {
                        BaseField::zero()
                    }
                })),
            }];
            return generate_trace::<FIB_SEQUENCE_LENGTH>(log_n_instances, &inputs);
        }
        let inputs = (0..(1 << (log_n_instances - LOG_N_LANES)))
            .map(|i| FibInput {
                a: PackedBaseField::one(),
                b: PackedBaseField::from_array(std::array::from_fn(|j| {
                    BaseField::from_u32_unchecked((i * 16 + j) as u32)
                })),
            })
            .collect_vec();
        generate_trace::<FIB_SEQUENCE_LENGTH>(log_n_instances, &inputs)
    }

    fn fibonacci_constraint_evaluator<const N: u32>(eval: AssertEvaluator<'_>) {
        WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
            eval_id: 1,
            log_n_rows: N
        }.evaluate(eval);
    }

    #[test]
    fn test_wide_fibonacci_constraints() {
        const LOG_N_INSTANCES: u32 = 6;
        let traces = TreeVec::new(vec![vec![], generate_test_trace(LOG_N_INSTANCES)]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            SecureField::zero(),
        );
    }

    #[test]
    #[should_panic]
    fn test_wide_fibonacci_constraints_fails() {
        const LOG_N_INSTANCES: u32 = 6;

        let mut trace = generate_test_trace(LOG_N_INSTANCES);
        // Modify the trace such that a constraint fail.
        trace[17].values.set(2, BaseField::one());
        let traces = TreeVec::new(vec![vec![], trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            SecureField::zero(),
        );
    }


    #[test_log::test]
    fn test_wide_fib_prove_with_blake_simd() {
        use crate::utils::get_env_var;
        use std::time::Instant;
        let min_log = get_env_var("MIN_LOG", 8u32);
        let max_log = get_env_var("MAX_LOG", 23u32);

        for log_n_instances in min_log..=max_log {
            let config = PcsConfig::default();
            let start = Instant::now();
            let (component, proof) = super::simd_prove_wide_fibonacci::<FIB_SEQUENCE_LENGTH>(
                log_n_instances,
                config,
            );
            println!(
                "wide-fibonacci(simd, blake2s) proving for 2^{} took {} ms",
                log_n_instances,
                start.elapsed().as_millis()
            );

            // Verify.
            let verifier_channel = &mut Blake2sChannel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_wide_fib_prove_with_poseidon_simd() {
        use crate::utils::get_env_var;
        use std::time::Instant;
        let min_log = get_env_var("MIN_LOG", 8u32);
        let max_log = get_env_var("MAX_LOG", 23u32);

        for log_n_instances in min_log..=max_log {
            let config = PcsConfig::default();
            let start = Instant::now();
            let (component, proof) = super::simd_prove_wide_fibonacci_poseidon::<FIB_SEQUENCE_LENGTH>(
                log_n_instances,
                config,
            );
            println!(
                "wide-fibonacci(simd, poseidon) proving for 2^{} took {} ms",
                log_n_instances,
                start.elapsed().as_millis()
            );

            // Verify using Poseidon channel and merkle.
            let verifier_channel = &mut Poseidon252Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(proof.config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }


    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_wide_fib_prove_with_poseidon_cuda() {
        use crate::utils::get_env_var;
        use std::time::Instant;
        let min_log = get_env_var("MIN_LOG", 8u32);
        let max_log = get_env_var("MAX_LOG", 23u32);

        for log_n_instances in min_log..=max_log {
            let config = PcsConfig::default();
            let start = Instant::now();
            let (component, proof) = super::cuda_prove_wide_fibonacci_poseidon::<FIB_SEQUENCE_LENGTH>(
                log_n_instances,
                config,
            );
            println!(
                "wide-fibonacci(cuda, poseidon) proving for 2^{} took {} ms",
                log_n_instances,
                start.elapsed().as_millis()
            );

            // Verify using Poseidon channel and merkle.
            let verifier_channel = &mut Poseidon252Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(proof.config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }


    #[test_log::test]
    fn test_wide_fib_prove_with_blake_cuda() {
        use crate::utils::get_env_var;
        use std::time::Instant;
        let min_log = get_env_var("MIN_LOG", 8u32);
        let max_log = get_env_var("MAX_LOG", 23u32);

        for log_n_instances in min_log..=max_log {
            let config = PcsConfig::default();
            let start = Instant::now();
            let (component, proof) = super::cuda_prove_wide_fibonacci::<FIB_SEQUENCE_LENGTH>(
                log_n_instances,
                config,
            );
            println!(
                "wide-fibonacci(cuda, blake2s) proving for 2^{} took {} ms",
                log_n_instances,
                start.elapsed().as_millis()
            );

            // Verify.
            let verifier_channel = &mut Blake2sChannel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }
}
