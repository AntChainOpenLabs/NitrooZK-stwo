//! AIR for Poseidon2 hash function from <https://eprint.iacr.org/2023/323.pdf>.

use std::ops::{Add, AddAssign, Mul, Sub};

use itertools::Itertools;
use num_traits::One;
use tracing::{info, span, Level};
use stwo_constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, Relation, RelationEntry,
    TraceLocationAllocator, fnv1a_eval_id_gen, LogupTraceGenerator,
};
use stwo::core::channel::Blake2sChannel;
#[cfg(not(target_arch = "wasm32"))]
use stwo::core::channel::Poseidon252Channel;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
#[cfg(not(target_arch = "wasm32"))]
use stwo::core::vcs::poseidon252_merkle::{Poseidon252MerkleChannel, Poseidon252MerkleHasher};
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove, CommitmentSchemeProver};

use stwo::stwo_cuda::bindings;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;

const N_LOG_INSTANCES_PER_ROW: usize = 0;
const N_INSTANCES_PER_ROW: usize = 1 << N_LOG_INSTANCES_PER_ROW;
const N_STATE: usize = 16;
const N_PARTIAL_ROUNDS: usize = 14;
const N_HALF_FULL_ROUNDS: usize = 4;
const FULL_ROUNDS: usize = 2 * N_HALF_FULL_ROUNDS;
const N_COLUMNS_PER_REP: usize = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS;
const N_COLUMNS: usize = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP;
const LOG_EXPAND: u32 = 2;
// TODO(shahars): Use poseidon's real constants.
const EXTERNAL_ROUND_CONSTS: [[BaseField; N_STATE]; 2 * N_HALF_FULL_ROUNDS] =
    [[BaseField::from_u32_unchecked(1234); N_STATE]; 2 * N_HALF_FULL_ROUNDS];
const INTERNAL_ROUND_CONSTS: [BaseField; N_PARTIAL_ROUNDS] =
    [BaseField::from_u32_unchecked(1234); N_PARTIAL_ROUNDS];

pub type PoseidonComponent = FrameworkComponent<PoseidonEval>;

relation!(PoseidonElements, N_STATE);

#[derive(Clone)]
#[repr(C)]
pub struct PoseidonEval {
    pub eval_id: u32,
    pub log_n_rows: u32,
    pub lookup_elements: PoseidonElements,
    pub claimed_sum: SecureField,
}
impl FrameworkEval for PoseidonEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + LOG_EXPAND
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        eval_poseidon_constraints(&mut eval, &self.lookup_elements);
        eval
    }
}

#[inline(always)]
/// Applies the M4 MDS matrix described in <https://eprint.iacr.org/2023/323.pdf> 5.1.
fn apply_m4<F>(x: [F; 4]) -> [F; 4]
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    let t0 = x[0].clone() + x[1].clone();
    let t02 = t0.clone() + t0.clone();
    let t1 = x[2].clone() + x[3].clone();
    let t12 = t1.clone() + t1.clone();
    let t2 = x[1].clone() + x[1].clone() + t1.clone();
    let t3 = x[3].clone() + x[3].clone() + t0.clone();
    let t4 = t12.clone() + t12.clone() + t3.clone();
    let t5 = t02.clone() + t02.clone() + t2.clone();
    let t6 = t3.clone() + t5.clone();
    let t7 = t2.clone() + t4.clone();
    [t6, t5, t7, t4]
}

/// Applies the external round matrix.
/// See <https://eprint.iacr.org/2023/323.pdf> 5.1 and Appendix B.
fn apply_external_round_matrix<F>(state: &mut [F; 16])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    // Applies circ(2M4, M4, M4, M4).
    for i in 0..4 {
        [
            state[4 * i],
            state[4 * i + 1],
            state[4 * i + 2],
            state[4 * i + 3],
        ] = apply_m4([
            state[4 * i].clone(),
            state[4 * i + 1].clone(),
            state[4 * i + 2].clone(),
            state[4 * i + 3].clone(),
        ]);
    }
    for j in 0..4 {
        let s =
            state[j].clone() + state[j + 4].clone() + state[j + 8].clone() + state[j + 12].clone();
        for i in 0..4 {
            state[4 * i + j] += s.clone();
        }
    }
}

// Applies the internal round matrix.
//   mu_i = 2^{i+1} + 1.
// See <https://eprint.iacr.org/2023/323.pdf> 5.2.
fn apply_internal_round_matrix<F>(state: &mut [F; 16])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    // TODO(shahars): Check that these coefficients are good according to section  5.3 of Poseidon2
    // paper.
    let sum = state[1..]
        .iter()
        .cloned()
        .fold(state[0].clone(), |acc, s| acc + s);
    state.iter_mut().enumerate().for_each(|(i, s)| {
        // TODO(andrew): Change to rotations.
        *s = s.clone() * BaseField::from_u32_unchecked(1 << (i + 1)) + sum.clone();
    });
}

fn pow5<F: FieldExpOps>(x: F) -> F {
    let x2 = x.clone() * x.clone();
    let x4 = x2.clone() * x2.clone();
    x4 * x.clone()
}

pub fn eval_poseidon_constraints<E: EvalAtRow>(eval: &mut E, lookup_elements: &PoseidonElements) {
    for _ in 0..N_INSTANCES_PER_ROW {
        let mut state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());

        // Require state lookup.
        let initial_state = state.clone();

        // 4 full rounds.
        (0..N_HALF_FULL_ROUNDS).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += EXTERNAL_ROUND_CONSTS[round][i];
            });
            apply_external_round_matrix(&mut state);
            // TODO(andrew) Apply round matrix after the pow5, as is the order in the paper.
            state = std::array::from_fn(|i| pow5(state[i].clone()));
            state.iter_mut().for_each(|s| {
                let m = eval.next_trace_mask();
                eval.add_constraint(s.clone() - m.clone());
                *s = m;
            });
        });

        // Partial rounds.
        (0..N_PARTIAL_ROUNDS).for_each(|round| {
            state[0] += INTERNAL_ROUND_CONSTS[round];
            apply_internal_round_matrix(&mut state);
            state[0] = pow5(state[0].clone());
            let m = eval.next_trace_mask();
            eval.add_constraint(state[0].clone() - m.clone());
            state[0] = m;
        });

        // 4 full rounds.
        (0..N_HALF_FULL_ROUNDS).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i];
            });
            apply_external_round_matrix(&mut state);
            state = std::array::from_fn(|i| pow5(state[i].clone()));
            state.iter_mut().for_each(|s| {
                let m = eval.next_trace_mask();
                eval.add_constraint(s.clone() - m.clone());
                *s = m;
            });
        });

        // Provide state lookups.
        eval.add_to_relation(RelationEntry::new(
            lookup_elements,
            E::EF::one(),
            &initial_state,
        ));
        eval.add_to_relation(RelationEntry::new(lookup_elements, -E::EF::one(), &state));
    }

    eval.finalize_logup_in_pairs();
}

pub struct LookupData {
    initial_state: [[BaseColumn; N_STATE]; N_INSTANCES_PER_ROW],
    final_state: [[BaseColumn; N_STATE]; N_INSTANCES_PER_ROW],
}
pub fn gen_trace(
    log_size: u32,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    LookupData,
) {
    let _span = span!(Level::INFO, "Generation").entered();
    assert!(log_size >= LOG_N_LANES);
    let mut trace = (0..N_COLUMNS)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();
    let mut lookup_data = LookupData {
        initial_state: std::array::from_fn(|_| {
            std::array::from_fn(|_| BaseColumn::zeros(1 << log_size))
        }),
        final_state: std::array::from_fn(|_| {
            std::array::from_fn(|_| BaseColumn::zeros(1 << log_size))
        }),
    };

    for vec_index in 0..(1 << (log_size - LOG_N_LANES)) {
        // Initial state.
        let mut col_index = 0;
        for rep_i in 0..N_INSTANCES_PER_ROW {
            let mut state: [_; N_STATE] = std::array::from_fn(|state_i| {
                PackedBaseField::from_array(std::array::from_fn(|i| {
                    BaseField::from_u32_unchecked((vec_index * 16 + i + state_i + rep_i) as u32)
                }))
            });
            state.iter().copied().for_each(|s| {
                trace[col_index].data[vec_index] = s;
                col_index += 1;
            });
            lookup_data.initial_state[rep_i]
                .iter_mut()
                .zip(state)
                .for_each(|(res, state_i)| res.data[vec_index] = state_i);

            // 4 full rounds.
            (0..N_HALF_FULL_ROUNDS).for_each(|round| {
                (0..N_STATE).for_each(|i| {
                    state[i] += PackedBaseField::broadcast(EXTERNAL_ROUND_CONSTS[round][i]);
                });
                apply_external_round_matrix(&mut state);
                state = std::array::from_fn(|i| pow5(state[i]));
                state.iter().copied().for_each(|s| {
                    trace[col_index].data[vec_index] = s;
                    col_index += 1;
                });
            });

            // Partial rounds.
            (0..N_PARTIAL_ROUNDS).for_each(|round| {
                state[0] += PackedBaseField::broadcast(INTERNAL_ROUND_CONSTS[round]);
                apply_internal_round_matrix(&mut state);
                state[0] = pow5(state[0]);
                trace[col_index].data[vec_index] = state[0];
                col_index += 1;
            });

            // 4 full rounds.
            (0..N_HALF_FULL_ROUNDS).for_each(|round| {
                (0..N_STATE).for_each(|i| {
                    state[i] += PackedBaseField::broadcast(
                        EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i],
                    );
                });
                apply_external_round_matrix(&mut state);
                state = std::array::from_fn(|i| pow5(state[i]));
                state.iter().copied().for_each(|s| {
                    trace[col_index].data[vec_index] = s;
                    col_index += 1;
                });
            });

            lookup_data.final_state[rep_i]
                .iter_mut()
                .zip(state)
                .for_each(|(res, state_i)| res.data[vec_index] = state_i);
        }
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_iter()
        .map(|eval| CircleEvaluation::new(domain, eval))
        .collect();
    (trace, lookup_data)
}

pub fn gen_interaction_trace(
    log_size: u32,
    lookup_data: LookupData,
    lookup_elements: &PoseidonElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let _span = span!(Level::INFO, "Generate interaction trace").entered();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    #[allow(clippy::needless_range_loop)]
    for rep_i in 0..N_INSTANCES_PER_ROW {
        let mut col_gen = logup_gen.new_col();
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let denom0: PackedSecureField = lookup_elements.combine(
                &lookup_data.initial_state[rep_i]
                    .each_ref()
                    .map(|s| s.data[vec_row]),
            );
            let denom1: PackedSecureField = lookup_elements.combine(
                &lookup_data.final_state[rep_i]
                    .each_ref()
                    .map(|s| s.data[vec_row]),
            );
            col_gen.write_frac(vec_row, denom1 - denom0, denom0 * denom1);
        }
        col_gen.finalize_col();
    }

    logup_gen.finalize_last()
}

pub fn prove_poseidon(
    log_n_instances: u32,
    config: PcsConfig,
) -> (PoseidonComponent, StarkProof<Blake2sMerkleHasher>) {
    assert!(log_n_instances >= N_LOG_INSTANCES_PER_ROW as u32);
    let log_n_rows = log_n_instances - N_LOG_INSTANCES_PER_ROW as u32;

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + LOG_EXPAND + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Setup protocol.
    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    let constant_trace = vec![];
    tree_builder.extend_evals(constant_trace);
    tree_builder.commit(channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let start = std::time::Instant::now();
    let (trace, lookup_data) = gen_trace(log_n_rows);
    stwo::bench_println!(
        "trace generation for 2^{:?} took {:?} ms",
        log_n_instances,
        start.elapsed().as_millis()
    );
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup elements.
    let lookup_elements = PoseidonElements::draw(channel);
    // println!("cpu lookup_elements: {:?}", lookup_elements);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let start = std::time::Instant::now();
    let (trace, claimed_sum) = gen_interaction_trace(log_n_rows, lookup_data, &lookup_elements);
    stwo::bench_println!(
        "interaction trace generation for 2^{:?} took {:?} ms",
        log_n_instances,
        start.elapsed().as_millis()
    );
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Prove constraints.
    let component = PoseidonComponent::new(
        &mut TraceLocationAllocator::default(),
        PoseidonEval {
            eval_id: fnv1a_eval_id_gen("poseidon_example"),
            log_n_rows,
            lookup_elements,
            claimed_sum,
        },
        claimed_sum,
    );
    info!("Poseidon component info:\n{}", component);
    let start = std::time::Instant::now();
    let proof = prove(&[&component], channel, commitment_scheme).unwrap();
    stwo::bench_println!(
        "Poseidon proof generation for 2^{:?} took {:?} ms",
        log_n_instances,
        start.elapsed().as_millis()
    );

    (component, proof)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn prove_poseidon_poseidon(
    log_n_instances: u32,
    config: PcsConfig,
) -> (PoseidonComponent, StarkProof<Poseidon252MerkleHasher>) {
    assert!(log_n_instances >= N_LOG_INSTANCES_PER_ROW as u32);
    let log_n_rows = log_n_instances - N_LOG_INSTANCES_PER_ROW as u32;

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + LOG_EXPAND + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Setup protocol with Poseidon commit channel.
    let channel = &mut Poseidon252Channel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<_, Poseidon252MerkleChannel>::new(config, &twiddles);

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    let constant_trace = vec![];
    tree_builder.extend_evals(constant_trace);
    tree_builder.commit(channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let start = std::time::Instant::now();
    let (trace, lookup_data) = gen_trace(log_n_rows);
    stwo::bench_println!(
        "poseidon(simd) trace generation for 2^{:?} took {:?} ms",
        log_n_instances,
        start.elapsed().as_millis()
    );
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup elements.
    let lookup_elements = PoseidonElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let start = std::time::Instant::now();
    let (trace, claimed_sum) = gen_interaction_trace(log_n_rows, lookup_data, &lookup_elements);
    stwo::bench_println!(
        "poseidon(simd) interaction trace generation for 2^{:?} took {:?} ms",
        log_n_instances,
        start.elapsed().as_millis()
    );
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Prove constraints.
    let component = PoseidonComponent::new(
        &mut TraceLocationAllocator::default(),
        PoseidonEval {
            eval_id: fnv1a_eval_id_gen("poseidon_example"),
            log_n_rows,
            lookup_elements,
            claimed_sum,
        },
        claimed_sum,
    );

    let start = std::time::Instant::now();
    let proof: StarkProof<Poseidon252MerkleHasher> =
        prove(&[&component], channel, commitment_scheme).unwrap();
    stwo::bench_println!(
        "poseidon(simd) proving for 2^{:?} took {:?} ms",
        log_n_instances,
        start.elapsed().as_millis()
    );

    (component, proof)
}

pub struct CudaLookupData {
    initial_state: [[BaseFieldVec; N_STATE]; N_INSTANCES_PER_ROW],
    final_state: [[BaseFieldVec; N_STATE]; N_INSTANCES_PER_ROW],
}

pub fn gen_cuda_interaction_trace(
    log_size: u32,
    lookup_data: CudaLookupData,
    lookup_elements: &PoseidonElements,
)  -> (
    ColumnVec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);
    let lookup_init_vec = lookup_data.initial_state
        .iter()
        .flat_map(|row| row.iter().map(|base_vec| base_vec.device_ptr))
        .collect_vec();

    let lookup_final_vec = lookup_data.final_state
        .iter()
        .flat_map(|row| row.iter().map(|base_vec| base_vec.device_ptr))
        .collect_vec();

    let interaction_trace = (0.. 4 * N_INSTANCES_PER_ROW)
        .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();

    let interaction_trace_vec = interaction_trace
        .iter()
        .map(|column_evaluations| column_evaluations.device_ptr)
        .collect_vec();

    unsafe {
        let lookup_elements_ptr = lookup_elements as *const _ as *mut std::os::raw::c_void;
            bindings::generate_poseidon_interaction_traces(
            lookup_elements_ptr,
            lookup_init_vec.as_ptr(),
            lookup_final_vec.as_ptr(),
            log_size,
            interaction_trace_vec.as_ptr(),
            cuda_claimed_sum.device_ptr,
        );
    }

    // interaction_trace.clone().into_iter().enumerate().for_each(|(i, column_evaluations)| {
    //     println!("trace:[{}], {:?}", i, column_evaluations.to_cpu());
    // });

    let claimed_sum_vec = cuda_claimed_sum.to_cpu();
    let claimed_sum =  SecureField::from_m31_array([claimed_sum_vec[0], claimed_sum_vec[1], claimed_sum_vec[2], claimed_sum_vec[3]]);

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = interaction_trace
        .into_iter()
        .map(|eval| CircleEvaluation::new(domain, eval))
        .collect();
    (trace, claimed_sum)
}

pub fn gen_cuda_trace(
    log_size: u32,
) -> (
    ColumnVec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>,
    CudaLookupData,
) {
    let _span = span!(Level::INFO, "Cuda Trace Generation").entered();
    assert!(log_size >= LOG_N_LANES);
    let trace = (0..N_COLUMNS)
        .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();
    let lookup_data = CudaLookupData {
        initial_state: std::array::from_fn(|_| {
            std::array::from_fn(|_| BaseFieldVec::zeros(1 << log_size))
        }),
        final_state: std::array::from_fn(|_| {
            std::array::from_fn(|_| BaseFieldVec::zeros(1 << log_size))
        }),
    };

    let traces_vec = trace
        .iter()
        .map(|column_evaluations| column_evaluations.device_ptr)
        .collect_vec();

    let lookup_init_vec = lookup_data.initial_state
        .iter()
        .flat_map(|row| row.iter().map(|base_vec| base_vec.device_ptr))
        .collect_vec();

    let lookup_final_vec = lookup_data.final_state
        .iter()
        .flat_map(|row| row.iter().map(|base_vec| base_vec.device_ptr))
        .collect_vec();

    unsafe {
        bindings::generate_poseidon_traces(
            traces_vec.as_ptr(),
            lookup_init_vec.as_ptr(),
            lookup_final_vec.as_ptr(),
            log_size as u32,
        );
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_iter()
        .map(|eval| CircleEvaluation::new(domain, eval))
        .collect();
    (trace, lookup_data)
}

pub fn cuda_prove_poseidon(
    log_n_instances: u32,
    config: PcsConfig,
) -> (PoseidonComponent, StarkProof<Blake2sMerkleHasher>) {
    assert!(log_n_instances >= N_LOG_INSTANCES_PER_ROW as u32);
    let log_n_rows = log_n_instances - N_LOG_INSTANCES_PER_ROW as u32;

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = CudaBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + LOG_EXPAND + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    span.exit();
    // Setup protocol.
    let prover_channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<CudaBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    let constant_trace = vec![];
    tree_builder.extend_evals(constant_trace);
    tree_builder.commit(prover_channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let start = std::time::Instant::now();
    let (trace, lookup_data) = gen_cuda_trace(log_n_rows);
    stwo::bench_println!(
        "cuda trace generation for 2^{:?} took {:?} ms",
        log_n_instances,
        start.elapsed().as_millis()
    );

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(prover_channel);
    span.exit();

    // Draw lookup elements.
    let lookup_elements = PoseidonElements::draw(prover_channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let start = std::time::Instant::now();
    let (trace, claimed_sum) = gen_cuda_interaction_trace(log_n_rows, lookup_data, &lookup_elements);
    // println!("cuda claimed_sum: {:?}", claimed_sum);
    stwo::bench_println!(
        "cuda interaction trace generation for 2^{:?} took {:?} ms",
        log_n_instances,
        start.elapsed().as_millis()
    );
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(prover_channel);
    span.exit();

    // Prove constraints.
    let component = PoseidonComponent::new(
        &mut TraceLocationAllocator::default(),
        PoseidonEval {
            eval_id: fnv1a_eval_id_gen("poseidon_example"),
            log_n_rows,
            lookup_elements,
            claimed_sum,
        },
        claimed_sum,
    );

    let start = std::time::Instant::now();
    let proof: StarkProof<Blake2sMerkleHasher> =  prove(&[&component], prover_channel, commitment_scheme).unwrap();
    stwo::bench_println!(
        "cuda proving for 2^{:?} took {:?} ms",
        log_n_instances,
        start.elapsed().as_millis()
    );

    (component, proof)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn cuda_prove_poseidon_poseidon(
    log_n_instances: u32,
    config: PcsConfig,
) -> (PoseidonComponent, StarkProof<Poseidon252MerkleHasher>) {
    assert!(log_n_instances >= N_LOG_INSTANCES_PER_ROW as u32);
    let log_n_rows = log_n_instances - N_LOG_INSTANCES_PER_ROW as u32;

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = CudaBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + LOG_EXPAND + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    span.exit();
    // Setup protocol with Poseidon commit channel.
    let prover_channel = &mut Poseidon252Channel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<CudaBackend, Poseidon252MerkleChannel>::new(config, &twiddles);

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    let constant_trace = vec![];
    tree_builder.extend_evals(constant_trace);
    tree_builder.commit(prover_channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let start = std::time::Instant::now();
    let (trace, lookup_data) = gen_cuda_trace(log_n_rows);
    stwo::bench_println!(
        "poseidon(cuda) trace generation for 2^{:?} took {:?} ms",
        log_n_instances,
        start.elapsed().as_millis()
    );

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(prover_channel);
    span.exit();

    // Draw lookup elements.
    let lookup_elements = PoseidonElements::draw(prover_channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let start = std::time::Instant::now();
    let (trace, claimed_sum) = gen_cuda_interaction_trace(log_n_rows, lookup_data, &lookup_elements);
    stwo::bench_println!(
        "poseidon(cuda) interaction trace generation for 2^{:?} took {:?} ms",
        log_n_instances,
        start.elapsed().as_millis()
    );
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(prover_channel);
    span.exit();

    // Prove constraints.
    let component = PoseidonComponent::new(
        &mut TraceLocationAllocator::default(),
        PoseidonEval {
            eval_id: fnv1a_eval_id_gen("poseidon_example"),
            log_n_rows,
            lookup_elements,
            claimed_sum,
        },
        claimed_sum,
    );

    let start = std::time::Instant::now();
    let proof: StarkProof<Poseidon252MerkleHasher> =
        prove(&[&component], prover_channel, commitment_scheme).unwrap();
    stwo::bench_println!(
        "poseidon(cuda) proving for 2^{:?} took {:?} ms",
        log_n_instances,
        start.elapsed().as_millis()
    );

    (component, proof)
}

#[cfg(test)]
mod tests {
    use std::array;

    use itertools::Itertools;
    use stwo::core::air::Component;
    use stwo::core::channel::Blake2sChannel;
    use stwo::core::fields::m31::M31;
    use stwo::core::fri::FriConfig;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use stwo::core::verifier::verify;
    use stwo_constraint_framework::assert_constraints_on_polys;

    use crate::poseidon::{
        apply_internal_round_matrix, apply_m4, eval_poseidon_constraints, gen_interaction_trace,
        gen_trace, prove_poseidon, PoseidonElements,
    };

    #[cfg(all(target_family = "wasm", not(target_os = "wasi")))]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn test_poseidon_prove_wasm() {
        const LOG_N_INSTANCES: u32 = 10;
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 1, 64),
        };

        // Prove.
        prove_poseidon(LOG_N_INSTANCES, config);
    }

    #[test]
    fn test_apply_m4() {
        let m4 = ndarray::arr2(&[
            [5, 7, 1, 3].map(M31),
            [4, 6, 1, 1].map(M31),
            [1, 3, 5, 7].map(M31),
            [1, 1, 4, 6].map(M31),
        ]);
        let state = [0, 1, 2, 3].map(M31);
        let expected_dot = m4.dot(&ndarray::arr2(&[state]).t());
        let expected_dot: [_; 4] = expected_dot.into_raw_vec_and_offset().0.try_into().unwrap();

        let actual_dot = apply_m4(state);

        assert_eq!(expected_dot, actual_dot);
    }

    #[test]
    fn test_apply_internal() {
        const W: usize = 16;
        let mut state = array::from_fn(|i| M31((i * 3 + 187) as u32));
        let mut internal_matrix = ndarray::arr2(&[[M31(1); W]; W]);
        for (i, elem) in internal_matrix.diag_mut().iter_mut().enumerate() {
            *elem += M31((1 << (i + 1)) as u32);
        }
        let expected_state = internal_matrix.dot(&ndarray::arr2(&[state]).t());
        let expected_state: [_; W] = expected_state
            .into_raw_vec_and_offset()
            .0
            .try_into()
            .unwrap();

        apply_internal_round_matrix(&mut state);

        assert_eq!(state, expected_state);
    }

    #[test]
    fn test_poseidon_constraints() {
        const LOG_N_ROWS: u32 = 8;

        // Trace.
        let (trace0, interaction_data) = gen_trace(LOG_N_ROWS);
        let lookup_elements = PoseidonElements::dummy();
        let (trace1, claimed_sum) =
            gen_interaction_trace(LOG_N_ROWS, interaction_data, &lookup_elements);

        let traces = TreeVec::new(vec![vec![], trace0, trace1]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());
        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_ROWS),
            |mut eval| {
                eval_poseidon_constraints(&mut eval, &lookup_elements);
            },
            claimed_sum,
        );
    }

    #[test_log::test]
    fn test_simd_poseidon_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS="
        //   -C target-cpu=native -C target-feature=+avx512f -C opt-level=3" cargo test
        //   test_simd_poseidon_prove -- --nocapture

        // Get from environment variable:
        let min_log = get_env_var("MIN_LOG", 7u32);
        let max_log = get_env_var("MAX_LOG", 25u32);
        for log_n_instances in min_log..=max_log {
            let config = PcsConfig {
                pow_bits: 10,
                fri_config: FriConfig::new(5, 1, 64),
            };
            // let config = PcsConfig::default();

            // Prove.
            let (component, proof) = prove_poseidon(log_n_instances, config);

            // Verify.
            // TODO: Create Air instance independently.
            let channel = &mut Blake2sChannel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(proof.config);

            // Decommit.
            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();

            // Preprocessed columns.
            commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
            // Trace columns.
            commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
            // Draw lookup element.
            let lookup_elements = PoseidonElements::draw(channel);
            assert_eq!(lookup_elements, component.lookup_elements);
            // Interaction columns.
            commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

            verify(&[&component], channel, commitment_scheme, proof).unwrap();
        }
    }

    use crate::poseidon::cuda_prove_poseidon;
    use crate::utils::get_env_var;
    #[test_log::test]
    fn test_poseidon_prove_with_blake_cuda() {
        let min_log = get_env_var("MIN_LOG", 7u32);
        let max_log = get_env_var("MAX_LOG", 23u32);

        for log_n_instances in min_log..=max_log {
            // let config = PcsConfig::default();
            let config = PcsConfig {
                pow_bits: 10,
                fri_config: FriConfig::new(5, 1, 64),
            };

            let (component, proof) = cuda_prove_poseidon(log_n_instances, config);

            // Verify.
            let channel = &mut Blake2sChannel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(proof.config);

            // Decommit.
            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();

            // Preprocessed columns.
            commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
            // Trace columns.
            commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
            // Draw lookup element.
            let lookup_elements = PoseidonElements::draw(channel);
            assert_eq!(lookup_elements, component.lookup_elements);
            // Interaction columns.
            commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

            verify(&[&component], channel, commitment_scheme, proof).unwrap();
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::channel::Poseidon252Channel;
    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::vcs::poseidon252_merkle::Poseidon252MerkleChannel;
    use crate::poseidon::{prove_poseidon_poseidon, cuda_prove_poseidon_poseidon};

    #[test_log::test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_simd_poseidon_prove_with_poseidon() {
        let min_log = get_env_var("MIN_LOG", 7u32);
        let max_log = get_env_var("MAX_LOG", 25u32);
        for log_n_instances in min_log..=max_log {
            let config = PcsConfig {
                pow_bits: 10,
                fri_config: FriConfig::new(5, 1, 64),
            };

            let (component, proof) = prove_poseidon_poseidon(log_n_instances, config);

            let channel = &mut Poseidon252Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(proof.config);
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
            let lookup_elements = PoseidonElements::draw(channel);
            assert_eq!(lookup_elements, component.lookup_elements);
            commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);
            verify(&[&component], channel, commitment_scheme, proof).unwrap();
        }
    }

    #[test_log::test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_poseidon_prove_with_poseidon_cuda() {
        let min_log = get_env_var("MIN_LOG", 7u32);
        let max_log = get_env_var("MAX_LOG", 23u32);
        for log_n_instances in min_log..=max_log {
            let config = PcsConfig {
                pow_bits: 10,
                fri_config: FriConfig::new(5, 1, 64),
            };

            let (component, proof) = cuda_prove_poseidon_poseidon(log_n_instances, config);

            let channel = &mut Poseidon252Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(proof.config);
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
            let lookup_elements = PoseidonElements::draw(channel);
            assert_eq!(lookup_elements, component.lookup_elements);
            commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);
            verify(&[&component], channel, commitment_scheme, proof).unwrap();
        }
    }

    #[test]
    fn poseidon_constraints_regression() {
        // This test verifies that Poseidon constraints are satisfied by the generated trace.
        // It serves as a regression test for the constraint evaluation logic.
        const LOG_N_ROWS: u32 = 4; // Small size for quick testing: 2^4 = 16 rows

        // Generate trace and interaction data using SIMD backend
        let (trace0, interaction_data) = gen_trace(LOG_N_ROWS);
        let lookup_elements = PoseidonElements::dummy();
        let (trace1, claimed_sum) =
            gen_interaction_trace(LOG_N_ROWS, interaction_data, &lookup_elements);

        // Verify constraints on the generated trace
        let traces = TreeVec::new(vec![vec![], trace0, trace1]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        // This will panic if constraints don't hold
        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_ROWS),
            |mut eval| {
                eval_poseidon_constraints(&mut eval, &lookup_elements);
            },
            claimed_sum,
        );

        println!("Poseidon constraints regression test passed for log_size={}", LOG_N_ROWS);
    }


    #[test]
    fn poseidon_constraints_expr_count() {
        // This test verifies Poseidon constraint count using ExprEvaluator.
        // Note: For complex AIRs like Poseidon, random_assignment() and degree_bounds()
        // can be very slow. This test only builds the expression tree.
        use stwo_constraint_framework::expr::ExprEvaluator;
        use stwo_constraint_framework::{FrameworkEval, fnv1a_eval_id_gen};
        use crate::poseidon::{PoseidonEval, PoseidonElements};
        use stwo::core::fields::qm31::QM31;
        use num_traits::Zero;

        const LOG_N_ROWS: u32 = 4;

        // Create evaluator with dummy lookup elements
        let eval = PoseidonEval {
            eval_id: fnv1a_eval_id_gen("poseidon_example"),
            log_n_rows: LOG_N_ROWS,
            lookup_elements: PoseidonElements::dummy(),
            claimed_sum: QM31::zero(),
        };

        // Build symbolic expression tree
        let expr_eval = eval.evaluate(ExprEvaluator::new());

        // Verify constraint count (regression test - will fail if constraints change)
        assert_eq!(expr_eval.constraints.len(), 143);

        println!("✓ Poseidon symbolic constraint expressions built successfully");
        println!("  Total constraints: {}", expr_eval.constraints.len());
    }

    #[test]
    fn poseidon_constraints_structure_test() {
        // This test verifies constraint structure without generating trace.
        // It uses InfoEvaluator to count constraints and analyze the AIR structure.
        use stwo_constraint_framework::{InfoEvaluator, FrameworkEval, fnv1a_eval_id_gen};
        use crate::poseidon::{PoseidonEval, PoseidonElements, N_INSTANCES_PER_ROW, N_COLUMNS_PER_REP};
        use stwo::core::fields::qm31::SecureField;
        use num_traits::Zero;

        const LOG_N_ROWS: u32 = 4;

        // Create evaluator with dummy lookup elements
        let lookup_elements = PoseidonElements::dummy();
        let eval = PoseidonEval {
            eval_id: fnv1a_eval_id_gen("poseidon_example"),
            log_n_rows: LOG_N_ROWS,
            lookup_elements: lookup_elements.clone(),
            claimed_sum: SecureField::zero(), // Doesn't matter for structure analysis
        };

        // Use InfoEvaluator to analyze constraint structure
        let info = eval.evaluate(InfoEvaluator::empty());

        // Verify constraint count
        // Poseidon has:
        // - 4 full rounds with 16 constraints each = 64
        // - 14 partial rounds with 1 constraint each = 14
        // - 4 more full rounds with 16 constraints each = 64
        // - finalize_logup_in_pairs with 2 entries batched = 1 logup constraint
        // Total: 64 + 14 + 64 + 1 = 143 constraints per instance
        const EXPECTED_CONSTRAINTS: usize = 143 * N_INSTANCES_PER_ROW;
        assert_eq!(
            info.n_constraints, EXPECTED_CONSTRAINTS,
            "Expected {} constraints, got {}",
            EXPECTED_CONSTRAINTS, info.n_constraints
        );

        // Verify logup counts (should have 2 relation entries per instance)
        let total_logup_entries: usize = info.logup_counts.iter().map(|(_, count)| count).sum();
        const EXPECTED_LOGUP_ENTRIES: usize = 2 * N_INSTANCES_PER_ROW;
        assert_eq!(
            total_logup_entries, EXPECTED_LOGUP_ENTRIES,
            "Expected {} logup entries, got {}",
            EXPECTED_LOGUP_ENTRIES, total_logup_entries
        );

        // Verify mask offsets structure
        // Should have ORIGINAL_TRACE_IDX and INTERACTION_TRACE_IDX
        assert!(
            info.mask_offsets.len() >= 2,
            "Expected at least 2 trace interactions (original + interaction)"
        );

        // Verify original trace has the right number of columns
        // N_COLUMNS = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP
        // N_COLUMNS_PER_REP = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS
        //                   = 16 * (1 + 8) + 14 = 144 + 14 = 158
        const EXPECTED_TRACE_COLS: usize = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP;
        if let Some(original_masks) = info.mask_offsets.get(stwo_constraint_framework::ORIGINAL_TRACE_IDX) {
            assert_eq!(
                original_masks.len(), EXPECTED_TRACE_COLS,
                "Expected {} trace columns, got {}",
                EXPECTED_TRACE_COLS, original_masks.len()
            );
        }

        println!("✓ Poseidon constraint structure verified:");
        println!("  - Constraints: {}", info.n_constraints);
        println!("  - Logup entries: {}", total_logup_entries);
        println!("  - Trace columns: {}", info.mask_offsets.get(1).map_or(0, |m| m.len()));
        println!("  - Arithmetic operations:");
        println!("    * EF×EF: {}", info.arithmetic_counts.n_ef_mul_ef);
        println!("    * EF×F:  {}", info.arithmetic_counts.n_ef_mul_f);
        println!("    * F×F:   {}", info.arithmetic_counts.n_f_mul_f);
    }

    #[test]
    fn cuda_poseidon_constraints_direct() {
        // This test validates CUDA trace using assert_constraints_on_polys_cuda
        // which directly calls the CUDA kernel to compute constraint quotients.
        use crate::poseidon::{gen_cuda_trace, gen_cuda_interaction_trace, PoseidonEval};
        use stwo_constraint_framework::{assert_constraints_on_polys_cuda, fnv1a_eval_id_gen, FrameworkEval};
        use stwo::core::channel::Blake2sChannel;

        const LOG_N_ROWS: u32 = 4; // Small size: 2^4 = 16 rows
        const N_INSTANCES_PER_ROW: usize = 1;

        println!("Testing CUDA Poseidon constraints with assert_constraints_on_polys_cuda for log_size={}", LOG_N_ROWS);

        // Generate CUDA trace
        let (cuda_trace0, cuda_lookup_data) = gen_cuda_trace(LOG_N_ROWS);

        // Draw lookup elements from channel (not dummy) to match prover behavior
        let mut channel = Blake2sChannel::default();
        let lookup_elements = PoseidonElements::draw(&mut channel);

        // Generate CUDA interaction trace
        let (cuda_trace1, cuda_claimed_sum) =
            gen_cuda_interaction_trace(LOG_N_ROWS, cuda_lookup_data, &lookup_elements);

        // Build trace tree
        let cuda_traces = TreeVec::new(vec![vec![], cuda_trace0, cuda_trace1]);
        let cuda_trace_polys =
            cuda_traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        // Create eval structure for CUDA kernel
        let poseidon_eval = PoseidonEval {
            eval_id: fnv1a_eval_id_gen("poseidon_example"),
            log_n_rows: LOG_N_ROWS,
            lookup_elements: lookup_elements.clone(),
            claimed_sum: cuda_claimed_sum,
        };

        // Get number of constraints (from Poseidon: 16 states * multiple rounds)
        // Poseidon has:
        // - 4 full rounds with 16 constraints each = 64
        // - 14 partial rounds with 1 constraint each = 14
        // - 4 more full rounds with 16 constraints each = 64
        // - finalize_logup_in_pairs with 2 entries batched = 1 logup constraint
        // Total: 64 + 14 + 64 + 1 = 143 constraints per instance
        let n_constraints = 143 * N_INSTANCES_PER_ROW;
        let logup_counts = 2; // Two logup entries per instance (initial and final state)

        // Verify constraints using CUDA kernel with CPU reference verification
        // eval_log_size = trace_log_size + LOG_EXPAND
        let eval_log_size = poseidon_eval.max_constraint_log_degree_bound();
        assert_constraints_on_polys_cuda(
            &cuda_trace_polys,
            CanonicCoset::new(LOG_N_ROWS),
            eval_log_size,
            &poseidon_eval,  // Pass evaluator directly, not as pointer
            n_constraints,
            cuda_claimed_sum,
            logup_counts,
        );

        println!("✓ CUDA constraints verified with assert_constraints_on_polys_cuda");
        println!("CUDA claimed sum: {:?}", cuda_claimed_sum);
    }

}
