use core::iter::zip;

use itertools::Itertools;
use std_shims::Vec;

use super::super::circle::CirclePoint;
use super::super::fields::qm31::SecureField;
use super::super::fri::{CirclePolyDegreeBound, FriVerifier};
use super::quotients::{fri_answers, PointSample};
use super::utils::TreeVec;
use super::PcsConfig;
use std_shims::BTreeMap;

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::pcs::quotients::CommitmentSchemeProof;
use crate::core::pcs::utils::prepare_preprocessed_query_positions;
use crate::core::vcs::MerkleHasher;
use crate::core::vcs::verifier::MerkleVerifier;
use crate::core::verifier::VerificationError;
use crate::core::ColumnVec;

/// The verifier side of a FRI polynomial commitment scheme. See [super].
#[derive(Default)]
pub struct CommitmentSchemeVerifier<MC: MerkleChannel> {
    pub trees: TreeVec<MerkleVerifier<MC::H>>,
    pub config: PcsConfig,
}

impl<MC: MerkleChannel> CommitmentSchemeVerifier<MC> {
    pub fn new(config: PcsConfig) -> Self {
        Self {
            trees: TreeVec::default(),
            config,
        }
    }

    /// A [TreeVec<ColumnVec>] of the log sizes of each column in each commitment tree.
    fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        self.trees
            .as_ref()
            .map(|tree| tree.column_log_sizes.clone())
    }

    /// Reads a commitment from the prover.
    pub fn commit(
        &mut self,
        commitment: <MC::H as MerkleHasher>::Hash,
        log_sizes: &[u32],
        channel: &mut MC::C,
    ) {
        MC::mix_root(channel, commitment);
        let extended_log_sizes = log_sizes
            .iter()
            .map(|&log_size| log_size + self.config.fri_config.log_blowup_factor)
            .collect();
        let verifier = MerkleVerifier::new(commitment, extended_log_sizes);
        self.trees.push(verifier);
    }

    pub fn verify_values(
        &self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        proof: CommitmentSchemeProof<MC::H>,
        channel: &mut MC::C,
    ) -> Result<(), VerificationError> {
        channel.mix_felts(&proof.sampled_values.clone().flatten_cols());
        let random_coeff = channel.draw_secure_felt();
        // The lifting log size is the length of the longest column which has at least one sample
        // (i.e. a column which is actually used in the constraints). Usually, the only columns
        // that have an empty vector of samples are among the preprocessed columns.
        let lifting_log_size = self
            .column_log_sizes()
            .zip_cols(&sampled_points)
            .flatten()
            .iter()
            .filter(|(_, sampled_points)| !sampled_points.is_empty())
            .map(|(log_size, _)| *log_size)
            .max()
            .unwrap();

        let bound =
            CirclePolyDegreeBound::new(lifting_log_size - self.config.fri_config.log_blowup_factor);

        // FRI commitment phase on OODS quotients.
        let mut fri_verifier =
            FriVerifier::<MC>::commit(channel, self.config.fri_config, proof.fri_proof, bound)?;

        // Verify proof of work.
        if !channel.verify_pow_nonce(self.config.pow_bits, proof.proof_of_work) {
            return Err(VerificationError::ProofOfWork);
        }
        channel.mix_u64(proof.proof_of_work);
        // Get FRI query positions.
        let query_positions = fri_verifier.sample_query_positions(channel);
        let preprocessed_query_positions = prepare_preprocessed_query_positions(
            &query_positions,
            lifting_log_size,
            self.column_log_sizes()[0]
                .iter()
                .max()
                .copied()
                .unwrap_or_default(),
        );

        // Build the query positions tree: the preprocessed tree needs a different treatment than
        // the other trees.
        let query_positions_tree = TreeVec::new(
            self.trees
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i == 0 {
                        preprocessed_query_positions.as_slice()
                    } else {
                        query_positions.as_slice()
                    }
                })
                .collect::<Vec<_>>(),
        );
        // Verify decommitments.
        // Build queries_per_log_size for each tree from flat query positions and column log sizes.
        self.trees
            .as_ref()
            .zip_eq(proof.decommitments)
            .zip_eq(proof.queried_values.clone())
            .zip_eq(query_positions_tree)
            .map(
                |(((tree, decommitment), queried_values), query_positions)| {
                    let queries_per_log_size =
                        build_queries_per_log_size(&tree.column_log_sizes, query_positions);
                    tree.verify(&queries_per_log_size, queried_values, decommitment)
                },
            )
            .0
            .into_iter()
            .collect::<Result<(), _>>()?;
        // Answer FRI queries.
        let samples = sampled_points.zip_cols(proof.sampled_values).map_cols(
            |(sampled_points, sampled_values)| {
                zip(sampled_points, sampled_values)
                    .map(|(point, value)| PointSample { point, value })
                    .collect_vec()
            },
        );

        let fri_answers = fri_answers(
            self.column_log_sizes(),
            samples,
            random_coeff,
            &query_positions,
            proof.queried_values,
            lifting_log_size,
        )?;

        fri_verifier.decommit(fri_answers)?;

        Ok(())
    }
}

/// Builds a `BTreeMap<u32, Vec<usize>>` mapping each unique column log_size to sorted, deduplicated
/// query positions at that log_size. The input `query_positions` are at the max log_size; smaller
/// columns have their positions shifted right.
fn build_queries_per_log_size(
    column_log_sizes: &[u32],
    query_positions: &[usize],
) -> BTreeMap<u32, Vec<usize>> {
    let max_log_size = column_log_sizes.iter().copied().max().unwrap_or(0);
    let mut queries_per_log_size = BTreeMap::new();
    for &log_size in column_log_sizes {
        queries_per_log_size.entry(log_size).or_insert_with(|| {
            let shift = max_log_size - log_size;
            query_positions
                .iter()
                .map(|&pos| (pos >> (shift + 1) << 1) + (pos & 1))
                .sorted()
                .dedup()
                .collect()
        });
    }
    queries_per_log_size
}
