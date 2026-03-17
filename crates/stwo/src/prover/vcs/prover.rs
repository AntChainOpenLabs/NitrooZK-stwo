use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use tracing::{span, Level};

use super::ops::MerkleOps;
use crate::core::fields::m31::BaseField;
use crate::core::utils::PeekableExt;
use crate::core::vcs::verifier::{
    ExtendedMerkleDecommitment, MerkleDecommitment, MerkleDecommitmentAux,
};
use crate::core::vcs::MerkleHasher;
use crate::prover::backend::{Col, Column};

pub struct MerkleProver<B: MerkleOps<H>, H: MerkleHasher> {
    /// Layers of the Merkle tree.
    /// The first layer is the root layer.
    /// The last layer is the largest layer.
    /// See [MerkleOps::commit_on_layer] for more details.
    pub layers: Vec<Col<B, H::Hash>>,
}
/// The MerkleProver struct represents a prover for a Merkle commitment scheme.
/// It is generic over the types `B` and `H`, which represent the Merkle operations and Merkle
/// hasher respectively.
impl<B: MerkleOps<H>, H: MerkleHasher> MerkleProver<B, H> {
    /// Commits to columns.
    /// Columns must be of power of 2 sizes.
    ///
    /// # Arguments
    ///
    /// * `columns` - A vector of references to columns.
    ///
    /// # Returns
    ///
    /// A new instance of `MerkleProver` with the committed layers.
    pub fn commit(columns: Vec<&Col<B, BaseField>>) -> Self {
        let _span = span!(Level::TRACE, "Merkle", class = "MerkleCommitment").entered();
        if columns.is_empty() {
            return Self {
                layers: vec![B::commit_on_layer(0, None, &[])],
            };
        }

        let columns = &mut columns
            .into_iter()
            .sorted_by_key(|c| Reverse(c.len()))
            .peekable();

        let mut layers: Vec<Col<B, H::Hash>> = Vec::new();

        let max_log_size = columns.peek().unwrap().len().ilog2();
        for log_size in (0..=max_log_size).rev() {
            // Take columns of the current log_size.
            let layer_columns = columns
                .peek_take_while(|column| column.len().ilog2() == log_size)
                .collect_vec();

            layers.push(B::commit_on_layer(log_size, layers.last(), &layer_columns));
        }
        layers.reverse();
        Self { layers }
    }

    /// Decommits to columns on the given queries.
    /// Queries are given as indices to the largest column.
    ///
    /// Uses a two-pass approach per layer to enable batch GPU fetches:
    /// Pass 1 computes all needed indices on CPU, Pass 2 batch-fetches from GPU.
    ///
    /// # Arguments
    ///
    /// * `queries_per_log_size` - Maps a log_size to a vector of queries for columns of that size.
    /// * `columns` - A vector of references to columns.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * A vector queried values sorted by the order they were queried from the largest layer to
    ///   the smallest.
    /// * A `MerkleDecommitment` containing the hash and column witnesses.
    pub fn decommit(
        &self,
        queries_per_log_size: &BTreeMap<u32, Vec<usize>>,
        columns: Vec<&Col<B, BaseField>>,
    ) -> (Vec<BaseField>, ExtendedMerkleDecommitment<H>) {
        let mut queried_values = vec![];
        let mut decommitment = MerkleDecommitment::empty();
        let mut all_node_values: Vec<HashMap<usize, <H as MerkleHasher>::Hash>> = vec![];

        // Sort columns by layer (largest first).
        let mut columns_by_layer = columns
            .iter()
            .sorted_by_key(|c| Reverse(c.len()))
            .peekable();

        // ================================================================
        // Phase 1: CPU-only index computation for ALL layers at once.
        // No GPU access — just build the index arrays.
        // ================================================================
        struct LayerPlan {
            layer_log_size: u32,
            has_prev_layer: bool,
            merged_nodes: Vec<usize>,
            children_to_fetch: Vec<usize>,
            witness_positions: Vec<usize>,
            queried_column_nodes: Vec<usize>,
            witness_column_nodes: Vec<usize>,
        }

        let mut plans: Vec<LayerPlan> = Vec::new();
        let mut last_layer_queries: Vec<usize> = vec![];

        for layer_log_size in (0..self.layers.len() as u32).rev() {
            let has_prev_layer = self.layers.get(layer_log_size as usize + 1).is_some();
            let prev_queries_set: HashSet<usize> =
                last_layer_queries.iter().copied().collect();
            let parent_queries: BTreeSet<usize> =
                last_layer_queries.iter().map(|&q| q / 2).collect();
            let column_queries_set: HashSet<usize> = queries_per_log_size
                .get(&layer_log_size)
                .map(|v| v.iter().copied().collect())
                .unwrap_or_default();

            let mut merged: BTreeSet<usize> = parent_queries;
            merged.extend(column_queries_set.iter().copied());
            let merged_nodes: Vec<usize> = merged.into_iter().collect();

            let mut children_to_fetch = Vec::with_capacity(merged_nodes.len() * 2);
            let mut witness_positions = vec![];
            let mut queried_column_nodes = vec![];
            let mut witness_column_nodes = vec![];

            for (i, &node_index) in merged_nodes.iter().enumerate() {
                if has_prev_layer {
                    children_to_fetch.push(2 * node_index);
                    children_to_fetch.push(2 * node_index + 1);
                    if !prev_queries_set.contains(&(2 * node_index)) {
                        witness_positions.push(2 * i);
                    }
                    if !prev_queries_set.contains(&(2 * node_index + 1)) {
                        witness_positions.push(2 * i + 1);
                    }
                }
                if column_queries_set.contains(&node_index) {
                    queried_column_nodes.push(node_index);
                } else {
                    witness_column_nodes.push(node_index);
                }
            }

            last_layer_queries = merged_nodes.clone();
            plans.push(LayerPlan {
                layer_log_size,
                has_prev_layer,
                merged_nodes,
                children_to_fetch,
                witness_positions,
                queried_column_nodes,
                witness_column_nodes,
            });
        }

        // ================================================================
        // Phase 2: Batch GPU fetches using pre-computed indices.
        // ================================================================
        for plan in &plans {
            let layer_columns = columns_by_layer
                .peek_take_while(|column| column.len().ilog2() == plan.layer_log_size)
                .collect_vec();

            let mut all_node_values_for_layer =
                HashMap::<usize, <H as MerkleHasher>::Hash>::new();

            if plan.has_prev_layer {
                let prev_hashes = &self.layers[plan.layer_log_size as usize + 1];
                let children = prev_hashes.batch_at(&plan.children_to_fetch);

                for (i, &node_index) in plan.merged_nodes.iter().enumerate() {
                    all_node_values_for_layer.insert(2 * node_index, children[2 * i]);
                    all_node_values_for_layer
                        .insert(2 * node_index + 1, children[2 * i + 1]);
                }
                for &pos in &plan.witness_positions {
                    decommitment.hash_witness.push(children[pos]);
                }
            }

            all_node_values.push(all_node_values_for_layer);

            if !plan.queried_column_nodes.is_empty() && !layer_columns.is_empty() {
                let col_refs: Vec<&Col<B, BaseField>> =
                    layer_columns.iter().map(|c| **c).collect();
                let gathered =
                    Col::<B, BaseField>::batch_at_multi(&col_refs, &plan.queried_column_nodes);
                queried_values.extend_from_slice(&gathered);
            }

            if !plan.witness_column_nodes.is_empty() && !layer_columns.is_empty() {
                let col_refs: Vec<&Col<B, BaseField>> =
                    layer_columns.iter().map(|c| **c).collect();
                let gathered =
                    Col::<B, BaseField>::batch_at_multi(&col_refs, &plan.witness_column_nodes);
                decommitment.column_witness.extend_from_slice(&gathered);
            }
        }

        (
            queried_values,
            ExtendedMerkleDecommitment {
                decommitment,
                aux: MerkleDecommitmentAux { all_node_values },
            },
        )
    }

    pub fn root(&self) -> H::Hash {
        self.layers.first().unwrap().at(0)
    }
}
