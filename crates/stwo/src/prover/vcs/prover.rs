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

        let mut last_layer_queries: Vec<usize> = vec![];
        for layer_log_size in (0..self.layers.len() as u32).rev() {
            let layer_columns = columns_by_layer
                .peek_take_while(|column| column.len().ilog2() == layer_log_size)
                .collect_vec();
            let previous_layer_hashes = self.layers.get(layer_log_size as usize + 1);

            // === Pass 1: Compute all indices on CPU (no GPU access) ===

            // Build set of previous-layer query indices for O(1) membership checks.
            let prev_queries_set: HashSet<usize> =
                last_layer_queries.iter().copied().collect();

            // Parent queries: each prev query q maps to parent node q/2.
            let parent_queries: BTreeSet<usize> =
                last_layer_queries.iter().map(|&q| q / 2).collect();

            // Column queries for this layer.
            let column_queries_set: HashSet<usize> = queries_per_log_size
                .get(&layer_log_size)
                .map(|v| v.iter().copied().collect())
                .unwrap_or_default();

            // Merge parent queries and column queries (sorted, deduplicated).
            let mut merged: BTreeSet<usize> = parent_queries;
            merged.extend(column_queries_set.iter().copied());
            let merged_nodes: Vec<usize> = merged.into_iter().collect();

            // Classify each node and build index arrays for batch fetches.
            let mut children_to_fetch: Vec<usize> = Vec::with_capacity(merged_nodes.len() * 2);
            let mut witness_positions: Vec<usize> = vec![];
            let mut queried_column_nodes: Vec<usize> = vec![];
            let mut witness_column_nodes: Vec<usize> = vec![];

            for (i, &node_index) in merged_nodes.iter().enumerate() {
                if previous_layer_hashes.is_some() {
                    children_to_fetch.push(2 * node_index);
                    children_to_fetch.push(2 * node_index + 1);

                    // Track which children positions are witnesses (not in prev queries).
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

            // === Pass 2: Batch fetch from GPU ===

            let mut all_node_values_for_layer =
                HashMap::<usize, <H as MerkleHasher>::Hash>::new();

            if let Some(prev_hashes) = previous_layer_hashes {
                // Single batch fetch for all children hashes.
                let children = prev_hashes.batch_at(&children_to_fetch);

                // Build all_node_values from the batch result.
                for (i, &node_index) in merged_nodes.iter().enumerate() {
                    all_node_values_for_layer.insert(2 * node_index, children[2 * i]);
                    all_node_values_for_layer
                        .insert(2 * node_index + 1, children[2 * i + 1]);
                }

                // Extract hash witnesses from the same batch result.
                for &pos in &witness_positions {
                    decommitment.hash_witness.push(children[pos]);
                }
            }

            all_node_values.push(all_node_values_for_layer);

            // Batch fetch column values for queried nodes (one batch per column).
            if !queried_column_nodes.is_empty() && !layer_columns.is_empty() {
                let batched: Vec<Vec<BaseField>> = layer_columns
                    .iter()
                    .map(|col| col.batch_at(&queried_column_nodes))
                    .collect();
                for node_idx in 0..queried_column_nodes.len() {
                    for col_values in &batched {
                        queried_values.push(col_values[node_idx]);
                    }
                }
            }

            // Batch fetch column values for witness nodes (one batch per column).
            if !witness_column_nodes.is_empty() && !layer_columns.is_empty() {
                let batched: Vec<Vec<BaseField>> = layer_columns
                    .iter()
                    .map(|col| col.batch_at(&witness_column_nodes))
                    .collect();
                for node_idx in 0..witness_column_nodes.len() {
                    for col_values in &batched {
                        decommitment.column_witness.push(col_values[node_idx]);
                    }
                }
            }

            // Propagate all merged nodes as queries to the next (smaller) layer.
            last_layer_queries = merged_nodes;
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
