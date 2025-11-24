use serde::{Deserialize, Serialize};
use crate::prover::backend::{Backend, BackendForChannel, simd::SimdBackend};
use crate::core::{
    channel::{Blake2sChannel, Poseidon252Channel},
    proof_of_work::GrindOps,
    vcs::blake2_merkle::Blake2sMerkleChannel,
    vcs::poseidon252_merkle::Poseidon252MerkleChannel,
};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct CudaBackend;

impl Backend for CudaBackend {}

impl GrindOps<Blake2sChannel> for CudaBackend {
    fn grind(channel: &Blake2sChannel, pow_bits: u32) -> u64 {
        SimdBackend::grind(channel, pow_bits)
    }
}

impl BackendForChannel<Blake2sMerkleChannel> for CudaBackend {}

impl GrindOps<Poseidon252Channel> for CudaBackend {
    fn grind(channel: &Poseidon252Channel, pow_bits: u32) -> u64 {
        SimdBackend::grind(channel, pow_bits)
    }
}

impl BackendForChannel<Poseidon252MerkleChannel> for CudaBackend {}
