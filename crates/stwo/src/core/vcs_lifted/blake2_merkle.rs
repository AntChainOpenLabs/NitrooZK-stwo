use blake2::{Blake2s256, Digest};
use serde::{Deserialize, Serialize};

use super::merkle_hasher::MerkleHasherLifted;
use crate::core::channel::{Blake2sChannelGeneric, MerkleChannel};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::{reduce_to_m31, Blake2sHash, Blake2sHasherGeneric};
use crate::core::vcs::MerkleHasher;

pub type Blake2sMerkleHasherGeneric<const IS_M31_OUTPUT: bool> =
    Blake2sHasherGeneric<IS_M31_OUTPUT>;

pub type Blake2sMerkleHasher = Blake2sMerkleHasherGeneric<false>;
/// Same as [Blake2sMerkleHasher], except that the hash output is taken modulo M31::P.
pub type Blake2sM31MerkleHasher = Blake2sMerkleHasherGeneric<true>;

impl<const IS_M31_OUTPUT: bool> MerkleHasherLifted for Blake2sMerkleHasherGeneric<IS_M31_OUTPUT> {
    type Hash = Blake2sHash;

    fn hash_children(children_hashes: (Self::Hash, Self::Hash)) -> Self::Hash {
        let mut hasher = Self::default();
        let (left_child, right_child) = children_hashes;
        hasher.update(&left_child.0);
        hasher.update(&right_child.0);

        hasher.finalize()
    }

    fn update_leaf(&mut self, column_values: &[BaseField]) {
        column_values
            .iter()
            .for_each(|x| self.update(&x.0.to_le_bytes()));
    }

    fn finalize(self) -> Self::Hash {
        self.finalize()
    }
}

/// Also implement non-lifted MerkleHasher for the lifted hasher type, so it can be used
/// with proof types that require MerkleHasher (e.g. StarkProof, CommitmentSchemeProof).
impl<const IS_M31_OUTPUT: bool> MerkleHasher for Blake2sMerkleHasherGeneric<IS_M31_OUTPUT> {
    type Hash = Blake2sHash;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash {
        let mut hasher = Blake2s256::new();

        if let Some((left_child, right_child)) = children_hashes {
            hasher.update(left_child);
            hasher.update(right_child);
        }

        for value in column_values {
            hasher.update(value.0.to_le_bytes());
        }

        let mut r: [u8; 32] = hasher.finalize().into();
        if IS_M31_OUTPUT {
            r = reduce_to_m31(r);
        }

        Blake2sHash(r)
    }
}

pub type Blake2sMerkleChannel = Blake2sMerkleChannelGeneric<false>;
/// Same as [Blake2sMerkleChannel], expect that the hash output is taken modulo M31::P.
pub type Blake2sM31MerkleChannel = Blake2sMerkleChannelGeneric<true>;

#[derive(Default)]
pub struct Blake2sMerkleChannelGeneric<const IS_M31_OUTPUT: bool>;

/// Implement non-lifted MerkleChannel for the lifted channel type, so it can be used
/// with prove/verify functions that require MerkleChannel (with H: MerkleHasher).
impl<const IS_M31_OUTPUT: bool> MerkleChannel for Blake2sMerkleChannelGeneric<IS_M31_OUTPUT> {
    type C = Blake2sChannelGeneric<IS_M31_OUTPUT>;
    type H = Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash) {
        channel.update_digest(Blake2sHasherGeneric::<IS_M31_OUTPUT>::concat_and_hash(
            &channel.digest(),
            &root,
        ));
    }
}

/// Dummy implementations of `Serialize` and `Deserialize` for `Blake2sMerkleHasherGeneric` (we
/// cannot simply derive them because its inner field doesn't implement these traits and is from an
/// external crate).
/// Note: remove this code when possible.
impl<const IS_M31_OUTPUT: bool> Serialize for Blake2sMerkleHasherGeneric<IS_M31_OUTPUT> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let ser = serializer.serialize_struct("Blake2sMerkleHasherGeneric", 1)?;
        serde::ser::SerializeStruct::end(ser)
    }
}

impl<'de, const IS_M31_OUTPUT: bool> Deserialize<'de>
    for Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>
{
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::default())
    }
}
