use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::prover::backend::{Column, ColumnOps};
use crate::core::{
    fields::{m31::BaseField, qm31::SecureField},
};
use itertools::izip;
use crate::stwo_cuda::{bindings, base_field_vec::BaseFieldVec, secure_field_vec::SecureFieldVec, blake_2s_hash_vec::Blake2sHashVec};
use crate::prover::backend::cuda::CudaBackend;
use crate::stwo_cuda as interface;

impl ColumnOps<BaseField> for CudaBackend {
    type Column = BaseFieldVec;

    fn bit_reverse_column(column: &mut Self::Column) {
        let size = column.len();
        assert!(size.is_power_of_two() && size < u32::MAX as usize);

        unsafe {
            interface::bindings::bit_reverse_base_field(column.device_ptr, size);
        }
    }
}

impl ColumnOps<SecureField> for CudaBackend {
    type Column = SecureFieldVec;
    fn bit_reverse_column(column: &mut Self::Column) {
        let size = column.len();
        assert!(size.is_power_of_two() && size < u32::MAX as usize);

        unsafe {
            interface::bindings::bit_reverse_secure_field(column.device_ptr, size);
        }
    }
}

impl Column<BaseField> for interface::base_field_vec::BaseFieldVec {
    fn zeros(len: usize) -> Self {
        Self::new_zeroes(len)
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        self.to_vec()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn at(&self, index: usize) -> BaseField {
        Self::get_data(self, index)
    }

    fn set(&mut self, _index: usize, _value: BaseField) {
        Self::set_data(self, _index, _value);
    }

    unsafe fn uninitialized(len: usize) -> Self {
        Self {
            device_ptr: bindings::cuda_malloc_uint32_t(len as u32),
            size: len,
        }
    }

}

impl FromIterator<BaseField> for BaseFieldVec {
    fn from_iter<T: IntoIterator<Item = BaseField>>(iter: T) -> Self {
        let vec: Vec<BaseField> = iter.into_iter().collect();
        BaseFieldVec::from_vec(vec)
    }
}

impl IntoIterator for BaseFieldVec {
    type Item = BaseField;

    type IntoIter = std::vec::IntoIter<BaseField>;

    fn into_iter(self) -> Self::IntoIter {
        self.to_cpu().into_iter()
    }
}

impl Column<SecureField> for SecureFieldVec {
    fn zeros(_len: usize) -> Self {
        Self::new_zeroes(_len)
    }

    fn to_cpu(&self) -> Vec<SecureField> {
        self.to_vec()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn at(&self, _index: usize) -> SecureField {
        Self::get_data(self, _index)
    }

    fn set(&mut self, _index: usize, _value: SecureField) {
        todo!()
    }

    unsafe fn uninitialized(len: usize) -> Self {
        Self {
            device_ptr: bindings::cuda_malloc_uint32_t(4 * len as u32),
            size: len,
        }
    }
}

impl FromIterator<SecureField> for SecureFieldVec {
    fn from_iter<T: IntoIterator<Item = SecureField>>(_iter: T) -> Self {
        todo!()
    }
}

impl Column<Blake2sHash> for Blake2sHashVec {
    fn zeros(len: usize) -> Self {
        Self::new_zeroes(len)
    }

    fn to_cpu(&self) -> Vec<Blake2sHash> {
        self.to_vec()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn at(&self, index: usize) -> Blake2sHash {
        Self::get_data(self, index)
    }

    fn set(&mut self, _index: usize, _value: Blake2sHash) {
        todo!()
    }

    unsafe fn uninitialized(len: usize) -> Self {
        Self {
            device_ptr: bindings::cuda_malloc_blake_2s_hash(len),
            size: len,
        }
    }

    /// Optimized batch retrieval for GPU backend
    fn batch_at(&self, indices: &[usize]) -> Vec<Blake2sHash> {
        self.batch_get(indices)
    }
}

impl FromIterator<Blake2sHash> for Blake2sHashVec {
    fn from_iter<T: IntoIterator<Item = Blake2sHash>>(_iter: T) -> Self {
        todo!()
    }
}
use crate::prover::secure_column::SecureColumnByCoords;
impl SecureColumnByCoords<CudaBackend> {

    pub fn to_vec(&self) -> Vec<SecureField> {
        izip!(
            self.columns[0].to_cpu(),
            self.columns[1].to_cpu(),
            self.columns[2].to_cpu(),
            self.columns[3].to_cpu(),
        )
        .map(|(a, b, c, d)| SecureField::from_m31_array([a, b, c, d]))
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::prover::backend::{Column, ColumnOps, CpuBackend};
    use crate::core::{
        fields::{m31::BaseField, qm31::SecureField},
    };

    use crate::prover::backend::cuda::CudaBackend;
    use crate::stwo_cuda::{base_field_vec::BaseFieldVec, secure_field_vec::SecureFieldVec};


    #[test]
    fn test_bit_reverse_base_field() {
        let size: usize = 1 << 10;
        let column_data = (0..size as u32).map(BaseField::from).collect::<Vec<_>>();
        let mut expected_result = column_data.clone();
        CpuBackend::bit_reverse_column(&mut expected_result);

        let mut column = BaseFieldVec::from_vec(column_data);
        <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut column);

        assert_eq!(column.to_cpu(), expected_result);
    }

    #[test]
    fn test_bit_reverse_secure_field() {
        let size: usize = 1 << 16;

        let from_raw = (1..(size + 1) as u32).collect::<Vec<u32>>();
        let from_cpu = from_raw
            .chunks(4)
            .map(|a| SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect::<Vec<_>>();
        let mut array_expected = from_cpu.clone();

        CpuBackend::bit_reverse_column(&mut array_expected);

        let mut array = SecureFieldVec::from_vec(from_cpu.clone());
        <CudaBackend as ColumnOps<SecureField>>::bit_reverse_column(&mut array);

        assert_eq!(array.to_cpu(), array_expected);
    }
}
