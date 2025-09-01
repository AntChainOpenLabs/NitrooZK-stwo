use std::ffi::c_void;

use crate::core::backend::ColumnOps;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use crate::core::vcs::ops::MerkleOps;

use crate::stwo_cuda::{bindings, base_field_vec::BaseFieldVec, blake_2s_hash_vec::Blake2sHashVec};
use crate::core::backend::cuda::CudaBackend;

impl ColumnOps<Blake2sHash> for CudaBackend {
    type Column = Blake2sHashVec;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Blake2sMerkleHasher> for CudaBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Blake2sHashVec>,
        columns: &[&BaseFieldVec],
    ) -> Blake2sHashVec {
        let size = 1 << log_size;
        let number_of_columns = columns.len();

        let result: Blake2sHashVec = Blake2sHashVec::new_uninitialized(size);
        unsafe {
            Self::commit_on_layer_using_gpu(
                size,
                number_of_columns,
                columns,
                prev_layer,
                result.device_ptr,
            );
        }

        result
    }
}

impl CudaBackend {
    unsafe fn commit_on_layer_using_gpu(
        size: usize,
        number_of_columns: usize,
        columns: &[&BaseFieldVec],
        prev_layer: Option<&Blake2sHashVec>,
        result_pointer: *const Blake2sHash,
    ) {
        let device_column_pointers_vector: Vec<*const u32> =
            columns.iter().map(|column| column.device_ptr).collect();

        let device_column_pointers: *const *const u32 =
            bindings::copy_device_pointer_vec_from_host_to_device(
                device_column_pointers_vector.as_ptr(),
                number_of_columns,
            );

        if let Some(previous_layer) = prev_layer {
            // println!("commit_on_layer_with_previous");
            bindings::commit_on_layer_with_previous(
                size,
                number_of_columns,
                device_column_pointers,
                previous_layer.device_ptr,
                result_pointer as *mut Blake2sHash,
            );
        } else {
            // println!("commit_on_first_layer");
            bindings::commit_on_first_layer(
                size,
                number_of_columns,
                device_column_pointers,
                result_pointer as *mut Blake2sHash,
            );
        }
        bindings::cuda_free_memory(device_column_pointers as *const c_void);
    }
}

#[cfg(test)]
mod tests {
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
    use crate::core::vcs::ops::MerkleOps;

    use crate::stwo_cuda::base_field_vec::BaseFieldVec;
    use crate::core::backend::cuda::CudaBackend;

    #[test]
    fn test_commit_on_first_layer_with_many_columns_compared_with_cpu() {
        let log_size = 16;
        let size = 1 << log_size;

        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(100, size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size,
            None,
            &cpu_columns_vector.iter().collect::<Vec<_>>(),
        );
        let result = CudaBackend::commit_on_layer(
            log_size,
            None,
            &gpu_columns_vector.iter().collect::<Vec<_>>(),
        );

        // println!("CUDA Result: {:?}", result.to_cpu());
        assert_eq!(result.to_cpu(), expected_result);
    }

    #[test]
    fn test_commit_on_layer_with_previous_layer_compared_with_cpu() {
        let current_layer_log_size = 10;
        let current_layer_size = 1 << current_layer_log_size;
        let previous_layer_log_size = current_layer_log_size + 1;
        let previous_layer_size = 1 << previous_layer_log_size;

        // First layer

        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(35, previous_layer_size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let cpu_previous_layer = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            previous_layer_log_size,
            None,
            &cpu_columns_vector.iter().collect::<Vec<_>>(),
        );
        let gpu_previous_layer = CudaBackend::commit_on_layer(
            previous_layer_log_size,
            None,
            &gpu_columns_vector.iter().collect::<Vec<_>>(),
        );

        // Current layer

        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(16, current_layer_size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            current_layer_log_size,
            Some(&cpu_previous_layer),
            &cpu_columns_vector.iter().collect::<Vec<_>>(),
        );
        let result = CudaBackend::commit_on_layer(
            current_layer_log_size,
            Some(&gpu_previous_layer),
            &gpu_columns_vector.iter().collect::<Vec<_>>(),
        );

        assert_eq!(result.to_cpu(), expected_result);
    }

    fn gpu_columns_from(columns: &Vec<Vec<BaseField>>) -> Vec<BaseFieldVec> {
        columns
            .clone()
            .into_iter()
            .map(|vector| BaseFieldVec::from_vec(vector))
            .collect()
    }

    fn columns_test_vector(
        number_of_columns: usize,
        size_of_columns: usize,
    ) -> Vec<Vec<BaseField>> {
        (0..number_of_columns)
            .map(|index_of_column| {
                (0..size_of_columns)
                    .map(|index_in_column| M31::from(index_in_column * index_of_column))
                    .collect()
            })
            .collect()
    }
}
