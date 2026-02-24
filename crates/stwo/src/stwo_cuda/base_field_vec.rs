use std::ffi::c_void;

use crate::core::fields::m31::BaseField;

use super::bindings;

#[derive(Debug)]
pub struct BaseFieldVec {
    pub device_ptr: *const u32,
    pub size: usize,
    pub owns_memory: bool,
}
unsafe impl Send for BaseFieldVec {}
unsafe impl Sync for BaseFieldVec {}

impl BaseFieldVec {
    pub fn new(device_ptr: *const u32, size: usize) -> Self {
        Self { device_ptr, size, owns_memory: true }
    }

    /// Create a BaseFieldVec that references existing device memory without owning it.
    /// The caller is responsible for ensuring the memory outlives this BaseFieldVec.
    /// Drop will NOT free the memory. Clone will produce an owned copy.
    pub fn from_borrowed_ptr(device_ptr: *const u32, size: usize) -> Self {
        Self { device_ptr, size, owns_memory: false }
    }

    pub fn from_vec(host_array: Vec<BaseField>) -> Self {
        let device_ptr = unsafe {
            bindings::copy_uint32_t_vec_from_host_to_device(
                host_array.as_ptr() as *const u32,
                host_array.len() as u32,
            )
        };
        let size = host_array.len();
        Self::new(device_ptr, size)
    }

    pub fn new_uninitialized(size: usize) -> Self {
        let device_ptr = unsafe {
            bindings::cuda_malloc_uint32_t(size as u32)
        };
        Self::new(device_ptr, size)
    }

    pub fn new_zeroes(size: usize) -> Self {
        let device_ptr = unsafe {
            bindings::cuda_alloc_zeroes_uint32_t(size as u32)
        };
        Self::new(device_ptr, size)
    }

    pub fn get_data(&self, index: usize) -> BaseField {
        let value = unsafe {
            bindings::cuda_get_uint32_t(self.device_ptr as *const c_void, index)
        };
        BaseField::from_u32_unchecked(value)
    }

    pub fn batch_get(&self, indices: &[usize]) -> Vec<BaseField> {
        if indices.is_empty() {
            return Vec::new();
        }
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let mut result: Vec<BaseField> = Vec::with_capacity(indices.len());
        unsafe {
            result.set_len(indices.len());
            bindings::cuda_batch_get_uint32_t(
                self.device_ptr,
                result.as_mut_ptr() as *mut u32,
                indices_u32.as_ptr(),
                indices.len() as u32,
            );
        }
        result
    }

    pub fn set_data(&mut self, index: usize, value: BaseField) {
        unsafe {
            bindings::cuda_set_uint32_t(self.device_ptr as *const c_void, index, value.0);
        }
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert!(self.size >= other.size);
        unsafe {
            bindings::copy_uint32_t_vec_from_device_to_device(
                other.device_ptr,
                self.device_ptr,
                other.size as u32,
            );
        }
    }

    pub fn copy_from_offset(&mut self, other: &Self, offset: usize) {
        assert!(self.size >= other.size);
        unsafe {
            bindings::copy_uint32_t_vec_from_device_to_device_offset(
                other.device_ptr,
                self.device_ptr,
                other.size as u32,
                offset as u32,
            );
        }
    }

    pub fn to_vec(&self) -> Vec<BaseField> {
        let mut host_data: Vec<BaseField> = Vec::with_capacity(self.size);
        unsafe {
            host_data.set_len(self.size);
            bindings::copy_uint32_t_vec_from_device_to_host(
                self.device_ptr,
                host_data.as_mut_ptr() as *const u32,
                self.size as u32,
            );
        }
        host_data
    }
    pub fn extend(&mut self, other: &Self) {
        let new_size = self.size + other.size;
        let mut new_vec = Self::new_uninitialized(new_size);
        new_vec.copy_from(self);
        new_vec.copy_from_offset(other, self.size);
        *self = new_vec;
    }

    /// Pads the vector to the target size by filling with zeros.
    /// If the current size is already >= target_size, does nothing.
    pub fn pad_to_size(&mut self, target_size: usize) {
        if self.size >= target_size {
            return;
        }
        // Create new zeroed buffer of target size
        let mut new_vec = Self::new_zeroes(target_size);
        // Copy existing data
        new_vec.copy_from(self);
        // The remaining elements are already zero (from new_zeroes)
        *self = new_vec;
    }
}

impl Clone for BaseFieldVec {
    fn clone(&self) -> Self {
        let mut cloned = Self::new_uninitialized(self.size);
        cloned.copy_from(self);
        cloned
    }
}

impl Drop for BaseFieldVec {
    fn drop(&mut self) {
        if self.owns_memory {
            unsafe {
                bindings::cuda_free_memory(self.device_ptr as *const c_void);
            }
        }
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct Uint32Vec {
    pub device_ptr: *const u32,
    pub size: usize,
}
unsafe impl Send for Uint32Vec {}
unsafe impl Sync for Uint32Vec {}

#[allow(unused_variables)]
impl Uint32Vec {
    pub fn new(device_ptr: *const u32, size: usize) -> Self {
        Self { device_ptr, size }
    }

    pub fn from_vec(host_array: Vec<u32>) -> Self {
        let device_ptr = unsafe {
            bindings::copy_uint32_t_vec_from_host_to_device(
                host_array.as_ptr() as *const u32,
                host_array.len() as u32,
            )
        };
        let size = host_array.len();
        Self::new(device_ptr, size)
    }

    pub fn new_uninitialized(size: usize) -> Self {
        Self::new(unsafe { bindings::cuda_malloc_uint32_t(size as u32) }, size)
    }

    pub fn new_zeroes(size: usize) -> Self {
        Self::new(
            unsafe { bindings::cuda_alloc_zeroes_uint32_t(size as u32) },
            size,
        )
    }

    pub fn get_data(&self, index: usize) -> u32 {
        let value = unsafe {
            bindings::cuda_get_uint32_t(self.device_ptr as *const c_void, index)
        };
        value
    }

    pub fn set_data(&mut self, index: usize, value: u32) {
        unsafe {
            bindings::cuda_set_uint32_t(self.device_ptr as *const c_void, index, value);
        }
    }

    pub fn increase_at(&self, address: u32) {
        unsafe {
            bindings::cuda_increase_at(self.device_ptr as *const c_void, address)
        }
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert!(self.size >= other.size);
        unsafe {
            bindings::copy_uint32_t_vec_from_device_to_device(
                other.device_ptr,
                self.device_ptr,
                other.size as u32,
            );
        }
    }

    pub fn copy_from_offset(&mut self, other: &Self, offset: usize) {
        assert!(self.size >= other.size);
        unsafe {
            bindings::copy_uint32_t_vec_from_device_to_device_offset(
                other.device_ptr,
                self.device_ptr,
                other.size as u32,
                offset as u32,
            );
        }
    }

    pub fn to_vec(&self) -> Vec<u32> {
        let mut host_data: Vec<u32> = Vec::with_capacity(self.size);
        unsafe {
            host_data.set_len(self.size);
            bindings::copy_uint32_t_vec_from_device_to_host(
                self.device_ptr,
                host_data.as_mut_ptr() as *const u32,
                self.size as u32,
            );
        }
        host_data
    }

    pub fn extend(&mut self, other: &Self) {
        let new_size = self.size + other.size;
        let mut new_vec = Self::new_uninitialized(new_size);
        new_vec.copy_from(self);
        new_vec.copy_from_offset(other, self.size);
        *self = new_vec;
    }
}

impl Clone for Uint32Vec {
    fn clone(&self) -> Self {
        let mut cloned = Self::new_uninitialized(self.size);
        cloned.copy_from(self);
        cloned
    }
}

impl Drop for Uint32Vec {
    fn drop(&mut self) {
        unsafe { bindings::cuda_free_memory(self.device_ptr as *const c_void) };
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct Uint128Vec {
    pub device_ptr: *const u32,
    pub size: usize,
}
unsafe impl Send for Uint128Vec {}
unsafe impl Sync for Uint128Vec {}

#[allow(unused_variables)]
impl Uint128Vec {
    pub fn new(device_ptr: *const u32, size: usize) -> Self {
        Self { device_ptr, size }
    }

    pub fn from_vec(host_array: Vec<u128>) -> Self {
        let device_ptr = unsafe {
            bindings::copy_uint32_t_vec_from_host_to_device(
                host_array.as_ptr() as *const u32,
                (host_array.len() * 4) as u32,
            )
        };
        let size = host_array.len();
        Self::new(device_ptr, size)
    }

    pub fn new_uninitialized(size: usize) -> Self {
        Self::new(unsafe { bindings::cuda_malloc_uint32_t(4 * size as u32) }, size)
    }

    pub fn new_zeroes(size: usize) -> Self {
        Self::new(
            unsafe { bindings::cuda_alloc_zeroes_uint32_t(4 * size as u32) },
            size,
        )
    }

    // pub fn get_data(&self, index: usize) -> u32 {
    //     let value = unsafe {
    //         bindings::cuda_get_uint32_t(self.device_ptr as *const c_void, index)
    //     };
    //     value
    // }

    // pub fn set_data(&mut self, index: usize, value: u32) {
    //     unsafe {
    //         bindings::cuda_set_uint32_t(self.device_ptr as *const c_void, index, value);
    //     }
    // }

    pub fn copy_from(&mut self, other: &Self) {
        assert!(self.size >= other.size);
        unsafe {
            bindings::copy_uint32_t_vec_from_device_to_device(
                other.device_ptr,
                self.device_ptr,
                4 * other.size as u32,
            );
        }
    }

    pub fn to_vec(&self) -> Vec<u128> {
        let mut host_data: Vec<u128> = Vec::with_capacity(self.size);
        unsafe {
            host_data.set_len(self.size);
            bindings::copy_uint32_t_vec_from_device_to_host(
                self.device_ptr,
                host_data.as_mut_ptr() as *const u32,
                4 * self.size as u32,
            );
        }
        host_data
    }
}

impl Clone for Uint128Vec {
    fn clone(&self) -> Self {
        let mut cloned = Self::new_uninitialized(self.size);
        cloned.copy_from(self);
        cloned
    }
}

impl Drop for Uint128Vec {
    fn drop(&mut self) {
        unsafe { bindings::cuda_free_memory(self.device_ptr as *const c_void) };
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_constructor() {
        let size = 1 << 25;
        let host_data = (0..size).map(BaseField::from).collect::<Vec<_>>();
        let base_field_vec = BaseFieldVec::from_vec(host_data.clone());
        assert_eq!(base_field_vec.to_vec(), host_data);
        assert_eq!(base_field_vec.size, host_data.len());
    }

    #[test]
    fn test_zeroes() {
        let size = 64;
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        let new_zeroes = BaseFieldVec::new_zeroes(size);
        for a in new_zeroes.to_vec().iter() {
            assert_eq!(a, &BaseField::from(0));
        }
    }
}
