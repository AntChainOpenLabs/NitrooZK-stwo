use crate::stwo_cuda::bindings::CudaSecureField;
use crate::core::backend::cuda::CudaBackend;
use ark_std::log2;
// use cudart::slice::CudaVariable;
use itertools::Itertools;
use crate::core::{
    backend::{Column},
    circle::{CirclePoint, Coset},
    fields::{m31::BaseField, qm31::SecureField},
    poly::{
        circle::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps},
        twiddles::TwiddleTree,
        BitReversedOrder,
    },
    ColumnVec,
};

use std::ffi::c_void;
pub trait CudaVariable<T> {
    /// # Safety
    /// do not dereference if the memory is located on the device
    unsafe fn as_ref(&self) -> &T;

    fn as_ptr(&self) -> *const T {
        unsafe { self.as_ref() }
    }

    fn as_c_void_ptr(&self) -> *const c_void {
        self.as_ptr() as *const c_void
    }
}

pub trait CudaVariableMut<T>: CudaVariable<T> {
    /// # Safety
    /// do not dereference if the memory is located on the device
    unsafe fn as_mut(&mut self) -> &mut T;

    fn as_mut_ptr(&mut self) -> *mut T {
        unsafe { self.as_mut() }
    }

    fn as_mut_c_void_ptr(&mut self) -> *mut c_void {
        self.as_mut_ptr() as *mut c_void
    }
}

impl<T> CudaVariable<T> for T {
    unsafe fn as_ref(&self) -> &T {
        self
    }
}

impl<T> CudaVariableMut<T> for T {
    unsafe fn as_mut(&mut self) -> &mut T {
        self
    }
}


use crate::stwo_cuda as interface;
use crate::stwo_cuda::base_field_vec::BaseFieldVec;
use crate::core::backend::cpu::CpuCirclePoly;
pub(crate) type CudaCircleEvaluation<F, EvalOrder> =
    CircleEvaluation<CudaBackend, F, EvalOrder>;
// fn interpolate_native(
//     eval: CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>,
//     twiddle_tree: &TwiddleTree<CudaBackend>,
// ) -> CirclePoly<CudaBackend> {
//     let values = eval.values;
//     assert!(eval.domain.half_coset.is_doubling_of(twiddle_tree.root_coset));
//     unsafe {
//         interface::bindings::interpolate(
//             eval.domain.half_coset.size() as u32,
//             values.device_ptr,
//             twiddle_tree.itwiddles.device_ptr,
//             twiddle_tree.itwiddles.len() as u32,
//             values.len() as u32,
//         );
//     }

//     CirclePoly::new(values)
// }

// fn interpolate_columns_native(
//     columns: impl IntoIterator<Item = CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>,
//     twiddles: &TwiddleTree<CudaBackend>,
// ) -> Vec<CirclePoly<CudaBackend>> {
//     let columns = columns.into_iter().collect_vec();
//     let values = columns
//         .iter()
//         .map(|column| column.values.device_ptr)
//         .collect_vec();
//     let number_of_rows = columns[0].len();
//     unsafe {
//         interface::bindings::interpolate_columns(
//             columns[0].domain.half_coset.size() as u32,
//             values.as_ptr(),
//             twiddles.itwiddles.device_ptr,
//             twiddles.itwiddles.len() as u32,
//             columns.len() as u32,
//             number_of_rows as u32,
//         );
//     }

//     columns
//         .into_iter()
//         .map(|column| CirclePoly::new(column.values))
//         .collect_vec()
// }


pub fn evaluate_native(
    poly: &CirclePoly<CudaBackend>,
    domain: CircleDomain,
    twiddle_tree: &TwiddleTree<CudaBackend>,
) -> CircleEvaluation<CudaBackend, BaseField, BitReversedOrder> {
    let values = poly.extend(domain.log_size()).coeffs;
    assert!(domain.half_coset.is_doubling_of(twiddle_tree.root_coset));
    unsafe {
        interface::bindings::ntt_n2b_native_batch(
            values.device_ptr.as_ptr() as *mut *mut u32,
            log2(values.len()) as u32,
            1,
            1,
            log2(values.len()) as u32,
            twiddle_tree.twiddles.device_ptr,
            twiddle_tree.twiddles.len() as u32,
            domain.half_coset.size() as u32,
        );

    }
    CircleEvaluation::new(domain, values)
}

pub fn evaluate_polynomials_native(
    polynomials: &mut ColumnVec<CirclePoly<CudaBackend>>,
    log_blowup_factor: u32,
    twiddles: &TwiddleTree<CudaBackend>,
) -> Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
    let mut values_pointers = Vec::new();
    let mut values_columns = Vec::new();
    let mut column_sizes = Vec::new();
    let mut domains = Vec::new();
    let mut eval_domain_sizes = Vec::new();

    for poly in polynomials.iter() {
        let domain = CanonicCoset::new(poly.log_size() + log_blowup_factor).circle_domain();
        let values = poly.extend(domain.log_size()).coeffs;

        values_pointers.push(values.device_ptr);
        column_sizes.push(values.len() as u32);
        values_columns.push(values);
        domains.push(domain);
        eval_domain_sizes.push(domain.half_coset.size() as u32);
    }

    unsafe {
        interface::bindings::evaluate_columns(
            eval_domain_sizes.as_ptr(),
            values_pointers.as_ptr(),
            twiddles.twiddles.device_ptr,
            twiddles.twiddles.len() as u32,
            values_columns.len() as u32,
            column_sizes.as_ptr(),
        );
    }

    domains
        .into_iter()
        .zip(values_columns.into_iter())
        .map(|(domain, values)| {
            CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(domain, values)
        })
        .collect_vec()
}

use crate::core::backend::cpu::CpuCircleEvaluation;
use crate::core::backend::CpuBackend;
use std::mem::transmute;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

impl PolyOps for CudaBackend {
    type Twiddles = BaseFieldVec;

    // fn new_canonical_ordered(
    //     coset: CanonicCoset,
    //     values: Col<Self, BaseField>,
    // ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
    //     let size = values.len();
    //     let device_ptr = unsafe {
    //         interface::bindings::sort_values_and_permute_with_bit_reverse_order(values.device_ptr, size)
    //     };
    //     let result = BaseFieldVec::new(device_ptr, size);
    //     CircleEvaluation::new(coset.circle_domain(), result)
    // }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddle_tree: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        assert!(eval.domain.half_coset.is_doubling_of(twiddle_tree.root_coset));

        if eval.domain.log_size() <= 3 {
            let cpu_eval = CpuCircleEvaluation::new(eval.domain, eval.values.to_cpu());

            let cpu_circle_poly = CpuBackend::interpolate(
                cpu_eval,
                unsafe { transmute(twiddle_tree) },
            );

            let cuda_coeffs = BaseFieldVec::from_vec(cpu_circle_poly.coeffs.to_vec());

            return CirclePoly::<CudaBackend>::new(cuda_coeffs);
        }

        let values = eval.values;
        unsafe {
            interface::bindings::ntt_b2n_column(
                values.device_ptr.as_ptr() as *mut *mut u32,
                log2(values.len()) as u32,
                1 as u32,
                twiddle_tree.itwiddles.device_ptr,
                twiddle_tree.itwiddles.len() as u32,
                eval.domain.half_coset.size() as u32,
            );
        }

        CirclePoly::new(values)
    }
    fn interpolate_columns(
        columns: impl IntoIterator<Item = CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CirclePoly<Self>> {
        columns
            .into_iter()
            .map(|eval| eval.interpolate_with_twiddles(twiddles))
            .collect()
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
        unsafe {
            interface::bindings::eval_at_point(
                poly.coeffs.device_ptr,
                poly.coeffs.len() as u32,
                CudaSecureField::from(point.x),
                CudaSecureField::from(point.y),
            )
            .into()
        }
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        let new_size = 1 << log_size;
        assert!(
            new_size >= poly.coeffs.len(),
            "New size must be larger than the old size"
        );

        let mut new_coeffs = BaseFieldVec::new_zeroes(new_size);
        new_coeffs.copy_from(&poly.coeffs);
        CirclePoly::new(new_coeffs)
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddle_tree: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        assert!(domain.half_coset.is_doubling_of(twiddle_tree.root_coset));
        if domain.log_size() <= 3 {
            let cpu_poly = CpuCirclePoly::new(poly.coeffs.to_cpu());

            let cpu_circle_eval = CpuBackend::evaluate(
                &cpu_poly,
                domain,
                unsafe { transmute(twiddle_tree) },
            );

            let cuda_eval_values = BaseFieldVec::from_vec(cpu_circle_eval.values.to_vec());
            return CudaCircleEvaluation::new(cpu_circle_eval.domain, cuda_eval_values);
        }
        let values = poly.extend(domain.log_size()).coeffs;
        unsafe {
            interface::bindings::ntt_n2b_columns(
                values.device_ptr.as_ptr() as *mut *mut u32,
                log2(values.len()) as u32,
                1,
                twiddle_tree.twiddles.device_ptr,
                twiddle_tree.twiddles.len() as u32,
                domain.half_coset.size() as u32,
            );
        }

        CircleEvaluation::new(domain, values)
    }

    fn evaluate_polynomials(
        polynomials: &ColumnVec<CirclePoly<Self>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleEvaluation<Self, BaseField, BitReversedOrder>> {
        #[cfg(not(feature = "parallel"))]
        let res = polynomials
            .iter()
            .map(|poly| {
                poly.evaluate_with_twiddles(
                    CanonicCoset::new(poly.log_size() + log_blowup_factor).circle_domain(),
                    twiddles,
                )
            })
            .collect_vec();

        #[cfg(feature = "parallel")]
        let res = polynomials
            .par_iter()
            .map(|poly| {
                poly.evaluate_with_twiddles(
                    CanonicCoset::new(poly.log_size() + log_blowup_factor).circle_domain(),
                    twiddles,
                )
            })
            .collect();
        res
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        unsafe {
            let twiddles = BaseFieldVec::new(
                interface::bindings::precompute_twiddles(
                    coset.initial.into(),
                    coset.step.into(),
                    coset.size(),
                ),
                coset.size(),
            );
            let itwiddles = BaseFieldVec::new_uninitialized(coset.size());
            interface::bindings::batch_inverse_base_field(
                twiddles.device_ptr,
                itwiddles.device_ptr,
                coset.size(),
            );
            TwiddleTree {
                root_coset: coset,
                twiddles,
                itwiddles,
            }
        }
    }
}
#[cfg(test)]
mod tests {
    // use itertools::Itertools;
    use crate::core::poly::circle::{
        CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
    };
    use crate::core::poly::twiddles::TwiddleTree;
    use crate::core::{
        backend::{Column, CpuBackend},
        circle::{CirclePoint, CirclePointIndex, Coset, },
        fields::m31::BaseField,
        // ColumnVec,
    };
    use test_log::test;

    // use crate::core::backend::cuda::poly::evaluate_native;
    use crate::core::backend::cuda::{
        CudaBackend,
        poly::BitReversedOrder,
    };
    use crate::stwo_cuda::base_field_vec::BaseFieldVec;

    // use ark_std::start_timer;
    // use ark_std::end_timer;

    // #[test]
    // fn test_new_canonical_ordered() {
    //     let log_size = 4;
    //     let coset = CanonicCoset::new(log_size);
    //     let size: usize = 1 << log_size;
    //     let column_data = (0..size as u32).map(BaseField::from).collect::<Vec<_>>();
    //     let cpu_values = column_data.clone();
    //     let expected_result = CpuBackend::new_canonical_ordered(coset, cpu_values.clone());

    //     let column = BaseFieldVec::from_vec(column_data);
    //     let result = CudaBackend::new_canonical_ordered(coset, column);

    //     assert_eq!(result.values.to_cpu(), expected_result.values);
    //     assert_eq!(
    //         result.domain.iter().collect::<Vec<_>>(),
    //         expected_result.domain.iter().collect::<Vec<_>>()
    //     );
    // }

    #[test]
    fn test_precompute_twiddles() {
        let log_size = 5;

        let half_coset = CanonicCoset::new(log_size).half_coset();
        let expected_result = CpuBackend::precompute_twiddles(half_coset);
        let twiddles = CudaBackend::precompute_twiddles(half_coset);

        assert_eq!(twiddles.twiddles.to_cpu(), expected_result.twiddles);
        assert_eq!(twiddles.itwiddles.to_cpu(), expected_result.itwiddles);
        assert_eq!(
            twiddles.root_coset.iter().collect::<Vec<_>>(),
            expected_result.root_coset.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_extend() {
        let log_size = 20;
        let size = 1 << log_size;
        let new_log_size = log_size + 5;
        let cpu_coeffs = (0..size).map(BaseField::from).collect::<Vec<_>>();
        let cuda_coeffs = BaseFieldVec::from_vec(cpu_coeffs.clone());
        let cpu_poly = CirclePoly::<CpuBackend>::new(cpu_coeffs);
        let cuda_poly = CirclePoly::<CudaBackend>::new(cuda_coeffs);
        let result = CudaBackend::extend(&cuda_poly, new_log_size);
        let expected_result = CpuBackend::extend(&cpu_poly, new_log_size);
        assert_eq!(result.coeffs.to_cpu(), expected_result.coeffs);
        assert_eq!(result.log_size(), expected_result.log_size());
    }

    // #[test]
    // fn test_interpolate() {
    //     let log_size = 20;

    //     let size = 1 << log_size;

    //     let cpu_values = (1..(size + 1) as u32)
    //         .map(BaseField::from)
    //         .collect::<Vec<_>>();
    //     let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

    //     let coset = CanonicCoset::new(log_size);
    //     let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
    //     let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

    //     let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
    //     let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

    //     let timer = start_timer!(|| format!("cpu backend interpolate, log_n:{}", log_size));
    //     let expected_result = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
    //     end_timer!(timer);

    //     let timer = start_timer!(|| format!("gpu backend interpolate, log_n:{}", log_size));
    //     let result = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);
    //     end_timer!(timer);

    //     assert_eq!(result.coeffs.to_cpu(), expected_result.coeffs);
    // }

    // #[test]
    // fn test_interpolate_2() {
    //     let log_size = 5;

    //     let cpu_values = vec![
    //         BaseField::from(1),
    //         BaseField::from(443693538),
    //         BaseField::from(793699796),
    //         BaseField::from(1631104375),
    //         BaseField::from(460025527),
    //         BaseField::from(98131605),
    //         BaseField::from(1292025643),
    //         BaseField::from(1056169651),
    //         BaseField::from(29),
    //         BaseField::from(1645907698),
    //         BaseField::from(300234932),
    //         BaseField::from(2113642380),
    //         BaseField::from(2031046861),
    //         BaseField::from(541052612),
    //         BaseField::from(1857203558),
    //         BaseField::from(5),
    //         BaseField::from(2),
    //         BaseField::from(187770177),
    //         BaseField::from(1190378570),
    //         BaseField::from(1107054997),
    //         BaseField::from(1436440899),
    //         BaseField::from(1555024221),
    //         BaseField::from(2002021885),
    //         BaseField::from(866),
    //         BaseField::from(750797),
    //         BaseField::from(1704111751),
    //         BaseField::from(1874758341),
    //         BaseField::from(960394553),
    //         BaseField::from(1365348280),
    //         BaseField::from(376645196),
    //         BaseField::from(2119137245),
    //         BaseField::from(1),
    //     ];
    //     let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

    //     let coset = CanonicCoset::new(log_size);
    //     let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
    //     let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

    //     let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
    //     let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

    //     let expected_result = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
    //     let result = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

    //     assert_eq!(result.coeffs.to_cpu(), expected_result.coeffs);
    // }


    // #[test]
    // fn test_interpolate_3() {

    //     for log_size in 4..30 {
    //         println!("log_size:{}", log_size);

    //         let size = 1 << log_size;

    //         let cpu_values = (1..(size + 1) as u32)
    //             .map(BaseField::from)
    //             .collect::<Vec<_>>();
    //         let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

    //         let coset = CanonicCoset::new(log_size);
    //         let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
    //         let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

    //         let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
    //         let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

    //         assert_eq!(cpu_twiddles.twiddles.to_vec(), gpu_twiddles.twiddles.to_vec());

    //         let timer = start_timer!(|| format!("cpu backend interpolate, log_n:{}", log_size));
    //         let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
    //         end_timer!(timer);

    //         let timer = start_timer!(|| format!("optimize gpu backend interpolate, log_n:{}", log_size));
    //         let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);
    //         end_timer!(timer);

    //         assert_eq!(gpu_poly.coeffs.to_vec(), cpu_poly.coeffs.to_vec());

    //     }

    // }

    // #[test]
    // #[allow(unused_variables)]
    // fn test_evaluate() {
    //     for log_size in 13..26 {
    //         println!("log_size:{}", log_size);

    //         let size = 1 << log_size;

    //         let cpu_values = (1..(size + 1) as u32)
    //             .map(BaseField::from)
    //             .collect::<Vec<_>>();
    //         let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());
    //         let gpu_values_optim = BaseFieldVec::from_vec(cpu_values.clone());


    //         let coset = CanonicCoset::new(log_size);
    //         let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
    //         let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);
    //         let gpu_evaluations_optim = CudaBackend::new_canonical_ordered(coset, gpu_values_optim);

    //         let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
    //         let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

    //         assert_eq!(cpu_twiddles.twiddles.to_vec(), gpu_twiddles.twiddles.to_vec());

    //         let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
    //         let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);
    //         let gpu_poly_optim = CudaBackend::interpolate(gpu_evaluations_optim, &gpu_twiddles);
    //         assert_eq!(gpu_poly_optim.coeffs.to_vec(), cpu_poly.coeffs.to_vec());

    //         let timer = start_timer!(|| format!("cpu backend interpolate, log_n:{}", log_size));
    //         let expected_result = CpuBackend::evaluate(&cpu_poly, coset.circle_domain(), &cpu_twiddles);
    //         end_timer!(timer);

    //         let timer = start_timer!(|| format!("native gpu backend  interpolate, log_n:{}", log_size));
    //         let result = evaluate_native(&gpu_poly, coset.circle_domain(), &gpu_twiddles);
    //         end_timer!(timer);

    //         let timer = start_timer!(|| format!("optimize gpu backend interpolate, log_n:{}", log_size));
    //         let result_optim = CudaBackend::evaluate(&gpu_poly_optim, coset.circle_domain(), &gpu_twiddles);
    //         end_timer!(timer);

    //         assert_eq!(result_optim.values.to_cpu(), expected_result.values);
    //     }

    // }

    // #[test]
    // fn test_eval_at_point() {
    //     let log_size = 20;

    //     let size = 1 << log_size;
    //     let coset = CanonicCoset::new(log_size);
    //     let point = SECURE_FIELD_CIRCLE_GEN;

    //     let cpu_values = (1..(size + 1) as u32)
    //         .map(BaseField::from)
    //         .collect::<Vec<_>>();

    //     let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());
    //     let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);
    //     let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());
    //     let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);
    //     let result = CudaBackend::eval_at_point(&gpu_poly, point);

    //     let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
    //     let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
    //     let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);


    //     let expected_result = CpuBackend::eval_at_point(&cpu_poly, point.clone());

    //     assert_eq!(result, expected_result);
    // }

    #[test]
    fn test_interpolate_from_fib() {
        let eval = CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
            CircleDomain {
                half_coset: Coset {
                    initial_index: CirclePointIndex(33554432),
                    initial: CirclePoint {
                        x: BaseField::from(579625837),
                        y: BaseField::from(1690787918),
                    },
                    step_size: CirclePointIndex(134217728),
                    step: CirclePoint {
                        x: BaseField::from(590768354),
                        y: BaseField::from(978592373),
                    },
                    log_size: 4,
                },
            },
            BaseFieldVec::from_vec(vec![
                BaseField::from(1),
                BaseField::from(443693538),
                BaseField::from(793699796),
                BaseField::from(1631104375),
                BaseField::from(460025527),
                BaseField::from(98131605),
                BaseField::from(1292025643),
                BaseField::from(1056169651),
                BaseField::from(29),
                BaseField::from(1645907698),
                BaseField::from(300234932),
                BaseField::from(2113642380),
                BaseField::from(2031046861),
                BaseField::from(541052612),
                BaseField::from(1857203558),
                BaseField::from(5),
                BaseField::from(2),
                BaseField::from(187770177),
                BaseField::from(1190378570),
                BaseField::from(1107054997),
                BaseField::from(1436440899),
                BaseField::from(1555024221),
                BaseField::from(2002021885),
                BaseField::from(866),
                BaseField::from(750797),
                BaseField::from(1704111751),
                BaseField::from(1874758341),
                BaseField::from(960394553),
                BaseField::from(1365348280),
                BaseField::from(376645196),
                BaseField::from(2119137245),
                BaseField::from(1),
            ]),
        );
        let twiddles = vec![
            BaseField::from(785043271),
            BaseField::from(1260750973),
            BaseField::from(736262640),
            BaseField::from(1553669210),
            BaseField::from(479120236),
            BaseField::from(225856549),
            BaseField::from(197700101),
            BaseField::from(1079800039),
            BaseField::from(1911378744),
            BaseField::from(1577470940),
            BaseField::from(1334497267),
            BaseField::from(2085743640),
            BaseField::from(477953613),
            BaseField::from(125103457),
            BaseField::from(1977033713),
            BaseField::from(2005527287),
            BaseField::from(251924953),
            BaseField::from(636875771),
            BaseField::from(48903418),
            BaseField::from(1896945393),
            BaseField::from(1514613395),
            BaseField::from(870936612),
            BaseField::from(1297878576),
            BaseField::from(583555490),
            BaseField::from(640817200),
            BaseField::from(1702126977),
            BaseField::from(1054411686),
            BaseField::from(648593218),
            BaseField::from(1014093253),
            BaseField::from(2137011181),
            BaseField::from(81378258),
            BaseField::from(789857006),
            BaseField::from(838195206),
            BaseField::from(1774253895),
            BaseField::from(1739004854),
            BaseField::from(262191051),
            BaseField::from(206059115),
            BaseField::from(212443077),
            BaseField::from(1796741361),
            BaseField::from(883753057),
            BaseField::from(2140339328),
            BaseField::from(404685994),
            BaseField::from(9803698),
            BaseField::from(68458636),
            BaseField::from(14530030),
            BaseField::from(228509164),
            BaseField::from(1038945916),
            BaseField::from(134155457),
            BaseField::from(579625837),
            BaseField::from(1690787918),
            BaseField::from(1641940819),
            BaseField::from(2121318970),
            BaseField::from(1952787376),
            BaseField::from(1580223790),
            BaseField::from(1013961365),
            BaseField::from(280947147),
            BaseField::from(1179735656),
            BaseField::from(1241207368),
            BaseField::from(1415090252),
            BaseField::from(2112881577),
            BaseField::from(590768354),
            BaseField::from(978592373),
            BaseField::from(32768),
            BaseField::from(1),
        ];
        let itwiddles = vec![
            BaseField::from(1541158724),
            BaseField::from(16208603),
            BaseField::from(62823040),
            BaseField::from(1642210396),
            BaseField::from(1631996251),
            BaseField::from(1007591000),
            BaseField::from(1874949287),
            BaseField::from(1849862501),
            BaseField::from(781334166),
            BaseField::from(132945364),
            BaseField::from(1278220752),
            BaseField::from(214347122),
            BaseField::from(1165838173),
            BaseField::from(2054194025),
            BaseField::from(1234096940),
            BaseField::from(1721693449),
            BaseField::from(622651690),
            BaseField::from(1373671071),
            BaseField::from(82740187),
            BaseField::from(1683898894),
            BaseField::from(1918467639),
            BaseField::from(1186332607),
            BaseField::from(1296073347),
            BaseField::from(401388709),
            BaseField::from(1383565722),
            BaseField::from(656788371),
            BaseField::from(1787268380),
            BaseField::from(1809670981),
            BaseField::from(99372120),
            BaseField::from(765975505),
            BaseField::from(774809712),
            BaseField::from(348924564),
            BaseField::from(2029303208),
            BaseField::from(959596234),
            BaseField::from(1051468699),
            BaseField::from(721860568),
            BaseField::from(1767118503),
            BaseField::from(218253990),
            BaseField::from(1356867335),
            BaseField::from(1955048591),
            BaseField::from(559361447),
            BaseField::from(1046725194),
            BaseField::from(448375059),
            BaseField::from(1036402186),
            BaseField::from(2138687850),
            BaseField::from(1268642696),
            BaseField::from(1381082522),
            BaseField::from(559888787),
            BaseField::from(248349974),
            BaseField::from(969924856),
            BaseField::from(1461702947),
            BaseField::from(655012266),
            BaseField::from(1385854532),
            BaseField::from(1859156789),
            BaseField::from(349252128),
            BaseField::from(421110815),
            BaseField::from(1160411471),
            BaseField::from(1518526074),
            BaseField::from(490549293),
            BaseField::from(1942501404),
            BaseField::from(991237807),
            BaseField::from(775648038),
            BaseField::from(65536),
            BaseField::from(1),
        ];
        let root_coset = Coset {
            initial_index: CirclePointIndex(8388608),
            initial: CirclePoint {
                x: BaseField::from(785043271),
                y: BaseField::from(1260750973),
            },
            step_size: CirclePointIndex(33554432),
            step: CirclePoint {
                x: BaseField::from(579625837),
                y: BaseField::from(1690787918),
            },
            log_size: 6,
        };
        let twiddle_tree = TwiddleTree::<CudaBackend> {
            root_coset,
            twiddles: BaseFieldVec::from_vec(twiddles),
            itwiddles: BaseFieldVec::from_vec(itwiddles),
        };

        let cpu_evaluation = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
            eval.domain,
            eval.values.to_cpu(),
        );
        let cpu_twiddle_tree = TwiddleTree::<CpuBackend> {
            root_coset: twiddle_tree.root_coset.clone(),
            twiddles: twiddle_tree.twiddles.to_cpu(),
            itwiddles: twiddle_tree.itwiddles.to_cpu(),
        };
        let expected_result = CpuBackend::interpolate(cpu_evaluation, &cpu_twiddle_tree);
        let result = CudaBackend::interpolate(eval, &twiddle_tree);
        assert_eq!(expected_result.coeffs, result.coeffs.to_cpu());
    }

    // #[test_log::test]
    // fn test_interpolate_columns() {
    //     // use crate::core::backend::cuda::poly::interpolate_columns_native;
    //     let log_number_of_columns = 7;

    //     for log_size in 13..17 {
    //         let size = 1 << log_size;
    //         let number_of_columns = 1 << log_number_of_columns;
    //         let cpu_values = (1..(size + 1) as u32).map(BaseField::from).collect_vec();
    //         let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

    //         let coset = CanonicCoset::new(log_size);
    //         let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
    //         let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

    //         let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
    //         let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

    //         let cpu_columns = (0..number_of_columns)
    //             .map(|_index| cpu_evaluations.clone())
    //             .collect_vec();
    //         let gpu_columns = (0..number_of_columns)
    //             .map(|_index| gpu_evaluations.clone())
    //             .collect_vec();

    //         let timer = start_timer!(|| format!("cpu backend interpolate_columns, column:{} log_n:{}", 1<<log_number_of_columns, log_size));
    //         let expected_result = CpuBackend::interpolate_columns(cpu_columns, &cpu_twiddles);
    //         end_timer!(timer);

    //         // let timer = start_timer!(|| format!("gpu backend native interpolate_columns, column:{} log_n:{}", 1<<log_number_of_columns, log_size));
    //         // let result = interpolate_columns_native(gpu_columns.clone(), &gpu_twiddles);
    //         // end_timer!(timer);


    //         let timer = start_timer!(|| format!("cuda backend optimize interpolate_columns, column:{} log_n:{}", 1<<log_number_of_columns, log_size));
    //         let result_optim = CudaBackend::interpolate_columns(gpu_columns, &gpu_twiddles);
    //         end_timer!(timer);


    //         let expected_coeffs = expected_result
    //             .iter()
    //             .map(|poly| poly.coeffs.clone())
    //             .collect_vec();
    //         let coeffs = result
    //             .iter()
    //             .map(|poly| poly.coeffs.clone().to_cpu())
    //             .collect_vec();
    //         let coeffs_optim = result_optim
    //             .iter()
    //             .map(|poly| poly.coeffs.clone().to_cpu())
    //             .collect_vec();

    //         assert_eq!(expected_coeffs, coeffs_optim);
    //     }
    // }

    // #[allow(unused_variables)]
    // #[test_log::test]
    // fn test_evaluate_columns() {
    //     let log_blowup_factor = 2;
    //     let log_number_of_columns = 7;

    //     for log_size in 13..20-log_blowup_factor {

    //         let size = 1 << log_size;
    //         let number_of_columns = 1 << log_number_of_columns;

    //         let cpu_values = (1..(size + 1) as u32)
    //             .map(BaseField::from)
    //             .collect::<Vec<_>>();
    //         let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

    //         let trace_coset = CanonicCoset::new(log_size);
    //         let cpu_evaluations = CpuBackend::new_canonical_ordered(trace_coset, cpu_values);
    //         let gpu_evaluations = CudaBackend::new_canonical_ordered(trace_coset, gpu_values.clone());
    //         let gpu_evaluations_ref = CudaBackend::new_canonical_ordered(trace_coset, gpu_values);

    //         let interpolation_coset = CanonicCoset::new(log_size + log_blowup_factor);
    //         let cpu_twiddles = CpuBackend::precompute_twiddles(interpolation_coset.half_coset());
    //         let gpu_twiddles = CudaBackend::precompute_twiddles(interpolation_coset.half_coset());

    //         let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
    //         let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

    //         assert_eq!(gpu_poly.coeffs.to_vec(), cpu_poly.coeffs.to_vec());

    //         let mut cpu_columns: Vec<CirclePoly<CpuBackend>> = ColumnVec::from(
    //             (0..number_of_columns)
    //                 .map(|_index| cpu_poly.clone())
    //                 .collect_vec(),
    //         );
    //         let mut gpu_columns = ColumnVec::from(
    //             (0..number_of_columns)
    //                 .map(|_index| gpu_poly.clone())
    //                 .collect_vec(),
    //         );

    //         let timer = start_timer!(|| format!("cpu backend optimize evaluate_polynomials, column:{} log_n:{}", 1<<log_number_of_columns, log_size));
    //         let expected_result = CpuBackend::evaluate_polynomials(&mut cpu_columns, log_blowup_factor, &cpu_twiddles);
    //         end_timer!(timer);

    //         let timer = start_timer!(|| format!("cuda backend evaluate_polynomials, column:{} log_n:{}", 1<<log_number_of_columns, log_size));
    //         let result = CudaBackend::evaluate_polynomials(&mut gpu_columns, log_blowup_factor, &gpu_twiddles);
    //         end_timer!(timer);

    //         let expected_values = expected_result
    //             .iter()
    //             .map(|eval| eval.clone().values)
    //             .collect_vec();
    //         let values = result
    //             .iter()
    //             .map(|eval| eval.clone().values.to_cpu())
    //             .collect_vec();

    //         assert_eq!(values, expected_values);
    //     }
    // }
}
