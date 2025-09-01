use crate::stwo_cuda::bindings::CudaSecureField;
use crate::stwo_cuda::{bindings, secure_field_vec::SecureFieldVec};
use crate::core::backend::cuda::CudaBackend;
use crate::core::{
    fields::qm31::SecureField,
    lookups::{
        gkr_prover::GkrOps,
        mle::Mle,
    },
};

#[allow(unused_variables)]
impl GkrOps for CudaBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        let y_size = y.len();
        let result_evals = SecureFieldVec::new_uninitialized(1 << y_size);

        unsafe {
            bindings::gen_eq_evals(
                v.into(),
                y.as_ptr() as *const CudaSecureField,
                y_size as u32,
                result_evals.device_ptr as *const CudaSecureField,
                result_evals.size as u32,
            );
        }

        Mle::new(result_evals)
    }

    fn next_layer(
        layer: &crate::core::lookups::gkr_prover::Layer<Self>,
    ) -> crate::core::lookups::gkr_prover::Layer<Self> {
        todo!()
    }

    fn sum_as_poly_in_first_variable(
        h: &crate::core::lookups::gkr_prover::GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> crate::core::lookups::utils::UnivariatePoly<SecureField> {
        todo!()
    }
}

mod tests {

    #[test]
    fn gen_eq_evals_matches_cpu() {
        use crate::core::backend::{Column, CpuBackend};
        use crate::core::fields::m31::BaseField;
        use crate::core::fields::qm31::SecureField;
        use crate::core::lookups::gkr_prover::GkrOps;
        use itertools::Itertools;
        use crate::core::backend::cuda::CudaBackend;

        let two = BaseField::from(2).into();

        let from_raw = [7, 3, 5, 6, 1, 1, 9].repeat(4);
        let y = from_raw
            .chunks(4)
            .map(|a| SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect_vec();

        let cpu_eq_evals = CpuBackend::gen_eq_evals(&y, two);
        let gpu_eq_evals = CudaBackend::gen_eq_evals(&y, two);

        assert_eq!(gpu_eq_evals.to_cpu(), *cpu_eq_evals);
    }
}
