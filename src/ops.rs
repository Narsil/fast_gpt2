use crate::tensor::{OwnedTensor, Tensor};

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

#[inline]
pub(crate) fn matmul(a: &OwnedTensor, b: &Tensor, c: &mut OwnedTensor) {
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    assert_eq!(k, b.shape()[0]);

    let ar = k as isize;
    let ac = 1;
    let br = n as isize;
    let bc = 1;
    let cr = n as isize;
    let cc = 1;
    #[cfg(not(feature = "cblas"))]
    unsafe {
        matrixmultiply::sgemm(m, k, n, 1.0, ap, ar, ac, bp, br, bc, 1.0, cp, cr, cc);
    }

    #[cfg(feature = "cblas")]
    unsafe {
        let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);
        let (layout, a_tr, b_tr, lda, ldb, ldc) = if cr < cc {
            let (lda, a_tr) = if ar < ac { (m, NoTr) } else { (k, Tr) };
            let (ldb, b_tr) = if br < bc { (k, NoTr) } else { (n, Tr) };
            (ColMajor, a_tr, b_tr, lda, ldb, m)
        } else {
            let (lda, a_tr) = if ar < ac { (m, Tr) } else { (k, NoTr) };
            let (ldb, b_tr) = if br < bc { (k, Tr) } else { (n, NoTr) };
            (RowMajor, a_tr, b_tr, lda, ldb, n)
        };
        sgemm(
            layout, a_tr, b_tr, m, n, k, 1.0, ap, lda, bp, ldb, 1.0, cp, ldc,
        )
    }
}

pub fn add(a: &Tensor, b: &mut OwnedTensor) {
    if a.shape() == b.shape() {
        a.data
            .iter()
            .zip(b.data.iter_mut())
            .for_each(|(left, right)| *right += left);
    } else if b.shape()[0] == 2 && &b.shape()[1..] == a.shape() {
        a.data
            .iter()
            .zip(b.data.iter_mut())
            .for_each(|(left, right)| *right *= left);
        a.data
            .iter()
            .zip(b.data.iter_mut().skip(a.shape()[0]))
            .for_each(|(left, right)| *right *= left);
    } else {
        todo!("Add broadcast")
    }
}

pub fn mul(a: &Tensor, b: &mut OwnedTensor) {
    if a.shape() == b.shape() {
        a.data
            .iter()
            .zip(b.data.iter_mut())
            .for_each(|(left, right)| *right *= left);
    } else if b.shape()[0] == 2 && &b.shape()[1..] == a.shape() {
        a.data
            .iter()
            .zip(b.data.iter_mut())
            .for_each(|(left, right)| *right *= left);
        a.data
            .iter()
            .zip(b.data.iter_mut().skip(a.shape()[0]))
            .for_each(|(left, right)| *right *= left);
    } else {
        todo!("mul broadcast")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_matmul() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let A = Tensor::new(&data, vec![2, 2]);

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let b = OwnedTensor::new(data, vec![2, 2]);
        let data = vec![0.0; 4];
        let mut c = OwnedTensor::new(data, vec![2, 2]);

        matmul(&A, &b, &mut c);
        assert_eq!(c.data, &[7.0, 10.0, 15.0, 22.0]);

        let data = [1.0, 2.0];
        let A = Tensor::new(&data, vec![2, 1]);
        let data = vec![3.0, 4.0];
        let b = OwnedTensor::new(data, vec![1, 2]);
        let data = vec![0.0; 4];
        let mut c = OwnedTensor::new(data, vec![2, 2]);
        matmul(&A, &b, &mut c);
        assert_eq!(c.data, &[3.0, 4.0, 6.0, 8.0])
    }
}

// impl super::VecVecKernel<f32> for Cpu {
//     fn forward<M: Dim, N: Dim>(
//         &self,
//         lhs: &Self::Storage<(M,), f32>,
//         rhs: &Self::Storage<(N,), f32>,
//     ) -> Result<Self::Storage<(M, N), f32>, Self::Err> {
//         let mut out = StridedArray::new((lhs.shape().0, rhs.shape().0))?;
//         matmul(lhs.view().br1(), rhs.view().br0(), &mut out.view_mut());
//         Ok(out)
//     }
//     fn backward<M: Dim, N: Dim>(
//         &self,
//         lhs: &Self::Storage<(M,), f32>,
//         grad_lhs: &mut Self::Storage<(M,), f32>,
//         rhs: &Self::Storage<(N,), f32>,
//         grad_rhs: &mut Self::Storage<(N,), f32>,
//         grad_out: &Self::Storage<(M, N), f32>,
//     ) -> Result<(), Self::Err> {
//         let grad_out = grad_out.view();
//         let lhs = lhs.view().br1().tr();
//         let rhs = rhs.view().br0().tr();
//         matmul(grad_out, rhs, &mut grad_lhs.view_mut().br1());
//         matmul(lhs, grad_out, &mut grad_rhs.view_mut().br0());
//         Ok(())
//     }
// }
//
// impl super::VecMatKernel<f32> for Cpu {
//     fn forward<const K: usize, N: Dim>(
//         &self,
//         lhs: &Self::Storage<(Const<K>,), f32>,
//         rhs: &Self::Storage<(Const<K>, N), f32>,
//     ) -> Result<Self::Storage<(N,), f32>, Self::Err> {
//         let mut out = StridedArray::new((rhs.shape.1,))?;
//         matmul(lhs.view().br0(), rhs.view(), &mut out.view_mut().br0());
//         Ok(out)
//     }
//     fn backward<const K: usize, N: Dim>(
//         &self,
//         lhs: &Self::Storage<(Const<K>,), f32>,
//         grad_lhs: &mut Self::Storage<(Const<K>,), f32>,
//         rhs: &Self::Storage<(Const<K>, N), f32>,
//         grad_rhs: &mut Self::Storage<(Const<K>, N), f32>,
//         grad_out: &Self::Storage<(N,), f32>,
//     ) -> Result<(), Self::Err> {
//         let grad_out = grad_out.view().br0();
//         matmul(grad_out, rhs.view().tr(), &mut grad_lhs.view_mut().br0());
//         matmul(lhs.view().br0().tr(), grad_out, &mut grad_rhs.view_mut());
//         Ok(())
//     }
// }
//
// impl super::MatMatKernel<f32> for Cpu {
//     fn forward<M: Dim, const K: usize, N: Dim>(
//         &self,
//         lhs: &Self::Storage<(M, Const<K>), f32>,
//         rhs: &Self::Storage<(Const<K>, N), f32>,
//     ) -> Result<Self::Storage<(M, N), f32>, Self::Err> {
//         let mut out = StridedArray::new((lhs.shape.0, rhs.shape.1))?;
//         matmul(lhs.view(), rhs.view(), &mut out.view_mut());
//         Ok(out)
//     }
//     fn backward<M: Dim, const K: usize, N: Dim>(
//         &self,
//         lhs: &Self::Storage<(M, Const<K>), f32>,
//         grad_lhs: &mut Self::Storage<(M, Const<K>), f32>,
//         rhs: &Self::Storage<(Const<K>, N), f32>,
//         grad_rhs: &mut Self::Storage<(Const<K>, N), f32>,
//         grad_out: &Self::Storage<(M, N), f32>,
//     ) -> Result<(), Self::Err> {
//         let grad_out = grad_out.view();
//         matmul(grad_out, rhs.view().tr(), &mut grad_lhs.view_mut());
//         matmul(lhs.view().tr(), grad_out, &mut grad_rhs.view_mut());
//         Ok(())
//     }
// }
//
// impl super::MatMatBrKernel<f32> for Cpu {
//     fn forward<B: Dim, M: Dim, const K: usize, N: Dim>(
//         &self,
//         lhs: &Self::Storage<(B, M, Const<K>), f32>,
//         rhs: &Self::Storage<(Const<K>, N), f32>,
//     ) -> Result<Self::Storage<(B, M, N), f32>, Self::Err> {
//         let (batch, seq, _) = *lhs.shape();
//         let (_, n) = *rhs.shape();
//         let mut out = StridedArray::new((batch, seq, n))?;
//         let a = lhs.view();
//         let b = rhs.view();
//         let mut c = out.view_mut();
//         for batch in 0..batch.size() {
//             matmul(a.idx(batch), b, &mut c.idx_mut(batch));
//         }
//         Ok(out)
//     }
//     fn backward<B: Dim, M: Dim, const K: usize, N: Dim>(
//         &self,
//         lhs: &Self::Storage<(B, M, Const<K>), f32>,
//         grad_lhs: &mut Self::Storage<(B, M, Const<K>), f32>,
//         rhs: &Self::Storage<(Const<K>, N), f32>,
//         grad_rhs: &mut Self::Storage<(Const<K>, N), f32>,
//         grad_out: &Self::Storage<(B, M, N), f32>,
//     ) -> Result<(), Self::Err> {
//         let batch_size = lhs.shape().0.size();
//         let lhs = lhs.view();
//         let mut grad_lhs = grad_lhs.view_mut();
//         let rhs = rhs.view().tr();
//         let mut grad_rhs = grad_rhs.view_mut();
//         let grad_out = grad_out.view();
//         for b in 0..batch_size {
//             let go = grad_out.idx(b);
//             matmul(go, rhs, &mut grad_lhs.idx_mut(b));
//             matmul(lhs.idx(b).tr(), go, &mut grad_rhs);
//         }
//         Ok(())
//     }
// }
//
// impl super::MatMatBatch3Kernel<f32> for Cpu {
//     fn forward<const B: usize, M: Dim, const K: usize, N: Dim>(
//         &self,
//         lhs: &Self::Storage<(Const<B>, M, Const<K>), f32>,
//         rhs: &Self::Storage<(Const<B>, Const<K>, N), f32>,
//     ) -> Result<Self::Storage<(Const<B>, M, N), f32>, Self::Err> {
//         let m: M = lhs.shape().1;
//         let n: N = rhs.shape().2;
//         let mut out = StridedArray::new((Const, m, n))?;
//         let a = lhs.view();
//         let b = rhs.view();
//         let mut c = out.view_mut();
//         for batch in 0..B {
//             matmul(a.idx(batch), b.idx(batch), &mut c.idx_mut(batch));
//         }
//         Ok(out)
//     }
//     fn backward<const B: usize, M: Dim, const K: usize, N: Dim>(
//         &self,
//         lhs: &Self::Storage<(Const<B>, M, Const<K>), f32>,
//         grad_lhs: &mut Self::Storage<(Const<B>, M, Const<K>), f32>,
//         rhs: &Self::Storage<(Const<B>, Const<K>, N), f32>,
//         grad_rhs: &mut Self::Storage<(Const<B>, Const<K>, N), f32>,
//         grad_out: &Self::Storage<(Const<B>, M, N), f32>,
//     ) -> Result<(), Self::Err> {
//         let lhs = lhs.view();
//         let mut grad_lhs = grad_lhs.view_mut();
//         let rhs = rhs.view();
//         let mut grad_rhs = grad_rhs.view_mut();
//         let grad_out = grad_out.view();
//         for b in 0..B {
//             let go = grad_out.idx(b);
//             matmul(go, rhs.idx(b).tr(), &mut grad_lhs.idx_mut(b));
//             matmul(lhs.idx(b).tr(), go, &mut grad_rhs.idx_mut(b));
//         }
//         Ok(())
//     }
// }
//
// impl super::MatMatBatch4Kernel<f32> for Cpu {
//     fn forward<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim>(
//         &self,
//         lhs: &Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
//         rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
//     ) -> Result<Self::Storage<(Const<B>, Const<S>, M, N), f32>, Self::Err> {
//         let m: M = lhs.shape.2;
//         let n: N = rhs.shape.3;
//         let mut out = StridedArray::new((Const, Const, m, n))?;
//         let lhs = lhs.view();
//         let rhs = rhs.view();
//         let mut out_view = out.view_mut();
//         for b in 0..B {
//             let l_b = lhs.idx(b);
//             let r_b = rhs.idx(b);
//             let mut o_b = out_view.idx_mut(b);
//             for s in 0..S {
//                 matmul(l_b.idx(s), r_b.idx(s), &mut o_b.idx_mut(s));
//             }
//         }
//         Ok(out)
//     }
//     fn backward<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim>(
//         &self,
//         lhs: &Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
//         grad_lhs: &mut Self::Storage<(Const<B>, Const<S>, M, Const<K>), f32>,
//         rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
//         grad_rhs: &mut Self::Storage<(Const<B>, Const<S>, Const<K>, N), f32>,
//         grad_out: &Self::Storage<(Const<B>, Const<S>, M, N), f32>,
//     ) -> Result<(), Self::Err> {
//         let lhs = lhs.view();
//         let mut grad_lhs = grad_lhs.view_mut();
//         let rhs = rhs.view();
//         let mut grad_rhs = grad_rhs.view_mut();
//         let grad_out = grad_out.view();
//         for b in 0..B {
//             let l_b = lhs.idx(b);
//             let mut gl_b = grad_lhs.idx_mut(b);
//             let r_b = rhs.idx(b);
//             let mut gr_b = grad_rhs.idx_mut(b);
//             let go_b = grad_out.idx(b);
//             for s in 0..S {
//                 matmul(go_b.idx(s), r_b.idx(s).tr(), &mut gl_b.idx_mut(s));
//                 matmul(l_b.idx(s).tr(), go_b.idx(s), &mut gr_b.idx_mut(s));
//             }
//         }
//         Ok(())
//     }
// }
