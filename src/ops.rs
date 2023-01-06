use crate::tensor::{OwnedTensor, Tensor};

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

#[inline]
pub fn addmm(x: &OwnedTensor, a: &Tensor, b: &Tensor, out: &mut OwnedTensor) {
    let m = x.shape()[0];
    let k = x.shape()[1];
    let n = a.shape()[1];
    assert_eq!(k, a.shape()[0]);
    assert_eq!(n, b.shape()[0]);
    assert_eq!(out.shape(), &[m, n]);

    matmul(x, a, out);
    add(b, out);
}

pub fn select(ids: &[u32], weights: &Tensor, out: &mut OwnedTensor) {
    let hidden_dim = weights.shape()[1];
    let sequence_length = ids.len();
    assert_eq!(out.shape(), [sequence_length, hidden_dim]);
    for (i, id) in ids.iter().enumerate() {
        let id = *id as usize;
        let weight_offset = id * hidden_dim;
        let data_offset = i * hidden_dim;
        out.data[data_offset..data_offset + hidden_dim]
            .copy_from_slice(&weights.data()[weight_offset..weight_offset + hidden_dim]);
    }
}

#[inline]
pub(crate) fn matmul_t(a: &OwnedTensor, b: &Tensor, c: &mut OwnedTensor) {
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[0];

    assert_eq!(k, b.shape()[1]);

    let ar = k as isize;
    let ac = 1;
    let br = 1;
    let bc = b.shape()[1] as isize;
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

#[inline]
pub(crate) fn matmul_t_owned(a: &OwnedTensor, b: &OwnedTensor, c: &mut OwnedTensor) {
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[0];

    assert_eq!(k, b.shape()[1]);

    let ar = k as isize;
    let ac = 1;
    let br = 1;
    let bc = b.shape()[1] as isize;
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

#[inline]
pub(crate) fn matmul_owned(a: &OwnedTensor, b: &OwnedTensor, c: &mut OwnedTensor) {
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

pub fn add_owned(a: &OwnedTensor, b: &mut OwnedTensor) {
    if a.shape() == b.shape() {
        a.data
            .iter()
            .zip(b.data.iter_mut())
            .for_each(|(left, right)| *right += left);
    } else if &b.shape()[1..] == a.shape() {
        let n = b.shape()[0];
        (0..n).for_each(|i| {
            a.data
                .iter()
                .zip(b.data.iter_mut().skip(i * a.shape()[0]))
                .for_each(|(left, right)| *right += left)
        });
    } else {
        todo!("Add broadcast")
    }
}

pub fn add(a: &Tensor, b: &mut OwnedTensor) {
    if a.shape() == b.shape() {
        a.data
            .iter()
            .zip(b.data.iter_mut())
            .for_each(|(left, right)| *right += left);
    } else if &b.shape()[1..] == a.shape() {
        let n = b.shape()[0];
        (0..n).for_each(|i| {
            a.data
                .iter()
                .zip(b.data.iter_mut().skip(i * a.shape()[0]))
                .for_each(|(left, right)| *right += left);
        });
    } else {
        todo!("add broadcast A {:?} B {:?}", a.shape(), b.shape());
    }
}

pub fn mul(a: &Tensor, b: &mut OwnedTensor) {
    if a.shape() == b.shape() {
        a.data
            .iter()
            .zip(b.data.iter_mut())
            .for_each(|(left, right)| *right *= left);
    } else if &b.shape()[1..] == a.shape() {
        let n = b.shape()[0];
        (0..n).for_each(|i| {
            a.data
                .iter()
                .zip(b.data.iter_mut().skip(i * a.shape()[0]))
                .for_each(|(left, right)| *right *= left);
        });
    } else {
        todo!("mul broadcast A {:?} B {:?}", a.shape(), b.shape());
    }
}

pub fn normalize(x: &mut OwnedTensor, epsilon: f32) {
    assert_eq!(x.shape().len(), 2);
    let size = x.shape()[1];

    let mut mean: Vec<f32> = vec![0.0; x.shape()[0]];
    let mut var: Vec<f32> = vec![0.0; x.shape()[0]];
    let mut sum = 0.0;
    for (i, v) in x.data.iter().enumerate() {
        sum += v;
        if (i + 1) % size == 0 {
            let value = sum / size as f32;
            mean[i / size] = value;
            sum = 0.0;
        }
    }
    sum = 0.0;
    for (i, v) in x.data.iter().enumerate() {
        let value = (v - mean[i / size]).powf(2.0);
        sum += value;
        if (i + 1) % size == 0 {
            let value = sum / size as f32;
            var[i / size] = value;
            sum = 0.0;
        }
    }

    x.data
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = (*v - mean[i / size]) / (var[i / size] + epsilon).sqrt());
}

#[cfg(test)]
#[inline]
pub(crate) fn softmax(x: &mut OwnedTensor) {
    assert_eq!(x.shape().len(), 2);

    let m = x.shape()[0];
    let n = x.shape()[1];
    let mut max = vec![0.0; m];
    let mut current_max = f32::NEG_INFINITY;
    for (i, &v) in x.data.iter().enumerate() {
        if v > current_max {
            current_max = v;
        }
        if (i + 1) % n == 0 {
            max[i / n] = current_max;
            current_max = f32::NEG_INFINITY;
        }
    }
    x.data
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v -= max[i / n]);
    x.data.iter_mut().for_each(|v| *v = (*v).exp());
    let mut softmax = vec![0.0; m];
    let mut sum = 0.0;
    for (i, v) in x.data.iter().enumerate() {
        sum += v;
        if (i + 1) % n == 0 {
            softmax[i / n] = sum;
            sum = 0.0;
        }
    }
    x.data
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v /= softmax[i / n]);
}

#[inline]
pub(crate) fn causal_softmax(x: &mut OwnedTensor) {
    assert_eq!(x.shape().len(), 2);

    let m = x.shape()[0];
    let n = x.shape()[1];
    let mut max = vec![0.0; m];
    let mut current_max = f32::NEG_INFINITY;
    for (i, &v) in x.data.iter().enumerate() {
        let j = i / n;
        let i = i % n;
        if i >= j {
            if v > current_max {
                current_max = v;
            }
            if (i + 1) % n == 0 {
                max[j] = current_max;
                current_max = f32::NEG_INFINITY;
            }
        }
    }
    x.data
        .iter_mut()
        .enumerate()
        // Technically we're removing the max from the masked values.
        // We don't care since this make this branchless and additions
        // are super fast and we're going to reset the values to zero anyway
        // at the end.
        .for_each(|(i, v)| *v -= max[i / n]);
    x.data.iter_mut().enumerate().for_each(|(i, v)| {
        let j = i / n;
        let i = i % n;
        if i >= j {
            *v = (*v).exp();
        }
    });
    let mut softmax = vec![0.0; m];
    let mut sum = 0.0;
    for (i, v) in x.data.iter().enumerate() {
        let j = i / n;
        let i = i % n;
        if i >= j {
            sum += v;
            if (i + 1) % n == 0 {
                softmax[j] = sum;
                sum = 0.0;
            }
        }
    }
    x.data.iter_mut().enumerate().for_each(|(i, v)| {
        let j = i / n;
        let i = i % n;
        if i >= j {
            *v /= softmax[j]
        } else {
            *v = 0.0;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::simplify;

    #[test]
    fn simple_matmul() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let a = OwnedTensor::new(data, vec![2, 2]);
        let data = [1.0, 2.0, 3.0, 4.0];
        let b = Tensor::new(&data, vec![2, 2]);
        let data = vec![0.0; 4];
        let mut c = OwnedTensor::new(data, vec![2, 2]);

        matmul(&a, &b, &mut c);
        assert_eq!(c.data, &[7.0, 10.0, 15.0, 22.0]);

        let data = vec![1.0, 2.0];
        let a = OwnedTensor::new(data, vec![2, 1]);
        let data = [3.0, 4.0];
        let b = Tensor::new(&data, vec![1, 2]);
        let data = vec![0.0; 4];
        let mut c = OwnedTensor::new(data, vec![2, 2]);
        matmul(&a, &b, &mut c);
        assert_eq!(c.data, &[3.0, 4.0, 6.0, 8.0])
    }

    #[test]
    fn simple_matmul_t() {
        let a = OwnedTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        // A.T
        let b = Tensor::new(&[1.0, 3.0, 2.0, 4.0], vec![2, 2]);
        let data = vec![0.0; 4];
        let mut c = OwnedTensor::new(data, vec![2, 2]);

        matmul_t(&a, &b, &mut c);
        assert_eq!(c.data, &[7.0, 10.0, 15.0, 22.0]);

        let a = OwnedTensor::new(vec![1.0, 2.0], vec![2, 1]);
        let b = Tensor::new(&[3.0, 4.0], vec![2, 1]);
        let data = vec![0.0; 4];
        let mut c = OwnedTensor::new(data, vec![2, 2]);
        matmul_t(&a, &b, &mut c);
        assert_eq!(c.data, &[3.0, 4.0, 6.0, 8.0])
    }

    #[test]
    fn simple_softmax() {
        let mut a = OwnedTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        softmax(&mut a);
        assert_eq!(
            simplify(&a.data[..]),
            // Values obtained through python
            [0.2689, 0.7311, 0.2689, 0.7311]
        );
    }

    #[test]
    fn simple_causal_softmax() {
        let mut a = OwnedTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        causal_softmax(&mut a);
        assert_eq!(
            simplify(&a.data[..]),
            // Values obtained through python
            [0.2689, 0.7311, 0.0000, 1.0000]
        );
    }
}
