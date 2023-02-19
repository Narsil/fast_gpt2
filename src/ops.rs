use crate::tensor::{OwnedTensor, PastKeyValue, Tensor, TensorMut};

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

#[inline]
pub fn addmm<X: Tensor, A: Tensor, B: Tensor, TM: TensorMut>(x: &X, a: &A, b: &B, out: &mut TM) {
    let m = x.shape()[0];
    let k = x.shape()[1];
    let n = a.shape()[1];
    assert_eq!(k, a.shape()[0]);
    assert_eq!(n, b.shape()[0]);
    assert_eq!(out.shape(), &[m, n]);

    matmul(x, a, out);
    add(b, out);
}

pub fn select<T: Tensor, TM: TensorMut>(ids: &[u32], weights: &T, out: &mut TM) {
    let hidden_dim = weights.shape()[1];
    let sequence_length = ids.len();
    assert_eq!(out.shape(), [sequence_length, hidden_dim]);
    for (i, id) in ids.iter().enumerate() {
        let id = *id as usize;
        let weight_offset = id * hidden_dim;
        let data_offset = i * hidden_dim;
        out.data_mut()[data_offset..data_offset + hidden_dim]
            .copy_from_slice(&weights.data()[weight_offset..weight_offset + hidden_dim]);
    }
}

pub fn matmul<A: Tensor, B: Tensor, TM: TensorMut>(a: &A, b: &B, c: &mut TM) {
    g_matmul::<false, A, B, TM>(a, b, c)
}

pub fn matmul_t<A: Tensor, B: Tensor, TM: TensorMut>(a: &A, b: &B, c: &mut TM) {
    g_matmul::<true, A, B, TM>(a, b, c)
}

#[inline]
pub fn g_matmul<const TRANSPOSE: bool, A: Tensor, B: Tensor, TM: TensorMut>(
    a: &A,
    b: &B,
    c: &mut TM,
) {
    let dim = a.shape().len();
    assert!(dim >= 2);
    assert_eq!(b.shape().len(), dim);
    assert_eq!(c.shape().len(), dim);
    assert_eq!(a.shape()[..dim - 2], b.shape()[..dim - 2]);
    assert_eq!(a.shape()[..dim - 2], c.shape()[..dim - 2]);

    let m = a.shape()[dim - 2];
    let k = a.shape()[dim - 1];

    let n = if TRANSPOSE {
        let n = b.shape()[dim - 2];
        assert_eq!(k, b.shape()[dim - 1]);
        n
    } else {
        let n = b.shape()[dim - 1];
        assert_eq!(k, b.shape()[dim - 2]);
        n
    };
    assert_eq!(c.shape()[dim - 2..], vec![m, n]);

    let batching: usize = a.shape()[..dim - 2].iter().product();
    let a_skip: usize = m * k;
    let b_skip: usize = n * k;
    let c_skip: usize = m * n;

    let ar = k as isize;
    let ac = 1;
    let (br, bc) = if TRANSPOSE {
        (1, b.shape()[dim - 1] as isize)
    } else {
        (b.shape()[dim - 1] as isize, 1)
    };
    let cr = n as isize;
    let cc = 1;

    (0..batching).for_each(|step| {
        let ap = a.data()[step * a_skip..].as_ptr();
        let bp = b.data()[step * b_skip..].as_ptr();
        let cp = c.data_mut()[step * c_skip..].as_mut_ptr();

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
    });
}

pub fn add<T: Tensor, TM: TensorMut>(a: &T, b: &mut TM) {
    if a.shape() == b.shape() {
        a.data()
            .iter()
            .zip(b.data_mut().iter_mut())
            .for_each(|(left, right)| *right += left);
    } else if &b.shape()[1..] == a.shape() {
        let n = b.shape()[0];
        (0..n).for_each(|i| {
            a.data()
                .iter()
                .zip(b.data_mut().iter_mut().skip(i * a.shape()[0]))
                .for_each(|(left, right)| *right += left);
        });
    } else {
        todo!("add broadcast A {:?} B {:?}", a.shape(), b.shape());
    }
}

pub fn mul<T: Tensor, TM: TensorMut>(a: &T, b: &mut TM) {
    if a.shape() == b.shape() {
        a.data()
            .iter()
            .zip(b.data_mut().iter_mut())
            .for_each(|(left, right)| *right *= left);
    } else if &b.shape()[1..] == a.shape() {
        let n = b.shape()[0];
        (0..n).for_each(|i| {
            a.data()
                .iter()
                .zip(b.data_mut().iter_mut().skip(i * a.shape()[0]))
                .for_each(|(left, right)| *right *= left);
        });
    } else {
        todo!("mul broadcast A {:?} B {:?}", a.shape(), b.shape());
    }
}

pub fn normalize<TM: TensorMut>(x: &mut TM, mean: &mut [f32], var: &mut [f32], epsilon: f32) {
    assert_eq!(x.shape().len(), 2);
    let m = x.shape()[0];
    let size = x.shape()[1];
    assert!(mean.len() >= m);
    assert!(var.len() >= m);

    let mut sum = 0.0;
    for (i, v) in x.data().iter().enumerate() {
        sum += v;
        if (i + 1) % size == 0 {
            let value = sum / size as f32;
            mean[i / size] = value;
            sum = 0.0;
        }
    }
    sum = 0.0;
    for (i, v) in x.data().iter().enumerate() {
        let value = (v - mean[i / size]).powf(2.0);
        sum += value;
        if (i + 1) % size == 0 {
            let value = sum / size as f32;
            var[i / size] = value;
            sum = 0.0;
        }
    }

    x.data_mut()
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = (*v - mean[i / size]) / (var[i / size] + epsilon).sqrt());
}

#[inline]
fn g_softmax<const CAUSAL: bool, TM: TensorMut>(
    x: &mut TM,
    max: &mut [f32],
    past_sequence_length: usize,
) {
    let dim = x.shape().len();

    let m = x.shape()[dim - 2];
    let n = x.shape()[dim - 1];
    let b: usize = x.shape()[..dim - 2].iter().product();
    assert!(max.len() >= b * m);
    let mut current_max = f32::NEG_INFINITY;
    for (ii, &v) in x.data().iter().enumerate() {
        let i = ii / n;
        let j = ii % n;
        if (!CAUSAL || i + past_sequence_length >= j) && v > current_max {
            current_max = v;
        }
        if (j + 1) % n == 0 {
            max[ii / n] = current_max;
            current_max = f32::NEG_INFINITY;
        }
    }
    x.data_mut()
        .iter_mut()
        .enumerate()
        // Technically we're removing the max from the masked values.
        // We don't care since this make this branchless and additions
        // are super fast and we're going to reset the values to zero anyway
        // at the end.
        .for_each(|(ii, v)| *v -= max[ii / n]);
    x.data_mut().iter_mut().for_each(|v| {
        // TODO Is skipping the causal ops faster ?
        *v = (*v).exp();
    });
    let softmax = max;
    let mut sum = 0.0;
    for (ii, v) in x.data().iter().enumerate() {
        let i = (ii / n) % m;
        let j = ii % n;
        if !CAUSAL || i + past_sequence_length >= j {
            sum += v;
        }
        if (j + 1) % n == 0 {
            softmax[ii / n] = sum;
            sum = 0.0;
        }
    }
    x.data_mut().iter_mut().enumerate().for_each(|(ii, v)| {
        let i = (ii / n) % m;
        let j = ii % n;
        if !CAUSAL || i + past_sequence_length >= j {
            *v /= softmax[ii / n];
        } else {
            *v = 0.0;
        }
    });
}

pub fn softmax<TM: TensorMut>(x: &mut TM, max: &mut [f32]) {
    g_softmax::<false, TM>(x, max, 0)
}
pub fn causal_softmax<TM: TensorMut>(x: &mut TM, max: &mut [f32], past_sequence_length: usize) {
    g_softmax::<true, TM>(x, max, past_sequence_length)
}

pub(crate) fn split_qkv<T: Tensor>(
    qkv: &T,
    past: &PastKeyValue,
) -> (OwnedTensor, OwnedTensor, OwnedTensor) {
    let sequence_length = qkv.shape()[0];
    let past_sequence_length = past.key.shape()[1];
    let hidden_dim3 = qkv.shape()[1];
    assert_eq!(hidden_dim3 % 3, 0);
    let hidden_dim = hidden_dim3 / 3;
    let num_heads = past.key.shape()[0];
    assert_eq!(hidden_dim % num_heads, 0);
    let head_dim = hidden_dim / num_heads;
    let mut query_data = vec![0.0; num_heads * sequence_length * head_dim];
    (0..num_heads).for_each(|i| {
        (0..sequence_length).for_each(|j| {
            (0..head_dim).for_each(|k| {
                let index = j * hidden_dim * 3 + i * head_dim + k;
                let out_index = i * sequence_length * head_dim + j * head_dim + k;
                let value = qkv.data()[index];
                query_data[out_index] = value;
            });
        });
    });
    let query = OwnedTensor::new(query_data, vec![num_heads, sequence_length, head_dim]);

    let mut key_data = vec![0.0; num_heads * (past_sequence_length + sequence_length) * head_dim];
    let mut value_data = vec![0.0; num_heads * (past_sequence_length + sequence_length) * head_dim];
    (0..num_heads).for_each(|i| {
        (0..past_sequence_length + sequence_length).for_each(|j| {
            (0..head_dim).for_each(|k| {
                let in_index =
                    i * (past_sequence_length + sequence_length) * head_dim + j * head_dim + k;
                if j < past_sequence_length {
                    let index = i * past_sequence_length * head_dim + j * head_dim + k;
                    let k_value = past.key.data()[index];
                    let v_value = past.value.data()[index];
                    key_data[in_index] = k_value;
                    value_data[in_index] = v_value;
                } else {
                    let sj = j - past_sequence_length;
                    let k_index = sj * hidden_dim * 3 + i * head_dim + hidden_dim + k;
                    let v_index = sj * hidden_dim * 3 + i * head_dim + hidden_dim * 2 + k;
                    let k_value = qkv.data()[k_index];
                    let v_value = qkv.data()[v_index];
                    key_data[in_index] = k_value;
                    value_data[in_index] = v_value;
                }
            });
        });
    });

    let key = OwnedTensor::new(
        key_data,
        vec![
            num_heads,
            (past_sequence_length + sequence_length),
            head_dim,
        ],
    );
    let value = OwnedTensor::new(
        value_data,
        vec![
            num_heads,
            (past_sequence_length + sequence_length),
            head_dim,
        ],
    );
    (query, key, value)
}

pub fn attention<T: Tensor, TM: TensorMut>(
    qkv: &T,
    qk: &mut TM,
    max: &mut [f32],
    past: &mut PastKeyValue,
    out: &mut OwnedTensor,
) {
    let sequence_length = qkv.shape()[0];
    let past_sequence_length = past.key.shape()[1];
    let hidden_dim3 = qkv.shape()[1];
    assert_eq!(hidden_dim3 % 3, 0);
    let hidden_dim = hidden_dim3 / 3;
    let num_heads = qk.shape()[0];
    assert_eq!(hidden_dim % num_heads, 0);
    let head_dim = hidden_dim / num_heads;

    assert_eq!(
        qk.shape(),
        vec![
            num_heads,
            sequence_length,
            (past_sequence_length + sequence_length)
        ]
    );
    // assert_eq!(out.shape(), vec![sequence_length, hidden_dim]);
    assert_eq!(
        past.key.shape(),
        vec![num_heads, past_sequence_length, head_dim]
    );
    assert_eq!(
        past.value.shape(),
        vec![num_heads, past_sequence_length, head_dim]
    );

    // let tmp = past.value.data();
    // if tmp.len() > 0 {
    //     println!("past value {:?} {:?}", &tmp[..5], &tmp[tmp.len() - 5..]);
    // }

    let (query, key, value) = split_qkv(qkv, past);

    // println!(
    //     "Q vec {:?} {:?}",
    //     &query.data()[..5],
    //     &query.data()[query.data().len() - 5..]
    // );
    // println!(
    //     "K vec {:?} {:?}",
    //     &key.data()[..5],
    //     &key.data()[key.data().len() - 5..]
    // );
    // println!(
    //     "V vec {:?} {:?}",
    //     &value.data()[..5],
    //     &value.data()[value.data().len() - 5..]
    // );

    matmul_t(&query, &key, qk);
    let head_dim = hidden_dim / num_heads;
    let scale = (head_dim as f32).sqrt();
    qk.data_mut().iter_mut().for_each(|v| *v /= scale);

    // causal_softmax(qk, max, past_sequence_length);
    // // TODO
    softmax(qk, max);
    // let tmp = qk.data();
    // println!("weights {:?} {:?}", &tmp[..5], &tmp[tmp.len() - 5..]);
    matmul(qk, &value, out);
    // let tmp = out.data();
    // println!("post value {:?} {:?}", &tmp[..5], &tmp[tmp.len() - 5..]);

    let mut new_out = vec![0.0; sequence_length * hidden_dim];
    (0..num_heads).for_each(|i| {
        (0..sequence_length).for_each(|j| {
            (0..head_dim).for_each(|k| {
                let in_index = i * sequence_length * head_dim + j * head_dim + k;
                let out_index = j * hidden_dim + i * head_dim + k;
                new_out[out_index] = out.data()[in_index];
            });
        });
    });
    *out = OwnedTensor::new(new_out, vec![sequence_length, hidden_dim]);
    // let tmp = out.data();
    // println!("tokens vec {:?} {:?}", &tmp[..5], &tmp[tmp.len() - 5..]);
    *past = PastKeyValue { key, value };
}

pub fn special_argmax<T: Tensor>(x: &T) -> usize {
    assert_eq!(x.shape().len(), 2);
    let n = x.shape()[0];
    let m = x.shape()[1];

    let mut max = f32::NEG_INFINITY;
    let mut max_id = usize::MAX;
    for (i, &v) in x.data().iter().skip((n - 1) * m).enumerate() {
        if v > max {
            max = v;
            max_id = i;
        }
    }
    max_id
}

#[inline]
pub fn faster_tanh(x: f32) -> f32 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;

    let a = x + (0.16489087 * x3) + (0.00985468 * x5);

    a / (1.0 + (a * a)).sqrt()
}

pub fn gelu<T: TensorMut>(x: &mut T) {
    x.data_mut().iter_mut().for_each(|v| {
        *v = 0.5
            * (*v)
            * (1.0
                + f32::tanh((2.0f32 / std::f32::consts::PI).sqrt() * (*v + 0.044715 * v.powf(3.0))))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{OwnedTensor, ViewTensor};
    use crate::tests::simplify;

    #[test]
    fn simple_matmul() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let a = OwnedTensor::new(data, vec![2, 2]);
        let data = [1.0, 2.0, 3.0, 4.0];
        let b = ViewTensor::new(&data, vec![2, 2]);
        let data = vec![0.0; 4];
        let mut c = OwnedTensor::new(data, vec![2, 2]);

        matmul(&a, &b, &mut c);
        assert_eq!(c.data, &[7.0, 10.0, 15.0, 22.0]);

        let data = vec![1.0, 2.0];
        let a = OwnedTensor::new(data, vec![2, 1]);
        let data = [3.0, 4.0];
        let b = ViewTensor::new(&data, vec![1, 2]);
        let data = vec![0.0; 4];
        let mut c = OwnedTensor::new(data, vec![2, 2]);
        matmul(&a, &b, &mut c);
        assert_eq!(c.data, &[3.0, 4.0, 6.0, 8.0]);

        let data: Vec<_> = (0..6).map(|i| i as f32).collect();
        let a = OwnedTensor::new(data, vec![2, 3]);
        let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
        let b = OwnedTensor::new(data, vec![3, 2]);
        let mut c = OwnedTensor::zeros(vec![2, 2]);
        matmul(&a, &b, &mut c);
        assert_eq!(c.data(), &[16., 19., 52., 64.]);

        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let a = OwnedTensor::new(data, vec![2, 2, 3]);
        let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
        let b = OwnedTensor::new(data, vec![2, 3, 2]);
        let mut c = OwnedTensor::zeros(vec![2, 2, 2]);
        matmul(&a, &b, &mut c);
        assert_eq!(c.data(), &[16., 19., 52., 64., 214., 235., 304., 334.]);
    }

    #[test]
    fn simple_matmul_t() {
        let a = OwnedTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        // A.T
        let b = ViewTensor::new(&[1.0, 3.0, 2.0, 4.0], vec![2, 2]);
        let mut c = OwnedTensor::zeros(vec![2, 2]);

        matmul_t(&a, &b, &mut c);
        assert_eq!(c.data, &[7.0, 10.0, 15.0, 22.0]);

        let a = OwnedTensor::new(vec![1.0, 2.0], vec![2, 1]);
        let b = ViewTensor::new(&[3.0, 4.0], vec![2, 1]);
        let mut c = OwnedTensor::zeros(vec![2, 2]);
        matmul_t(&a, &b, &mut c);
        assert_eq!(c.data, &[3.0, 4.0, 6.0, 8.0]);

        let data: Vec<_> = (0..6).map(|i| i as f32).collect();
        let a = OwnedTensor::new(data, vec![2, 3]);
        let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
        let b = OwnedTensor::new(data, vec![2, 3]);
        let mut c = OwnedTensor::zeros(vec![2, 2]);
        matmul_t(&a, &b, &mut c);
        assert_eq!(c.data(), &[11., 20., 38., 74.]);

        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let a = OwnedTensor::new(data, vec![2, 2, 3]);
        let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
        let b = OwnedTensor::new(data, vec![2, 2, 3]);
        let mut c = OwnedTensor::zeros(vec![2, 2, 2]);
        matmul_t(&a, &b, &mut c);
        assert_eq!(c.data(), &[11., 20., 38., 74., 191., 254., 272., 362.]);
    }

    #[test]
    fn simple_softmax() {
        let mut a = OwnedTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut max = vec![0.0; 2];
        softmax(&mut a, &mut max);
        assert_eq!(
            simplify(&a.data[..]),
            // Values obtained through python
            [0.2689, 0.7311, 0.2689, 0.7311]
        );
    }

    #[test]
    fn simple_causal_softmax() {
        let mut a = OwnedTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        // Large enough for the second test
        let mut max = vec![0.0; 3 * 2];
        causal_softmax(&mut a, &mut max, 0);
        assert_eq!(
            simplify(&a.data[..]),
            // Values obtained through python
            [1.0000, 0.0000, 0.2689, 0.7311]
        );

        let mut a = OwnedTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        causal_softmax(&mut a, &mut max, 1);
        assert_eq!(
            simplify(&a.data[..]),
            // Values obtained through python
            [0.2689, 0.7311, 0.2689, 0.7311]
        );

        let data: Vec<_> = (0..12).map(|i| (i + 1) as f32).collect();
        let mut a = OwnedTensor::new(data, vec![3, 2, 2]);
        causal_softmax(&mut a, &mut max, 0);
        assert_eq!(
            simplify(&a.data[..]),
            // Values obtained through python
            [
                1.0000, 0.0000, 0.2689, 0.7311, 1.0000, 0.0000, 0.2689, 0.7311, 1.0000, 0.0000,
                0.2689, 0.7311
            ]
        );

        let data: Vec<_> = (0..12).map(|i| (i + 1) as f32).collect();
        let mut a = OwnedTensor::new(data, vec![2, 2, 3]);
        causal_softmax(&mut a, &mut max, 1);
        assert_eq!(
            simplify(&a.data[..]),
            // Values obtained through python
            [
                0.2689, 0.7311, 0.0, 0.09, 0.2447, 0.6652, 0.2689, 0.7311, 0.0, 0.09, 0.2447,
                0.6652
            ]
        );
    }

    #[test]
    fn simple_select() {
        let a = ViewTensor::new(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut tensor = OwnedTensor::new(vec![0.0; 6], vec![3, 2]);
        select(&[1, 0, 0], &a, &mut tensor);
        assert_eq!(
            simplify(&tensor.data[..]),
            // Values obtained through python
            [3.0, 4.0, 1.0, 2.0, 1.0, 2.0]
        );
    }
}
