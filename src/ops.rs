use crate::tensor::{Tensor, TensorMut};

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

#[inline]
pub fn addmm<X: Tensor, A: Tensor, B: Tensor, TM: Tensor + TensorMut>(
    x: &X,
    a: &A,
    b: &B,
    out: &mut TM,
) {
    let m = x.shape()[0];
    let k = x.shape()[1];
    let n = a.shape()[1];
    assert_eq!(k, a.shape()[0]);
    assert_eq!(n, b.shape()[0]);
    assert_eq!(out.shape(), &[m, n]);

    matmul(x, a, out);
    add(b, out);
}

pub fn select<T: Tensor, TM: Tensor + TensorMut>(ids: &[u32], weights: &T, out: &mut TM) {
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

#[inline]
pub fn matmul_t<A: Tensor, B: Tensor, TM: Tensor + TensorMut>(a: &A, b: &B, c: &mut TM) {
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
pub fn matmul<A: Tensor, B: Tensor, TM: TensorMut>(a: &A, b: &B, c: &mut TM) {
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

pub fn add<T: Tensor, TM: Tensor + TensorMut>(a: &T, b: &mut TM) {
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

pub fn mul<T: Tensor, TM: Tensor + TensorMut>(a: &T, b: &mut TM) {
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

pub fn normalize<TM: Tensor + TensorMut>(x: &mut TM, epsilon: f32) {
    assert_eq!(x.shape().len(), 2);
    let size = x.shape()[1];

    let mut mean: Vec<f32> = vec![0.0; x.shape()[0]];
    let mut var: Vec<f32> = vec![0.0; x.shape()[0]];
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
fn g_softmax<const CAUSAL: bool, TM: Tensor + TensorMut>(x: &mut TM) {
    let m = x.shape()[x.shape().len() - 2];
    let n = x.shape()[x.shape().len() - 1];
    let mut b = 1;
    for s in &x.shape()[..x.shape().len() - 2] {
        b *= s;
    }
    let mut max = vec![0.0; m * b];
    let mut current_max = f32::NEG_INFINITY;
    for (ii, &v) in x.data().iter().enumerate() {
        let i = ii / n;
        let j = ii % n;
        if (!CAUSAL || i >= j) && v > current_max {
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
    let mut softmax = vec![0.0; m * b];
    let mut sum = 0.0;
    for (ii, v) in x.data().iter().enumerate() {
        let i = (ii / n) % m;
        let j = ii % n;
        if !CAUSAL || i >= j {
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
        if !CAUSAL || i >= j {
            *v /= softmax[ii / n];
        } else {
            *v = 0.0;
        }
    });
}

#[cfg(test)]
pub fn softmax<TM: Tensor + TensorMut>(x: &mut TM) {
    g_softmax::<false, TM>(x)
}
pub fn causal_softmax<TM: Tensor + TensorMut>(x: &mut TM) {
    g_softmax::<true, TM>(x)
}

fn attention_matmul_qk<T: Tensor, TM: Tensor + TensorMut>(qkv: &T, qk: &mut TM) {
    // qkv = [S, 3H]
    // qk = [NH, S, S]
    // q = qkv[:, :H], k = qkv[:, H: 2H], v = qkv[:, 2H: 3H]
    //
    // --HEADS--
    // q[S, H] -> q[S, NH, HH] -> q[NH, S, HH]
    // k[S, H] -> k[S, NH, HH] -> k[NH, S, HH]
    // qk[i, j, k] = sum(q[i, j, l] * k[i, k, l]) over l
    let num_heads = qk.shape()[0];
    let sequence_length = qk.shape()[1];
    let hidden_dim3 = qkv.shape()[1];
    assert_eq!(qk.shape()[2], sequence_length);
    assert_eq!(qkv.shape()[0], sequence_length);
    assert_eq!(hidden_dim3 % 3, 0);
    let hidden_dim = hidden_dim3 / 3;
    assert_eq!(hidden_dim % num_heads, 0);
    let head_dim = hidden_dim / num_heads;

    (0..num_heads).for_each(|i| {
        (0..sequence_length).for_each(|j| {
            (0..sequence_length).for_each(|k| {
                let index = i * sequence_length * sequence_length + j * sequence_length + k;
                let sum = (0..head_dim)
                    .map(|l| {
                        let index_q = j * hidden_dim3 + i * head_dim + l;
                        let index_k = k * hidden_dim3 + i * head_dim + l + hidden_dim;
                        let q = qkv.data()[index_q];
                        let k = qkv.data()[index_k];
                        q * k
                    })
                    .sum();
                qk.data_mut()[index] = sum;
            });
        });
    });
}

fn attention_matmul_qkv<QK: Tensor, T: Tensor, TM: Tensor + TensorMut>(
    qk: &QK,
    qkv: &T,
    out: &mut TM,
) {
    // qkv = [S, 3H]
    // v = qkv[S, 2H: 3H]
    // out = [S, H]
    //
    // --HEADS--
    // v[S, H] -> v[S, NH, HH] -> v[NH, S, HH]
    // out = [S, H]
    // qk = [NH, S, S]
    // v[NH, S, HH]
    // out[i, j, k] = sum(qk[i, j, l] * v[i, l, k]) over l
    let num_heads = qk.shape()[0];
    let sequence_length = qk.shape()[1];
    let hidden_dim = out.shape()[1];
    assert_eq!(qk.shape()[2], sequence_length);
    assert_eq!(out.shape()[0], sequence_length);
    assert_eq!(hidden_dim % num_heads, 0);
    let head_dim = hidden_dim / num_heads;
    assert_eq!(qkv.shape(), vec![sequence_length, hidden_dim * 3]);
    (0..num_heads).for_each(|head_index| {
        (0..sequence_length).for_each(|lseq_index| {
            (0..head_dim).for_each(|hh| {
                let index = lseq_index * hidden_dim + head_index * head_dim + hh;
                // println!("Index {index:?}");
                out.data_mut()[index] = (0..sequence_length)
                    .map(|l| {
                        let index_qk = head_index * sequence_length * sequence_length
                            + lseq_index * sequence_length
                            + l;
                        let index_v =
                            l * hidden_dim * 3 + head_index * head_dim + hh + hidden_dim * 2;
                        let qk = qk.data()[index_qk];
                        let value = qkv.data()[index_v];
                        // println!("Value {value:?}");
                        qk * value
                    })
                    .sum();
            });
        })
    });
}

pub fn attention<T: Tensor, TM: Tensor + TensorMut, OUT: Tensor + TensorMut>(
    qkv: &T,
    qk: &mut TM,
    out: &mut OUT,
) {
    let sequence_length = qkv.shape()[0];
    let hidden_dim3 = qkv.shape()[1];
    assert_eq!(hidden_dim3 % 3, 0);
    let hidden_dim = hidden_dim3 / 3;
    let num_heads = qk.shape()[0];
    assert_eq!(
        qk.shape(),
        vec![num_heads, sequence_length, sequence_length]
    );
    assert_eq!(out.shape(), vec![sequence_length, hidden_dim]);

    attention_matmul_qk(qkv, qk);
    // println!("After qk {:?}", &qk.data()[qk.data().len() - 10..]);
    // Need to do scale_weights
    let head_dim = hidden_dim / num_heads;
    let scale = (head_dim as f32).sqrt();
    qk.data_mut().iter_mut().for_each(|v| *v /= scale);
    // println!("After scale {:?}", &qk.data()[qk.data().len() - 10..]);
    causal_softmax(qk);
    // println!(
    //     "After causal_softmax {:?}",
    //     &qk.data()[qk.data().len() - 10..]
    // );
    attention_matmul_qkv(qk, qkv, out);
    // println!("After values {:?}", out.shape());
    // println!("After values {:?}", &out.data()[out.data().len() - 10..]);
    // println!("---------")
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

pub fn gelu<T: TensorMut>(x: &mut T) {
    x.data_mut().iter_mut().for_each(|v| {
        *v = 0.5
            * (*v)
            * (1.0
                + ((2.0f32 / std::f32::consts::PI).sqrt() * (*v + 0.044715 * v.powf(3.0))).tanh())
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Linear;
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
        assert_eq!(c.data, &[3.0, 4.0, 6.0, 8.0])
    }

    #[test]
    fn simple_matmul_t() {
        let a = OwnedTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        // A.T
        let b = ViewTensor::new(&[1.0, 3.0, 2.0, 4.0], vec![2, 2]);
        let data = vec![0.0; 4];
        let mut c = OwnedTensor::new(data, vec![2, 2]);

        matmul_t(&a, &b, &mut c);
        assert_eq!(c.data, &[7.0, 10.0, 15.0, 22.0]);

        let a = OwnedTensor::new(vec![1.0, 2.0], vec![2, 1]);
        let b = ViewTensor::new(&[3.0, 4.0], vec![2, 1]);
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
            [1.0000, 0.0000, 0.2689, 0.7311]
        );

        let data: Vec<_> = (0..12).map(|i| (i + 1) as f32).collect();
        let mut a = OwnedTensor::new(data, vec![3, 2, 2]);
        causal_softmax(&mut a);
        assert_eq!(
            simplify(&a.data[..]),
            // Values obtained through python
            [
                1.0000, 0.0000, 0.2689, 0.7311, 1.0000, 0.0000, 0.2689, 0.7311, 1.0000, 0.0000,
                0.2689, 0.7311
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

    #[test]
    fn simple_attention_matmul_qk() {
        let hidden_dim = 8;
        let num_heads = 2;
        let data = (0..hidden_dim * hidden_dim * 3)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let weight = ViewTensor::new(&data, vec![hidden_dim, hidden_dim * 3]);
        let data = (0..hidden_dim * 3).map(|i| i as f32).collect::<Vec<_>>();
        let bias = ViewTensor::new(&data, vec![hidden_dim * 3]);
        let c_attn = Linear::new(weight, bias);

        let sequence_length = 3;
        let mut qkv = OwnedTensor::new(
            vec![1.0; sequence_length * hidden_dim],
            vec![sequence_length, hidden_dim],
        );
        c_attn.forward(&mut qkv);
        let mut qk = OwnedTensor::new(
            vec![0.0; num_heads * sequence_length * sequence_length],
            vec![num_heads, sequence_length, sequence_length],
        );
        attention_matmul_qk(&qkv, &mut qk);
        assert_eq!(
            qk.data(),
            // from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Config
            // import torch
            //
            // config = GPT2Config(n_embd=8, n_head=2)
            // attn = GPT2Attention(config)
            // attn.c_attn.weight = torch.nn.Parameter(torch.arange(attn.c_attn.weight.nelement()).view(attn.c_attn.weight.shape).float())
            // attn.c_attn.bias = torch.nn.Parameter(torch.arange(attn.c_attn.bias.nelement()).view(attn.c_attn.bias.shape).float())
            //
            // hidden_states = torch.ones((1, 3, 8))
            // qkv = attn.c_attn(hidden_states)
            // query, key, value = qkv.split(attn.split_size, dim=2)
            //
            // query = attn._split_heads(query, attn.num_heads, attn.head_dim)
            // key = attn._split_heads(key, attn.num_heads, attn.head_dim)
            // value = attn._split_heads(value, attn.num_heads, attn.head_dim)
            // key = key.transpose(-1, -2)
            // attn_weights = torch.matmul(query, key)
            // print(attn_weights.view(-1))
            [
                2077470., 2077470., 2077470., 2077470., 2077470., 2077470., 2077470., 2077470.,
                2077470., 2290446., 2290446., 2290446., 2290446., 2290446., 2290446., 2290446.,
                2290446., 2290446.
            ]
        );
    }

    #[test]
    fn simple_attention_matmul_qkv() {
        let hidden_dim = 8;
        let num_heads = 2;
        let sequence_length = 3;

        let data = (0..sequence_length * hidden_dim * 3)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let qkv = OwnedTensor::new(data, vec![sequence_length, hidden_dim * 3]);
        let data = (0..num_heads * sequence_length * sequence_length)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let qk = OwnedTensor::new(data, vec![num_heads, sequence_length, sequence_length]);
        let mut out = OwnedTensor::new(
            vec![0.0; sequence_length * hidden_dim],
            vec![sequence_length, hidden_dim],
        );
        attention_matmul_qkv(&qk, &qkv, &mut out);
        assert_eq!(
            // from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Config
            // import torch
            //
            // config = GPT2Config(n_embd=8, n_head=2)
            // attn = GPT2Attention(config)
            //
            // attn_weights = torch.ones((1, 2, 3, 3))
            // attn_weights = torch.arange(attn_weights.nelement()).view(attn_weights.shape).float()
            //
            // qkv = torch.ones((1, 3, 24))
            // qkv = torch.arange(qkv.nelement()).view(qkv.shape).float()
            //
            // query, key, value = qkv.split(attn.split_size, dim=2)
            // value = attn._split_heads(value, attn.num_heads, attn.head_dim)
            // attn_output = torch.matmul(attn_weights, value)
            // attn_output = attn._merge_heads(attn_output, attn.num_heads, attn.head_dim)
            // print(attn_output)
            // print(attn_output.view(-1))
            simplify(out.data()),
            [
                168., 171., 174., 177., 1368., 1398., 1428., 1458., 528., 540., 552., 564., 1764.,
                1803., 1842., 1881., 888., 909., 930., 951., 2160., 2208., 2256., 2304.
            ]
        );
    }

    #[test]
    fn simple_attention() {
        let hidden_dim = 8;
        let num_heads = 2;
        let data = (0..hidden_dim * hidden_dim * 3)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let weight = ViewTensor::new(&data, vec![hidden_dim, hidden_dim * 3]);
        let data = (0..hidden_dim * 3).map(|i| i as f32).collect::<Vec<_>>();
        let bias = ViewTensor::new(&data, vec![hidden_dim * 3]);
        let c_attn = Linear::new(weight, bias);

        let sequence_length = 3;
        let mut qkv = OwnedTensor::new(
            vec![1.0; sequence_length * hidden_dim],
            vec![sequence_length, hidden_dim],
        );
        c_attn.forward(&mut qkv);
        let mut qk = OwnedTensor::new(
            vec![0.0; num_heads * sequence_length * sequence_length],
            vec![num_heads, sequence_length, sequence_length],
        );

        let mut qv = OwnedTensor::new(
            vec![0.0; sequence_length * hidden_dim],
            vec![sequence_length, hidden_dim],
        );
        attention(&qkv, &mut qk, &mut qv);
        assert_eq!(
            // Values gotten from Python
            // ```python
            // from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Config
            // import torch
            //
            // config = GPT2Config(n_embd=8, n_head=2)
            // attn = GPT2Attention(config)
            // attn.c_attn.weight = torch.nn.Parameter(torch.arange(attn.c_attn.weight.nelement()).view(attn.c_attn.weight.shape).float())
            // attn.c_attn.bias = torch.nn.Parameter(torch.arange(attn.c_attn.bias.nelement()).view(attn.c_attn.bias.shape).float())
            // input = torch.ones((1, 1, 8))
            // print(attn.c_attn(input)[0].view(-1)[:10])
            // print(attn.c_attn(input)[0].view(-1)[-10:])
            // ```
            qv.data(),
            [
                816.0, 825.0, 834.0, 843.0, 852.0, 861.0, 870.0, 879.0, 816.0, 825.0, 834.0, 843.0,
                852.0, 861.0, 870.0, 879.0, 816.0, 825.0, 834.0, 843.0, 852.0, 861.0, 870.0, 879.0
            ]
        );
    }
}
