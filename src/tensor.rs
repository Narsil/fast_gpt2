use crate::ops::{add, add_owned, matmul, matmul_t};
use safetensors::tensor::{Dtype, TensorView};

pub struct Tensor<'data> {
    pub(crate) shape: Vec<usize>,
    pub(crate) data: &'data [f32],
}

impl<'data> Tensor<'data> {
    pub fn new(data: &'data [f32], shape: Vec<usize>) -> Self {
        Self { shape, data }
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        self.data
    }
}

impl<'data> From<TensorView<'data>> for Tensor<'data> {
    fn from(view: TensorView<'data>) -> Self {
        assert_eq!(view.dtype(), Dtype::F32);
        let v = view.data();
        let data: &[f32] = if (v.as_ptr() as usize) % 4 == 0 {
            // SAFETY This is safe because we just checked that this
            // was correctly aligned.
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) }
        } else {
            let mut c = Vec::with_capacity(v.len() / 4);
            let mut i = 0;
            while i < v.len() {
                c.push(f32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]));
                i += 4;
            }
            let c: &'static Vec<f32> = Box::leak(Box::new(c));
            c
        };

        Self::new(data, view.shape().to_vec())
    }
}

#[derive(Debug)]
pub struct OwnedTensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) data: Vec<f32>,
}

impl OwnedTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { shape, data }
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[cfg(test)]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn add_tensor(&mut self, other: &OwnedTensor) {
        add_owned(other, self)
    }

    pub fn addmm(&self, a: &Tensor, b: &Tensor) -> OwnedTensor {
        let m = self.shape()[0];
        let k = self.shape()[1];
        let n = a.shape()[1];
        assert_eq!(k, a.shape()[0]);
        assert_eq!(n, b.shape()[0]);

        let shape = vec![m, n];
        let len = m * n;
        let mut c = OwnedTensor::new(vec![0.0; len], shape);
        matmul(self, a, &mut c);
        add(b, &mut c);
        c
    }

    pub fn matmul_t(&self, a: &Tensor) -> OwnedTensor {
        let m = self.shape()[0];
        let k = self.shape()[1];
        let n = a.shape()[0];
        assert_eq!(
            k,
            a.shape()[1],
            "matmul transposed A {:?} B {:?}",
            self.shape(),
            a.shape()
        );

        let shape = vec![m, n];
        let len = m * n;
        let mut c = OwnedTensor::new(vec![0.0; len], shape);
        matmul_t(self, a, &mut c);
        c
    }

    pub fn select(mut self, ids: &[u32], weights: &Tensor) -> OwnedTensor {
        let _vocab_size = weights.shape()[0];
        let hidden_dim = weights.shape()[1];
        let sequence_length = ids.len();
        assert_eq!(self.shape(), [sequence_length, hidden_dim]);
        for (i, id) in ids.iter().enumerate() {
            let id = *id as usize;
            let weight_offset = id * hidden_dim;
            let data_offset = i * hidden_dim;
            self.data[data_offset..data_offset + hidden_dim]
                .copy_from_slice(&weights.data()[weight_offset..weight_offset + hidden_dim]);
        }
        self
    }
}
