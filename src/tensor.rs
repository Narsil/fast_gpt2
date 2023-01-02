use crate::ops::{add, matmul};
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

    pub fn _item(&self) -> f32 {
        self.data[0]
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<'data> From<TensorView<'data>> for Tensor<'data> {
    fn from(view: TensorView<'data>) -> Self {
        assert_eq!(view.dtype(), Dtype::F32);
        let v = view.data();
        let data: &[f32] =
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) };

        Self {
            shape: view.shape().to_vec(),
            data,
        }
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

    pub fn add(&mut self, value: f32) {
        self.data.iter_mut().for_each(|i| *i += value);
    }

    pub fn addmm(&self, a: &Tensor, b: &Tensor) -> OwnedTensor {
        let m = self.shape()[0];
        let k = self.shape()[1];
        let n = a.shape()[1];

        let shape = vec![m, n];
        let len = m * n;
        let mut c = OwnedTensor::new(vec![0.0; len], shape);
        matmul(self, a, &mut c);
        add(b, &mut c);
        c
    }
}
