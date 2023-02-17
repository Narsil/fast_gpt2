use safetensors::tensor::{Dtype, TensorView};

pub struct PastKeyValue {
    pub key: OwnedTensor,
    pub value: OwnedTensor,
}

impl PastKeyValue {
    pub fn new(num_heads: usize, past_sequence_length: usize, head_dim: usize) -> Self {
        let key = OwnedTensor::new(vec![], vec![num_heads, past_sequence_length, head_dim]);
        let value = OwnedTensor::new(vec![], vec![num_heads, past_sequence_length, head_dim]);
        Self { key, value }
    }
}

pub type PastKeyValues = Vec<PastKeyValue>;

pub trait Tensor {
    fn as_ptr(&self) -> *const f32 {
        self.data().as_ptr()
    }
    fn shape(&self) -> &[usize];
    fn data(&self) -> &[f32];
}

pub trait TensorMut: Tensor {
    fn data_mut(&mut self) -> &mut [f32];
    fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data_mut().as_mut_ptr()
    }
    fn zeros(shape: Vec<usize>) -> Self;
}

#[derive(Clone)]
pub struct ViewTensor<'data> {
    pub shape: Vec<usize>,
    pub data: &'data [f32],
}

impl<'data> Tensor for ViewTensor<'data> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> &[f32] {
        self.data
    }
}

impl<'data> ViewTensor<'data> {
    pub fn new(data: &'data [f32], shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { shape, data }
    }
}

pub fn to_f32<'data>(view: &TensorView<'data>) -> &'data [f32] {
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
    data
}
impl<'data> From<TensorView<'data>> for ViewTensor<'data> {
    fn from(view: TensorView<'data>) -> Self {
        let data = to_f32(&view);
        Self::new(data, view.shape().to_vec())
    }
}

#[derive(Debug, Clone)]
pub struct OwnedTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor for OwnedTensor {
    fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> &[f32] {
        &self.data
    }
}

impl TensorMut for OwnedTensor {
    fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
    fn zeros(shape: Vec<usize>) -> Self {
        let nelement: usize = shape.iter().product();
        let data = vec![0.0; nelement];
        Self { shape, data }
    }
}
impl OwnedTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { shape, data }
    }
}
