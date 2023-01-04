use crate::ops::{add, mul};
use crate::tensor::{OwnedTensor, Tensor};
use safetensors::tensor::{SafeTensors, TensorView};

pub struct Mlp<'a> {
    c_fc: Linear<'a>,
    c_proj: Linear<'a>,
}

impl<'a> Mlp<'a> {
    fn from_tensors(index: usize, tensors: &'a SafeTensors<'a>) -> Self {
        let c_fc = Linear::from(
            tensors
                .tensor(&format!("h.{index}.mlp.c_fc.weight"))
                .unwrap(),
            tensors.tensor(&format!("h.{index}.mlp.c_fc.bias")).unwrap(),
        );
        let c_proj = Linear::from(
            tensors
                .tensor(&format!("h.{index}.mlp.c_proj.weight"))
                .unwrap(),
            tensors
                .tensor(&format!("h.{index}.mlp.c_proj.bias"))
                .unwrap(),
        );
        Self { c_fc, c_proj }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        self.c_fc.forward(tensor);
        self.c_proj.forward(tensor);
    }
}

pub struct Attention<'a> {
    bias: Tensor<'a>,
    c_attn: Linear<'a>,
    c_proj: Linear<'a>,
}

impl<'a> Attention<'a> {
    fn from_tensors(index: usize, tensors: &'a SafeTensors<'a>) -> Self {
        let bias: Tensor = tensors
            .tensor(&format!("h.{index}.attn.bias"))
            .unwrap()
            .into();
        // TODO Implement it for real with `c_attn` tensor
        let c_attn = Linear::from(
            tensors
                .tensor(&format!("h.{index}.attn.c_proj.weight"))
                .unwrap(),
            tensors
                .tensor(&format!("h.{index}.attn.c_proj.bias"))
                .unwrap(),
        );
        let c_proj = Linear::from(
            tensors
                .tensor(&format!("h.{index}.attn.c_proj.weight"))
                .unwrap(),
            tensors
                .tensor(&format!("h.{index}.attn.c_proj.bias"))
                .unwrap(),
        );
        Self {
            c_attn,
            c_proj,
            bias,
        }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        // println!("Bias {:?}", self.bias.shape());
        tensor.add(self.bias._item());
        // println!("c_attn {:?}", self.c_attn);
        self.c_attn.forward(tensor);
        // println!("c_proj {:?}", self.c_attn);
        self.c_proj.forward(tensor);
    }
}

pub struct Gpt2Layer<'a> {
    ln_1: LayerNorm<'a>,
    ln_2: LayerNorm<'a>,
    mlp: Mlp<'a>,
    attention: Attention<'a>,
}

impl<'a> Gpt2Layer<'a> {
    fn from_tensors(index: usize, tensors: &'a SafeTensors<'a>) -> Self {
        let ln_1 = LayerNorm::from(
            tensors.tensor(&format!("h.{index}.ln_1.weight")).unwrap(),
            tensors.tensor(&format!("h.{index}.ln_1.bias")).unwrap(),
        );
        let ln_2 = LayerNorm::from(
            tensors.tensor(&format!("h.{index}.ln_2.weight")).unwrap(),
            tensors.tensor(&format!("h.{index}.ln_2.bias")).unwrap(),
        );
        let mlp = Mlp::from_tensors(index, tensors);
        let attention = Attention::from_tensors(index, tensors);
        Self {
            ln_1,
            ln_2,
            mlp,
            attention,
        }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        self.ln_1.forward(tensor);
        self.attention.forward(tensor);
        self.ln_2.forward(tensor);
        self.mlp.forward(tensor);
    }
}

pub struct Gpt2Model<'a> {
    layers: Vec<Gpt2Layer<'a>>,
}

impl<'a> Gpt2Model<'a> {
    fn from_tensors(tensors: &'a SafeTensors<'a>) -> Self {
        let layers: Vec<_> = (0..12)
            .map(|i| Gpt2Layer::from_tensors(i, tensors))
            .collect();
        Self { layers }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        for layer in &self.layers {
            layer.forward(tensor);
        }
    }
}

pub struct Linear<'a> {
    weight: Tensor<'a>,
    bias: Tensor<'a>,
}

impl<'a> std::fmt::Debug for Linear<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("shape", &self.weight.shape())
            .finish()
    }
}

impl<'a> Linear<'a> {
    fn from(weight: TensorView<'a>, bias: TensorView<'a>) -> Self {
        let weight: Tensor = weight.into();
        let bias: Tensor = bias.into();
        Self { weight, bias }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        let c = tensor.addmm(&self.weight, &self.bias);
        *tensor = c;
    }
}

pub struct LayerNorm<'a> {
    weight: Tensor<'a>,
    bias: Tensor<'a>,
}

impl<'a> LayerNorm<'a> {
    fn from(weight: TensorView<'a>, bias: TensorView<'a>) -> Self {
        let weight: Tensor = weight.into();
        let bias: Tensor = bias.into();
        Self { weight, bias }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        mul(&self.weight, tensor);
        add(&self.bias, tensor);
    }
}

pub struct Gpt2<'a> {
    h: Gpt2Model<'a>,
    ln_f: LayerNorm<'a>,
}

impl<'a> Gpt2<'a> {
    pub fn from_tensors(tensors: &'a SafeTensors<'a>) -> Self {
        let h = Gpt2Model::from_tensors(tensors);
        let ln_f = LayerNorm::from(
            tensors.tensor("ln_f.weight").unwrap(),
            tensors.tensor("ln_f.bias").unwrap(),
        );
        Self { h, ln_f }
    }
}

impl<'a> Gpt2<'a> {
    pub fn forward(&self, _ids: &[u32]) -> OwnedTensor {
        let mut tensor = OwnedTensor::new(vec![0.0; 768 * 2], vec![2, 768]);
        self.h.forward(&mut tensor);
        self.ln_f.forward(&mut tensor);
        tensor
    }
}
