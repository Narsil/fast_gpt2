use crate::tensor::to_f32;
use dfdx::nn::{BuildOnDevice, Linear, Module, UnbiasedLinear};
use dfdx::prelude::{
    Axis, BuildModule, Const, Device, Dtype, Embedding, LayerNorm1D, Rank0, Rank1, Rank2, Rank3,
    Shape, Tensor, TensorFrom, ZerosTensor,
};
use dfdx::shapes::{Dim, Dyn, HasShape};
use dfdx::tensor::AsArray;
use dfdx::tensor_ops::{GatherTo, MaxTo, PermuteTo, ReshapeTo, TryMatMul};
use safetensors::tensor::{SafeTensorError, SafeTensors, TensorView};
use std::borrow::Cow;

#[cfg(not(feature = "cuda"))]
use dfdx::prelude::Cpu;
#[cfg(not(feature = "cuda"))]
type Dev = Cpu;

#[cfg(feature = "cuda")]
use dfdx::prelude::Cuda;
#[cfg(feature = "cuda")]
type Dev = Cuda;
const HIDDEN_DIM: usize = 768;
const NUM_HEADS: usize = 12;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS;
const NUM_LAYERS: usize = 12;
const VOCAB_SIZE: usize = 50257;
const FF_DIM: usize = HIDDEN_DIM * 4;
const PAST: char = 'P';
const SEQ: char = 'S';
const PAST_PLUS_SEQ: char = 'P';
type HiddenShape = (Dyn<SEQ>, Const<HIDDEN_DIM>);

pub struct PastKeyValue {
    pub key: Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, Dyn<PAST>), f32, Dev>,
    pub value: Tensor<(Const<NUM_HEADS>, Dyn<PAST>, Const<HEAD_DIM>), f32, Dev>,
}

impl PastKeyValue {
    pub fn new(past_sequence_length: usize) -> Self {
        let dev: Dev = Default::default();
        let past_sequence_length = Dyn::<PAST>(past_sequence_length);
        let key: Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, Dyn<PAST>), _, _> =
            dev.zeros_like(&(Const, Const, past_sequence_length));
        let value: Tensor<(Const<NUM_HEADS>, Dyn<PAST>, Const<HEAD_DIM>), _, _> =
            dev.zeros_like(&(Const, past_sequence_length, Const));
        Self { key, value }
    }
}

pub type PastKeyValues = Vec<PastKeyValue>;

#[derive(Clone)]
pub struct Mlp {
    c_fc: Linear<HIDDEN_DIM, FF_DIM, Dev>,
    c_proj: Linear<FF_DIM, HIDDEN_DIM, Dev>,
}

fn linear_from<const I: usize, const O: usize, D: Device<f32>>(
    weight: TensorView,
    bias: TensorView,
) -> Linear<I, O, D> {
    let dev: D = Default::default();
    let mut linear: Linear<I, O, D> = BuildModule::build(&dev);
    linear.weight.copy_from(to_f32(&weight));
    linear.bias.copy_from(to_f32(&bias));
    linear
}

fn layer_norm_from<const M: usize, D: Device<f32>>(
    weight: TensorView,
    bias: TensorView,
) -> LayerNorm1D<M, D> {
    let dev: D = Default::default();
    let mut layer_norm: LayerNorm1D<M, D> = BuildModule::build(&dev);
    layer_norm.gamma.copy_from(to_f32(&weight));
    layer_norm.beta.copy_from(to_f32(&bias));
    layer_norm
}

impl Mlp {
    fn from_tensors(index: usize, tensors: &SafeTensors) -> Result<Self, SafeTensorError> {
        let dev: Dev = Default::default();
        let c_fc = linear_from(
            tensors.tensor(&format!("h.{index}.mlp.c_fc.weight"))?,
            tensors.tensor(&format!("h.{index}.mlp.c_fc.bias"))?,
        );
        let c_proj = linear_from(
            tensors.tensor(&format!("h.{index}.mlp.c_proj.weight"))?,
            tensors.tensor(&format!("h.{index}.mlp.c_proj.bias"))?,
        );
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, tensor: Tensor<HiddenShape, f32, Dev>) -> Tensor<HiddenShape, f32, Dev> {
        let tensor = self.c_fc.forward(tensor);
        let tensor = tensor.gelu();
        self.c_proj.forward(tensor)
    }
}

#[derive(Clone)]
pub struct Attention {
    c_attn: Linear<HIDDEN_DIM, { 3 * HIDDEN_DIM }, Dev>,
    c_proj: Linear<HIDDEN_DIM, HIDDEN_DIM, Dev>,
}

impl Attention {
    fn from_tensors(
        index: usize,
        tensors: &SafeTensors,
        num_heads: usize,
    ) -> Result<Self, SafeTensorError> {
        let c_attn = linear_from(
            tensors.tensor(&format!("h.{index}.attn.c_attn.weight"))?,
            tensors.tensor(&format!("h.{index}.attn.c_attn.bias"))?,
        );
        let c_proj = linear_from(
            tensors.tensor(&format!("h.{index}.attn.c_proj.weight"))?,
            tensors.tensor(&format!("h.{index}.attn.c_proj.bias"))?,
        );
        Ok(Self { c_attn, c_proj })
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<HiddenShape, f32, Dev>,
        past: &mut PastKeyValue,
    ) -> Tensor<HiddenShape, f32, Dev> {
        type PreSplitHead = (Dyn<SEQ>, Const<NUM_HEADS>, Const<HEAD_DIM>);
        type SplitQuery = (Const<NUM_HEADS>, Dyn<SEQ>, Const<HEAD_DIM>);
        type SplitKeys = (Const<NUM_HEADS>, Const<HEAD_DIM>, Dyn<PAST_PLUS_SEQ>);
        type SplitValues = (Const<NUM_HEADS>, Dyn<PAST_PLUS_SEQ>, Const<HEAD_DIM>);
        type Weights = (Const<NUM_HEADS>, Dyn<SEQ>, Const<HEAD_DIM>);

        let sequence_length = hidden_states.shape().0;
        let past_sequence_length = past.key.shape().2;
        let total_length =
            Dyn::<PAST_PLUS_SEQ>(sequence_length.size() + past_sequence_length.size());

        let dev: Dev = Default::default();
        let qkv = self.c_attn.forward(hidden_states);

        let mut qkv_vec = vec![0.0; qkv.shape().num_elements()];
        qkv.copy_into(&mut qkv_vec);

        let mut q: Tensor<SplitQuery, f32, Dev> = dev.zeros_like(&(Const, sequence_length, Const));

        let mut k: Tensor<SplitKeys, f32, Dev> = dev.zeros_like(&(Const, Const, total_length));
        let mut v: Tensor<SplitValues, f32, Dev> = dev.zeros_like(&(Const, total_length, Const));
        let mut q_vec = vec![0.0; q.shape().num_elements()];
        let mut k_vec = vec![0.0; k.shape().num_elements()];
        let mut v_vec = vec![0.0; v.shape().num_elements()];
        let mut past_key_vec = vec![0.0; past.key.shape().num_elements()];
        let mut past_value_vec = vec![0.0; past.value.shape().num_elements()];
        past.key.copy_into(&mut past_key_vec);
        past.value.copy_into(&mut past_value_vec);

        let head_dim = HEAD_DIM;
        let hidden_dim = HIDDEN_DIM;
        let num_heads = NUM_HEADS;
        (0..num_heads).for_each(|i| {
            (0..sequence_length.size()).for_each(|j| {
                (0..head_dim).for_each(|k| {
                    let index = j * hidden_dim * 3 + i * head_dim + k;
                    let out_index = i * sequence_length.size() * head_dim + j * head_dim + k;
                    let value = qkv_vec[index];
                    q_vec[out_index] = value;
                });
            });
        });
        (0..num_heads).for_each(|i| {
            (0..past_sequence_length.size() + sequence_length.size()).for_each(|j| {
                (0..head_dim).for_each(|k| {
                    let in_index_k =
                        i * (past_sequence_length.size() + sequence_length.size()) * head_dim
                            + k * (past_sequence_length.size() + sequence_length.size())
                            + j;

                    let in_index_v =
                        i * (past_sequence_length.size() + sequence_length.size()) * head_dim
                            + j * head_dim
                            + k;
                    if j < past_sequence_length.size() {
                        let k_index = i * past_sequence_length.size() * head_dim
                            + k * past_sequence_length.size()
                            + j;
                        let k_value = past_key_vec[k_index];
                        k_vec[in_index_k] = k_value;

                        let v_index = i * past_sequence_length.size() * head_dim + j * head_dim + k;
                        let v_value = past_value_vec[v_index];
                        v_vec[in_index_v] = v_value;
                    } else {
                        let sj = j - past_sequence_length.size();
                        let k_index = sj * hidden_dim * 3 + i * head_dim + hidden_dim + k;
                        let k_value = qkv_vec[k_index];
                        k_vec[in_index_k] = k_value;

                        let v_index = sj * hidden_dim * 3 + i * head_dim + hidden_dim * 2 + k;
                        let v_value = qkv_vec[v_index];
                        v_vec[in_index_v] = v_value;
                    }
                });
            });
        });
        q.copy_from(&q_vec);
        k.copy_from(&k_vec);
        v.copy_from(&v_vec);

        println!("Q vec {:?} {:?}", &q_vec[..5], &q_vec[q_vec.len() - 5..]);

        past.key = k.clone();
        past.value = v.clone();

        // Get weights
        let scalar: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
        let weights = q.matmul(k) * scalar;
        let weights = weights.softmax::<Axis<2>>();

        // Get new tokens
        let tokens: Tensor<Weights, f32, Dev> = weights.try_matmul(v).unwrap();
        let tokens = tokens.permute::<PreSplitHead, _>();
        // TODO XXX: For some reason this fails.
        // let tokens = tokens.reshape::<HiddenShape>();
        let mut tokens2: Tensor<HiddenShape, f32, Dev> = dev.zeros_like(&(sequence_length, Const));

        let mut tokens_vec = vec![0.0; tokens.shape().num_elements()];
        tokens.copy_into(&mut tokens_vec);
        tokens2.copy_from(&tokens_vec);

        self.c_proj.forward(tokens2)
    }
}

#[derive(Clone)]
pub struct Gpt2Layer {
    ln_1: LayerNorm1D<HIDDEN_DIM, Dev>,
    ln_2: LayerNorm1D<HIDDEN_DIM, Dev>,
    mlp: Mlp,
    attention: Attention,
}

impl Gpt2Layer {
    fn from_tensors(
        index: usize,
        tensors: &SafeTensors,
        num_heads: usize,
    ) -> Result<Self, SafeTensorError> {
        let dev: Dev = Default::default();
        let ln_1 = layer_norm_from(
            tensors.tensor(&format!("h.{index}.ln_1.weight"))?,
            tensors.tensor(&format!("h.{index}.ln_1.bias"))?,
        );
        let ln_2 = layer_norm_from(
            tensors.tensor(&format!("h.{index}.ln_2.weight"))?,
            tensors.tensor(&format!("h.{index}.ln_2.bias"))?,
        );
        let mlp = Mlp::from_tensors(index, tensors)?;
        let attention = Attention::from_tensors(index, tensors, num_heads)?;
        Ok(Self {
            ln_1,
            ln_2,
            mlp,
            attention,
        })
    }

    fn forward(
        &self,
        tensor: Tensor<HiddenShape, f32, Dev>,
        past_key_value: &mut PastKeyValue,
    ) -> Tensor<HiddenShape, f32, Dev> {
        let residual = tensor.clone();
        let tensor = self.ln_1.forward(tensor);
        let tensor = self.attention.forward(tensor, past_key_value);
        let tensor = tensor + residual;
        let residual = tensor.clone();
        let tensor = self.ln_2.forward(tensor);
        let tensor = self.mlp.forward(tensor);
        let tensor = tensor + residual;
        tensor
    }
}
#[derive(Clone)]
pub struct Gpt2Model {
    layers: Vec<Gpt2Layer>,
}

impl Gpt2Model {
    fn from_tensors(tensors: &SafeTensors) -> Result<Self, SafeTensorError> {
        let layers: Result<Vec<_>, _> = (0..NUM_LAYERS)
            .map(|i| -> Result<Gpt2Layer, SafeTensorError> {
                Gpt2Layer::from_tensors(i, tensors, NUM_HEADS)
            })
            .collect();
        let layers = layers?;
        Ok(Self { layers })
    }

    fn forward(
        &self,
        mut tensor: Tensor<HiddenShape, f32, Dev>,
        past_key_values: &mut PastKeyValues,
    ) -> Tensor<HiddenShape, f32, Dev> {
        for (layer, past_key_value) in self.layers.iter().zip(past_key_values.iter_mut()) {
            tensor = layer.forward(tensor, past_key_value);
        }
        tensor
    }
}

#[derive(Clone)]
pub struct Gpt2 {
    wte: Embedding<VOCAB_SIZE, HIDDEN_DIM, Dev>,
    wpe: Embedding<VOCAB_SIZE, HIDDEN_DIM, Dev>,
    h: Gpt2Model,
    ln_f: LayerNorm1D<HIDDEN_DIM, Dev>,
    lm_head: UnbiasedLinear<HIDDEN_DIM, VOCAB_SIZE, Dev>,
}

impl Gpt2 {
    pub fn from_tensors<'a>(tensors: &SafeTensors<'a>, num_heads: usize) -> Self {
        assert_eq!(num_heads, NUM_HEADS);
        let dev: Dev = Default::default();
        let wte = BuildModule::build(&dev);
        let wpe = BuildModule::build(&dev);
        let h = Gpt2Model::from_tensors(tensors).unwrap();
        let ln_f = BuildModule::build(&dev);
        let lm_head = BuildModule::build(&dev);
        Self {
            wte,
            wpe,
            h,
            ln_f,
            lm_head,
        }
    }

    pub fn empty_past_key_values(&self) -> PastKeyValues {
        (0..NUM_LAYERS).map(|_| PastKeyValue::new(0)).collect()
    }

    pub fn forward(&self, ids: &[u32], past: &mut PastKeyValues) -> usize {
        let dev: Dev = Default::default();
        let n = ids.len();
        let past_sequence_length = past[0].key.shape().1.size();
        let ids: Vec<usize> = ids.iter().map(|id| *id as usize).collect();
        let positions: Vec<usize> = (0..ids.len()).map(|i| (i + past_sequence_length)).collect();

        let nn = Dyn::<SEQ>(n);
        let mut input_ids: Tensor<(Dyn<SEQ>,), usize, Dev> = dev.zeros_like(&(nn,));
        input_ids.copy_from(&ids);

        let mut position_ids: Tensor<(Dyn<SEQ>,), usize, Dev> = dev.zeros_like(&(nn,));
        position_ids.copy_from(&positions);

        let input_embeds = self.wte.forward(input_ids);
        let position_embeds = self.wpe.forward(position_ids);
        let embeds = input_embeds + position_embeds;
        let x = self.h.forward(embeds, past);
        let x = self.ln_f.forward(x);
        let y = self.lm_head.forward(x);
        let last_logits: Tensor<Rank2<1, VOCAB_SIZE>, _, _> = y.gather(dev.tensor([n - 1]));

        let mut argmax = 0;
        let mut max = f32::NEG_INFINITY;
        for (index, value) in last_logits.array()[0].into_iter().enumerate() {
            if value > max {
                max = value;
                argmax = index;
            }
        }
        argmax
    }
}
