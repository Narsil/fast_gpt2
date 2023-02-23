use crate::tensor::to_f32;
use dfdx::nn::{
    modules::{Embedding, LayerNorm1D, Linear, UnbiasedLinear},
    Module,
};
use dfdx::prelude::{Axis, BuildModule, Const, Rank2, Shape, Tensor, TensorFrom, ZerosTensor};
use dfdx::shapes::{Dim, Dyn, HasShape};
use dfdx::tensor::TensorFromVec;
use dfdx::tensor::{AsArray, AsVec};
use dfdx::tensor_ops::{GatherTo, PermuteTo, ReshapeTo, TryCat, TryMatMul};
use safetensors::tensor::{SafeTensorError, SafeTensors, TensorView};

#[cfg(not(feature = "cuda"))]
use dfdx::prelude::Cpu;
#[cfg(not(feature = "cuda"))]
pub type Dev = Cpu;

#[cfg(feature = "cuda")]
use dfdx::prelude::Cuda;
#[cfg(feature = "cuda")]
pub type Dev = Cuda;

type FTensor<S> = Tensor<S, f32, Dev>;

#[cfg(feature = "gpt2")]
const HIDDEN_DIM: usize = 768;
#[cfg(feature = "gpt2")]
const NUM_HEADS: usize = 12;
#[cfg(feature = "gpt2")]
const NUM_LAYERS: usize = 12;

#[cfg(feature = "gpt2-medium")]
const HIDDEN_DIM: usize = 1024;
#[cfg(feature = "gpt2-medium")]
const NUM_HEADS: usize = 16;
#[cfg(feature = "gpt2-medium")]
const NUM_LAYERS: usize = 24;

#[cfg(feature = "gpt2-large")]
const HIDDEN_DIM: usize = 1280;
#[cfg(feature = "gpt2-large")]
const NUM_HEADS: usize = 20;
#[cfg(feature = "gpt2-large")]
const NUM_LAYERS: usize = 36;

const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS;
const VOCAB_SIZE: usize = 50257;
const MAX_POSITIONS: usize = 1024;
const FF_DIM: usize = HIDDEN_DIM * 4;
const PAST: char = 'P';
const SEQ: char = 'S';
const PRESENT: char = 'T';
type HiddenShape = (Dyn<SEQ>, Const<HIDDEN_DIM>);
type QkvShape = (Dyn<SEQ>, Const<{ HIDDEN_DIM * 3 }>);
type PastKeyShape = (Const<NUM_HEADS>, Const<HEAD_DIM>, usize);
type PastValueShape = (Const<NUM_HEADS>, usize, Const<HEAD_DIM>);
type SplitQuery = (Const<NUM_HEADS>, Dyn<SEQ>, Const<HEAD_DIM>);
type SplitKeys = (Const<NUM_HEADS>, Const<HEAD_DIM>, usize);
type SplitValues = (Const<NUM_HEADS>, usize, Const<HEAD_DIM>);
type Weights = (Const<NUM_HEADS>, Dyn<SEQ>, Const<HEAD_DIM>);
type UnsplitWeights = (Dyn<SEQ>, Const<NUM_HEADS>, Const<HEAD_DIM>);

pub struct PastKeyValue {
    pub key: FTensor<PastKeyShape>,
    pub value: FTensor<PastValueShape>,
}

impl PastKeyValue {
    pub fn new(past_sequence_length: usize, dev: &Dev) -> Self {
        let past_sequence_length = Dyn::<PAST>(past_sequence_length);
        let key: FTensor<PastKeyShape> =
            dev.zeros_like(&(Const, Const, past_sequence_length.size()));
        let value: FTensor<PastValueShape> =
            dev.zeros_like(&(Const, past_sequence_length.size(), Const));
        Self { key, value }
    }
}

pub type PastKeyValues = Vec<PastKeyValue>;

fn linear_from<const I: usize, const O: usize>(
    weight: TensorView,
    bias: TensorView,
    dev: &Dev,
) -> Linear<I, O, f32, Dev> {
    let mut linear: Linear<I, O, f32, Dev> = BuildModule::build(dev);
    let mut weight_tensor: FTensor<(Const<I>, Const<O>)> = dev.zeros_like(&(Const, Const));
    weight_tensor.copy_from(to_f32(&weight));
    let weight_t: FTensor<(Const<O>, Const<I>)> = weight_tensor.permute();
    linear.weight = weight_t;
    linear.bias.copy_from(to_f32(&bias));
    linear
}

fn unbiased_linear_from<const I: usize, const O: usize>(
    weight: TensorView,
    dev: &Dev,
) -> UnbiasedLinear<I, O, f32, Dev> {
    let mut unbiased_linear: UnbiasedLinear<I, O, f32, Dev> = BuildModule::build(dev);
    unbiased_linear.weight.copy_from(to_f32(&weight));
    unbiased_linear
}

fn layer_norm_from<const M: usize>(
    weight: TensorView,
    bias: TensorView,
    dev: &Dev,
) -> LayerNorm1D<M, f32, Dev> {
    let mut layer_norm: LayerNorm1D<M, f32, Dev> = BuildModule::build(dev);
    layer_norm.gamma.copy_from(to_f32(&weight));
    layer_norm.beta.copy_from(to_f32(&bias));
    layer_norm
}

fn embedding_from<const I: usize, const O: usize>(
    weight: TensorView,
    dev: &Dev,
) -> Embedding<I, O, f32, Dev> {
    let mut embedding: Embedding<I, O, f32, Dev> = BuildModule::build(dev);
    embedding.weight.copy_from(to_f32(&weight));
    embedding
}

fn attention_reshape(
    qkv: &FTensor<QkvShape>,
    past_key: &FTensor<PastKeyShape>,
    past_value: &FTensor<PastValueShape>,
) -> (
    FTensor<SplitQuery>,
    FTensor<SplitKeys>,
    FTensor<SplitValues>,
) {
    let dev = qkv.device().clone();
    let sequence_length = qkv.shape().0;
    let past_sequence_length = past_key.shape().2;
    let total_length = Dyn::<PRESENT>(sequence_length.size() + past_sequence_length.size());

    // let k: FTensor<SplitKeys> = dev.zeros_like(&(Const, Const, total_length.size()));
    // let v: FTensor<SplitValues> = dev.zeros_like(&(Const, total_length.size(), Const));
    // let mut k_vec = vec![0.0; k.shape().num_elements()];
    // let mut v_vec = vec![0.0; v.shape().num_elements()];
    // let mut past_key_vec = vec![0.0; past_key.shape().num_elements()];
    // let mut past_value_vec = vec![0.0; past_value.shape().num_elements()];
    // let mut qkv_vec = vec![0.0; qkv.shape().num_elements()];
    // past_key.copy_into(&mut past_key_vec);
    // past_value.copy_into(&mut past_value_vec);
    // qkv.copy_into(&mut qkv_vec);

    let qkv: FTensor<(Dyn<SEQ>, Const<3>, Const<HIDDEN_DIM>)> = qkv
        .clone()
        .try_reshape_like(&(sequence_length, Const, Const))
        .unwrap();
    let qkv: FTensor<(Const<3>, Dyn<SEQ>, Const<HIDDEN_DIM>)> = qkv.permute();

    let q: FTensor<(Const<1>, Dyn<SEQ>, Const<HIDDEN_DIM>)> = qkv.clone().gather(dev.tensor([0]));
    let q: FTensor<(Dyn<SEQ>, Const<HIDDEN_DIM>)> =
        q.try_reshape_like(&(sequence_length, Const)).unwrap();
    let q: FTensor<(Dyn<SEQ>, Const<NUM_HEADS>, Const<HEAD_DIM>)> = q
        .try_reshape_like(&(sequence_length, Const, Const))
        .unwrap();
    let q: FTensor<(Const<NUM_HEADS>, Dyn<SEQ>, Const<HEAD_DIM>)> = q.permute();

    let k: FTensor<(Const<1>, Dyn<SEQ>, Const<HIDDEN_DIM>)> = qkv.clone().gather(dev.tensor([1]));
    let k: FTensor<(Dyn<SEQ>, Const<HIDDEN_DIM>)> =
        k.try_reshape_like(&(sequence_length, Const)).unwrap();
    let k = if past_sequence_length > 0 {
        let past_key: FTensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize)> = past_key.clone();
        let past_key: FTensor<(Const<HIDDEN_DIM>, usize)> = past_key
            .try_reshape_like(&(Const, past_sequence_length.size()))
            .unwrap();
        let past_key: FTensor<(usize, Const<HIDDEN_DIM>)> = past_key.permute();
        let past_key: FTensor<(usize, Const<HIDDEN_DIM>)> =
            dev.tensor_from_vec(past_key.as_vec(), *past_key.shape());
        let k: FTensor<(usize, Const<HIDDEN_DIM>)> = dev.cat(past_key, k);
        k
    } else {
        let k: FTensor<(usize, Const<HIDDEN_DIM>)> = k.reshape_like(&(total_length.size(), Const));
        let k: FTensor<(usize, Const<HIDDEN_DIM>)> = dev.tensor_from_vec(k.as_vec(), *k.shape());
        k
    };
    let k: FTensor<(usize, Const<NUM_HEADS>, Const<HEAD_DIM>)> = k
        .try_reshape_like(&(total_length.size(), Const, Const))
        .unwrap();
    let k: FTensor<SplitKeys> = k.permute();
    let k: FTensor<SplitKeys> = dev.tensor_from_vec(k.as_vec(), *k.shape());
    // let mut k: FTensor<SplitKeys> = dev.zeros_like(&(Const, Const, total_length.size()));

    let v: FTensor<(Const<1>, Dyn<SEQ>, Const<HIDDEN_DIM>)> = qkv.clone().gather(dev.tensor([2]));
    let v: FTensor<(Dyn<SEQ>, Const<HIDDEN_DIM>)> =
        v.try_reshape_like(&(sequence_length, Const)).unwrap();
    let v = if past_sequence_length > 0 {
        let past_value: FTensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>)> = past_value.clone();
        let past_value: FTensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize)> = past_value.permute();
        let past_value: FTensor<(Const<HIDDEN_DIM>, usize)> = past_value
            .try_reshape_like(&(Const, past_sequence_length.size()))
            .unwrap();
        let past_value: FTensor<(usize, Const<HIDDEN_DIM>)> = past_value.permute();
        let past_value: FTensor<(usize, Const<HIDDEN_DIM>)> =
            dev.tensor_from_vec(past_value.as_vec(), *past_value.shape());
        let v: FTensor<(usize, Const<HIDDEN_DIM>)> = dev.cat(past_value, v);
        v
    } else {
        let v: FTensor<(usize, Const<HIDDEN_DIM>)> = v.reshape_like(&(total_length.size(), Const));
        v
    };
    let v: FTensor<(usize, Const<NUM_HEADS>, Const<HEAD_DIM>)> = v
        .try_reshape_like(&(total_length.size(), Const, Const))
        .unwrap();
    let v: FTensor<SplitValues> = v.permute();
    let v: FTensor<SplitValues> = dev.tensor_from_vec(v.as_vec(), *v.shape());
    // let mut v: FTensor<SplitValues> = dev.zeros_like(&(Const, total_length.size(), Const));

    // let head_dim = HEAD_DIM;
    // let hidden_dim = HIDDEN_DIM;
    // let num_heads = NUM_HEADS;
    // (0..num_heads).for_each(|i| {
    //     (0..past_sequence_length.size() + sequence_length.size()).for_each(|j| {
    //         (0..head_dim).for_each(|k| {
    //             let in_index_k =
    //                 i * (past_sequence_length.size() + sequence_length.size()) * head_dim
    //                     + k * (past_sequence_length.size() + sequence_length.size())
    //                     + j;

    //             let in_index_v =
    //                 i * (past_sequence_length.size() + sequence_length.size()) * head_dim
    //                     + j * head_dim
    //                     + k;
    //             if j < past_sequence_length.size() {
    //                 let k_index = i * past_sequence_length.size() * head_dim
    //                     + k * past_sequence_length.size()
    //                     + j;
    //                 let k_value = past_key_vec[k_index];
    //                 k_vec[in_index_k] = k_value;

    //                 let v_index = i * past_sequence_length.size() * head_dim + j * head_dim + k;
    //                 let v_value = past_value_vec[v_index];
    //                 v_vec[in_index_v] = v_value;
    //             } else {
    //                 let sj = j - past_sequence_length.size();
    //                 let k_index = sj * hidden_dim * 3 + i * head_dim + hidden_dim + k;
    //                 let k_value = qkv_vec[k_index];
    //                 k_vec[in_index_k] = k_value;

    //                 let v_index = sj * hidden_dim * 3 + i * head_dim + hidden_dim * 2 + k;
    //                 let v_value = qkv_vec[v_index];
    //                 v_vec[in_index_v] = v_value;
    //             }
    //         });
    //     });
    // });
    // k.copy_from(&k_vec);
    // v.copy_from(&v_vec);

    (q, k, v)
}

#[derive(Clone)]
pub struct Mlp {
    c_fc: Linear<HIDDEN_DIM, FF_DIM, f32, Dev>,
    c_proj: Linear<FF_DIM, HIDDEN_DIM, f32, Dev>,
}

impl Mlp {
    fn from_tensors(
        index: usize,
        tensors: &SafeTensors,
        dev: &Dev,
    ) -> Result<Self, SafeTensorError> {
        let c_fc = linear_from(
            tensors.tensor(&format!("h.{index}.mlp.c_fc.weight"))?,
            tensors.tensor(&format!("h.{index}.mlp.c_fc.bias"))?,
            dev,
        );
        let c_proj = linear_from(
            tensors.tensor(&format!("h.{index}.mlp.c_proj.weight"))?,
            tensors.tensor(&format!("h.{index}.mlp.c_proj.bias"))?,
            dev,
        );
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, tensor: FTensor<HiddenShape>) -> FTensor<HiddenShape> {
        let tensor = self.c_fc.forward(tensor);
        let tensor = tensor.gelu();
        // println!("===");
        // let mut tmp = vec![0.0; tensor.shape().num_elements()];
        // tensor.copy_into(&mut tmp);
        // println!("After gelu {:?} {:?}", &tmp[..5], &tmp[tmp.len() - 5..]);
        // println!("Before c_proj");
        let tensor = self.c_proj.forward(tensor);
        tensor
    }
}

#[derive(Clone)]
pub struct Attention {
    c_attn: Linear<HIDDEN_DIM, { 3 * HIDDEN_DIM }, f32, Dev>,
    c_proj: Linear<HIDDEN_DIM, HIDDEN_DIM, f32, Dev>,
}

impl Attention {
    fn from_tensors(
        index: usize,
        tensors: &SafeTensors,
        dev: &Dev,
    ) -> Result<Self, SafeTensorError> {
        let c_attn = linear_from(
            tensors.tensor(&format!("h.{index}.attn.c_attn.weight"))?,
            tensors.tensor(&format!("h.{index}.attn.c_attn.bias"))?,
            dev,
        );
        let c_proj = linear_from(
            tensors.tensor(&format!("h.{index}.attn.c_proj.weight"))?,
            tensors.tensor(&format!("h.{index}.attn.c_proj.bias"))?,
            dev,
        );
        Ok(Self { c_attn, c_proj })
    }

    pub fn forward(
        &self,
        hidden_states: FTensor<HiddenShape>,
        past: &mut PastKeyValue,
    ) -> FTensor<HiddenShape> {
        // if past.value.shape().num_elements() > 0 {
        //     let mut tmp = vec![0.0; past.value.shape().num_elements()];
        //     past.value.copy_into(&mut tmp);
        //     println!("Past value {:?} {:?}", &tmp[..5], &tmp[tmp.len() - 5..]);
        // }

        let sequence_length = hidden_states.shape().0;

        let qkv = self.c_attn.forward(hidden_states);
        // let mut tmp = vec![0.0; qkv.shape().num_elements()];
        // qkv.copy_into(&mut tmp);
        // println!("Qkv {:?} {:?}", &tmp[..5], &tmp[tmp.len() - 5..]);

        let (q, k, v) = attention_reshape(&qkv, &past.key, &past.value);
        past.key = k.clone();
        past.value = v.clone();

        // Get weights
        let scalar: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
        let weights = q.matmul(k) * scalar;
        let weights = weights.softmax::<Axis<2>>();

        // let mut tmp = vec![0.0; weights.shape().num_elements()];
        // weights.copy_into(&mut tmp);
        // println!("Weights {:?} {:?}", &tmp[..5], &tmp[tmp.len() - 5..]);

        // Get new tokens
        let hidden_states: FTensor<Weights> = weights.try_matmul(v).unwrap();
        let hidden_states: FTensor<UnsplitWeights> = hidden_states.permute();
        let hidden_states: FTensor<HiddenShape> = hidden_states
            .try_reshape_like(&(sequence_length, Const))
            .unwrap();

        self.c_proj.forward(hidden_states)
    }
}

#[derive(Clone)]
pub struct Gpt2Layer {
    ln_1: LayerNorm1D<HIDDEN_DIM, f32, Dev>,
    ln_2: LayerNorm1D<HIDDEN_DIM, f32, Dev>,
    mlp: Mlp,
    attention: Attention,
}

impl Gpt2Layer {
    fn from_tensors(
        index: usize,
        tensors: &SafeTensors,
        dev: &Dev,
    ) -> Result<Self, SafeTensorError> {
        let ln_1 = layer_norm_from(
            tensors.tensor(&format!("h.{index}.ln_1.weight"))?,
            tensors.tensor(&format!("h.{index}.ln_1.bias"))?,
            dev,
        );
        let ln_2 = layer_norm_from(
            tensors.tensor(&format!("h.{index}.ln_2.weight"))?,
            tensors.tensor(&format!("h.{index}.ln_2.bias"))?,
            dev,
        );
        let mlp = Mlp::from_tensors(index, tensors, dev)?;
        let attention = Attention::from_tensors(index, tensors, dev)?;
        Ok(Self {
            ln_1,
            ln_2,
            mlp,
            attention,
        })
    }

    fn forward(
        &self,
        tensor: FTensor<HiddenShape>,
        past_key_value: &mut PastKeyValue,
    ) -> FTensor<HiddenShape> {
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
    fn from_tensors(tensors: &SafeTensors, dev: &Dev) -> Result<Self, SafeTensorError> {
        let layers: Result<Vec<_>, _> = (0..NUM_LAYERS)
            .map(|i| -> Result<Gpt2Layer, SafeTensorError> {
                Gpt2Layer::from_tensors(i, tensors, dev)
            })
            .collect();
        let layers = layers?;
        Ok(Self { layers })
    }

    fn forward(
        &self,
        mut tensor: FTensor<HiddenShape>,
        past_key_values: &mut PastKeyValues,
    ) -> FTensor<HiddenShape> {
        for (layer, past_key_value) in self.layers.iter().zip(past_key_values.iter_mut()) {
            tensor = layer.forward(tensor, past_key_value);
        }
        tensor
    }
}

#[derive(Clone)]
pub struct Gpt2 {
    wte: Embedding<VOCAB_SIZE, HIDDEN_DIM, f32, Dev>,
    wpe: Embedding<MAX_POSITIONS, HIDDEN_DIM, f32, Dev>,
    h: Gpt2Model,
    ln_f: LayerNorm1D<HIDDEN_DIM, f32, Dev>,
    lm_head: UnbiasedLinear<HIDDEN_DIM, VOCAB_SIZE, f32, Dev>,
}

impl Gpt2 {
    pub fn from_tensors<'a>(tensors: &SafeTensors<'a>, num_heads: usize, dev: &Dev) -> Self {
        assert_eq!(num_heads, NUM_HEADS);
        let wte = embedding_from(tensors.tensor(&format!("wte.weight")).unwrap(), dev);
        let wpe = embedding_from(tensors.tensor(&format!("wpe.weight")).unwrap(), dev);
        let h = Gpt2Model::from_tensors(tensors, dev).unwrap();
        let ln_f = layer_norm_from(
            tensors.tensor("ln_f.weight").unwrap(),
            tensors.tensor("ln_f.bias").unwrap(),
            dev,
        );
        let lm_head = unbiased_linear_from(tensors.tensor("wte.weight").unwrap(), dev);
        Self {
            wte,
            wpe,
            h,
            ln_f,
            lm_head,
        }
    }

    pub fn empty_past_key_values(&self, dev: &Dev) -> PastKeyValues {
        (0..NUM_LAYERS).map(|_| PastKeyValue::new(0, dev)).collect()
    }

    pub fn forward(&self, ids: &[u32], past: &mut PastKeyValues) -> usize {
        let n = ids.len();
        let past_sequence_length = past[0].key.shape().2.size();
        let ids: Vec<usize> = ids.iter().map(|id| *id as usize).collect();
        let positions: Vec<usize> = (0..ids.len()).map(|i| (i + past_sequence_length)).collect();

        let nn = Dyn::<SEQ>(n);
        let dev = past[0].key.device().clone();
        let mut input_ids: Tensor<(Dyn<SEQ>,), usize, Dev> = dev.zeros_like(&(nn,));
        input_ids.copy_from(&ids);

        let mut position_ids: Tensor<(Dyn<SEQ>,), usize, Dev> = dev.zeros_like(&(nn,));
        position_ids.copy_from(&positions);

        let input_embeds = self.wte.forward(input_ids);
        // let mut embeds_vec = vec![0.0; input_embeds.shape().num_elements()];
        // input_embeds.copy_into(&mut embeds_vec);
        // println!(
        //     "embeds {:?} {:?}",
        //     &embeds_vec[..5],
        //     &embeds_vec[embeds_vec.len() - 5..]
        // );
        let position_embeds = self.wpe.forward(position_ids);
        let embeds = input_embeds + position_embeds;
        // let mut embeds_vec = vec![0.0; embeds.shape().num_elements()];
        // embeds.copy_into(&mut embeds_vec);
        // println!(
        //     "embeds {:?} {:?}",
        //     &embeds_vec[..5],
        //     &embeds_vec[embeds_vec.len() - 5..]
        // );
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
