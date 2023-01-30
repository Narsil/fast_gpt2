use dfdx::nn::{BuildOnDevice, Linear, Module, UnbiasedLinear};
use dfdx::prelude::{
    Axis, BuildModule, Const, Device, Embedding, LayerNorm1D, Rank0, Rank1, Rank2, Rank3, Shape,
    Tensor, TensorFromArray, ZerosTensor,
};
use dfdx::shapes::{Dim, Dyn, HasShape};
use dfdx::tensor::AsArray;
use dfdx::tensor_ops::{GatherTo, MaxTo, PermuteTo, ReshapeTo, TryMatMul};
use safetensors::tensor::SafeTensors;

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
type HiddenShape = (Dyn<SEQ>, Const<HIDDEN_DIM>);

pub struct PastKeyValue {
    pub key: Tensor<(Const<NUM_HEADS>, Dyn<PAST>, Const<HEAD_DIM>), f32, Dev>,
    pub value: Tensor<(Const<NUM_HEADS>, Dyn<PAST>, Const<HEAD_DIM>), f32, Dev>,
}

impl PastKeyValue {
    pub fn new(past_sequence_length: usize) -> Self {
        let dev: Dev = Default::default();
        let past_sequence_length = Dyn::<PAST>(past_sequence_length);
        let key: Tensor<(Const<NUM_HEADS>, Dyn<PAST>, Const<HEAD_DIM>), _, _> =
            dev.zeros_like(&(Const, past_sequence_length, Const));
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

impl Mlp {
    fn from_tensors(index: usize, tensors: &SafeTensors) -> Self {
        let dev: Dev = Default::default();
        let c_fc = BuildModule::build(&dev);
        let c_proj = BuildModule::build(&dev);
        // let c_fc = Linear::from(
        //     tensors
        //         .tensor(&format!("h.{index}.mlp.c_fc.weight"))
        //         .unwrap(),
        //     tensors.tensor(&format!("h.{index}.mlp.c_fc.bias")).unwrap(),
        // );
        // let c_proj = Linear::from(
        //     tensors
        //         .tensor(&format!("h.{index}.mlp.c_proj.weight"))
        //         .unwrap(),
        //     tensors
        //         .tensor(&format!("h.{index}.mlp.c_proj.bias"))
        //         .unwrap(),
        // );
        Self { c_fc, c_proj }
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
    fn from_tensors(index: usize, tensors: &SafeTensors, num_heads: usize) -> Self {
        let dev: Dev = Default::default();
        let c_attn = BuildModule::build(&dev);
        let c_proj = BuildModule::build(&dev);
        // let c_attn = Linear::from(
        //     tensors
        //         .tensor(&format!("h.{index}.attn.c_attn.weight"))
        //         .unwrap(),
        //     tensors
        //         .tensor(&format!("h.{index}.attn.c_attn.bias"))
        //         .unwrap(),
        // );
        // let c_proj = Linear::from(
        //     tensors
        //         .tensor(&format!("h.{index}.attn.c_proj.weight"))
        //         .unwrap(),
        //     tensors
        //         .tensor(&format!("h.{index}.attn.c_proj.bias"))
        //         .unwrap(),
        // );
        // Self { c_attn, c_proj }
        Self { c_attn, c_proj }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<HiddenShape, f32, Dev>,
        past: &mut PastKeyValue,
    ) -> Tensor<HiddenShape, f32, Dev> {
        type PreSplitHead = (Dyn<SEQ>, Const<NUM_HEADS>, Const<HEAD_DIM>);
        type SplitQuery = (Const<NUM_HEADS>, Dyn<SEQ>, Const<HEAD_DIM>);
        type SplitKeys = (Const<NUM_HEADS>, Const<HEAD_DIM>, Dyn<'T'>);
        type SplitValues = (Const<NUM_HEADS>, Dyn<'T'>, Const<HEAD_DIM>);
        type Weights = (Const<NUM_HEADS>, Dyn<SEQ>, Const<HEAD_DIM>);

        let sequence_length = hidden_states.shape().0;
        let past_sequence_length = past.key.shape().1;
        let total_length = Dyn::<'T'>(sequence_length.size() + past_sequence_length.size());

        let dev: Dev = Default::default();
        let qkv = self.c_attn.forward(hidden_states);

        let q: Tensor<SplitQuery, f32, Dev> = dev.zeros_like(&(Const, sequence_length, Const));
        let k: Tensor<SplitKeys, f32, Dev> = dev.zeros_like(&(Const, Const, total_length));

        // Get weights
        let scalar: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
        let weights = q.matmul(k) * scalar;
        let weights = weights.softmax::<Axis<2>>();

        let v: Tensor<SplitValues, f32, Dev> = dev.zeros_like(&(Const, total_length, Const));

        // Get new tokens
        let tokens: Tensor<Weights, f32, Dev> = weights.try_matmul(v).unwrap();
        let tokens = tokens.permute::<PreSplitHead, _>();
        // TODO XXX: For some reason this fails.
        // let tokens = tokens.reshape::<HiddenShape>();
        let tokens: Tensor<HiddenShape, f32, Dev> = dev.zeros_like(&(sequence_length, Const));

        self.c_proj.forward(tokens)
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
    fn from_tensors(index: usize, tensors: &SafeTensors, num_heads: usize) -> Self {
        let dev: Dev = Default::default();
        let ln_1 = BuildModule::build(&dev);
        let ln_2 = BuildModule::build(&dev);
        // let ln_1 = LayerNorm::from(
        //     tensors.tensor(&format!("h.{index}.ln_1.weight")).unwrap(),
        //     tensors.tensor(&format!("h.{index}.ln_1.bias")).unwrap(),
        // );
        // let ln_2 = LayerNorm::from(
        //     tensors.tensor(&format!("h.{index}.ln_2.weight")).unwrap(),
        //     tensors.tensor(&format!("h.{index}.ln_2.bias")).unwrap(),
        // );
        let mlp = Mlp::from_tensors(index, tensors);
        let attention = Attention::from_tensors(index, tensors, num_heads);
        Self {
            ln_1,
            ln_2,
            mlp,
            attention,
        }
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
    fn from_tensors(tensors: &SafeTensors) -> Self {
        let layers: Vec<_> = (0..NUM_LAYERS)
            .map(|i| Gpt2Layer::from_tensors(i, tensors, NUM_HEADS))
            .collect();
        Self { layers }
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
        let h = Gpt2Model::from_tensors(tensors);
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
