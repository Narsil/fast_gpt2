use dfdx::nn::BuildOnDevice;
use dfdx::prelude::{Const, Cpu, Tensor, Transformer, ZerosTensor};
use safetensors::tensor::SafeTensors;

type Dev = Cpu;
const HIDDEN_DIM: usize = 768;
const NUM_HEADS: usize = 12;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS;
const NUM_LAYERS: usize = 12;
const VOCAB_SIZE: usize = 50257;
const FF_DIM: usize = HIDDEN_DIM * 4;

pub struct PastKeyValue {
    pub key: Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>)>,
    pub value: Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>)>,
}

impl PastKeyValue {
    pub fn new(past_sequence_length: usize) -> Self {
        let dev: Dev = Default::default();
        let key: Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>)> =
            dev.zeros_like(&(Const, past_sequence_length, Const));
        let value: Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>)> =
            dev.zeros_like(&(Const, past_sequence_length, Const));
        Self { key, value }
    }
}

pub type PastKeyValues = Vec<PastKeyValue>;

type Model = Transformer<HIDDEN_DIM, NUM_HEADS, 0, NUM_LAYERS, FF_DIM>;

pub struct Gpt2 {
    transformer: Model,
}

impl Gpt2 {
    pub fn from_tensors<'a>(tensors: &SafeTensors<'a>, num_heads: usize) -> Self {
        assert_eq!(num_heads, NUM_HEADS);
        let dev: Dev = Default::default();
        let transformer = Model::build_on_device(&dev);
        Self { transformer }
    }

    pub fn empty_past_key_values(&self) -> PastKeyValues {
        (0..NUM_LAYERS).map(|_| PastKeyValue::new(0)).collect()
    }

    pub fn forward(&self, ids: &[u32], past: &mut PastKeyValues) -> usize {
        0
    }
}
