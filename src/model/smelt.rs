use safetensors::tensor::{SafeTensors, TensorView};
use smelt::ops::{add, causal_softmax, gelu, matmul, matmul_t, mul, normalize, select};
use smelt::tensor::{OwnedTensor, Tensor, TensorMut, ViewTensor};

/// A special structure handy for Past Key values for text-generation
pub struct PastKeyValue {
    /// The cached key tensor. Shape is `NUM_HEADS, PAST_SEQUENCE_LENGTH, HEAD_DIM`.
    pub key: OwnedTensor,
    /// The cached value tensor. Shape is `NUM_HEADS, PAST_SEQUENCE_LENGTH, HEAD_DIM`.
    pub value: OwnedTensor,
}

impl PastKeyValue {
    pub fn new(num_heads: usize, past_sequence_length: usize, head_dim: usize) -> Self {
        let key = OwnedTensor::zeros(vec![num_heads, past_sequence_length, head_dim]);
        let value = OwnedTensor::zeros(vec![num_heads, past_sequence_length, head_dim]);
        Self { key, value }
    }
}

pub type PastKeyValues = Vec<PastKeyValue>;

fn attention<T: Tensor, TM: TensorMut>(
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
    assert_eq!(
        past.key.shape(),
        vec![num_heads, past_sequence_length, head_dim]
    );
    assert_eq!(
        past.value.shape(),
        vec![num_heads, past_sequence_length, head_dim]
    );

    let (query, key, value) = split_qkv(qkv, past);

    matmul_t(&query, &key, qk).unwrap();
    let head_dim = hidden_dim / num_heads;
    let scale = (head_dim as f32).sqrt();
    qk.data_mut().iter_mut().for_each(|v| *v /= scale);

    causal_softmax(qk, max, past_sequence_length).unwrap();
    matmul(qk, &value, out).unwrap();

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
    *out = OwnedTensor::new(new_out, vec![sequence_length, hidden_dim]).unwrap();
    *past = PastKeyValue { key, value };
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
    let query = OwnedTensor::new(query_data, vec![num_heads, sequence_length, head_dim]).unwrap();

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
    )
    .unwrap();
    let value = OwnedTensor::new(
        value_data,
        vec![
            num_heads,
            (past_sequence_length + sequence_length),
            head_dim,
        ],
    )
    .unwrap();
    (query, key, value)
}

#[derive(Clone)]
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
        gelu(tensor);
        // let tmp = tensor.data();
        // println!("After gelu {:?} {:?}", &tmp[..5], &tmp[tmp.len() - 5..]);
        self.c_proj.forward(tensor);
        // let tmp = tensor.data();
        // println!("After MLP {:?} {:?}", &tmp[..5], &tmp[tmp.len() - 5..]);
    }
}

#[derive(Clone)]
pub struct Attention<'a> {
    c_attn: Linear<'a>,
    c_proj: Linear<'a>,
    num_heads: usize,
}

impl<'a> Attention<'a> {
    fn from_tensors(index: usize, tensors: &'a SafeTensors<'a>, num_heads: usize) -> Self {
        let c_attn = Linear::from(
            tensors
                .tensor(&format!("h.{index}.attn.c_attn.weight"))
                .unwrap(),
            tensors
                .tensor(&format!("h.{index}.attn.c_attn.bias"))
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
            num_heads,
        }
    }

    pub fn forward(&self, hidden_states: &mut OwnedTensor, past: &mut PastKeyValue) {
        assert_eq!(hidden_states.shape().len(), 2);
        let sequence_length = hidden_states.shape()[0];
        let hidden_dim = hidden_states.shape()[1];
        self.c_attn.forward(hidden_states);
        let qkv = hidden_states;

        // let tmp = qkv.data();
        // println!("qkv {:?} {:?}", &tmp[..5], &tmp[tmp.len() - 5..]);

        let num_heads = self.num_heads;
        assert_eq!(hidden_dim % num_heads, 0);
        let head_dim = hidden_dim / num_heads;
        let past_sequence_length = past.key.shape()[1];
        let mut qk = OwnedTensor::zeros(vec![
            num_heads,
            sequence_length,
            past_sequence_length + sequence_length,
        ]);
        let mut qv = OwnedTensor::zeros(vec![num_heads, sequence_length, head_dim]);
        let mut max = vec![0.0; (past_sequence_length + sequence_length) * num_heads];
        attention(qkv, &mut qk, &mut max, past, &mut qv);
        self.c_proj.forward(&mut qv);
        *qkv = qv;
    }
}

#[derive(Clone)]
pub struct Gpt2Layer<'a> {
    ln_1: LayerNorm<'a>,
    ln_2: LayerNorm<'a>,
    mlp: Mlp<'a>,
    attention: Attention<'a>,
}

impl<'a> Gpt2Layer<'a> {
    fn from_tensors(index: usize, tensors: &'a SafeTensors<'a>, num_heads: usize) -> Self {
        let ln_1 = LayerNorm::from(
            tensors.tensor(&format!("h.{index}.ln_1.weight")).unwrap(),
            tensors.tensor(&format!("h.{index}.ln_1.bias")).unwrap(),
        );
        let ln_2 = LayerNorm::from(
            tensors.tensor(&format!("h.{index}.ln_2.weight")).unwrap(),
            tensors.tensor(&format!("h.{index}.ln_2.bias")).unwrap(),
        );
        let mlp = Mlp::from_tensors(index, tensors);
        let attention = Attention::from_tensors(index, tensors, num_heads);
        Self {
            ln_1,
            ln_2,
            mlp,
            attention,
        }
    }

    fn forward(&self, tensor: &mut OwnedTensor, past_key_value: &mut PastKeyValue) {
        let residual = tensor.clone();
        self.ln_1.forward(tensor);
        self.attention.forward(tensor, past_key_value);
        add(&residual, tensor).unwrap();
        let residual = tensor.clone();
        self.ln_2.forward(tensor);

        self.mlp.forward(tensor);
        add(&residual, tensor).unwrap();
    }
}

#[derive(Clone)]
pub struct Gpt2Model<'a> {
    layers: Vec<Gpt2Layer<'a>>,
}

impl<'a> Gpt2Model<'a> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, num_heads: usize) -> Self {
        let layers: Vec<_> = (0..12)
            .map(|i| Gpt2Layer::from_tensors(i, tensors, num_heads))
            .collect();
        Self { layers }
    }

    fn forward(&self, tensor: &mut OwnedTensor, past_key_values: &mut PastKeyValues) {
        for (layer, past_key_value) in self.layers.iter().zip(past_key_values.iter_mut()) {
            layer.forward(tensor, past_key_value);
        }
    }
}

#[derive(Clone)]
pub struct Linear<'a> {
    weight: ViewTensor<'a>,
    bias: ViewTensor<'a>,
}

impl<'a> std::fmt::Debug for Linear<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("shape", &self.weight.shape())
            .finish()
    }
}

impl<'a> Linear<'a> {
    pub fn new(weight: ViewTensor<'a>, bias: ViewTensor<'a>) -> Self {
        Self { weight, bias }
    }

    fn from(weight: TensorView<'a>, bias: TensorView<'a>) -> Self {
        let weight: ViewTensor = weight.into();
        let bias: ViewTensor = bias.into();
        Self::new(weight, bias)
    }

    pub fn forward(&self, tensor: &mut OwnedTensor) {
        assert_eq!(tensor.shape().len(), 2);
        let m = tensor.shape()[0];
        let n = self.weight.shape()[1];
        let mut c = OwnedTensor::zeros(vec![m, n]);
        matmul(tensor, &self.weight, &mut c).unwrap();
        add(&self.bias, &mut c).unwrap();
        *tensor = c;
    }
}

#[derive(Clone)]
pub struct UnbiasedLinear<'a> {
    weight: ViewTensor<'a>,
}

impl<'a> std::fmt::Debug for UnbiasedLinear<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnbiasedLinear")
            .field("shape", &self.weight.shape())
            .finish()
    }
}

impl<'a> UnbiasedLinear<'a> {
    fn from(weight: TensorView<'a>) -> Self {
        let weight: ViewTensor = weight.into();
        Self { weight }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        let m = tensor.shape()[0];
        let n = self.weight.shape()[0];
        let mut c = OwnedTensor::zeros(vec![m, n]);
        matmul_t(tensor, &self.weight, &mut c).unwrap();
        *tensor = c;
    }
}

#[derive(Clone)]
pub struct Embedding<'a> {
    weight: ViewTensor<'a>,
}

impl<'a> Embedding<'a> {
    fn from(weight: TensorView<'a>) -> Self {
        let weight: ViewTensor = weight.into();
        Self { weight }
    }

    fn forward(&self, ids: &[u32]) -> OwnedTensor {
        let _vocab_size = self.weight.shape()[0];
        let hidden_dim = self.weight.shape()[1];
        let shape = vec![ids.len(), hidden_dim];
        let mut tensor = OwnedTensor::zeros(shape);
        select(ids, &self.weight, &mut tensor).unwrap();
        tensor
    }
}

#[derive(Clone)]
pub struct LayerNorm<'a> {
    weight: ViewTensor<'a>,
    bias: ViewTensor<'a>,
    epsilon: f32,
}

impl<'a> LayerNorm<'a> {
    fn from(weight: TensorView<'a>, bias: TensorView<'a>) -> Self {
        let weight: ViewTensor = weight.into();
        let bias: ViewTensor = bias.into();
        let epsilon = 1e-5;
        Self {
            weight,
            bias,
            epsilon,
        }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        let m = tensor.shape()[0];
        let mut mean = vec![0.0; m];
        let mut var = vec![0.0; m];
        normalize(tensor, &mut mean, &mut var, self.epsilon).unwrap();
        mul(&self.weight, tensor).unwrap();
        add(&self.bias, tensor).unwrap();
    }
}

#[derive(Clone)]
pub struct Gpt2<'a> {
    wte: Embedding<'a>,
    wpe: Embedding<'a>,
    h: Gpt2Model<'a>,
    ln_f: LayerNorm<'a>,
    lm_head: UnbiasedLinear<'a>,
    num_heads: usize,
}

impl<'a> Gpt2<'a> {
    pub fn from_tensors(tensors: &'a SafeTensors<'a>, num_heads: usize) -> Self {
        let wte = Embedding::from(tensors.tensor("wte.weight").unwrap());
        let wpe = Embedding::from(tensors.tensor("wpe.weight").unwrap());
        let h = Gpt2Model::from_tensors(tensors, num_heads);
        let ln_f = LayerNorm::from(
            tensors.tensor("ln_f.weight").unwrap(),
            tensors.tensor("ln_f.bias").unwrap(),
        );
        let lm_head = UnbiasedLinear::from(tensors.tensor("wte.weight").unwrap());
        Self {
            h,
            ln_f,
            wte,
            wpe,
            lm_head,
            num_heads,
        }
    }
}

/// Argmax of the last dimension of tensor `x `.
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

impl<'a> Gpt2<'a> {
    pub fn empty_past_key_values(&self) -> PastKeyValues {
        let num_layers = self.h.layers.len();
        let hidden_dim = self.wte.weight.shape()[1];
        let num_heads = self.num_heads;
        assert_eq!(hidden_dim % num_heads, 0);
        let head_dim = hidden_dim / num_heads;
        (0..num_layers)
            .map(|_| PastKeyValue::new(num_heads, 0, head_dim))
            .collect()
    }

    pub fn forward(&self, ids: &[u32], past: &mut PastKeyValues) -> usize {
        let mut tensor = self.wte.forward(ids);
        let past_sequence_length = past[0].key.shape()[1];
        let positions: Vec<u32> = (0..ids.len())
            .map(|i| (i + past_sequence_length) as u32)
            .collect();
        let position_embeddings = self.wpe.forward(&positions[..]);
        add(&position_embeddings, &mut tensor).unwrap();
        // println!(
        //     "embeds {:?} {:?}",
        //     &tensor.data()[..5],
        //     &tensor.data()[tensor.data().len() - 5..]
        // );
        self.h.forward(&mut tensor, past);
        self.ln_f.forward(&mut tensor);
        self.lm_head.forward(&mut tensor);
        let logits = tensor;
        special_argmax(&logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::simplify;
    use memmap2::MmapOptions;
    use smelt::tensor::{OwnedTensor, TensorMut, ViewTensor};

    #[test]
    fn tensor_values() {
        let filename = "model.safetensors";
        let file = std::fs::File::open(filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer).unwrap();
        let tensor: ViewTensor = tensors.tensor("ln_f.weight").unwrap().into();
        let data = tensor.data();
        assert_eq!(
            simplify(&data[..10]),
            // Values obtained through python
            [1.3971, 1.3750, 1.8870, 1.1688, 1.2724, 1.2508, 9.4198, 1.4371, 1.4527, 1.1856]
        );
        assert_eq!(
            simplify(&data[data.len() - 10..]),
            // Values obtained through python
            [1.1758, 1.4514, 1.1525, 1.1731, 4.2194, 1.1660, 1.1625, 1.1034, 1.0980, 1.2070]
        );
    }

    #[test]
    fn embedding() {
        let filename = "model.safetensors";
        let file = std::fs::File::open(filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer).unwrap();
        let tensor = tensors.tensor("wte.weight").unwrap();
        let embedding = Embedding::from(tensor);
        assert_eq!(
            simplify(&embedding.weight.data()[..10]),
            // Values obtained through python
            [
                -0.1101, -0.0393, 0.0331, 0.1338, -0.0485, -0.0789, -0.2398, -0.0895, 0.0253,
                -0.1074
            ]
        );
        let out = embedding.forward(&[1, 256, 50256]);
        let data = out.data();
        assert_eq!(out.shape(), [3, 768]);
        assert_eq!(
            simplify(&data[..10]),
            // Values obtained through python
            [0.0403, -0.0486, 0.0462, -0.0990, 0.0826, 0.0768, -0.2202, -0.0110, 0.0592, 0.0354]
        );
        assert_eq!(
            simplify(&data[data.len() - 10..]),
            // Values obtained through python
            [-0.0499, 0.0689, 0.0123, -0.2156, -0.1742, -0.0373, 0.0930, 0.0070, 0.1552, 0.1207]
        );
    }

    #[test]
    fn layer_norm() {
        let filename = "model.safetensors";
        let file = std::fs::File::open(filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer).unwrap();
        let layer_norm = LayerNorm::from(
            tensors.tensor("ln_f.weight").unwrap(),
            tensors.tensor("ln_f.bias").unwrap(),
        );
        let data = layer_norm.weight.data();
        assert_eq!(
            simplify(&data[..10]),
            // Values obtained through python
            [1.3971, 1.3750, 1.8870, 1.1688, 1.2724, 1.2508, 9.4198, 1.4371, 1.4527, 1.1856]
        );
        assert_eq!(
            simplify(&data[data.len() - 10..]),
            // Values obtained through python
            [1.1758, 1.4514, 1.1525, 1.1731, 4.2194, 1.1660, 1.1625, 1.1034, 1.0980, 1.2070]
        );

        let weight = ViewTensor::new(&[-1.0, 4.0], vec![2]);
        let bias = ViewTensor::new(&[1.0, 2.0], vec![2]);
        let epsilon = 1e-5;
        let layer_norm = LayerNorm {
            weight,
            bias,
            epsilon,
        };

        let mut input = OwnedTensor::new(vec![10.0, 1.0, 1.0, 1.0], vec![2, 2]);
        layer_norm.forward(&mut input);
        assert_eq!(
            simplify(input.data()),
            // Values obtained through python
            [0.0, -2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn attention_data() {
        let filename = "model.safetensors";
        let file = std::fs::File::open(filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer).unwrap();
        let attention = Attention::from_tensors(0, &tensors, 12);
        let data = attention.c_attn.weight.data();
        assert_eq!(
            simplify(&data[..10]),
            // Values obtained through python
            [
                -0.4738, -0.2614, -0.0978, -0.3499, 0.2243, -0.0429, 0.4187, 0.1744, -0.1883,
                0.1836
            ]
        );
        assert_eq!(
            simplify(&data[data.len() - 10..]),
            // Values obtained through python
            [0.0015, -0.0719, 0.0741, 0.0541, 0.0540, 0.0205, 0.0176, -0.0046, 0.0070, 0.0198]
        );
    }

    #[test]
    fn simple_attention() {
        // Values gotten from Python
        // ```python
        // import torch
        // from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Config
        // config = GPT2Config(n_embd=8, n_head=2)
        // attn = GPT2Attention(config)
        // # remove dropout
        // attn.eval()
        // attn.c_attn.weight = torch.nn.Parameter(torch.arange(attn.c_attn.weight.nelement()).view(attn.c_attn.weight.shape).float())
        // attn.c_attn.bias = torch.nn.Parameter(torch.arange(attn.c_attn.bias.nelement()).view(attn.c_attn.bias.shape).float())
        // attn.c_proj.weight = torch.nn.Parameter(torch.arange(attn.c_proj.weight.nelement()).view(attn.c_proj.weight.shape).float())
        // attn.c_proj.bias = torch.nn.Parameter(torch.arange(attn.c_proj.bias.nelement()).view(attn.c_proj.bias.shape).float())
        // input = torch.ones((1, 3, 8))
        // attn_weights, (past_key, past_value) = attn(input, use_cache=True)
        // print(attn_weights.view(-1))
        // print(past_key.shape)
        // print(past_key.reshape(-1))
        // print(past_value.shape)
        // print(past_value.reshape(-1))
        //
        //
        // print()
        // print("Second pass")
        // new_input = torch.ones((1, 1, 8))
        // attn_weights2, (past_key, past_value) = attn(new_input, layer_past = (past_key, past_value), use_cache=True)
        // print(attn_weights2.view(-1))
        // print(past_key.shape)
        // print(past_key.view(-1))
        // print(past_value.shape)
        // print(past_value.view(-1))
        // ```
        let hidden_dim = 8;
        let num_heads = 2;
        let head_dim = hidden_dim / num_heads;
        let data_w = (0..hidden_dim * hidden_dim * 3)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let weight = ViewTensor::new(&data_w, vec![hidden_dim, hidden_dim * 3]);
        let data_b = (0..hidden_dim * 3).map(|i| i as f32).collect::<Vec<_>>();
        let bias = ViewTensor::new(&data_b, vec![hidden_dim * 3]);
        let c_attn = Linear::new(weight, bias);

        let data_w2 = (0..hidden_dim * hidden_dim)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let weight = ViewTensor::new(&data_w2, vec![hidden_dim, hidden_dim]);
        let data_b2 = (0..hidden_dim).map(|i| i as f32).collect::<Vec<_>>();
        let bias = ViewTensor::new(&data_b2, vec![hidden_dim]);
        let c_proj = Linear::new(weight, bias);

        let attention = Attention {
            c_attn,
            c_proj,
            num_heads,
        };
        let sequence_length = 3;
        let mut input = OwnedTensor::new(
            vec![1.0; hidden_dim * sequence_length],
            vec![sequence_length, hidden_dim],
        );

        let key = OwnedTensor::zeros(vec![num_heads, 0, head_dim]);
        let value = OwnedTensor::zeros(vec![num_heads, 0, head_dim]);
        let mut past = PastKeyValue { key, value };
        attention.forward(&mut input, &mut past);
        assert_eq!(
            input.data(),
            &[
                192864., 199645., 206426., 213207., 219988., 226769., 233550., 240331., 192864.,
                199645., 206426., 213207., 219988., 226769., 233550., 240331., 192864., 199645.,
                206426., 213207., 219988., 226769., 233550., 240331.
            ]
        );

        assert_eq!(past.key.shape(), vec![2, 3, 4]);
        assert_eq!(
            past.key.data(),
            [
                744., 753., 762., 771., 744., 753., 762., 771., 744., 753., 762., 771., 780., 789.,
                798., 807., 780., 789., 798., 807., 780., 789., 798., 807.
            ]
        );
        assert_eq!(past.value.shape(), vec![2, 3, 4]);
        assert_eq!(
            past.value.data(),
            [
                816., 825., 834., 843., 816., 825., 834., 843., 816., 825., 834., 843., 852., 861.,
                870., 879., 852., 861., 870., 879., 852., 861., 870., 879.
            ]
        );

        // Second pass
        let sequence_length = 1;
        let mut input = OwnedTensor::new(vec![1.0; hidden_dim], vec![sequence_length, hidden_dim]);
        attention.forward(&mut input, &mut past);
        assert_eq!(
            input.data(),
            &[192864., 199645., 206426., 213207., 219988., 226769., 233550., 240331.]
        );
        assert_eq!(past.key.shape(), vec![2, 4, 4]);
        assert_eq!(
            past.key.data(),
            &[
                744., 753., 762., 771., 744., 753., 762., 771., 744., 753., 762., 771., 744., 753.,
                762., 771., 780., 789., 798., 807., 780., 789., 798., 807., 780., 789., 798., 807.,
                780., 789., 798., 807.
            ]
        );
        assert_eq!(past.value.shape(), vec![2, 4, 4]);
        assert_eq!(
            past.value.data(),
            &[
                816., 825., 834., 843., 816., 825., 834., 843., 816., 825., 834., 843., 816., 825.,
                834., 843., 852., 861., 870., 879., 852., 861., 870., 879., 852., 861., 870., 879.,
                852., 861., 870., 879.
            ]
        );
    }

    #[test]
    fn mlp() {
        let hidden_dim = 8;
        let data = (0..hidden_dim * hidden_dim * 4)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let weight = ViewTensor::new(&data, vec![hidden_dim, hidden_dim * 4]);
        let data = (0..hidden_dim * 4).map(|i| i as f32).collect::<Vec<_>>();
        let bias = ViewTensor::new(&data, vec![hidden_dim * 4]);
        let c_fc = Linear::new(weight, bias);

        let data = (0..hidden_dim * hidden_dim * 4)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let weight = ViewTensor::new(&data, vec![hidden_dim * 4, hidden_dim]);
        let data = (0..hidden_dim).map(|i| i as f32).collect::<Vec<_>>();
        let bias = ViewTensor::new(&data, vec![hidden_dim]);
        let c_proj = Linear::new(weight, bias);

        let mlp = Mlp { c_fc, c_proj };
        let mut input = OwnedTensor::new(vec![1.0; hidden_dim], vec![1, hidden_dim]);
        mlp.forward(&mut input);
        assert_eq!(
            input.data(),
            // Values gotten from Python
            // ```python
            // import torch
            // from transformers.models.gpt2.modeling_gpt2 import GPT2MLP, GPT2Config
            // config = GPT2Config(n_embd=8, n_head=2, activation_function="gelu_new")
            // mlp = GPT2MLP(config=config, intermediate_size = config.n_embd * 4)
            // # remove dropout
            // mlp.eval()
            // mlp.c_fc.weight = torch.nn.Parameter(torch.arange(mlp.c_fc.weight.nelement()).view(mlp.c_fc.weight.shape).float())
            // mlp.c_fc.bias = torch.nn.Parameter(torch.arange(mlp.c_fc.bias.nelement()).view(mlp.c_fc.bias.shape).float())
            // mlp.c_proj.weight = torch.nn.Parameter(torch.arange(mlp.c_proj.weight.nelement()).view(mlp.c_proj.weight.shape).float())
            // mlp.c_proj.bias = torch.nn.Parameter(torch.arange(mlp.c_proj.bias.nelement()).view(mlp.c_proj.bias.shape).float())
            // input = torch.ones((1, 1, 8))
            // print(mlp(input)[0].view(-1))
            // ```
            &[4305280., 4338417., 4371554., 4404691., 4437828., 4470965., 4504102., 4537239.]
        );
    }

    #[test]
    fn simple_attention_qk() {
        // from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Config
        // import torch
        //
        // config = GPT2Config(n_embd=8, n_head=2)
        // attn = GPT2Attention(config)
        // attn.c_attn.weight = torch.nn.Parameter(torch.arange(attn.c_attn.weight.nelement()).view(attn.c_attn.weight.shape).float())
        // attn.c_attn.bias = torch.nn.Parameter(torch.arange(attn.c_attn.bias.nelement()).view(attn.c_attn.bias.shape).float())
        //
        // hidden_states = torch.arange(24).view((1, 3, 8)).float() / 24
        // qkv = attn.c_attn(hidden_states)
        // print(qkv.view(-1))
        // query, key, value = qkv.split(attn.split_size, dim=2)
        //
        // query = attn._split_heads(query, attn.num_heads, attn.head_dim)
        // key = attn._split_heads(key, attn.num_heads, attn.head_dim)
        // value = attn._split_heads(value, attn.num_heads, attn.head_dim)
        //
        // print(query.reshape(-1))
        // print(key.reshape(-1))
        // key = key.transpose(-1, -2)
        // attn_weights = torch.matmul(query, key)
        // print(attn_weights.view(-1))
        let hidden_dim = 8;
        let num_heads = 2;
        let head_dim = hidden_dim / num_heads;
        let data = (0..hidden_dim * hidden_dim * 3)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let weight = ViewTensor::new(&data, vec![hidden_dim, hidden_dim * 3]);
        let data = (0..hidden_dim * 3).map(|i| i as f32).collect::<Vec<_>>();
        let bias = ViewTensor::new(&data, vec![hidden_dim * 3]);
        let c_attn = Linear::new(weight, bias);

        let sequence_length = 3;
        let data = (0..sequence_length * hidden_dim)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let mut qkv = OwnedTensor::new(data, vec![sequence_length, hidden_dim]);
        c_attn.forward(&mut qkv);
        assert_eq!(
            qkv.data(),
            [
                3360., 3389., 3418., 3447., 3476., 3505., 3534., 3563., 3592., 3621., 3650., 3679.,
                3708., 3737., 3766., 3795., 3824., 3853., 3882., 3911., 3940., 3969., 3998., 4027.,
                8736., 8829., 8922., 9015., 9108., 9201., 9294., 9387., 9480., 9573., 9666., 9759.,
                9852., 9945., 10038., 10131., 10224., 10317., 10410., 10503., 10596., 10689.,
                10782., 10875., 14112., 14269., 14426., 14583., 14740., 14897., 15054., 15211.,
                15368., 15525., 15682., 15839., 15996., 16153., 16310., 16467., 16624., 16781.,
                16938., 17095., 17252., 17409., 17566., 17723.
            ]
        );
        let mut qk = OwnedTensor::zeros(vec![num_heads, sequence_length, sequence_length]);
        let past = PastKeyValue::new(num_heads, 0, head_dim);
        let (query, key, _) = split_qkv(&qkv, &past);
        assert_eq!(
            query.data(),
            [
                3360., 3389., 3418., 3447., 8736., 8829., 8922., 9015., 14112., 14269., 14426.,
                14583., 3476., 3505., 3534., 3563., 9108., 9201., 9294., 9387., 14740., 14897.,
                15054., 15211.
            ]
        );
        assert_eq!(
            key.data(),
            [
                3592., 3621., 3650., 3679., 9480., 9573., 9666., 9759., 15368., 15525., 15682.,
                15839., 3708., 3737., 3766., 3795., 9852., 9945., 10038., 10131., 15996., 16153.,
                16310., 16467.
            ]
        );
        matmul_t(&query, &key, &mut qk);
        qk.data()
            .iter()
            .zip([
                49497900.0,
                130973350.0,
                212448800.0,
                129081000.0,
                341554720.0,
                554028500.0,
                208664100.0,
                552136100.0,
                895608200.0,
                52817820.0,
                140673820.0,
                228529820.0,
                138781470.0,
                369628860.0,
                600476200.0,
                224745120.0,
                598583900.0,
                972422660.0,
            ])
            .for_each(|(&l, r)| {
                assert!((l - r).abs() / l < 1e-7);
            });
    }

    #[test]
    fn simple_attention_ops() {
        // Values gotten from Python
        // ```python
        // from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Config
        // import torch
        //
        // config = GPT2Config(n_embd=8, n_head=2)
        // attn = GPT2Attention(config)
        // attn.eval()
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
        // attn_output, _ = attn._attn(query, key, value)
        // attn_output = attn._merge_heads(attn_output, attn.num_heads, attn.head_dim)
        //
        // print(key.reshape(-1))
        // print(value.reshape(-1))
        // print(attn_output.view(-1))
        // ```
        let hidden_dim = 8;
        let num_heads = 2;
        let head_dim = hidden_dim / num_heads;
        let data = (0..hidden_dim * hidden_dim * 3)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let weight = ViewTensor::new(&data, vec![hidden_dim, hidden_dim * 3]);
        let data = (0..hidden_dim * 3).map(|i| i as f32).collect::<Vec<_>>();
        let bias = ViewTensor::new(&data, vec![hidden_dim * 3]);
        let c_attn = Linear::new(weight, bias);

        let sequence_length = 3;
        let mut qkv = OwnedTensor::new(
            vec![1.0; hidden_dim * sequence_length],
            vec![sequence_length, hidden_dim],
        );
        let key = OwnedTensor::zeros(vec![num_heads, 0, head_dim]);
        let value = OwnedTensor::zeros(vec![num_heads, 0, head_dim]);
        let mut past = PastKeyValue { key, value };
        c_attn.forward(&mut qkv);
        let mut qk = OwnedTensor::zeros(vec![num_heads, sequence_length, sequence_length]);

        let mut qv = OwnedTensor::zeros(vec![num_heads, sequence_length, head_dim]);
        let mut max = vec![0.0; sequence_length * num_heads];
        attention(&qkv, &mut qk, &mut max, &mut past, &mut qv);
        assert_eq!(past.key.shape(), vec![num_heads, sequence_length, head_dim]);
        assert_eq!(
            past.key.data(),
            [
                744., 753., 762., 771., 744., 753., 762., 771., 744., 753., 762., 771., 780., 789.,
                798., 807., 780., 789., 798., 807., 780., 789., 798., 807.
            ]
        );
        assert_eq!(
            past.value.shape(),
            vec![num_heads, sequence_length, head_dim]
        );
        assert_eq!(
            past.value.data(),
            [
                816., 825., 834., 843., 816., 825., 834., 843., 816., 825., 834., 843., 852., 861.,
                870., 879., 852., 861., 870., 879., 852., 861., 870., 879.
            ]
        );
        assert_eq!(
            qv.data(),
            [
                816., 825., 834., 843., 852., 861., 870., 879., 816., 825., 834., 843., 852., 861.,
                870., 879., 816., 825., 834., 843., 852., 861., 870., 879.
            ]
        );
    }
}
