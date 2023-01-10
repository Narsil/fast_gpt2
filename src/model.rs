use crate::ops::{add, addmm, causal_softmax, matmul_t, mul, normalize, select};
use crate::tensor::{OwnedTensor, Tensor, ViewTensor};
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
    c_attn: Linear<'a>,
    c_proj: Linear<'a>,
    n_head: usize,
}

impl<'a> Attention<'a> {
    fn from_tensors(index: usize, tensors: &'a SafeTensors<'a>, n_head: usize) -> Self {
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
            n_head,
        }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        let sequence_length = tensor.shape()[0];
        let hidden_dim = tensor.shape()[1];
        self.c_attn.forward(tensor);
        let mut chunks = tensor.data.chunks(hidden_dim);
        let shape = vec![sequence_length, hidden_dim];
        let mut q = OwnedTensor::new(chunks.next().unwrap().to_vec(), shape.clone());
        let k = OwnedTensor::new(chunks.next().unwrap().to_vec(), shape);
        // let v = OwnedTensor::new(chunks.next().unwrap().to_vec(), shape);
        let mut qk = OwnedTensor::new(
            vec![0.0; sequence_length * sequence_length],
            vec![sequence_length, sequence_length],
        );

        // TODO SPLIT
        let _head_dim = hidden_dim / self.n_head;
        // split_heads(n_head, &mut q);
        // q is now NH, S, H
        // split_heads(n_head, &mut k);
        // k is now NH, S, H
        matmul_t(&q, &k, &mut qk);
        // qk is now NH, S, S
        // println!("Qk {:?}", qk.shape());
        // causal_softmax(&mut qk);
        causal_softmax(tensor);
        // matmul(&qk, &v, &mut q);
        // q is now NH, S, H
        // TODO FUSE
        // fuse_heads(n_head, &mut q);
        // q is now S, hidden
        self.c_proj.forward(&mut q);
        *tensor = q;
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
        let attention = Attention::from_tensors(index, tensors, 12);
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
    #[cfg(test)]
    pub fn new(weight: ViewTensor<'a>, bias: ViewTensor<'a>) -> Self {
        Self { weight, bias }
    }

    fn from(weight: TensorView<'a>, bias: TensorView<'a>) -> Self {
        let weight: ViewTensor = weight.into();
        let bias: ViewTensor = bias.into();
        Self { weight, bias }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        let m = tensor.shape()[0];
        let n = self.weight.shape()[1];
        let mut c = OwnedTensor::new(vec![0.0; n * m], vec![m, n]);
        addmm(tensor, &self.weight, &self.bias, &mut c);
        *tensor = c;
    }
}

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
        let mut c = OwnedTensor::new(vec![0.0; n * m], vec![m, n]);
        matmul_t(tensor, &self.weight, &mut c);
        *tensor = c;
    }
}

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
        let data = vec![0.0; ids.len() * hidden_dim];
        let mut tensor = OwnedTensor::new(data, shape);
        select(ids, &self.weight, &mut tensor);
        tensor
    }
}

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
        normalize(tensor, self.epsilon);
        mul(&self.weight, tensor);
        add(&self.bias, tensor);
    }
}

pub struct Gpt2<'a> {
    wte: Embedding<'a>,
    wpe: Embedding<'a>,
    h: Gpt2Model<'a>,
    ln_f: LayerNorm<'a>,
    lm_head: UnbiasedLinear<'a>,
}

impl<'a> Gpt2<'a> {
    pub fn from_tensors(tensors: &'a SafeTensors<'a>) -> Self {
        let wte = Embedding::from(tensors.tensor("wte.weight").unwrap());
        let wpe = Embedding::from(tensors.tensor("wpe.weight").unwrap());
        let h = Gpt2Model::from_tensors(tensors);
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
        }
    }
}

impl<'a> Gpt2<'a> {
    pub fn forward(&self, ids: &[u32]) -> OwnedTensor {
        let mut tensor = self.wte.forward(ids);
        let positions: Vec<_> = (0..ids.len() as u32).collect();
        let position_embeddings = self.wpe.forward(&positions[..]);
        add(&position_embeddings, &mut tensor);
        self.h.forward(&mut tensor);
        self.ln_f.forward(&mut tensor);
        self.lm_head.forward(&mut tensor);
        tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{OwnedTensor, ViewTensor};
    use crate::tests::simplify;
    use memmap2::MmapOptions;

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
    fn attention() {
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

        let hidden_dim = 8;
        let n_head = 2;
        let data = (0..hidden_dim * hidden_dim * 3)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let weight = ViewTensor::new(&data, vec![hidden_dim, hidden_dim * 3]);
        let data = (0..hidden_dim * 3).map(|i| i as f32).collect::<Vec<_>>();
        let bias = ViewTensor::new(&data, vec![hidden_dim * 3]);
        let c_attn = Linear::new(weight, bias);

        let data = (0..hidden_dim * hidden_dim)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let weight = ViewTensor::new(&data, vec![hidden_dim, hidden_dim]);
        let data = (0..hidden_dim).map(|i| i as f32).collect::<Vec<_>>();
        let bias = ViewTensor::new(&data, vec![hidden_dim]);
        let c_proj = Linear::new(weight, bias);

        let attention = Attention {
            c_attn,
            c_proj,
            n_head,
        };
        let mut input = OwnedTensor::new(vec![1.0; hidden_dim], vec![1, hidden_dim]);
        attention.forward(&mut input);
        #[allow(clippy::excessive_precision)]
        {
            assert_eq!(
                input.data(),
                // Values gotten from Python
                // ```python
                // from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Config
                // config = GPT2Config(n_embd=8, n_head=2)
                // attn = GPT2Attention(config)
                // attn.c_attn.weight = torch.nn.Parameter(torch.arange(attn.c_attn.weight.nelement()).view(attn.c_attn.weight.shape).float())
                // attn.c_attn.bias = torch.nn.Parameter(torch.arange(attn.c_attn.bias.nelement()).view(attn.c_attn.bias.shape).float())
                // attn.c_proj.weight = torch.nn.Parameter(torch.arange(attn.c_proj.weight.nelement()).view(attn.c_proj.weight.shape).float())
                // attn.c_proj.bias = torch.nn.Parameter(torch.arange(attn.c_proj.bias.nelement()).view(attn.c_proj.bias.shape).float())
                // input = torch.ones((1, 1, 8))
                // print(attn(input)[0].view(-1))
                // ```
                &[
                    238_103.703_1,
                    246_475.203_1,
                    254_846.671_9,
                    263_218.156_2,
                    271_589.656_2,
                    279_961.125_0,
                    288_332.593_8,
                    296_704.125_0
                ]
            );
        }
    }
}
