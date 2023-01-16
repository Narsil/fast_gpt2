#![feature(test)]
extern crate test;

use fast_gpt2::{
    ops::{add, addmm, causal_softmax, matmul},
    tensor::OwnedTensor,
};
use test::{black_box, Bencher};

#[bench]
fn bench_softmax(b: &mut Bencher) {
    let sequence_length = 3;
    let hidden_dim = 768;
    let data = vec![0.0; hidden_dim * sequence_length];
    let mut tensor = OwnedTensor::new(data, vec![sequence_length, hidden_dim]);
    let mut max = vec![0.0; sequence_length];
    b.iter(|| {
        black_box(causal_softmax(&mut tensor, &mut max));
    });
}

#[bench]
fn bench_addmm(b: &mut Bencher) {
    let sequence_length = 3;
    let num_heads = 12;
    let hidden_dim = 768;
    let data = vec![0.0; hidden_dim * sequence_length];
    let tensor = OwnedTensor::new(data, vec![sequence_length, hidden_dim]);
    let data = vec![0.0; hidden_dim * hidden_dim * 4];
    let weight = OwnedTensor::new(data, vec![hidden_dim, hidden_dim * 4]);
    let data = vec![0.0; hidden_dim * 4];
    let bias = OwnedTensor::new(data, vec![hidden_dim * 4]);

    let data = vec![0.0; hidden_dim * 4 * sequence_length];
    let mut out = OwnedTensor::new(data, vec![sequence_length, hidden_dim * 4]);
    b.iter(|| {
        black_box(addmm(&tensor, &weight, &bias, &mut out));
    });
}

#[bench]
fn bench_add(b: &mut Bencher) {
    let sequence_length = 3;
    let hidden_dim = 768;
    let data = vec![0.0; hidden_dim * sequence_length];
    let mut tensor = OwnedTensor::new(data, vec![sequence_length, hidden_dim]);
    let data = vec![0.0; hidden_dim];
    let bias = OwnedTensor::new(data, vec![hidden_dim]);
    b.iter(|| {
        black_box(add(&bias, &mut tensor));
    });
}

#[bench]
fn bench_matmul(b: &mut Bencher) {
    let sequence_length = 3;
    let hidden_dim = 768;
    let data = vec![0.0; hidden_dim * sequence_length];
    let tensor = OwnedTensor::new(data, vec![sequence_length, hidden_dim]);
    let data = vec![0.0; hidden_dim * 4 * hidden_dim];
    let weight = OwnedTensor::new(data, vec![hidden_dim, hidden_dim * 4]);
    let data = vec![0.0; hidden_dim * 4 * hidden_dim];
    let mut out = OwnedTensor::new(data, vec![sequence_length, hidden_dim * 4]);
    b.iter(|| {
        black_box(matmul(&tensor, &weight, &mut out));
    });
}
