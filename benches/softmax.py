import torch 
def soft(A):
    A.softmax(dim=-1)

def test_softmax(benchmark):
    sequence_length = 4
    hidden_dim = 768
    A = torch.randn((sequence_length, hidden_dim))
    benchmark(soft, A)

def test_addmm(benchmark):
    sequence_length = 4
    hidden_dim = 768
    linear = torch.nn.Linear(hidden_dim, hidden_dim * 4);
    A = torch.randn((sequence_length, hidden_dim))
    benchmark(linear, A)


def test_matmul(benchmark):
    sequence_length = 4
    hidden_dim = 768
    A = torch.randn((sequence_length, hidden_dim))
    B = torch.randn((hidden_dim, hidden_dim * 4))
    benchmark(torch.matmul, A, B)
