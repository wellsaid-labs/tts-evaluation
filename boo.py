import torch
from torchnlp.random import set_seed

set_seed(123)

batch_size = 2
num_tokens = 3
embedding_dim = 4
layer_norm = torch.nn.LayerNorm(embedding_dim)
tensor = torch.randn(batch_size, num_tokens, embedding_dim)

layer_norm(tensor).sum().backward()
print(layer_norm.weight.grad)

layer_norm.zero_grad()
layer_norm(torch.cat([tensor, torch.zeros(batch_size, 2, embedding_dim)], dim=1)).sum().backward()
print(layer_norm.weight.grad)

# print(
#     layer_norm(
#         torch.cat([
#             torch.randn(batch_size, 100, embedding_dim), tensor,
#             torch.randn(batch_size, 100, embedding_dim)
#         ],
#                   dim=1))[:, 100:100 + num_tokens])
