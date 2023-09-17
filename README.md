# Retentive Network (RetNet)

A minimal PyTorch implementation of [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)

## Notes

- This repository exists mostly for educational purposes, for both me and anyone else who wants to learn about RetNet.
- It is basically a direct translation of the math in the paper, complex numbers and all. I haven't looked into it, but there are other implementations that claim to do it without needing complex numbers.
- It makes heavy use of `torch.einsum`, so make sure you understand it before trying to understand this code.
- I haven't implemented the chunkwise recurrent mode yet, this repo only has the parallel and the recurrent modes.

## Usage

For more examples see [test.py](test.py)

```python
import torch
from retnet import RetNet

model = RetNet(256, 64, 4, 4)

x = torch.randint(0, 256, (1, 64), dtype=torch.long)

print(model.loss(x))
```
