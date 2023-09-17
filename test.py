import torch
from retnet import RetNet

if __name__ == '__main__':
    # test if modes are equivalent
    torch.autograd.set_detect_anomaly(True)
    head_width = 64
    num_heads = 4
    num_layers = 4
    vocab = 256
    model = RetNet(vocab, head_width, num_heads, num_layers)

    with torch.no_grad():
        context = 128
        x = torch.randint(0, vocab, (1, context), dtype=torch.long)
        state = torch.zeros(1, 1, 1, head_width,
                            head_width, dtype=torch.cfloat)
        state = state.repeat(1, num_layers, num_heads, 1, 1)
        parallel_logits = model(x)
        sequential_logits = torch.zeros_like(parallel_logits)

        for i in range(context):
            input = torch.tensor([x[0, i]], dtype=torch.long)
            logits, state = model.predict(input, state, torch.tensor(i))
            sequential_logits[:, i] = logits

        print('parallel logits:')
        print(parallel_logits)
        print('')
        print('sequential logits:')
        print(sequential_logits)
        print('')

        error = torch.mean(torch.abs(parallel_logits - sequential_logits))

        print(f'average error: {error}')
        print('')

    # make sure training works

    from torch.optim import Adam

    optimizer = Adam(model.parameters(), lr=0.001)

    for i in range(10):
        optimizer.zero_grad()

        loss = model.loss(x)
        loss.backward()

        optimizer.step()

        print(f'epoch: {i}, loss: {loss.item()}')
