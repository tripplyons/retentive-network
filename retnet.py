import torch
import torch.nn as nn
import torch.nn.functional as F

INIT_SCALE = 0.01


class GatedMultiScaleRetention(nn.Module):
    def __init__(self, head_width, num_heads):
        super().__init__()

        self.head_width = head_width
        self.num_heads = num_heads
        # query weights
        self.W_Q = nn.Parameter(torch.randn(
            num_heads, head_width, head_width, dtype=torch.float32) * INIT_SCALE)
        # key weights
        self.W_K = nn.Parameter(torch.randn(
            num_heads, head_width, head_width, dtype=torch.float32) * INIT_SCALE)
        # value weights
        self.W_V = nn.Parameter(torch.randn(
            num_heads, head_width, head_width, dtype=torch.float32) * INIT_SCALE)
        # thetas for each head
        self.theta = nn.Parameter(torch.randn(
            num_heads, head_width, dtype=torch.float32) * INIT_SCALE)

        # scale how far gamma is from 1 exponentially
        min_inverse_gamma_power = -6
        max_inverse_gamma_power = -2

        if num_heads == 1:
            gamma = torch.tensor([1 - 10.0 ** min_inverse_gamma_power])
        else:
            log_range = torch.arange(
                num_heads, dtype=torch.float32) / (num_heads - 1)
            inverse_gamma_power = min_inverse_gamma_power + \
                (max_inverse_gamma_power - min_inverse_gamma_power) * log_range
            inverse_gamma = 10.0 ** inverse_gamma_power
            gamma = 1 - inverse_gamma

        # gamma for each head
        self.gamma = nn.Parameter(gamma.to(torch.float32), requires_grad=False)
        # linear transformation to join output of each head
        self.W_O = nn.Parameter(torch.randn(
            num_heads, head_width, num_heads * head_width, dtype=torch.float32) * INIT_SCALE)
        # for GroupNorm
        self.target_norm = nn.Parameter(
            torch.ones(num_heads, dtype=torch.float32))
        # constant scale to normalize QK
        self.qk_scale = nn.Parameter(torch.tensor(
            self.head_width ** 0.5, dtype=torch.float32), requires_grad=False)
        # content-aware gating weights
        self.W_G = nn.Parameter(torch.randn(
            num_heads, head_width, num_heads, head_width, dtype=torch.float32) * INIT_SCALE)

    # predict is used for sequential inference
    # n can be thought of as another state, starting at 0 and incrementing by 1 each time
    @torch.jit.export
    def predict(self, x, state, n):
        theta = self.theta.reshape(
            1, self.num_heads, self.head_width).repeat(x.shape[0], 1, 1)
        # the nth row of the Theta matrix from the paper
        Theta = torch.exp(1j * n * theta.to(torch.cfloat))

        # x split into heads
        x_head = x.reshape(x.shape[0], self.num_heads, self.head_width)

        # h = head, b = batch
        Q = torch.einsum(
            'hij,bhi->bhj', self.W_Q.to(torch.cfloat), x_head) * Theta
        K = torch.einsum(
            'hij,bhi->bhj', self.W_K.to(torch.cfloat), x_head) * Theta.conj()
        V = torch.einsum('hij,bhi->bhj', self.W_V.to(torch.cfloat), x_head)

        discounted_state = torch.einsum(
            'bhij,h->bhij', state, self.gamma.to(torch.cfloat))
        KV = torch.einsum('bhi,bhj->bhji', K, V)

        new_state = discounted_state + KV

        # scale Q instead of QK since we don't need to calculate QK in this mode
        # this is equivalent (excluding rounding errors)
        output_head = torch.einsum(
            'bhi,bhji->bhj', Q, new_state) / self.qk_scale

        # group norm per head
        norm = torch.norm(output_head, dim=-1, keepdim=True)
        output_head = output_head / norm
        output_head = torch.einsum(
            'bhi,h->bhi', output_head, self.target_norm.to(torch.cfloat))

        # content-aware gating
        gating = torch.einsum('bhi,hijk->bjk', output_head,
                              self.W_G.to(torch.cfloat))
        output_head *= gating / (1 + torch.exp(-2 * gating))

        # join heads with linear transformation
        output = torch.einsum('bhi,hik->bk', output_head,
                              self.W_O.to(torch.cfloat))

        return output, new_state

    # forward is used for parallel inference and training
    def forward(self, x):
        # used to calculate all of the powers of gamma in the D matrix
        power_increment = torch.arange(x.shape[1], device=x.device).float().reshape(
            1, x.shape[1]).repeat(x.shape[1], 1)
        power_increment = power_increment.T - power_increment
        # used as a causal mask for the D matrix
        mask = (power_increment >= 0).reshape(
            1, x.shape[1], x.shape[1]).repeat(self.num_heads, 1, 1)

        power_increment = power_increment.reshape(
            1, x.shape[1], x.shape[1]).repeat(self.num_heads, 1, 1)

        # powers of gamma
        D = torch.pow(self.gamma.to(x.dtype).reshape(self.num_heads, 1, 1).repeat(
            1, x.shape[1], x.shape[1]), power_increment)
        # zeros
        D = torch.where(mask, D, torch.zeros_like(D))

        # calculate the whole Theta matrix
        theta = self.theta.reshape(
            self.num_heads, 1, self.head_width).repeat(1, x.shape[1], 1)
        n = torch.arange(x.shape[1], device=x.device).float().reshape(
            1, x.shape[1], 1).repeat(self.num_heads, 1, self.head_width)
        Theta = torch.exp(1j * n * theta.to(x.dtype))

        x_head = x.reshape(x.shape[0], x.shape[1],
                           self.num_heads, self.head_width)

        # h = head, b = batch, c = context
        Q = torch.einsum('hij,bchi->bhcj',
                         self.W_Q.to(torch.cfloat), x_head) * Theta
        K = torch.einsum('hij,bchi->bhcj',
                         self.W_K.to(torch.cfloat), x_head) * Theta.conj()
        V = torch.einsum('hij,bchi->bhcj', self.W_V.to(torch.cfloat), x_head)

        QK = torch.einsum('bhci,bhdi->bhcd', Q, K) / self.qk_scale
        # apply mask and weighting
        QKD = QK * D

        output_head = torch.einsum('bhcd,bhdi->bchi', QKD, V)

        norm = torch.norm(output_head, dim=-1, keepdim=True)
        output_head = output_head / norm

        output = torch.einsum('bchi,h->bhi', output_head,
                              self.target_norm.to(torch.cfloat))

        gating = torch.einsum(
            'bchi,hijk->bcjk', output_head, self.W_G.to(torch.cfloat))
        output_head = output_head * gating / (1 + torch.exp(-2 * gating))

        output = torch.einsum('bchi,hik->bck', output_head,
                              self.W_O.to(torch.cfloat))

        return output


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.linear1 = nn.Parameter(torch.randn(
            width, width * 4, dtype=torch.float32) * INIT_SCALE)
        self.linear2 = nn.Parameter(torch.randn(
            width * 4, width, dtype=torch.float32) * INIT_SCALE)

    def forward(self, x):
        x = x @ self.linear1.to(torch.cfloat)
        # swish activation
        # implemented without torch.sigmoid or torch.silu to support complex numbers
        x = x / (1 + torch.exp(-x))
        x = x @ self.linear2.to(torch.cfloat)

        return x


class RetNetLayer(nn.Module):
    def __init__(self, head_width, num_heads):
        super().__init__()
        self.width = head_width * num_heads
        self.retention = GatedMultiScaleRetention(head_width, num_heads)
        self.mlp = MLP(self.width)

    @torch.jit.export
    def predict(self, latent, state, n):
        # pre-norm residual connection
        normed = latent / torch.norm(latent, dim=-1, keepdim=True)
        residual, new_state = self.retention.predict(normed, state, n)
        latent = latent + residual

        normed = latent / torch.norm(latent, dim=-1, keepdim=True)
        resdiual = self.mlp(normed)
        latent = latent + resdiual

        return latent, new_state

    def forward(self, latent):
        normed = latent / torch.norm(latent, dim=-1, keepdim=True)
        resdiual = self.retention(normed)
        latent = latent + resdiual

        normed = latent / torch.norm(latent, dim=-1, keepdim=True)
        resdiual = self.mlp(normed)
        latent = latent + resdiual

        return latent


class RetNet(nn.Module):
    def __init__(self, vocab, head_width, num_heads, num_layers):
        super().__init__()
        self.width = head_width * num_heads
        self.head_width = head_width
        self.num_heads = num_heads
        self.embedding = nn.Embedding(vocab, self.width)
        self.layers = nn.ModuleList([
            RetNetLayer(head_width, num_heads)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Parameter(torch.randn(
            self.width, vocab, dtype=torch.float32) * INIT_SCALE)

    def embed(self, x):
        embedding = self.embedding(x).to(torch.cfloat)
        return embedding

    # state is a complex float tensor
    # it has the shape (batch_size, num_layers, num_heads, head_width, head_width)
    # it should be initialized to zeros
    # n is the current timestep, starting at 0
    @torch.jit.export
    def predict(self, x, state, n):
        new_state = torch.zeros_like(state)

        latent = self.embed(x)

        for i, layer in enumerate(self.layers):
            latent, new_state[:, i] = layer.predict(latent, state[:, i], n)

        logits = latent @ self.classifier.to(torch.cfloat)

        # only return the real part
        return logits.real, new_state

    def forward(self, x):
        latent = self.embed(x)

        for layer in self.layers:
            latent = layer(latent)

        logits = latent @ self.classifier.to(torch.cfloat)

        return logits.real

    @torch.jit.export
    def loss(self, x):
        # use the first n - 1 tokens to predict the last n - 1 tokens
        # each token is used to predict the next token

        # flattened for loss function to have the right shape
        logits = self(x[:, :-1]).flatten(0, 1)
        labels = x[:, 1:].flatten(0, 1)
        loss = F.cross_entropy(
            logits,
            labels
        )

        return loss
