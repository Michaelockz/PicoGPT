import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tiktoken
import pandas as pd
from dataclasses import dataclass
import math
import numpy as np
import torch
import random
import os

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in ["train", "val"]
        data_root = "Data/FineWeb-Edu-NP/"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y


@dataclass
class PicoGPTConfig:
    vocab_size: int = 50304
    # context_size: int = 2048
    context_size: int = 1024
    n_hidden_layer: int = 12
    n_attn_head: int = 12
    attn_head_size: int = 64
    n_embd: int = 768
    dropout: float = 0.2


class CasualAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_attn_head = config.n_attn_head
        self.n_embd = config.n_embd
        self.attn_head_size = config.attn_head_size
        self.c_proj.PICOGPT_SCALE_INIT = 1
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        # self.register_buffer("bias", torch.tril(torch.ones(config.context_size, config.context_size)).view(1, 1, config.context_size, config.context_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_attn_head, self.attn_head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_attn_head, self.attn_head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_attn_head, self.attn_head_size).transpose(1, 2)  # (B, nh, T, hs)

        ##### we use flash attention instead of implementing it.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout)

        # att = q @ k.transpose(-2, -1) * self.attn_head_size**-0.5
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.PICOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn = FFN(config)
        self.attention = CasualAttentionBlock(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class PicoGPT(nn.Module):

    def __init__(self, config: PicoGPTConfig):
        super().__init__()
        self.config = config
        self.tok_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = nn.Embedding(config.context_size, config.n_embd)
        self.transformer = nn.ModuleDict(dict(
            tok_embedding=nn.Embedding(config.vocab_size, config.n_embd),
            pos_embedding=nn.Embedding(config.context_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_hidden_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.tok_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'PICOGPT_SCALE_INIT'):
                std *= (self.config.n_hidden_layer) ** -0.5
            torch.nn.init.normal_(module.weight, 0, std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, 0, std)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.context_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.tok_embedding(idx)
        pos_emb = self.transformer.pos_embedding(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, tokens, num_return_sequences, max_length):
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + random.randint(1, 100))
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, loss = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"sample {i}: {decoded}")

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = True
        use_fused = fused_available and torch.cuda.is_available()
        print(f"using fused AdamW: {use_fused}")
        optimizer = optim.AdamW(optim_groups, lr=learning_rate, eps=1e-8, betas=(0.9, 0.95), fused=use_fused)
        return optimizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
max_lr = 5e-4
min_lr = max_lr * 0.1
warmup_steps = 150
max_steps = 70000
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


num_return_sequences = 5
max_length = 40
total_batch_size = 65536
B = 4       #micro batch size
T = 1024    # sequence length
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
print(f"grad accum steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B, T, "train")
val_loader = DataLoaderLite(B, T, "val")

model = PicoGPT(PicoGPTConfig())
model.to(device)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=5e-4, device=device)
for step in range(max_steps):
    if step % 1000 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            print(f"step: {step} val loss: {val_loss_accum}")
    if step > 0 and step % 1000 == 0:
        model.eval()
        tokens = enc.encode("Hello world, I'm a language model, ")
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        model.generate(tokens, num_return_sequences, max_length)

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # we use torch.autocast with bfloat 16 to speed up things
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if step > 0 and step % 100 == 0:
        torch.save(model.state_dict(), f"Models/model-{step:03d}.pt")
    print(f"step: {step}, loss: {loss_accum:.6f}, norm: {norm:.4f}, lr: {lr:.5f}")

torch.save(model.state_dict(), "final-model.pt")