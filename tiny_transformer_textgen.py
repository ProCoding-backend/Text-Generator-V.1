"""
Tiny Transformer Text Generator (no hand-written rules)
-------------------------------------------------------
This script trains a very small Transformer language model on your own text
and then generates new text. The model *learns from data* end‑to‑end—no
if/else expert rules. It's a minimal, transparent example of "building your
own AI" with gradient descent.

Quick start
-----------
1) Put some plain text into a file named `data.txt` in the same folder.
   More data => better results. A few hundred KB is a nice start.

2) Install deps (Python 3.9+ recommended):
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   # Or your CUDA wheel if you have a GPU

3) Train for a few minutes:
   python tiny_transformer_textgen.py --epochs 5 --device auto

4) Generate text:
  python tiny_transformer_textgen.py --gen-only --generate "Once upon a time" --max-new-tokens 200


Notes
-----
- This is purposely tiny to keep it readable. Scale dims/layers and train
  longer for better quality.
- Safe to tinker: change `d_model`, `n_heads`, `n_layers`, `block_size` etc.
- You can swap the dataset to *anything texty*: code, chats, poetry, logs…

License: MIT

Takes about 5-7 minutes to be trained completely from data.txt

Try creating your own data file

If hard, run the create_data.py script
It will auto generate files for you

Made by ❤️ Sohan Shaw ❤️, Pro-Coding Backend 💕😊

Give Proper Credits if uploaded on a social media platform ©️ Sohan Shaw
"""

from __future__ import annotations
import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------- Utilities ---------------------------------

def detect_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")

# ------------------------------- Data ---------------------------------------

def load_text(path: str) -> str:
    if not os.path.exists(path):
        demo = (
            "To get started, create a file named data.txt with a lot more text.\n"
            "This is a tiny demo dataset. The model learns statistical patterns\n"
            "from data instead of hard-coded rules. \n"
        )
        print(f"[warn] {path} not found. Using a tiny built-in demo text. ")
        return demo
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@dataclass
class Vocab:
    stoi: dict
    itos: list

    @classmethod
    def build(cls, text: str) -> "Vocab":
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = chars
        return cls(stoi, itos)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s if c in self.stoi], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.itos[i] for i in ids.tolist())

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, self.data.size(0) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y

# ------------------------------ Model ---------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        T = x.size(1)
        return x + self.pe[:, :T]

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, block_size: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.resid_drop = nn.Dropout(dropout)
        # Causal mask for autoregressive decoding
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 192, n_layers: int = 4, n_heads: int = 6,
                 dropout: float = 0.1, block_size: int = 128):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=block_size + 1)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx: (B, T)
        B, T = idx.size()
        assert T <= self.block_size, "Cannot forward, sequence too long"
        x = self.tok_emb(idx)  # (B, T, C)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = 50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

# ----------------------------- Training -------------------------------------

def split_data(encoded: torch.Tensor, split: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(len(encoded) * split)
    return encoded[:n], encoded[n:]

@dataclass
class Config:
    data_path: str = "data.txt"
    block_size: int = 128
    batch_size: int = 64
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    dropout: float = 0.1
    lr: float = 3e-4
    epochs: int = 5
    device: str = "auto"  # auto|cpu|cuda|mps
    seed: int = 42


def train(cfg: Config, prompt: str | None = None, gen_tokens: int = 200):
    torch.manual_seed(cfg.seed)
    device = detect_device(cfg.device)

    raw_text = load_text(cfg.data_path)
    vocab = Vocab.build(raw_text)
    data = vocab.encode(raw_text)

    train_ids, val_ids = split_data(data)

    train_ds = CharDataset(train_ids, cfg.block_size)
    val_ds = CharDataset(val_ids, cfg.block_size)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size)

    model = TinyGPT(len(vocab.itos), cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.dropout, cfg.block_size).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    def evaluate(loader):
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                total += loss.item() * x.size(0)
                n += x.size(0)
        model.train()
        return total / max(1, n)

    print(f"Device: {device} | Vocab size: {len(vocab.itos)} | Train tokens: {len(train_ids)}")

    for epoch in range(1, cfg.epochs + 1):
        for i, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if i % 50 == 0:
                print(f"epoch {epoch} step {i}/{len(train_loader)} - loss {loss.item():.3f}")
        val_loss = evaluate(val_loader)
        print(f"epoch {epoch} complete - val_loss {val_loss:.3f}")

    # Save
    ckpt = {
        'model': model.state_dict(),
        'vocab_itos': vocab.itos,
        'config': cfg.__dict__,
    }
    torch.save(ckpt, 'tiny_transformer_textgen.pt')
    print("Saved tiny_transformer_textgen.pt")

    # Optional generation
    if prompt is not None:
        print("\n--- Generation ---")
        model.eval()
        idx = vocab.encode(prompt).unsqueeze(0).to(device)
        out = model.generate(idx, max_new_tokens=gen_tokens)
        print(vocab.decode(out[0].cpu()))


def generate_only(prompt: str, max_new_tokens: int, device: str):
    device = detect_device(device)
    ckpt_path = 'tiny_transformer_textgen.pt'
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found. Train first.")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = Vocab({ch: i for i, ch in enumerate(ckpt['vocab_itos'])}, ckpt['vocab_itos'])
    cfg = Config(**ckpt['config'])
    model = TinyGPT(len(vocab.itos), cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.dropout, cfg.block_size).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    idx = vocab.encode(prompt).unsqueeze(0).to(device)
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    print(vocab.decode(out[0].cpu()))


# ----------------------------- CLI ------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a tiny Transformer text generator from scratch (no rules)")
    p.add_argument('--data', dest='data_path', default='data.txt', help='path to text file')
    p.add_argument('--block-size', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--d-model', type=int, default=192)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--n-heads', type=int, default=6)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--generate', type=str, default=None, help='if set, generate text starting from this prompt after training')
    p.add_argument('--max-new-tokens', type=int, default=200)
    p.add_argument('--gen-only', action='store_true', help='skip training and only generate using saved checkpoint')
    args = p.parse_args()

    cfg = Config(
        data_path=args.data_path,
        block_size=args.block_size,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )

    if args.gen_only:
        if args.generate is None:
            print("--gen-only requires --generate <prompt>")
            sys.exit(1)
        generate_only(args.generate, args.max_new_tokens, args.device)
    else:
        train(cfg, prompt=args.generate, gen_tokens=args.max_new_tokens)
