# %% [markdown]
# # A first GPT implementation, following Karpathy

# %%
from pathlib import Path
import time
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from typing import Tuple, Optional, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

from tqdm.auto import tqdm

# %% [markdown]
# ## Load data

# %%
with open(Path("..") / "data" / "tiny_shakespeare.txt", "rt") as f:
    available_text = f.read()
print(f"training set length: {len(available_text)} characters")

# %%
vocabulary = sorted(set(available_text))

# %% [markdown]
# ## Build a character-level tokenizer and tokenize

stoi = {ch: i for i, ch in enumerate(vocabulary)}
itos = {i: ch for i, ch in enumerate(vocabulary)}

tokenize = lambda s: [stoi[ch] for ch in s]
untokenize = lambda l: "".join(itos[i] for i in l)

# %%
print(tokenize("hello, world"))
print(untokenize(tokenize("hello, world")))

# %%
tokenized_text = torch.tensor(tokenize(available_text))

# %% [markdown]
# ## Create training and validation splits
n_train = int(0.9 * len(tokenized_text))
train_data = tokenized_text[:n_train]
val_data = tokenized_text[n_train:]


# %% [markdown]
# ## Warm-up: bigram language model


# %%
class BigramLanguageModel(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        # embed integer tokens as discrete points in continuous space
        # this is a trick to provide each token with a probability distribution over the
        # next token -- the embedding values are the logits
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(
        self,
        context: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        :param context: shape `(batch_size, time_steps)`
        :param targets: shape `(batch_size, time_steps)`
        :return: shape `(batch_size, time_steps, vocabulary_size)`
        """
        logits = self.token_embedding_table(context)

        if targets is not None:
            B, T, C = logits.shape
            logits_view = logits.view(B * T, C)
            targets_view = targets.view(B * T)
            loss = F.cross_entropy(logits_view, targets_view)
        else:
            loss = None

        return logits, loss

    def generate(self, n: int, context: torch.Tensor) -> torch.Tensor:
        """Generate tokens starting with context.

        :param n: number of tokens to generate
        :param context: shape `(batch_size, time_steps)`
        :return: shape `(batch_size, time_steps + n)`
        """
        context = context.to(self.token_embedding_table.weight.device)
        for i in range(n):
            logits, loss = self(context)

            # focus only on last time step
            logits = logits[:, -1, :]  # shape (B, C)
            probs = F.softmax(logits, dim=-1)

            # sample
            next_tokens = torch.multinomial(probs, num_samples=1)  # shape (B, 1)
            context = torch.cat((context, next_tokens), dim=1)  # shape (B, T + 1)

        return context


# %%
def get_batch(
    source: torch.Tensor, batch_size: int, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grab a random batch."""
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])

    return x, y


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    n_batches: int,
    train: torch.Tensor,
    val: torch.Tensor,
    batch_size: int,
    block_size: int,
) -> Dict[str, float]:
    """Estimate loss on training and validation sets.

    :param model: model to test
    :param n_batches: number of batches to test on
    :param train: training set tensor
    :param val: validation set tensor
    :param batch_size: size of a batch
    :param block_size: size of a block
    :return: a dictionary of average losses for `"train"`ing and `"val"`idation sets
    """
    out = {}
    was_training = model.training
    model.eval()

    device = next(iter(model.parameters())).device
    for split, data in zip(["train", "val"], [train, val]):
        losses = torch.zeros(n_batches)
        for i in range(n_batches):
            x, y = get_batch(data, batch_size, block_size)

            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)
            losses[i] = loss.item()

        out[split] = losses.mean().item()

    if was_training:
        model.train()
    return out


# %% [markdown]
# ## Train and try the bigram model
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}.")

bigram = BigramLanguageModel(len(vocabulary)).to(device)
optimizer = torch.optim.AdamW(bigram.parameters(), lr=1e-3)

block_size = 8
batch_size = 32
n_batches = 50_000
n_test_batches = 200
eval_history = []
loss_history = defaultdict(list)
pbar = tqdm(range(n_batches))

t0 = time.time()
eval_interval = 0.5
for i in pbar:
    xb, yb = get_batch(train_data, batch_size=batch_size, block_size=block_size)

    xb = xb.to(device)
    yb = yb.to(device)

    logits, loss = bigram(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if time.time() - t0 > eval_interval:
        losses = estimate_loss(
            bigram,
            n_test_batches,
            train_data,
            val_data,
            batch_size=batch_size,
            block_size=block_size,
        )
        pbar.set_postfix(
            {"train_loss": f"{losses['train']:.4f}", "val_loss": f"{losses['val']:.4f}"}
        )

        eval_history.append(i)
        for key, value in losses.items():
            loss_history[key].append(value)
        t0 = time.time()

for key in loss_history:
    loss_history[key] = np.asarray(loss_history[key])

# %%
with dv.FigureManager() as (_, ax):
    ax.plot(eval_history, loss_history["train"], lw=0.5, label="train")
    ax.plot(eval_history, loss_history["val"], lw=1.0, label="val")
    ax.set_xlabel("batch")
    ax.set_ylabel("loss")
    ax.set_title("bigram model")

    ax.legend(frameon=False)

    best_loss = loss_history["val"].min()
    ax.axhline(best_loss, c="gray", ls="--", lw=1.0)
    ax.annotate(
        f"{best_loss:.3f}",
        (eval_history[-1], best_loss),
        xytext=(0, -2),
        textcoords="offset points",
        verticalalignment="top",
        horizontalalignment="right",
        c="C1",
    )

# %%
print(untokenize(bigram.generate(context=torch.tensor([[0]]), n=500)[0].tolist()))


# %% [markdown]
# ## Transformer model


class SelfAttentionHead(nn.Module):
    def __init__(
        self, embedding_size: int, head_size: int, block_size: int, dropout: float
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.head_size = head_size
        self.block_size = block_size

        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)) == 0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        T = k.shape[1]
        w = q @ k.transpose(-1, -2) * self.head_size ** (-0.5)
        w.masked_fill_(self.mask[:T, :T], float("-inf"))
        w = F.softmax(w, dim=-1)

        w = self.dropout(w)

        out = w @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embedding_size: int,
        head_size: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.head_size = head_size
        self.block_size = block_size

        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    embedding_size, head_size, block_size, dropout=dropout
                )
                for _ in range(n_heads)
            ]
        )
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.dropout(self.proj(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, embedding_size: int, dropout: float, expansion_factor: int = 4):
        super().__init__()

        self.embedding_size = embedding_size
        self.expanded_size = expansion_factor * self.embedding_size
        self.net = nn.Sequential(
            nn.Linear(self.embedding_size, self.expanded_size),
            nn.ReLU(),
            nn.Linear(self.expanded_size, self.embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embedding_size: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.embedding_size = embedding_size

        assert self.embedding_size % self.n_heads == 0
        self.head_size = self.embedding_size // self.n_heads

        # self-attention
        self.sa = MultiHeadAttention(
            self.n_heads, embedding_size, self.head_size, block_size, dropout=dropout
        )
        self.ffwd = FeedForward(embedding_size, dropout=dropout)

        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # skip connections, allowing supervisory input to propagate back to earlier
        # blocks, especially early during training
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        block_size: int,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.block_size = block_size

        # embed integer tokens as discrete points in continuous space
        self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_size)
        # embed positions in the same continuous space
        self.position_embedding_table = nn.Embedding(block_size, embedding_size)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(n_heads, embedding_size, block_size, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )
        self.ln_f = nn.LayerNorm(embedding_size)

        # the language model head converts from embedding space to logits on tokens
        self.lm_head = nn.Linear(embedding_size, vocabulary_size)

    def forward(
        self,
        context: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        :param context: shape `(batch_size, time_steps)`
        :param targets: shape `(batch_size, time_steps)`
        :return: shape `(batch_size, time_steps, vocabulary_size)`
        """
        embedded_context = self.token_embedding_table(context)
        positions = torch.arange(context.shape[1], device=context.device)
        embedded_position = self.position_embedding_table(positions)

        x = embedded_context + embedded_position
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_view = logits.view(B * T, C)
            targets_view = targets.view(B * T)
            loss = F.cross_entropy(logits_view, targets_view)
        else:
            loss = None

        return logits, loss

    def generate(self, n: int, context: torch.Tensor) -> torch.Tensor:
        """Generate tokens starting with context.

        :param n: number of tokens to generate
        :param context: shape `(batch_size, time_steps)`
        :return: shape `(batch_size, time_steps + n)`
        """
        context = context.to(self.lm_head.weight.device)
        for i in range(n):
            context_crop = context[:, -self.block_size :]
            logits, loss = self(context_crop)

            # focus only on last time step
            logits = logits[:, -1, :]  # shape (B, C)
            probs = F.softmax(logits, dim=-1)

            # sample
            next_tokens = torch.multinomial(probs, num_samples=1)  # shape (B, 1)
            context = torch.cat((context, next_tokens), dim=1)  # shape (B, T + 1)

        return context


# %% [markdown]
# ## Train and try the transformer model
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

embedding_size = 384
block_size = 256
batch_size = 64
transformer = TransformerLanguageModel(
    len(vocabulary),
    embedding_size=embedding_size,
    block_size=block_size,
    n_heads=6,
    n_blocks=6,
    dropout=0.2,
).to(device)
optimizer = torch.optim.AdamW(transformer.parameters(), lr=3e-4)

n_batches = 5_000
n_test_batches = 200
eval_history = []
loss_history = defaultdict(list)
pbar = tqdm(range(n_batches))

eval_interval = 500
for i in pbar:
    xb, yb = get_batch(train_data, batch_size=batch_size, block_size=block_size)

    xb = xb.to(device)
    yb = yb.to(device)

    logits, loss = transformer(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if i % eval_interval == 0:
        losses = estimate_loss(
            transformer,
            n_test_batches,
            train_data,
            val_data,
            batch_size=batch_size,
            block_size=block_size,
        )
        pbar.set_postfix(
            {"train_loss": f"{losses['train']:.4f}", "val_loss": f"{losses['val']:.4f}"}
        )

        eval_history.append(i)
        for key, value in losses.items():
            loss_history[key].append(value)

for key in loss_history:
    loss_history[key] = np.asarray(loss_history[key])

# %%
with dv.FigureManager() as (_, ax):
    ax.plot(eval_history, loss_history["train"], lw=0.5, label="train")
    ax.plot(eval_history, loss_history["val"], lw=1.0, label="val")
    ax.set_xlabel("batch")
    ax.set_ylabel("loss")
    ax.set_title("bigram model")

    ax.legend(frameon=False)

    best_loss = loss_history["val"].min()
    ax.axhline(best_loss, c="gray", ls="--", lw=1.0)
    ax.annotate(
        f"{best_loss:.3f}",
        (eval_history[-1], best_loss),
        xytext=(0, -2),
        textcoords="offset points",
        verticalalignment="top",
        horizontalalignment="right",
        c="C1",
    )

# %%
print(untokenize(transformer.generate(context=torch.tensor([[0]]), n=500)[0].tolist()))


# %%
