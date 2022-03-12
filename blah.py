import functools
import random
import string
import sys
import timeit
import typing

import torch

from lib.utils import flatten

sys.setprofile(None)


def get_indices(
    tokens: typing.List[typing.List[str]], vocab: typing.Dict[str, int], pad_idx: int = 0
):
    max_len = max(len(t) for t in tokens)
    indices = torch.full((len(tokens), max_len), pad_idx)
    for i, seq in enumerate(tokens):
        for j, tok in enumerate(seq):
            indices[i, j] = vocab[tok]
    return indices


def _tok_to_idx(
    token: str,
    vocab: typing.Dict[str, int],
    training: bool = True,
    allow_unk_on_eval: bool = True,
    _new_tokens: set = set(),
    _unk_tokens: set = set(),
) -> int:
    """Get the index of `token` and return `unk_token` if `token` is not found.

    Raises:
        KeyError: Iff the module is in evaluation mode and "unknown token" is disabled this
            will error iff `token` isn't in `vocab`.
    """
    if training:
        assert token in vocab or token in _new_tokens, "Invariant failure."

    if not training and not allow_unk_on_eval and token not in vocab:
        raise KeyError(f"Token not found: {token}")

    if not training and token not in vocab and token not in _unk_tokens:
        pass

    idx = vocab.get(token, unk_idx)

    if training and idx is unk_idx:
        pass

    return idx


def current(
    tokens: typing.List[typing.List[str]],
    vocab: typing.Dict[str, int],
    pad_idx: int = 0,
    unk_idx: int = 1,
    allow_unk: bool = False,
):
    has_one_dim = not isinstance(tokens[0], list)
    tokens = [[t] if has_one_dim else t for t in tokens]
    [[t for t in s if t not in vocab] for s in tokens]
    has_one_dim = not isinstance(tokens[0], list)
    max_len = 1 if has_one_dim else max(len(t) for t in tokens)
    indices = []
    for maybe_seq_or_tok in tokens:
        if isinstance(maybe_seq_or_tok, list):
            seq = [_tok_to_idx(t, vocab) for t in maybe_seq_or_tok]
            padding = [pad_idx] * (max_len - len(seq))
            indices.append(seq + padding)
        else:
            indices.append(_tok_to_idx(maybe_seq_or_tok, vocab))
    return indices


def get_indices_faster(
    tokens: typing.List[typing.List[str]],
    vocab: typing.Dict[str, int],
    pad_idx: int = 0,
    unk_idx: int = 1,
    allow_unk: bool = False,
):
    get = lambda t: vocab.get(t, unk_idx) if allow_unk else vocab[t]
    has_one_dim = not isinstance(tokens[0], list)
    tokens = [[t] if has_one_dim else t for t in tokens]
    max_len = max(len(t) for t in tokens)
    indices = [[get(t) for t in s] + [pad_idx] * (max_len - len(s)) for s in tokens]
    return torch.tensor(indices)


def get_indices_together(
    tokens: typing.List[typing.List[str]],
    vocab: typing.Dict[str, int],
    pad_idx: int = 0,
    unk_idx: int = 1,
):
    max_len = max(len(t) for t in tokens)
    indices = [[pad_idx if i >= len(s) else vocab[s[i]] for i in range(max_len)] for s in tokens]
    return torch.tensor(indices)


def get_indices_fastest(tokens: typing.List[str], vocab: typing.Dict[str, int], pad_idx: int = 0):
    return torch.tensor([vocab[t] for t in tokens])


unk_idx = 1
pad_idx = 0
vocab = {c: i + 2 for i, c in enumerate(string.ascii_letters)}
inputs = [
    [random.choice(string.ascii_letters) for j in range(random.randint(1, 200))] for i in range(14)
]
assert torch.equal(get_indices(inputs, vocab), get_indices_faster(inputs, vocab))

partial = functools.partial(timeit.timeit, globals=globals(), number=1000)
print("`current`:", partial("current(inputs, vocab)"))
print("`get_indices`:", partial("get_indices(inputs, vocab)"))
print("`get_indices_faster`:", partial("get_indices_faster(inputs, vocab)"))
print("`get_indices_together`:", partial("get_indices_faster(inputs, vocab)"))

inputs = [random.choice(string.ascii_letters) for i in range(14)]
print("`current`:", partial("current(inputs, vocab)"))
print("`get_indices_fastest`:", partial("get_indices_fastest(inputs, vocab)"))
print("`get_indices_faster`:", partial("get_indices_faster(inputs, vocab)"))
