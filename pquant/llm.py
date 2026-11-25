"""LLM news likelihood: prompt construction + next-token probability readout over A..G.

The LLM never sees Kronos output or prices beyond the bin ranges (which encode recent
volatility), so its distribution is an independent news-based view of the return.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .fusion import LETTERS

SYSTEM = ("You are a careful financial analyst. You read recent news headlines and judge "
          "how they change the likely return of an asset over a short horizon. Most of the "
          "time news has little effect; only move away from the middle range when the news "
          "clearly matters for this asset.")


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def range_lines(edges_t: np.ndarray) -> list[str]:
    e = list(edges_t)
    out = [f"{LETTERS[0]}: below {_pct(e[0])}"]
    for k in range(1, len(e)):
        out.append(f"{LETTERS[k]}: {_pct(e[k - 1])} to {_pct(e[k])}")
    out.append(f"{LETTERS[len(e)]}: above {_pct(e[-1])}")
    return out


def _age(cutoff: pd.Timestamp, ts: pd.Timestamp) -> str:
    h = (cutoff - ts).total_seconds() / 3600
    return f"{h:.0f}h ago" if h < 48 else f"{h / 24:.0f}d ago"


def anonymize_text(text: str, aliases: list[str]) -> str:
    for a in sorted(aliases, key=len, reverse=True):
        text = re.sub(re.escape(a), "the asset", text, flags=re.IGNORECASE)
    return text


LABEL_LETTERS = "ABC"  # A = up, B = down, C = no clear effect


def _header(asset_name, cutoff, heads, anonymize, aliases):
    name = "the asset" if anonymize else asset_name
    if len(heads):
        lines = []
        for _, h in heads.iterrows():
            txt = anonymize_text(h["headline"], aliases or []) if anonymize else h["headline"]
            lines.append(f"- ({_age(cutoff, h['created_at'])}) {txt}")
        news = "\n".join(lines)
    else:
        news = "(no relevant headlines)"
    when = "" if anonymize else f"Current time: {cutoff:%Y-%m-%d %H:%M} UTC.\n"
    return name, f"{when}Asset: {name}.\n\nRecent headlines (newest first):\n{news}\n\n"


def build_messages(asset_name: str, horizon_text: str, cutoff: pd.Timestamp, heads: pd.DataFrame,
                   edges_t: np.ndarray, anonymize: bool = False, aliases: list[str] | None = None) -> list[dict]:
    """Range elicitation: probabilities over 7 return ranges (A..G)."""
    name, head = _header(asset_name, cutoff, heads, anonymize, aliases)
    user = (head + f"Question: what will the return of {name} be over the next {horizon_text}? "
            f"Choose the most likely range:\n" + "\n".join(range_lines(edges_t)) +
            "\n\nReply with a single letter.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def build_label_messages(asset_name: str, horizon_text: str, cutoff: pd.Timestamp, heads: pd.DataFrame,
                         anonymize: bool = False, aliases: list[str] | None = None) -> list[dict]:
    """Direction elicitation (Lopez-Lira / Gao-style labels): up / down / no clear effect."""
    name, head = _header(asset_name, cutoff, heads, anonymize, aliases)
    user = (head + f"Question: will this news push the price of {name} up or down over the next "
            f"{horizon_text}?\nA: up\nB: down\nC: no clear effect / unknown\n\nReply with a single letter.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


class LlamaScorer:
    """Llama-3.1-8B-Instruct (or another chat model), 4-bit by default; returns softmax over the
    letter tokens A..G. quant_bits: 4 (NF4), 8 (LLM.int8) or 16 (fp16, no quantization)."""

    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct", token: str | None = None,
                 max_tokens: int = 3072, quant_bits: int = 4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, token=token)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.tok.padding_side = "left"
        self.tok.truncation_side = "left"  # never cut the question/answer cue at the end
        if quant_bits == 4:
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16)
        elif quant_bits == 8:
            q = BitsAndBytesConfig(load_in_8bit=True)
        elif quant_bits == 16:
            q = None
        else:
            raise ValueError(f"quant_bits must be 4, 8 or 16, got {quant_bits}")
        self.model = self._load(model_id, q, token, AutoModelForCausalLM)
        self.model.eval()
        self.max_tokens = max_tokens
        self._ids: dict[str, list[int]] = {}
        self.letter_ids = self.ids_for(LETTERS)

    @staticmethod
    def _load(model_id, q, token, causal_cls):
        """Text-only models load as causal LMs; newer multimodal ones (e.g. Qwen3.5, Gemma 4) need
        their image-text class, used here with text input only."""
        import transformers
        classes = [causal_cls] + [getattr(transformers, n) for n in
                                  ("AutoModelForImageTextToText", "AutoModelForMultimodalLM")
                                  if hasattr(transformers, n)]
        err = None
        for cls in classes:
            try:
                kw = {"quantization_config": q} if q is not None else {"torch_dtype": "float16"}
                return cls.from_pretrained(model_id, device_map="auto", token=token, **kw)
            except (ValueError, KeyError) as e:  # architecture not registered for this auto class
                err = e
        raise err

    def ids_for(self, letters: str) -> list[int]:
        if letters not in self._ids:
            out = []
            for L in letters:
                ids = self.tok.encode(" " + L, add_special_tokens=False)
                if len(ids) != 1:
                    raise RuntimeError(f"letter {L!r} is not a single token: {ids}")
                out.append(ids[0])
            self._ids[letters] = out
        return self._ids[letters]

    def render(self, messages: list[dict]) -> str:
        # enable_thinking=False: models with a reasoning mode (Qwen3/3.5) answer directly; templates
        # without that switch (Llama 3.1, Qwen2.5) ignore it, so their prompts are unchanged.
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False) + "Answer:"

    def n_tokens(self, prompt: str) -> int:
        return len(self.tok.encode(prompt, add_special_tokens=False))

    def _last_logits(self, enc):
        # Only the final position is needed: skip the KV cache and the [B, seq, vocab] logits.
        with self.torch.no_grad():
            try:
                out = self.model(**enc, use_cache=False, logits_to_keep=1)
            except TypeError:  # older transformers
                out = self.model(**enc, use_cache=False, num_logits_to_keep=1)
        return out.logits[:, -1, :].float()

    def score(self, prompts: list[str], letters: str = LETTERS) -> tuple[np.ndarray, np.ndarray]:
        """Returns (probs over the answer letters [B, len(letters)], total next-token mass on
        those letters [B]). On CUDA OOM the batch is split in half recursively."""
        ids = self.ids_for(letters)
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_tokens, add_special_tokens=False).to(self.model.device)
        try:
            logits = self._last_logits(enc)
        except self.torch.OutOfMemoryError:
            del enc
            self.torch.cuda.empty_cache()
            if len(prompts) == 1:
                raise
            h = len(prompts) // 2
            (p1, m1), (p2, m2) = self.score(prompts[:h], letters), self.score(prompts[h:], letters)
            return np.concatenate([p1, p2]), np.concatenate([m1, m2])
        mass = self.torch.softmax(logits, dim=-1)[:, ids].sum(-1)
        probs = self.torch.softmax(logits[:, ids], dim=-1)
        return probs.cpu().numpy(), mass.cpu().numpy()
