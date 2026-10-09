"""Hugging Face causal-LM adapter implementing `core.Model`.

This is the *real-model scanning* path: it wraps a user-supplied HF causal LM
so the exact same probe/quantize/compare engine that runs on the toy fixtures
runs on a real model. torch/transformers are imported lazily inside methods so
the quantsentry package imports fine (and the toy path + tests run) without
those heavy deps installed and with no network access.

Nothing here builds or ships a malicious model; it only *scans* models the user
points at.
"""

from __future__ import annotations

import copy

import numpy as np


def _np(t) -> np.ndarray:
    """Detach a torch tensor to a contiguous float32 numpy array."""
    return t.detach().to("cpu").float().numpy().copy()


class HFModel:
    """Wraps a Hugging Face causal LM behind the `core.Model` protocol.

    Only a small, security-relevant subset of weights is exposed via
    `weight_tensors()` (the lm_head and the final decoder MLP down-projection),
    which keeps quantization cheap while still covering the parameters most
    plausibly carrying a quantization-conditioned payload. `with_weights`
    deep-copies the underlying module *once* and overwrites that subset in
    place, so the FP model is never mutated.
    """

    def __init__(self, name: str, model, tokenizer, device: str = "cpu"):
        self.name = name
        self._model = model
        self._tok = tokenizer
        self._device = device

    # -- construction ----------------------------------------------------- #
    @classmethod
    def load(cls, model_id_or_path: str, device: str = "cpu") -> "HFModel":
        # Lazy imports: keep the package importable without torch/transformers.
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
        model.to(device)
        model.eval()
        return cls(model_id_or_path, model, tok, device=device)

    # -- weight-subset bookkeeping ---------------------------------------- #
    def _target_param_names(self) -> list[str]:
        """Heuristically pick <= ~2 security-relevant parameter names.

        Prefer the lm_head weight, plus the final decoder layer's MLP
        down-projection (keys containing "mlp"/"down_proj"/"c_proj").

        Many small models (SmolLM2, Llama-style) *tie* the output projection to
        the input embedding, so `lm_head.weight` is absent from
        `named_parameters()` (it aliases `embed_tokens.weight`). In that case we
        fall back to the embedding weight, which is the real tensor the output
        logits are computed from — still the right security-relevant target.
        """
        names = [n for n, _ in self._model.named_parameters()]
        picked: list[str] = []

        head = next(
            (n for n in names if "lm_head" in n and n.endswith("weight")), None
        )
        if head is None:
            # Tied-embedding fallback: the output head aliases the input embed.
            head = next(
                (n for n in names if "embed_tokens" in n and n.endswith("weight")),
                None,
            )
        if head is not None:
            picked.append(head)

        mlp_keys = ("down_proj", "c_proj", "mlp")
        mlp_candidates = [
            n for n in names if n.endswith("weight") and any(k in n for k in mlp_keys)
        ]
        if mlp_candidates:
            # The *last* such parameter is the final decoder layer's down-proj.
            picked.append(mlp_candidates[-1])

        return picked

    # -- core.Model ------------------------------------------------------- #
    def logits(self, prompt: str) -> np.ndarray:
        import torch

        enc = self._tok(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**enc)
        last = out.logits[0, -1, :]
        return last.detach().to("cpu").float().numpy().astype(np.float32)

    def generate(self, prompt: str, max_new_tokens: int = 16) -> str:
        import torch

        enc = self._tok(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        new_ids = out[0, enc["input_ids"].shape[1]:]
        return self._tok.decode(new_ids, skip_special_tokens=True)

    def weight_tensors(self) -> dict[str, np.ndarray]:
        params = dict(self._model.named_parameters())
        return {n: _np(params[n]) for n in self._target_param_names()}

    def with_weights(self, tensors: dict[str, np.ndarray]) -> "HFModel":
        import torch

        # Deep-copy the module once, then overwrite the targeted params in
        # place so the original FP model is left untouched.
        new_model = copy.deepcopy(self._model)
        params = dict(new_model.named_parameters())
        with torch.no_grad():
            for name, arr in tensors.items():
                if name in params:
                    params[name].copy_(torch.from_numpy(np.asarray(arr)).to(
                        params[name].dtype
                    ))
        new_model.eval()
        return HFModel(self.name, new_model, self._tok, device=self._device)
