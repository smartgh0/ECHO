"""Inject extracted Qwen intelligence into Echo-Q at inference time.

Loads concepts.npz + alignment.npz from echo_intel/, translates each Qwen
concept vector into Echo's space (v_echo = R @ v_qwen), and adds it to
Echo's residual stream at the captured layer via a forward hook.

Echo's weights are never touched — this is pure runtime math.
"""

from __future__ import annotations

import numpy as np
import torch

ROOT = __file__.rsplit("/", 1)[0]
INTEL_DIR = f"{ROOT}/echo_intel"


class IntelInjector:
    """Runtime concept-vector injection into Echo's residual stream."""

    def __init__(self, model, intel_dir: str = INTEL_DIR):
        self.model = model
        self.intel_dir = intel_dir
        self.concepts: dict[str, np.ndarray] = {}
        self.R: np.ndarray | None = None
        self.echo_layer: int = 0
        self.active: dict[str, float] = {}  # name -> scale
        self._handle = None
        self._translated: dict[str, torch.Tensor] = {}
        self._load()

    def _load(self):
        import os

        cpath = os.path.join(self.intel_dir, "concepts.npz")
        apath = os.path.join(self.intel_dir, "alignment.npz")
        if not (os.path.exists(cpath) and os.path.exists(apath)):
            raise FileNotFoundError(
                f"missing {cpath} or {apath} — run echo_intel_extract.py first"
            )
        cdata = np.load(cpath)
        adata = np.load(apath)
        self.R = adata["R"]
        self.echo_layer = int(adata["echo_layer"])
        for key in cdata.files:
            if key in ("qwen_layer", "echo_layer"):
                continue
            self.concepts[key] = cdata[key]
        print(
            f"IntelInjector: {len(self.concepts)} concepts, "
            f"R{self.R.shape}, echo layer {self.echo_layer}"
        )

    def _translate(self, name: str) -> torch.Tensor:
        """v_echo = R @ v_qwen, cached per concept."""
        if name not in self._translated:
            v_q = torch.from_numpy(self.concepts[name]).float()
            R = torch.from_numpy(self.R).float()
            # R maps Qwen(1536) -> Echo(1024): v_e = v_q @ R
            v_e = (v_q.unsqueeze(0) @ R).squeeze(0)
            v_e = v_e / v_e.norm().clamp_min(1e-8)
            self._translated[name] = v_e
        return self._translated[name]

    def _hook_fn(self, _module, _inputs, output):
        if not self.active:
            return output
        # Echo blocks return the residual tensor directly.
        hidden = output
        inject = None
        for name, scale in self.active.items():
            v = self._translate(name).to(hidden.device, hidden.dtype)
            inject = v if inject is None else inject + v
        if inject is None:
            return output
        # Broadcast [d_model] over [batch, seq, d_model].
        return hidden + inject * _current_scale(self.active, self._translated, self.R)

    def enable(self, concepts: dict[str, float]):
        """Activate injection: {concept_name: scale}."""
        for name in concepts:
            if name not in self.concepts:
                raise ValueError(
                    f"unknown concept {name!r}; have: {sorted(self.concepts)}"
                )
        self.active = {k: float(v) for k, v in concepts.items()}
        if self._handle is None:
            block = self.model.blocks[self.echo_layer]
            self._handle = block.register_forward_hook(self._hook_fn)
        print(f"IntelInjector: active {self.active}")

    def disable(self):
        self.active = {}
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        print("IntelInjector: disabled")

    def status(self) -> dict:
        return {
            "concepts": sorted(self.concepts.keys()),
            "active": dict(self.active),
            "echo_layer": self.echo_layer,
        }


def _current_scale(active, translated, R):
    """Mean of active scales — keeps injection magnitude stable."""
    vals = list(active.values())
    return sum(vals) / max(len(vals), 1)