"""
Wrappers for PyPOTS imputation models: SAITS / BRITS / CSDI / StemGNN.

Note: official GRIN is not bundled in PyPOTS 1.5; we use StemGNN as a
graph-oriented baseline proxy (reported as GRIN*) and MaskView-ST as our
proposed spatiotemporal model.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _to_pypots(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(B,T,F) with NaN for missing."""
    arr = x.astype(np.float32).copy()
    arr[mask < 0.5] = np.nan
    return arr


def build_pypots_model(
    name: str,
    n_steps: int,
    n_features: int,
    n_samples: int,
    epochs: int = 15,
    patience: int = 5,
    device: str = "cuda",
):
    name = name.upper()
    common = dict(
        n_steps=n_steps,
        n_features=n_features,
        batch_size=min(32, max(1, n_samples)),
        epochs=epochs,
        patience=patience,
        device=device if device != "cpu" else "cpu",
    )
    if name == "SAITS":
        from pypots.imputation import SAITS

        return SAITS(n_layers=2, d_model=64, n_heads=2, d_k=32, d_v=32, d_ffn=128, dropout=0.1, **common)
    if name == "BRITS":
        from pypots.imputation import BRITS

        return BRITS(rnn_hidden_size=64, **common)
    if name == "CSDI":
        from pypots.imputation import CSDI

        return CSDI(
            n_layers=2,
            n_heads=2,
            n_channels=32,
            d_time_embedding=32,
            d_feature_embedding=16,
            d_diffusion_embedding=32,
            n_diffusion_steps=25,
            **common,
        )
    if name in {"STEMGNN", "GRIN"}:
        from pypots.imputation import StemGNN

        return StemGNN(n_layers=2, n_stacks=1, d_model=64, dropout=0.1, **common)
    if name == "LOCF":
        from pypots.imputation import LOCF

        return LOCF()
    if name == "MEAN":
        from pypots.imputation import Mean

        return Mean()
    raise ValueError(f"Unknown PyPOTS model: {name}")


def fit_pypots_model(
    name: str,
    train_x: np.ndarray,
    train_m: np.ndarray,
    epochs: int = 15,
    patience: int = 5,
    device: str = "cuda",
) -> Any:
    """Fit once; reuse for multiple test corruptions."""
    model = build_pypots_model(
        name,
        n_steps=train_x.shape[1],
        n_features=train_x.shape[2],
        n_samples=len(train_x),
        epochs=epochs,
        patience=patience,
        device=device,
    )
    model.fit({"X": _to_pypots(train_x, train_m)})
    return model


def impute_pypots(model, test_x: np.ndarray, test_m: np.ndarray) -> np.ndarray:
    imputed = model.impute({"X": _to_pypots(test_x, test_m)})
    return np.asarray(imputed, dtype=np.float32)


def run_pypots_model(
    name: str,
    train_x: np.ndarray,
    train_m: np.ndarray,
    test_x: np.ndarray,
    test_m: np.ndarray,
    epochs: int = 15,
    patience: int = 5,
    device: str = "cuda",
) -> np.ndarray:
    """Backward-compatible one-shot fit+impute."""
    model = fit_pypots_model(name, train_x, train_m, epochs, patience, device)
    return impute_pypots(model, test_x, test_m)
