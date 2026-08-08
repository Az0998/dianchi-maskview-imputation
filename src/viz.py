"""Benchmark visualization helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def plot_benchmark_heatmap(df: pd.DataFrame, metric: str, out: Path):
    """df columns: model, pattern, rate, metric"""
    pattern_order = ["point", "block_time", "sensor", "station", "mixed"]
    model_order = ["Mean", "LOCF", "Linear", "SAITS", "BRITS", "GRIN*", "MaskView-ST"]
    sub = df.copy()
    sub["setting"] = sub["pattern"] + "@" + sub["rate"].map(lambda x: f"{int(round(float(x) * 100))}%")
    pivot = sub.pivot_table(index="model", columns="setting", values=metric, aggfunc="mean")
    col_order = [
        f"{p}@{r}%"
        for p in pattern_order
        for r in (10, 20, 30)
        if f"{p}@{r}%" in pivot.columns
    ]
    row_order = [m for m in model_order if m in pivot.index] + [
        m for m in pivot.index if m not in model_order
    ]
    pivot = pivot.reindex(index=row_order, columns=col_order)
    # Paper-readable: large canvas + small annotations (talk context makes digits overlap).
    with sns.axes_style("white"), sns.plotting_context("paper"):
        fig, ax = plt.subplots(figsize=(14.5, 5.2))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu_r",
            ax=ax,
            linewidths=0.4,
            linecolor="white",
            annot_kws={"size": 7.5, "color": "black"},
            cbar_kws={"shrink": 0.85, "label": metric},
        )
        ax.set_title(f"Imputation {metric} (lower better)", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_model_bars(df: pd.DataFrame, metric: str, out: Path, pattern: str = "mixed", rate: float = 0.2):
    sub = df[(df["pattern"] == pattern) & (np.isclose(df["rate"], rate))].sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=sub, x="model", y=metric, hue="model", legend=False, ax=ax, palette="viridis")
    ax.set_title(f"{metric} | pattern={pattern}, rate={rate:.0%}")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_pattern_radar_or_lines(df: pd.DataFrame, model: str, metric: str, out: Path):
    sub = df[df["model"] == model]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.lineplot(data=sub, x="rate", y=metric, hue="pattern", marker="o", ax=ax)
    ax.set_title(f"{model}: {metric} vs missing rate")
    ax.set_xlabel("Artificial missing rate")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_imputation_example(t, obs, imp, mask_eval, var_name: str, station: str, out: Path):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, obs, label="observed/true", color="#264653", lw=1.4)
    ax.plot(t, imp, label="imputed", color="#e76f51", lw=1.2, alpha=0.9)
    hid = mask_eval > 0.5
    if hid.any():
        ax.scatter(np.asarray(t)[hid], np.asarray(obs)[hid], c="#e9c46a", s=28, zorder=3, label="eval targets")
    ax.set_title(f"{station} — {var_name}")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_mask_view_schematic(out: Path):
    """Simple schematic of Mask-View pattern bank."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    rng = np.random.default_rng(0)
    titles = ["point", "block_time", "sensor", "station"]
    for ax, title in zip(axes, titles):
        m = np.ones((16, 10))
        if title == "point":
            idx = rng.choice(160, 40, replace=False)
            m.ravel()[idx] = 0
        elif title == "block_time":
            m[5:11, 2:7] = 0
        elif title == "sensor":
            m[:, 7:9] = 0
        else:
            m[:, 3:5] = 0
        ax.imshow(m.T, aspect="auto", cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("node×var")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Mask-View pattern bank", y=1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
