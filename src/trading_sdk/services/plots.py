import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from ..shared.constants import ROOT_DIR

RESULTS_DIR = ROOT_DIR / "data" / "results"

def plot_learning_curve(
    losses: List[float],
    rewards: List[float],
    results_dir: str | Path | None = None,
) -> None:
    output_dir = _ensure_results_dir(results_dir)
    path = output_dir / "learning_curve.png"
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = "tab:red"
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(losses, color=color, alpha=0.6)
    ax1.tick_params(axis="y", labelcolor=color)
    
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Reward (Moving Avg)", color=color)
    ax2.plot(rewards, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    
    fig.tight_layout()
    plt.title("DQN Learning Curve")
    plt.savefig(path)
    plt.close()

def plot_backtest_results(results: Dict, results_dir: str | Path | None = None) -> None:
    output_dir = _ensure_results_dir(results_dir)
    path = output_dir / "backtest_results.png"
    equity = results["agent_equity"]
    bnh = results["bnh_equity"]
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label="DQN Agent Equity", color="blue")
    plt.plot(bnh, label="Buy and Hold Benchmark", color="orange", alpha=0.7)
    
    plt.title("Backtest: Agent vs Buy and Hold")
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _ensure_results_dir(results_dir: str | Path | None = None) -> Path:
    target = RESULTS_DIR if results_dir is None else ROOT_DIR / Path(results_dir)
    os.makedirs(target, exist_ok=True)
    return target
