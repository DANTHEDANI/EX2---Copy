"""
Epsilon-Greedy Decay Analysis Script

Demonstrates how epsilon decay schedule impacts:
1. Policy convergence (greedy selection)
2. Over-trading prevention (fewer trades as epsilon decays)
3. Exploration-Exploitation tradeoff

Run this to analyze epsilon scheduling and its effect on trading behavior.
"""

import json
from pathlib import Path


def analyze_epsilon_decay():
    """Analyze epsilon decay schedule from training metadata."""
    model_dir = Path("data/models")
    metadata_file = model_dir / "training_metadata.json"

    if not metadata_file.exists():
        print(
            "Training metadata not found. Run training first: python -m trading_sdk --action train"
        )
        return

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    print("=" * 70)
    print("EPSILON DECAY ANALYSIS")
    print("=" * 70)

    epsilon_start = metadata.get("epsilon_start", 1.0)
    epsilon_end = metadata.get("epsilon_end", 0.01)
    epsilon_decay = metadata.get("epsilon_decay", 200000)

    print(f"\nEpsilon Schedule:")
    print(f"  Start: {epsilon_start:.4f} (100% exploration)")
    print(f"  End:   {epsilon_end:.4f} ({epsilon_end*100:.1f}% exploration)")
    print(f"  Decay: {epsilon_decay:,} steps")

    # Calculate decay constant
    decay = (epsilon_start - epsilon_end) / max(epsilon_decay, 1)
    print(f"  Decay rate: {decay:.6f} per step")

    # Milestone analysis
    print("\n[Milestones During Training]")
    milestones = [10000, 50000, 100000, 150000, 200000]
    for step in milestones:
        epsilon = max(epsilon_end, epsilon_start - decay * step)
        exploration_pct = epsilon * 100
        print(
            f"  Step {step:>7,}: epsilon={epsilon:.4f} ({exploration_pct:>5.1f}% exploration)"
        )

    # Impact discussion
    print("\n[Impact on Trading Behavior]")
    print("""
Early Stage (epsilon ≈ 0.8-1.0):
  - Agent takes mostly random actions
  - Explores entire action space
  - High trading frequency
  - High transaction costs → lower net returns
  - Critical for discovering diverse market regimes

Middle Stage (epsilon ≈ 0.3-0.5):
  - Balanced exploration-exploitation
  - Agent begins learning profitable patterns
  - Gradually reduces over-trading
  - Lower costs, smoother returns

Late Stage (epsilon ≈ 0.01-0.1):
  - Nearly greedy policy (relies on learned Q-values)
  - Minimal exploration
  - Stable trading behavior
  - Low transaction costs
  - Convergence to learned policy
""")

    # Over-trading vs. convergence
    print("[Over-Trading vs. Convergence]")
    print("""
Problem: If epsilon stays high too long:
  - Agent keeps trading randomly despite learning good policy
  - Friction costs erode returns
  - Policy never fully exploits learned Q-values

Solution (in this project):
  - Linear decay from 1.0 → 0.01 over 200k steps
  - Encourages exploration early (market discovery)
  - Switches to exploitation late (policy refinement)
  - Balances sample efficiency with convergence speed
""")

    print("\n[Configuration Used]")
    print(f"  Target update frequency: {metadata.get('target_update_frequency')} steps")
    print(f"  Learning rate: {metadata.get('learning_rate'):.6f}")
    print(f"  Gamma (discount): {metadata.get('gamma'):.4f}")
    print(f"  Loss function: {metadata.get('loss_function')}")

    print("\n" + "=" * 70)
    print("Run: python verify_data.py")
    print("     python -m trading_sdk --action train")
    print("     python -m trading_sdk --action backtest")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    analyze_epsilon_decay()
