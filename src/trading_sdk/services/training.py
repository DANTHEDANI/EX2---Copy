import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..env.trading_env import TradingEnv
from ..memory.replay_buffer import ReplayBuffer
from ..model.network import DuelingDQNNetwork
from ..shared.config import ConfigManager
from ..shared.constants import ROOT_DIR


class TrainingService:
    """Train Dueling DQN with Double DQN, replay memory, target network."""

    def __init__(self, config: ConfigManager) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.hyper = self.config.setup["hyperparameters"]
        self.env_cfg = self.config.setup["environment"]
        self.paths_cfg = self.config.setup["paths"]
        self.memory = ReplayBuffer(self.hyper["memory_size"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = self.hyper["epsilon_start"]
        self.best_val_reward = -float("inf")
        self.epsilon_history = []

    def train(self, states: np.ndarray, prices: np.ndarray) -> dict[str, list[float]]:
        env = TradingEnv(
            states_3d=states,
            prices=prices,
            initial_balance=self.env_cfg["initial_balance"],
            commission_fee=self.env_cfg["commission_fee"],
            slippage_fee=self.env_cfg["slippage_fee"],
            sharpe_lambda=self.env_cfg["sharpe_lambda"],
            invalid_action_penalty=self.env_cfg["invalid_action_penalty"],
        )
        policy = DuelingDQNNetwork(action_dim=env.action_space.n).to(self.device)
        target = DuelingDQNNetwork(action_dim=env.action_space.n).to(self.device)
        target.load_state_dict(policy.state_dict())
        optimizer = optim.Adam(policy.parameters(), lr=self.hyper["learning_rate"])
        loss_fn = nn.SmoothL1Loss() if self.hyper["loss"] == "huber" else nn.MSELoss()
        rewards, losses, steps = [], [], 0
        for ep in range(self.hyper["episodes"]):
            state, _ = env.reset()
            ep_reward = 0.0
            while True:
                if random.random() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        action = int(policy(state_t).argmax(dim=1).item())
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward
                steps += 1
                self._decay_epsilon()
                if len(self.memory) >= max(
                    self.hyper["batch_size"],
                    self.hyper["warmup_steps"],
                ):
                    loss = self._optimize(policy, target, optimizer, loss_fn)
                    losses.append(loss)
                    if steps % self.hyper["target_update_interval"] == 0:
                        self._soft_update(policy, target)
                if done:
                    break
            rewards.append(ep_reward)
            self.epsilon_history.append(self.epsilon)
            if ep_reward > self.best_val_reward:
                self.best_val_reward = ep_reward
                self._save_model(policy, is_best=True)
        self._save_model(policy)
        self.logger.info(
            "Training complete. episodes=%s epsilon=%.4f best_reward=%.2f",
            len(rewards),
            self.epsilon,
            self.best_val_reward,
        )
        return {"losses": losses, "rewards": rewards, "epsilon_history": self.epsilon_history}

    def _optimize(
        self,
        policy: nn.Module,
        target: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
    ) -> float:
        """Bellman target with Double DQN: Q(s,a) = r + γ·Q_target(s',a*)
        where a* = argmax_a' Q_policy(s',a')"""
        states, actions, rewards, next_states, dones = self.memory.sample(self.hyper["batch_size"])
        s = torch.tensor(states, dtype=torch.float32, device=self.device)
        a = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        ns = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device)
        q_sa = policy(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            next_actions = policy(ns).argmax(dim=1, keepdim=True)
            next_q = target(ns).gather(1, next_actions).squeeze(1)
            target_q = r + (1 - d) * self.hyper["gamma"] * next_q
        loss = loss_fn(q_sa, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def _decay_epsilon(self) -> None:
        decay = (
            self.hyper["epsilon_start"] - self.hyper["epsilon_end"]
        ) / max(self.hyper["epsilon_decay"], 1)
        self.epsilon = max(self.hyper["epsilon_end"], self.epsilon - decay)

    def _soft_update(self, policy: nn.Module, target: nn.Module) -> None:
        tau = self.hyper["tau"]
        for t_param, p_param in zip(
            target.parameters(),
            policy.parameters(),
            strict=False,
        ):
            t_param.data.copy_(tau * p_param.data + (1 - tau) * t_param.data)

    def _save_model(self, policy: nn.Module, is_best: bool = False) -> None:
        model_dir = ROOT_DIR / Path(self.paths_cfg["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        filename = "dueling_dqn_best.pt" if is_best else self.paths_cfg["model_filename"]
        torch.save(policy.state_dict(), model_dir / filename)
        if not is_best:
            metadata = {
                "num_episodes": len(self.epsilon_history),
                "target_update_freq": self.hyper["target_update_interval"],
                "epsilon_decay": self.hyper["epsilon_decay"],
                "lr": self.hyper["learning_rate"],
                "gamma": self.hyper["gamma"],
                "loss": self.hyper["loss"],
                "best_reward": self.best_val_reward,
                "final_epsilon": float(self.epsilon),
            }
            with open(model_dir / "training_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
