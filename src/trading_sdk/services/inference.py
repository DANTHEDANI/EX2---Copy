import logging
from pathlib import Path

import numpy as np
import torch

from ..model.network import DuelingDQNNetwork
from ..shared.config import ConfigManager
from ..shared.constants import ROOT_DIR


class InferenceService:
    """Loads a trained policy and predicts greedy action."""

    def __init__(self, config: ConfigManager) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        paths = self.config.setup["paths"]
        model_path = ROOT_DIR / Path(paths["model_dir"]) / paths["model_filename"]
        self.model = DuelingDQNNetwork(action_dim=3).to(self.device)
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info("Loaded model for inference from %s", model_path)
        else:
            self.logger.warning("Model checkpoint not found at %s", model_path)
        self.model.eval()

    def predict(self, current_state: np.ndarray | None) -> int:
        if current_state is None:
            return 1
        state_t = torch.tensor(current_state, dtype=torch.float32, device=self.device)
        if state_t.ndim == 2:
            state_t = state_t.unsqueeze(0)
        with torch.no_grad():
            return int(self.model(state_t).argmax(dim=1).item())

    def predict_detailed(self, current_state: np.ndarray | None) -> dict:
        if current_state is None:
            return {"action": 1, "q_values": [0.0, 0.0, 0.0]}
        state_t = torch.tensor(current_state, dtype=torch.float32, device=self.device)
        if state_t.ndim == 2:
            state_t = state_t.unsqueeze(0)
        with torch.no_grad():
            q_vals = self.model(state_t)[0].cpu().numpy()
            return {"action": int(np.argmax(q_vals)), "q_values": [float(x) for x in q_vals]}
