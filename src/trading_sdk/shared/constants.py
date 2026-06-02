from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data" / "raw"

# Config Files
SETUP_FILE = CONFIG_DIR / "setup.json"
RATE_LIMITS_FILE = CONFIG_DIR / "rate_limits.json"
LOGGING_CONFIG_FILE = CONFIG_DIR / "logging_config.json"

# Trading Constants
STATE_DIM = 10  # Selected features per time step
ACTIONS = ["SELL", "HOLD", "BUY"]
ACTION_MAP = dict(enumerate(ACTIONS))

# Model Checkpoints
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
