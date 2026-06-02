import json
import logging
import logging.config
from pathlib import Path
from typing import Any

from .constants import LOGGING_CONFIG_FILE, RATE_LIMITS_FILE, SETUP_FILE


class ConfigManager:
    """
    Centralized configuration manager loading settings from JSON files.
    Enforces the zero-hardcoding constraint.
    """

    def __init__(self) -> None:
        """Initialize the layout by loading JSON config files."""
        self._setup_config = self._load_json(SETUP_FILE)
        self._rate_limits = self._load_json(RATE_LIMITS_FILE)
        self._init_logging()

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Load and parse JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file {path} not found.")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _init_logging(self) -> None:
        """Initialize standard logging configuration."""
        logging_config = self._load_json(LOGGING_CONFIG_FILE)
        logging.config.dictConfig(logging_config)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Configuration Manager initialized successfully.")

    @property
    def setup(self) -> dict[str, Any]:
        """Provides environment setup configuration."""
        return self._setup_config

    @property
    def rate_limits(self) -> dict[str, Any]:
        """Provides external API rate limits."""
        return self._rate_limits
