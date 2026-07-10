"""
Central config loader. Everything downstream reads thresholds/paths
from here instead of hardcoding them, so behavior is tunable via
config.yaml alone.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


@dataclass
class ArgosConfig:
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ArgosConfig":
        path = path or DEFAULT_CONFIG_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw=raw)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a nested value using dot notation, e.g. 'thresholds.flag_score'."""
        node = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
