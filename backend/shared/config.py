# backend/shared/config.py
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv, find_dotenv

# Automatically find and load the .env file globally.
load_dotenv(find_dotenv())


def get_backend_root() -> Path:
    """Find the backend root from the loaded .env path when available."""
    dotenv_path = find_dotenv()
    if dotenv_path:
        return Path(dotenv_path).resolve().parent
    return Path(__file__).resolve().parents[1]


def _env_key(agent_name: str, suffix: str) -> str:
    if agent_name.upper().endswith("_AGENT") and suffix == "AGENT_ID":
        return f"BAND_{agent_name.upper()}_ID"
    return f"BAND_{agent_name.upper()}_{suffix}"


def load_agent_credentials(agent_name: str) -> tuple[str, str]:
    """
    Load Band credentials from environment variables first, then fall back to
    backend/agent_config.yaml for backward compatibility.
    """
    agent_id = os.getenv(_env_key(agent_name, "AGENT_ID"), "").strip()
    api_key = os.getenv(_env_key(agent_name, "API_KEY"), "").strip()
    if agent_id and api_key:
        return agent_id, api_key

    root = get_backend_root()
    config_path = root / "agent_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing credentials for {agent_name}. Set "
            f"{_env_key(agent_name, 'AGENT_ID')} and {_env_key(agent_name, 'API_KEY')}, "
            f"or provide {config_path}."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if agent_name not in config:
        raise KeyError(f"Agent '{agent_name}' configuration block not found.")

    return config[agent_name]["agent_id"], config[agent_name]["api_key"]
