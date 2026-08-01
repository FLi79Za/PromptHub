"""Secure local configuration for PromptHub's Integration API."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
import secrets
from pathlib import Path
from typing import Any


CONFIG_FILENAME = "integration_api.json"
TOKEN_FILENAME = "integration_api.token"
SECRET_KEY_FILENAME = "flask_secret.key"

DEFAULT_CONFIG: dict[str, Any] = {
    "api_enabled": True,
    "read_enabled": True,
    "write_enabled": True,
    "allowed_origins": [],
    "host": "127.0.0.1",
    "port": None,
    "max_json_bytes": 1024 * 1024,
    "debug": False,
}


def config_dir() -> Path:
    override = os.environ.get("PROMPTHUB_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "PromptHub"
    return Path.home() / ".prompthub"


def _write_private_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    temporary.replace(path)


def _normalise_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if isinstance(raw, dict):
        for key in DEFAULT_CONFIG:
            if key in raw:
                config[key] = raw[key]

    for key in ("api_enabled", "read_enabled", "write_enabled", "debug"):
        config[key] = bool(config[key])

    origins = config.get("allowed_origins")
    if not isinstance(origins, list):
        origins = []
    config["allowed_origins"] = sorted(
        {str(origin).strip() for origin in origins if str(origin).strip()}
    )

    host = str(config.get("host") or "127.0.0.1").strip()
    config["host"] = host if host in {"127.0.0.1", "localhost", "::1"} else "127.0.0.1"

    port = config.get("port")
    if port in (None, "", 0, "0"):
        config["port"] = None
    else:
        try:
            parsed_port = int(port)
        except (TypeError, ValueError):
            parsed_port = 0
        config["port"] = parsed_port if 1 <= parsed_port <= 65535 else None

    try:
        max_bytes = int(config.get("max_json_bytes") or DEFAULT_CONFIG["max_json_bytes"])
    except (TypeError, ValueError):
        max_bytes = DEFAULT_CONFIG["max_json_bytes"]
    config["max_json_bytes"] = min(max(max_bytes, 16 * 1024), 4 * 1024 * 1024)
    return config


def load_config(create: bool = True) -> dict[str, Any]:
    directory = config_dir()
    path = directory / CONFIG_FILENAME
    raw: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            raw = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            raw = {}
    config = _normalise_config(raw)
    if create and (not path.exists() or raw != config):
        _write_private_text(path, json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config


def save_config(config: dict[str, Any]) -> dict[str, Any]:
    normalised = _normalise_config(config)
    _write_private_text(
        config_dir() / CONFIG_FILENAME,
        json.dumps(normalised, indent=2, sort_keys=True) + "\n",
    )
    return normalised


def get_or_create_token() -> str:
    path = config_dir() / TOKEN_FILENAME
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if len(token) >= 32:
            return token
    return regenerate_token()


def regenerate_token() -> str:
    token = secrets.token_urlsafe(48)
    _write_private_text(config_dir() / TOKEN_FILENAME, token + "\n")
    return token


def get_or_create_secret_key() -> str:
    path = config_dir() / SECRET_KEY_FILENAME
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if len(value) >= 32:
            return value
    value = secrets.token_urlsafe(64)
    _write_private_text(path, value + "\n")
    return value


def public_status() -> dict[str, Any]:
    config = load_config()
    directory = config_dir()
    return {
        **config,
        "config_directory": str(directory),
        "config_file": str(directory / CONFIG_FILENAME),
        "token_file": str(directory / TOKEN_FILENAME),
        "secret_key_file": str(directory / SECRET_KEY_FILENAME),
        "token_present": (directory / TOKEN_FILENAME).exists(),
        "secret_key_present": (directory / SECRET_KEY_FILENAME).exists(),
    }


def configure_integration_logging(debug: bool = False) -> Path:
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "integration_api.log"
    integration_logger = logging.getLogger("prompthub.integration")
    integration_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(getattr(handler, "baseFilename", "")) == log_path
        for handler in integration_logger.handlers
    ):
        handler = RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        integration_logger.addHandler(handler)
    integration_logger.propagate = False
    return log_path
