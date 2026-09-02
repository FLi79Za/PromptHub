"""View and manage PromptHub Integration API local configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from integration_config import (  # noqa: E402
    get_or_create_token,
    load_config,
    public_status,
    regenerate_token,
    save_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage PromptHub Integration API configuration.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show non-secret configuration status.")
    subparsers.add_parser("show-token", help="Explicitly display the current bearer token.")
    subparsers.add_parser("regenerate-token", help="Revoke the old token and display its replacement once.")

    set_parser = subparsers.add_parser("set", help="Update a non-secret setting.")
    set_parser.add_argument("name", choices=(
        "api_enabled", "read_enabled", "write_enabled", "host", "port",
        "allowed_origins", "max_json_bytes", "debug",
    ))
    set_parser.add_argument("value")
    args = parser.parse_args()

    if args.command == "status":
        get_or_create_token()
        print(json.dumps(public_status(), indent=2, sort_keys=True))
        return
    if args.command == "show-token":
        print(get_or_create_token())
        return
    if args.command == "regenerate-token":
        print("The previous Integration API token has been revoked.")
        print(regenerate_token())
        return

    config = load_config()
    if args.name in {"api_enabled", "read_enabled", "write_enabled", "debug"}:
        lowered = args.value.strip().lower()
        if lowered not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            raise SystemExit("Boolean values must be true or false.")
        value = lowered in {"true", "1", "yes", "on"}
    elif args.name in {"port", "max_json_bytes"}:
        value = None if args.name == "port" and args.value.strip().lower() in {"none", "auto", "0"} else int(args.value)
    elif args.name == "allowed_origins":
        value = [item.strip() for item in args.value.split(",") if item.strip()]
    else:
        value = args.value.strip()
    config[args.name] = value
    saved = save_config(config)
    print(json.dumps({args.name: saved[args.name]}, indent=2))
    print("Restart PromptHub for host, port, CORS, or debug logging changes to take full effect.")


if __name__ == "__main__":
    main()
