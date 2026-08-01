"""Create a consistent SQLite backup and verify it."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source), timeout=30) as source_db:
        with sqlite3.connect(str(destination)) as backup_db:
            source_db.backup(backup_db)
            result = backup_db.execute("PRAGMA integrity_check").fetchone()[0]
    if result != "ok":
        raise SystemExit(f"Backup integrity check failed: {result}")
    print("integrity_check=ok")


if __name__ == "__main__":
    main()
