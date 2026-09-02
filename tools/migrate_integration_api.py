"""Back up PromptHub's live SQLite database and apply Integration API migrations."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = PROJECT_ROOT.parent / "Plib_Backups"
DATABASE_PATH = PROJECT_ROOT / "prompts.db"
sys.path.insert(0, str(PROJECT_ROOT))

from prompt_service import apply_integration_migrations, integration_schema_ready  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not DATABASE_PATH.is_file():
        raise SystemExit(f"Database not found: {DATABASE_PATH}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_ROOT / f"Plib_PreIntegrationMigration_{timestamp}"
    backup_path.mkdir(parents=True, exist_ok=False)
    backup_database = backup_path / DATABASE_PATH.name

    source = sqlite3.connect(str(DATABASE_PATH), timeout=30)
    source.row_factory = sqlite3.Row
    try:
        destination = sqlite3.connect(str(backup_database))
        try:
            source.backup(destination)
            integrity = destination.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            destination.close()
        if integrity != "ok":
            raise SystemExit(f"Pre-migration backup integrity check failed: {integrity}")

        sidecars = []
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(DATABASE_PATH) + suffix)
            if sidecar.exists():
                copied = backup_path / sidecar.name
                shutil.copy2(sidecar, copied)
                sidecars.append({"name": copied.name, "bytes": copied.stat().st_size, "sha256": sha256(copied)})

        manifest = {
            "original_database": str(DATABASE_PATH),
            "backup_path": str(backup_path),
            "created_at": datetime.now().astimezone().isoformat(),
            "backup_method": "Python sqlite3 online backup API",
            "integrity_check": integrity,
            "database": {
                "name": backup_database.name,
                "bytes": backup_database.stat().st_size,
                "sha256": sha256(backup_database),
            },
            "sidecars": sidecars,
        }
        (backup_path / "DATABASE_BACKUP_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        source.execute("BEGIN IMMEDIATE")
        applied = apply_integration_migrations(source)
        source.commit()
        if not integration_schema_ready(source):
            raise SystemExit("Migration completed without the required schema.")
        live_integrity = source.execute("PRAGMA integrity_check").fetchone()[0]
        if live_integrity != "ok":
            raise SystemExit(f"Post-migration integrity check failed: {live_integrity}")
    except Exception:
        source.rollback()
        raise
    finally:
        source.close()

    print(f"PRE_MIGRATION_BACKUP={backup_path}")
    print(f"MIGRATIONS_APPLIED={','.join(applied) if applied else 'none_already_current'}")
    print("INTEGRITY_CHECK=ok")


if __name__ == "__main__":
    main()
