#!/usr/bin/env python3
"""Validate and append prompt-craft knowledge updates without duplicates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

MODES = {"t2i", "i2i", "edit", "inpaint", "composite", "text", "reference"}
HARDWARE = {"official", "community-tested", "unknown", "not-applicable"}
SOURCE_TYPES = {"official-docs", "official-repository", "official-model-card", "community-evidence", "user-supplied-contract"}
CONFIDENCE = {"high", "medium", "provisional"}
REQUIRED = {"checked_at", "model", "mode", "development", "prompting_impact", "hardware_impact_16gb", "source_type", "source_url", "skill_target", "confidence"}


def fail(message: str) -> None:
    raise ValueError(message)


def validate(entry: object, index: int) -> dict:
    if not isinstance(entry, dict):
        fail(f"entry {index}: expected an object")
    keys = set(entry)
    missing, extra = REQUIRED - keys, keys - REQUIRED
    if missing:
        fail(f"entry {index}: missing {', '.join(sorted(missing))}")
    if extra:
        fail(f"entry {index}: unsupported fields {', '.join(sorted(extra))}")
    for key in ("checked_at", "model", "development", "prompting_impact", "source_url", "skill_target"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            fail(f"entry {index}: {key} must be a non-empty string")
    if not isinstance(entry["mode"], list) or not entry["mode"] or any(mode not in MODES for mode in entry["mode"]):
        fail(f"entry {index}: mode must be a non-empty array of supported modes")
    if entry["hardware_impact_16gb"] not in HARDWARE:
        fail(f"entry {index}: unsupported hardware_impact_16gb")
    if entry["source_type"] not in SOURCE_TYPES:
        fail(f"entry {index}: unsupported source_type")
    if entry["confidence"] not in CONFIDENCE:
        fail(f"entry {index}: unsupported confidence")
    parsed = urlparse(entry["source_url"])
    if parsed.scheme != "https" or not parsed.netloc:
        fail(f"entry {index}: source_url must be an absolute HTTPS URL")
    normalised = dict(entry)
    normalised["model"] = entry["model"].strip()
    normalised["mode"] = sorted(set(entry["mode"]))
    for key in ("checked_at", "development", "prompting_impact", "source_url", "skill_target"):
        normalised[key] = entry[key].strip()
    return normalised


def fingerprint(entry: dict) -> str:
    basis = "\n".join((entry["model"].casefold(), entry["source_url"], entry["development"].casefold()))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def load_fingerprints(path: Path) -> set[str]:
    fingerprints: set[str] = set()
    if not path.exists():
        return fingerprints
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"existing knowledge line {line_number} is invalid JSON: {exc}")
        fingerprints.add(entry.get("id") or fingerprint(entry))
    return fingerprints


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSON object or array of update objects")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--database", type=Path, default=Path(__file__).resolve().parent.parent / "references" / "knowledge-updates.jsonl")
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        raw_entries = payload if isinstance(payload, list) else [payload]
        entries = [validate(item, index) for index, item in enumerate(raw_entries, start=1)]
        known = load_fingerprints(args.database)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 2
    imported, skipped = [], 0
    for entry in entries:
        entry_id = fingerprint(entry)
        if entry_id in known:
            skipped += 1
            continue
        imported.append({"id": entry_id, **entry})
        known.add(entry_id)
    if not args.dry_run and imported:
        args.database.parent.mkdir(parents=True, exist_ok=True)
        with args.database.open("a", encoding="utf-8", newline="\n") as handle:
            for entry in imported:
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    action = "would_import" if args.dry_run else "imported"
    print(json.dumps({action: len(imported), "duplicates_skipped": skipped}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
