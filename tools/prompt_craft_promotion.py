"""Local, manual promotion of one archived Prompt Craft finding into PromptHub."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:5000/api/integration/v1"


def request(base, token, method, path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=body, method=method,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"PromptHub HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    if not result.get("success"):
        raise RuntimeError(str(result.get("error")))
    return result["data"]


def stage(base, token, skill, finding, directory):
    record = json.loads(Path(finding).read_text(encoding="utf-8"))
    required = ("discovery_id", "revision", "primary_source", "release", "target_reference", "approved_text")
    if any(not record.get(key) for key in required):
        raise ValueError("Finding needs: " + ", ".join(required))
    if record.get("synthetic") or any(t in {"synthetic", "test", "demo", "setup-test"} for t in record.get("tags", [])) or record["release"] == "setup-test-v1" or record["discovery_id"] == "d2fc33b564d5d72ac03f80ab":
        raise ValueError("Synthetic/test findings cannot be promoted")
    target = Path(record["target_reference"])
    if target.is_absolute() or ".." in target.parts or target.suffix.lower() != ".md" or target.parts[0] != "references":
        raise ValueError("target_reference must be a Markdown file below references/")
    if not str(record["primary_source"]).startswith("https://"):
        raise ValueError("primary_source must be HTTPS")
    health = request(base, token, "GET", "/health")
    if not health["availability"]["write"]:
        raise RuntimeError("PromptHub API write access is disabled")
    installed = request(base, token, "GET", "/skills/" + urllib.parse.quote(skill, safe=""))
    source = Path(installed["package_path"]).resolve()
    if not (source / "SKILL.md").is_file():
        raise RuntimeError("Installed skill package is unavailable on this computer")
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    shutil.copytree(source, directory, symlinks=False)
    destination = directory / target
    if destination.exists():
        shutil.rmtree(directory)
        raise FileExistsError("Reference already exists; prepare a reviewed full-package edit manually")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(f"# Archived prompt craft finding\n\nSource: {record['primary_source']}\nRelease: {record['release']}\nArchive: {record['discovery_id']} revision {record['revision']}\n\n{record['approved_text'].strip()}\n", encoding="utf-8")
    comparison = request(base, token, "POST", "/skills/" + urllib.parse.quote(skill, safe="") + "/compare", {"source": str(directory)})["comparison"]
    files = comparison["files"]
    if comparison["state"] != "NEWER" or files["removed"] or files["modified"] or files["added"] != [target.as_posix()]:
        raise RuntimeError("Unexpected comparison; review manually. Staging retained at " + str(directory))
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {"skill": skill, "source": str(directory), "reference": target.as_posix(), "sha256": digest,
                "discovery_id": record["discovery_id"], "revision": record["revision"],
                "installed_hash": installed["content_hash"]}
    (directory.parent / (directory.name + ".promotion.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def promote(base, token, manifest_path, approval):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if approval != manifest["sha256"]:
        raise ValueError("Approval must equal the staged reference SHA-256 printed by stage")
    source = Path(manifest["source"])
    if hashlib.sha256((source / manifest["reference"]).read_bytes()).hexdigest() != approval:
        raise RuntimeError("Staged reference changed since approval")
    path = "/skills/" + urllib.parse.quote(manifest["skill"], safe="")
    installed = request(base, token, "GET", path)
    if installed["content_hash"] != manifest["installed_hash"]:
        raise RuntimeError("Installed skill changed since staging; start again")
    comparison = request(base, token, "POST", path + "/compare", {"source": str(source)})["comparison"]
    files = comparison["files"]
    if comparison["state"] != "NEWER" or files["added"] != [manifest["reference"]] or files["removed"] or files["modified"]:
        raise RuntimeError("Comparison changed; no update made")
    request(base, token, "POST", path + "/update", {"source": str(source)})
    current = request(base, token, "GET", path)
    if current["content_hash"] != comparison["incoming"]["content_hash"]:
        raise RuntimeError("Update returned but read-back hash differs; leave archive pending")
    return {"skill": manifest["skill"], "reference": manifest["reference"], "content_hash": current["content_hash"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE, help="PromptHub loopback Integration API URL")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("stage")
    prepare.add_argument("skill")
    prepare.add_argument("finding", help="Locally reviewed JSON finding")
    prepare.add_argument("directory", help="New staging directory")
    publish = sub.add_parser("promote")
    publish.add_argument("manifest", help="Stage-generated .promotion.json")
    publish.add_argument("--approve-sha256", required=True)
    args = parser.parse_args(argv)
    token = os.environ.get("PROMPTHUB_API_TOKEN")
    if not token:
        parser.error("Set PROMPTHUB_API_TOKEN locally; never store it in a finding or repository")
    result = stage(args.base, token, args.skill, args.finding, args.directory) if args.command == "stage" else promote(args.base, token, args.manifest, args.approve_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Promotion stopped: {exc}", file=sys.stderr)
        sys.exit(1)
