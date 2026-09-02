"""Portable Agent Skill management and execution for PromptHub.

The source package is kept on disk; SQLite stores catalogue/provenance only.
Imported scripts are never executed by this module.
"""
from __future__ import annotations

import hashlib
import difflib
import json
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from capability_providers import CapabilityRegistry, ProviderError

SKILL_SCHEMA_VERSION = 1
MAX_FILES = 2000
MAX_FILE_BYTES = 10 * 1024 * 1024
CAPABILITIES = ("image_generation", "web_search", "file_read", "file_write", "code_execution", "memory", "structured_output")
SKILL_OPERATIONS = ("create", "transform", "convert", "refine", "diagnose")
SUPPORTED_INPUT_TYPES = ("text",)


class SkillError(Exception):
    def __init__(self, message: str, code: str = "SKILL_ERROR", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code, self.details = code, details or {}


def now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="microseconds")


def _safe_rel(name: str) -> str:
    p = Path(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
        raise SkillError(f"Unsafe Skill path: {name}", "UNSAFE_PATH")
    return "/".join(part for part in p.parts if part not in ("", "."))


def _frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) != 3:
        return {}, text
    meta: dict[str, Any] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"\'')
        if value.startswith("[") and value.endswith("]"):
            value = [x.strip().strip('"\'') for x in value[1:-1].split(",") if x.strip()]
        meta[key.strip()] = value
    return meta, parts[2].lstrip()


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".prompthub" not in path.parts:
            rel = path.relative_to(root).as_posix()
            digest.update(rel.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _copy_checked(source: Path, destination: Path) -> list[str]:
    if not source.is_dir():
        raise SkillError("Skill source directory was not found.", "INVALID_SOURCE")
    files = [p for p in source.rglob("*") if p.is_file()]
    if len(files) > MAX_FILES:
        raise SkillError("Skill contains too many files.", "PACKAGE_TOO_LARGE")
    rels: list[str] = []
    for path in files:
        rel = _safe_rel(path.relative_to(source).as_posix())
        if path.stat().st_size > MAX_FILE_BYTES:
            raise SkillError(f"Skill file is too large: {rel}", "PACKAGE_TOO_LARGE")
        target = destination / Path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        rels.append(rel)
    return sorted(rels)


def _extract_zip(source: Path) -> Path:
    temp = Path(tempfile.mkdtemp(prefix="prompthub-skill-"))
    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES:
                raise SkillError("Skill archive contains too many files.", "PACKAGE_TOO_LARGE")
            for info in infos:
                rel = _safe_rel(info.filename)
                if not rel or info.is_dir():
                    continue
                if info.file_size > MAX_FILE_BYTES:
                    raise SkillError(f"Skill file is too large: {rel}", "PACKAGE_TOO_LARGE")
                target = temp / Path(rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
        return temp
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def inspect_skill(source: str | Path) -> dict[str, Any]:
    path = Path(source)
    temp: Path | None = None
    root = path
    if path.is_file() and path.suffix.lower() == ".zip":
        temp = _extract_zip(path)
        root = temp
    try:
        skill_files = list(root.rglob("SKILL.md"))
        if not skill_files:
            raise SkillError("Skill package must contain SKILL.md.", "INVALID_SKILL")
        if len(skill_files) != 1:
            raise SkillError("Skill package must contain exactly one SKILL.md.", "INVALID_SKILL")
        main = skill_files[0]
        metadata, body = _frontmatter(main.read_text(encoding="utf-8"))
        name = str(metadata.get("name") or root.name).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,99}", name):
            raise SkillError("Skill name is missing or invalid.", "INVALID_SKILL")
        files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        resources = [f for f in files if f != "SKILL.md" and not f.startswith(".prompthub/")]
        capabilities = sorted({cap for cap in CAPABILITIES if re.search(rf"\b{re.escape(cap.replace('_', ' '))}\b|{re.escape(cap)}", body, re.I)})
        return {"name": name, "display_name": str(metadata.get("display_name") or name),
                "description": str(metadata.get("description") or "").strip(),
                "version": str(metadata.get("version") or "").strip() or None,
                "metadata": metadata, "files": files, "resources": resources,
                "capabilities": capabilities, "content_hash": _hash_tree(root),
                "skill_markdown": body, "source_root": str(root)}
    finally:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)


def apply_skill_migrations(conn) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS skills (
        id TEXT PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE, display_name TEXT NOT NULL,
        description TEXT, version TEXT, source_type TEXT NOT NULL, source_location TEXT,
        source_platform TEXT, package_path TEXT NOT NULL, tags_json TEXT NOT NULL DEFAULT '[]',
        dependencies_json TEXT NOT NULL DEFAULT '[]', capabilities_json TEXT NOT NULL DEFAULT '[]',
        targets_json TEXT NOT NULL DEFAULT '[]', trust_state TEXT NOT NULL DEFAULT 'unreviewed',
        content_hash TEXT NOT NULL, provenance_json TEXT NOT NULL DEFAULT '{}',
        locally_modified INTEGER NOT NULL DEFAULT 0, compatibility_state TEXT NOT NULL DEFAULT 'unknown',
        runtime_config_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name COLLATE NOCASE)")
    existing = {row[1] for row in conn.execute("PRAGMA table_info(skills)")}
    for definition in ("runtime_status TEXT NOT NULL DEFAULT 'INSTALLED'", "capability_config_json TEXT NOT NULL DEFAULT '{}'", "package_baseline_hash TEXT"):
        if definition.split()[0] not in existing:
            conn.execute(f"ALTER TABLE skills ADD COLUMN {definition}")
    conn.execute("""CREATE TABLE IF NOT EXISTS skill_execution_traces (
        id TEXT PRIMARY KEY, skill_id TEXT NOT NULL, model TEXT NOT NULL, request TEXT NOT NULL,
        parameters_json TEXT NOT NULL, resources_json TEXT NOT NULL, capabilities_json TEXT NOT NULL,
        result_json TEXT, status TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL,
        FOREIGN KEY(skill_id) REFERENCES skills(id) ON DELETE CASCADE
    )""")
    trace_columns = {row[1] for row in conn.execute("PRAGMA table_info(skill_execution_traces)")}
    for definition in ("parent_execution_id TEXT", "root_execution_id TEXT", "duration_ms INTEGER", "schema_path TEXT", "provider_results_json TEXT NOT NULL DEFAULT '[]'"):
        if definition.split()[0] not in trace_columns:
            conn.execute(f"ALTER TABLE skill_execution_traces ADD COLUMN {definition}")
    trace_columns = {row[1] for row in conn.execute("PRAGMA table_info(skill_execution_traces)")}
    for definition in ("operation TEXT NOT NULL DEFAULT 'create'", "inputs_json TEXT NOT NULL DEFAULT '[]'", "target TEXT", "source_prompt_id INTEGER"):
        if definition.split()[0] not in trace_columns:
            conn.execute(f"ALTER TABLE skill_execution_traces ADD COLUMN {definition}")
    conn.execute("""CREATE TABLE IF NOT EXISTS prompt_skill_derivations (
        id TEXT PRIMARY KEY, source_prompt_id INTEGER NOT NULL, derived_prompt_id INTEGER,
        execution_id TEXT NOT NULL, skill_id TEXT NOT NULL, skill_version TEXT,
        operation TEXT NOT NULL, target TEXT, execution_model TEXT NOT NULL,
        action TEXT NOT NULL DEFAULT 'created', metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
        FOREIGN KEY(source_prompt_id) REFERENCES prompts(id) ON DELETE CASCADE,
        FOREIGN KEY(derived_prompt_id) REFERENCES prompts(id) ON DELETE SET NULL,
        FOREIGN KEY(execution_id) REFERENCES skill_execution_traces(id) ON DELETE CASCADE,
        FOREIGN KEY(skill_id) REFERENCES skills(id) ON DELETE RESTRICT
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prompt_skill_derivations_source ON prompt_skill_derivations(source_prompt_id, created_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prompt_skill_derivations_derived ON prompt_skill_derivations(derived_prompt_id)")
    conn.execute("""CREATE TABLE IF NOT EXISTS skill_capability_providers (
        id TEXT PRIMARY KEY, provider_id TEXT NOT NULL, capability TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 0,
        config_json TEXT NOT NULL DEFAULT '{}', trust_requirement TEXT NOT NULL DEFAULT 'trusted', created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        UNIQUE(provider_id, capability)
    )""")


def _library_root(base: Path) -> Path:
    root = base / "skill_packages"
    root.mkdir(parents=True, exist_ok=True)
    return root


def import_skill(conn, source: str | Path, *, base_dir: str | Path, source_type: str = "portable", source_platform: str = "unknown", tags: Iterable[str] = ()) -> dict[str, Any]:
    source_path = Path(source)
    info = inspect_skill(source_path)
    existing = conn.execute("SELECT * FROM skills WHERE name=? COLLATE NOCASE", (info["name"],)).fetchone()
    if existing and existing["content_hash"] == info["content_hash"]:
        return {"operation": "identical", "skill": dict(existing), "inspection": info}
    if existing and existing["locally_modified"]:
        return {"operation": "conflict", "skill": dict(existing), "inspection": info, "reason": "local_modifications"}
    root = _library_root(Path(base_dir)) / info["name"]
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    temp: Path | None = None
    copy_source = source_path
    if source_path.is_file():
        temp = _extract_zip(source_path)
        copy_source = temp
    try:
        _copy_checked(copy_source, root)
    finally:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
    t = now()
    provenance = {"source_type": source_type, "source_platform": source_platform, "source_location": str(source_path), "imported_at": t, "content_hash": info["content_hash"]}
    dependencies = _dependencies_from_metadata(info["metadata"], info["skill_markdown"])
    values = (str(uuid.uuid4()).upper(), info["name"], info["display_name"], info["description"], info["version"], source_type, str(source_path), source_platform, str(root), json.dumps(list(tags)), json.dumps(dependencies), json.dumps(info["capabilities"]), "[]", "unreviewed", info["content_hash"], json.dumps(provenance), 0, "unknown", "{}", t, t)
    if existing:
        skill_id = existing["id"]
        conn.execute("""UPDATE skills SET display_name=?,description=?,version=?,source_type=?,source_location=?,source_platform=?,package_path=?,tags_json=?,dependencies_json=?,capabilities_json=?,content_hash=?,provenance_json=?,locally_modified=0,package_baseline_hash=?,updated_at=? WHERE id=?""", (values[2], values[3], values[4], values[5], values[6], values[7], values[8], values[9], values[10], values[11], values[14], values[15], values[14], t, skill_id))
        operation = "updated"
    else:
        skill_id = values[0]
        conn.execute("""INSERT INTO skills(id,name,display_name,description,version,source_type,source_location,source_platform,package_path,tags_json,dependencies_json,capabilities_json,targets_json,trust_state,content_hash,provenance_json,locally_modified,compatibility_state,runtime_config_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)
        conn.execute("UPDATE skills SET package_baseline_hash=? WHERE id=?", (info["content_hash"], skill_id))
        operation = "imported"
    return {"operation": operation, "skill_id": skill_id, "inspection": info, "package_path": str(root)}


def list_skills(conn, query: str = "") -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM skills WHERE (?='' OR name LIKE ? OR display_name LIKE ? OR description LIKE ?) ORDER BY display_name COLLATE NOCASE", (query, f"%{query}%", f"%{query}%", f"%{query}%")).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        for key in ("tags_json", "dependencies_json", "capabilities_json", "targets_json", "provenance_json", "runtime_config_json"):
            item[key[:-5] if key.endswith("_json") else key] = json.loads(item.pop(key) or ("{}" if key.endswith("config_json") or key == "provenance_json" else "[]"))
        try:
            item["runtime"] = skill_runtime_status(conn, item["id"])
        except SkillError as exc:
            item["runtime"] = {"status": "BLOCKED", "reason": exc.code}
        item["operations"] = skill_supported_operations(item)
        item["targets"] = discover_skill_targets(item)
        latest = conn.execute("SELECT id,model,resources_json,status,created_at FROM skill_execution_traces WHERE skill_id=? ORDER BY created_at DESC LIMIT 1", (item["id"],)).fetchone()
        item["recent_execution"] = ({**dict(latest), "resources": json.loads(latest["resources_json"] or "[]")} if latest else None)
        result.append(item)
    return result


def resolve_skill_dependencies(conn, skill_id: str) -> list[dict[str, Any]]:
    """Return dependency records in execution order, detecting cycles."""
    seen: set[str] = set()
    active: set[str] = set()
    resolved: list[dict[str, Any]] = []
    def visit(identifier: str) -> None:
        row = conn.execute("SELECT * FROM skills WHERE id=? OR name=? COLLATE NOCASE", (identifier, identifier)).fetchone()
        if not row:
            raise SkillError(f"Missing Skill dependency: {identifier}", "MISSING_DEPENDENCY")
        key = row["id"]
        if key in active:
            raise SkillError(f"Circular Skill dependency detected at {row['name']}", "CIRCULAR_DEPENDENCY")
        if key in seen:
            return
        active.add(key)
        for dependency in json.loads(row["dependencies_json"] or "[]"):
            visit(str(dependency))
        active.remove(key)
        seen.add(key)
        resolved.append(dict(row))
    visit(skill_id)
    return resolved


def _route_resources(root: Path, body: str, request: str, parameters: dict[str, Any]) -> list[str]:
    target = " ".join(str(v) for v in parameters.values()).lower()
    query = f"{target} {request}".lower()
    files = [p for p in root.joinpath("references").rglob("*") if p.is_file()] if root.joinpath("references").exists() else []
    selected: list[str] = []
    for path in sorted(files):
        stem = path.stem.lower().replace("_", "-")
        tokens = set(re.findall(r"[a-z0-9]+", stem))
        query_tokens = {token for token in re.findall(r"[a-z0-9]+", query) if not token.isdigit()}
        semantic = (("action", "choreography") if "fight" in query_tokens or "action" in query_tokens else ())
        lyric_context = bool(tokens & {"lyrics", "lyric", "structuring"} and query_tokens & {"lyrics", "lyric", "verse", "chorus", "bridge", "hook", "cadence", "singable", "song"})
        if tokens & query_tokens or set(semantic) & tokens or lyric_context:
            selected.append(path.relative_to(root).as_posix())
    # The main skill can declare an explicit model/resource mapping in headings.
    for match in re.finditer(r"(?:resource|reference)\s*[:=]\s*([^\s]+)", body, re.I):
        candidate = match.group(1).strip("`[]()")
        if (root / candidate).is_file() and candidate not in selected:
            selected.append(candidate)
    return selected


def _read_resources(root: Path, names: list[str]) -> list[dict[str, str]]:
    result = []
    for name in names:
        rel = _safe_rel(name)
        path = root / rel
        if path.is_file() and path.stat().st_size <= MAX_FILE_BYTES:
            result.append({"path": rel, "content": path.read_text(encoding="utf-8", errors="replace")})
    return result


def _humanise_identifier(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").replace("-", " ")).strip().title()


def skill_supported_operations(skill: dict[str, Any]) -> list[str]:
    """Return PromptHub task semantics supported by a Skill without changing its package."""
    adapter = skill.get("runtime_config")
    if adapter is None:
        adapter = json.loads(skill.get("runtime_config_json") or "{}")
    configured = adapter.get("supported_operations") or []
    if configured:
        return [str(value).lower() for value in configured if str(value).lower() in SKILL_OPERATIONS]
    text = f"{skill.get('name', '')} {skill.get('description', '')}".lower()
    operations = ["create"]
    vocabulary = {
        "transform": ("transform", "rewrite", "apply"),
        "convert": ("convert", "conversion"),
        "refine": ("refine", "improve", "rewrite"),
        "diagnose": ("diagnose", "analyse", "analyze", "repair", "fix"),
    }
    for operation, terms in vocabulary.items():
        if any(term in text for term in terms):
            operations.append(operation)
    return operations


def discover_skill_targets(skill: dict[str, Any]) -> list[dict[str, str]]:
    """Discover target choices from local adapter/catalogue metadata, then package profiles."""
    configured = skill.get("targets")
    if configured is None:
        configured = json.loads(skill.get("targets_json") or "[]")
    if configured:
        return [{"id": str(item.get("id")), "label": str(item.get("label") or _humanise_identifier(str(item.get("id")))), "resource": str(item.get("resource") or "")} for item in configured if isinstance(item, dict) and item.get("id")]
    adapter = skill.get("runtime_config")
    if adapter is None:
        adapter = json.loads(skill.get("runtime_config_json") or "{}")
    configured = adapter.get("targets") or []
    if configured:
        return [{"id": str(item.get("id")), "label": str(item.get("label") or _humanise_identifier(str(item.get("id")))), "resource": str(item.get("resource") or "")} for item in configured if isinstance(item, dict) and item.get("id")]
    root = Path(str(skill.get("package_path") or ""))
    references = root / "references"
    if not references.is_dir():
        return []
    generic = {"core", "common", "source", "sources", "local", "model-profile-template", "portable-llm-prompts", "import-policy", "knowledge-updates", "update-schema"}
    targets: list[dict[str, str]] = []
    for path in sorted(references.glob("*.md")):
        stem = path.stem.lower()
        if stem in generic or any(stem.startswith(prefix + "-") for prefix in ("core", "common", "source", "local", "import", "knowledge", "portable")):
            continue
        first_heading = next((line[2:].strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("# ")), "")
        targets.append({"id": stem.replace("-", "_"), "label": first_heading or _humanise_identifier(stem), "resource": path.relative_to(root).as_posix()})
    return targets


def normalise_skill_execution(*, operation: str = "create", inputs: list[dict[str, Any]] | None = None, request: str = "", target: str | None = None, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    operation = str(operation or "create").strip().lower()
    if operation not in SKILL_OPERATIONS:
        raise SkillError("Unsupported Skill operation.", "UNSUPPORTED_SKILL_OPERATION", {"operation": operation, "allowed": list(SKILL_OPERATIONS)})
    values = list(inputs or [])
    if not values and str(request or "").strip():
        values = [{"type": "text", "role": "brief" if operation == "create" else "source", "content": str(request).strip()}]
    normalised: list[dict[str, str]] = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise SkillError("Each Skill input must be an object.", "INVALID_SKILL_INPUT", {"index": index})
        input_type = str(item.get("type") or "text").strip().lower()
        if input_type not in SUPPORTED_INPUT_TYPES:
            raise SkillError("This Skill input type is not supported yet.", "UNSUPPORTED_SKILL_INPUT_TYPE", {"index": index, "type": input_type, "supported": list(SUPPORTED_INPUT_TYPES)})
        role = str(item.get("role") or ("brief" if operation == "create" else "source")).strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            raise SkillError("Text Skill inputs require content.", "INVALID_SKILL_INPUT", {"index": index})
        normalised.append({"type": input_type, "role": role, "content": content})
    required_role = "brief" if operation == "create" else "source"
    # Editor executions are intentionally transient.  ``draft`` is a first-class
    # source input and must not be converted into a database prompt merely to
    # satisfy the historical saved-prompt contract.
    valid_source_roles = {required_role} if operation == "create" else {"source", "draft"}
    if not any(item["role"] in valid_source_roles for item in normalised):
        raise SkillError(f"{operation.upper()} requires a {required_role} text input.", "MISSING_SKILL_INPUT", {"required_role": required_role})
    merged_parameters = dict(parameters or {})
    target_value = str(target or merged_parameters.get("target_model") or merged_parameters.get("target") or "").strip() or None
    if target_value:
        merged_parameters["target_model"] = target_value
    return {"operation": operation, "inputs": normalised, "target": target_value, "parameters": merged_parameters}


def _execution_request_text(operation: str, inputs: list[dict[str, str]], target: str | None) -> str:
    semantics = {
        "create": "Create new content from the brief.",
        "transform": "Transform the source while preserving its core intent.",
        "convert": "Convert the source for the selected target while preserving its core intent.",
        "refine": "Improve the source while preserving its current target and intent.",
        "diagnose": "Analyse the source, identify material problems, and provide the requested diagnosis or corrected result.",
    }
    sections = [f"OPERATION: {operation.upper()}", f"TASK SEMANTICS: {semantics[operation]}"]
    if target:
        sections.append(f"TARGET: {target}")
    for item in inputs:
        sections.append(f"INPUT ({item['role']}, {item['type']}):\n{item['content']}")
    return "\n\n".join(sections)


def _capability_report(skill: dict[str, Any], registry: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    registry = registry or {}
    result = []
    for capability in json.loads(skill.get("capabilities_json") or "[]"):
        entry = registry.get(capability, {})
        result.append({"capability": capability, "status": entry.get("status", "UNAVAILABLE"), "provider": entry.get("provider")})
    return result


def export_skill(conn, skill_id: str, destination: str | Path, *, include_local_metadata: bool = False) -> str:
    row = conn.execute("SELECT * FROM skills WHERE id=? OR name=? COLLATE NOCASE", (skill_id, skill_id)).fetchone()
    if not row: raise SkillError("Skill was not found.", "SKILL_NOT_FOUND")
    root, dest = Path(row["package_path"]), Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob("*"):
            if path.is_file() and (include_local_metadata or ".prompthub" not in path.parts):
                archive.write(path, path.relative_to(root).as_posix())
    return str(dest)


# Phase 2 additions.  Source packages remain canonical; all local orchestration
# and provider policy is stored in PromptHub's catalogue/configuration tables.
def _dependencies_from_metadata(metadata: dict[str, Any], body: str) -> list[dict[str, Any]]:
    raw = metadata.get("dependencies") or metadata.get("depends_on") or []
    if isinstance(raw, str): raw = [raw]
    values: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        values.append({"skill": str(item), "required": True} if not isinstance(item, dict) else {"skill": str(item.get("skill") or item.get("name") or item.get("id") or ""), "required": bool(item.get("required", True)), "version": item.get("version"), "role": item.get("role")})
    # Codex's current portable Director expresses delegation in instructions,
    # not front matter. Preserve that source fact as an inferred local record.
    for name in re.findall(r"\$([A-Za-z0-9][A-Za-z0-9._-]+)", body):
        if not any(value["skill"] == name for value in values):
            values.append({"skill": name, "required": "only" not in body[max(0, body.find(name)-120):body.find(name)].lower(), "inferred": True})
    return [value for value in values if value.get("skill")]


def _provider_records(conn) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM skill_capability_providers").fetchall()
    result = []
    for row in rows:
        item = dict(row); item.update(json.loads(item.pop("config_json") or "{}")); item["enabled"] = bool(item.get("enabled")); result.append(item)
    return result


def skill_runtime_status(conn, skill_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM skills WHERE id=? OR name=? COLLATE NOCASE", (skill_id, skill_id)).fetchone()
    if not row: raise SkillError("Skill was not found.", "SKILL_NOT_FOUND")
    skill = dict(row)
    if not (Path(skill["package_path"]) / "SKILL.md").is_file(): return {"status": "BLOCKED", "reason": "missing_package"}
    try: dependencies = json.loads(skill["dependencies_json"] or "[]")
    except json.JSONDecodeError: return {"status": "BLOCKED", "reason": "invalid_dependencies"}
    missing = []
    for dep in dependencies:
        identifier = dep.get("skill") if isinstance(dep, dict) else dep
        required = dep.get("required", True) if isinstance(dep, dict) else True
        if required and not conn.execute("SELECT 1 FROM skills WHERE id=? OR name=? COLLATE NOCASE", (identifier, identifier)).fetchone(): missing.append(identifier)
    if missing: return {"status": "BLOCKED", "reason": "missing_dependencies", "dependencies": missing}
    registry = CapabilityRegistry(_provider_records(conn))
    reports = [registry.report(cap) for cap in json.loads(skill["capabilities_json"] or "[]")]
    unavailable = [r["capability"] for r in reports if r["status"] != "PRESERVED"]
    return {"status": "DEGRADED" if unavailable else "FULLY_OPERATIONAL", "capabilities": reports, "dependencies": dependencies}


def compare_skill(conn, skill_id: str, source: str | Path) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM skills WHERE id=? OR name=? COLLATE NOCASE", (skill_id, skill_id)).fetchone()
    if not row: raise SkillError("Skill was not found.", "SKILL_NOT_FOUND")
    incoming = inspect_skill(source); root = Path(row["package_path"])
    local_files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file() and ".prompthub" not in p.parts}
    source_root = Path(incoming["source_root"])
    incoming_files = {p.relative_to(source_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_root.rglob("*") if p.is_file()}
    added, removed = sorted(set(incoming_files)-set(local_files)), sorted(set(local_files)-set(incoming_files))
    modified = sorted(k for k in set(local_files)&set(incoming_files) if local_files[k] != incoming_files[k])
    baseline = row["package_baseline_hash"] or row["content_hash"]
    if incoming["content_hash"] == row["content_hash"]: state = "IDENTICAL"
    elif row["locally_modified"] or (baseline and row["content_hash"] != baseline): state = "CONFLICT"
    elif row["version"] and incoming.get("version") and str(incoming["version"]) < str(row["version"]): state = "OLDER"
    else: state = "NEWER"
    return {"state": state, "incoming": incoming, "installed": dict(row), "files": {"added": added, "removed": removed, "modified": modified, "unchanged": sorted(set(local_files)&set(incoming_files)-set(modified)), "skill_markdown_changed": "SKILL.md" in modified}}


def update_skill(conn, skill_id: str, source: str | Path, *, base_dir: str | Path) -> dict[str, Any]:
    comparison = compare_skill(conn, skill_id, source)
    if comparison["state"] == "CONFLICT": raise SkillError("Skill update conflicts with local portable-source modifications.", "SKILL_UPDATE_CONFLICT", comparison)
    row = comparison["installed"]
    # Preserve PromptHub-only runtime config/adapters and catalogue identity.
    result = import_skill(conn, source, base_dir=base_dir, source_type=row["source_type"], source_platform=row["source_platform"])
    conn.execute("UPDATE skills SET runtime_config_json=?, capability_config_json=?, package_baseline_hash=content_hash WHERE id=?", (row["runtime_config_json"], row.get("capability_config_json") or "{}", row["id"]))
    return {"comparison": comparison, "result": result}


def configure_provider(conn, provider_id: str, capability: str, config: dict[str, Any], *, enabled: bool = False, trust_requirement: str = "trusted") -> dict[str, Any]:
    existing = conn.execute("SELECT id FROM skill_capability_providers WHERE provider_id=? AND capability=?", (provider_id, capability)).fetchone()
    t = now()
    if existing:
        conn.execute("UPDATE skill_capability_providers SET enabled=?,config_json=?,trust_requirement=?,updated_at=? WHERE id=?", (int(enabled), json.dumps(config), trust_requirement, t, existing["id"]))
        return {"id": existing["id"], "operation": "updated"}
    provider_id_value = str(uuid.uuid4()).upper()
    conn.execute("INSERT INTO skill_capability_providers(id,provider_id,capability,enabled,config_json,trust_requirement,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)", (provider_id_value, provider_id, capability, int(enabled), json.dumps(config), trust_requirement, t, t))
    return {"id": provider_id_value, "operation": "created"}


def run_skill(conn, skill_id: str, request: str, model: str, parameters: dict[str, Any], ollama_generate: Callable[..., str], *, operation: str = "create", inputs: list[dict[str, Any]] | None = None, target: str | None = None, source_prompt_id: int | None = None, capability_registry: dict[str, dict[str, Any]] | None = None, parent_execution_id: str | None = None, _stack: tuple[str, ...] = ()) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM skills WHERE id=? OR name=? COLLATE NOCASE", (skill_id, skill_id)).fetchone()
    if not row: raise SkillError("Skill was not found.", "SKILL_NOT_FOUND")
    skill, root = dict(row), Path(row["package_path"])
    if skill["id"] in _stack: raise SkillError("Circular Skill dependency detected.", "CIRCULAR_DEPENDENCY")
    execution = normalise_skill_execution(operation=operation, inputs=inputs, request=request, target=target, parameters=parameters)
    operation, inputs, target, parameters = execution["operation"], execution["inputs"], execution["target"], execution["parameters"]
    if operation not in skill_supported_operations(skill):
        raise SkillError("This Skill does not support the requested operation.", "UNSUPPORTED_SKILL_OPERATION", {"operation": operation, "supported": skill_supported_operations(skill)})
    request = _execution_request_text(operation, inputs, target)
    status = skill_runtime_status(conn, skill["id"])
    if status["status"] == "BLOCKED": raise SkillError("Skill is blocked by runtime requirements.", "SKILL_BLOCKED", status)
    execution_id, started = str(uuid.uuid4()).upper(), time.monotonic()
    conn.execute("INSERT INTO skill_execution_traces(id,skill_id,model,request,parameters_json,resources_json,capabilities_json,status,parent_execution_id,root_execution_id,created_at,operation,inputs_json,target,source_prompt_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (execution_id, skill["id"], model, request, json.dumps(parameters), "[]", "[]", "running", parent_execution_id, parent_execution_id or execution_id, now(), operation, json.dumps(inputs), target, source_prompt_id))
    try:
        dependencies = json.loads(skill["dependencies_json"] or "[]")
        dependent_results = []
        for dep in dependencies:
            detail = dep if isinstance(dep, dict) else {"skill": dep, "required": True}
            try:
                dependent_results.append(run_skill(conn, str(detail["skill"]), "", model, {**parameters, "caller_skill": skill["name"], "execution_role": detail.get("role")}, ollama_generate, operation=operation, inputs=inputs, target=target, source_prompt_id=source_prompt_id, parent_execution_id=execution_id, _stack=_stack+(skill["id"],)))
            except SkillError:
                if detail.get("required", True): raise
        main = (root / "SKILL.md").read_text(encoding="utf-8", errors="replace"); _, body = _frontmatter(main)
        selected = _route_resources(root, body, request, parameters); resources = _read_resources(root, selected)
        registry = CapabilityRegistry(_provider_records(conn)); capabilities = [registry.report(cap) for cap in json.loads(skill["capabilities_json"] or "[]")]
        adapter = json.loads(skill.get("runtime_config_json") or "{}")
        system = body + "\n\nYou are executing this portable Agent Skill locally. Follow its instructions, do not invent unavailable tools, and return only the requested result."
        if adapter.get("instruction_override"): system += "\n\nLocal runtime adapter:\n" + str(adapter["instruction_override"])
        context = [f"USER REQUEST:\n{request}", f"PARAMETERS:\n{json.dumps(parameters, sort_keys=True)}"]
        if dependent_results: context.append("DEPENDENT SKILL RESULTS (validated operational records):\n" + json.dumps([{k: v for k, v in item.items() if k in {"skill", "structured", "content", "resources", "execution_id"}} for item in dependent_results], ensure_ascii=False))
        if resources: context.append("RELEVANT SKILL RESOURCES:\n" + "\n\n".join(f"[{r['path']}]\n{r['content']}" for r in resources))
        budget = int(adapter.get("context_budget_chars") or parameters.get("context_budget_chars") or 30000)
        prompt = "\n\n---\n\n".join(context)
        if len(system) + len(prompt) > budget: raise SkillError("Skill execution exceeds the configured local-model context budget.", "CONTEXT_BUDGET_EXCEEDED", {"budget": budget, "required": len(system)+len(prompt)})
        schema_path = None; schema = adapter.get("result_schema")
        if not schema:
            schemas = list(root.joinpath("schemas").glob("*.json")) if root.joinpath("schemas").exists() else []
            if schemas:
                schema_path = schemas[0].relative_to(root).as_posix(); schema = json.loads(schemas[0].read_text(encoding="utf-8"))
        raw = ollama_generate(prompt=prompt, model=model, system=system, think=False, **({"format": schema} if schema else {}))
        structured = None
        if schema:
            try: structured = json.loads(raw)
            except json.JSONDecodeError as exc: raise SkillError("Skill returned invalid structured JSON.", "INVALID_SKILL_RESULT", {"error": str(exc)}) from exc
        provider_results = []
        for request_spec in parameters.get("capability_requests") or []:
            capability = str(request_spec.get("capability") or "")
            if capability not in json.loads(skill["capabilities_json"] or "[]"): raise SkillError("Skill is not permitted to request this capability.", "CAPABILITY_NOT_DECLARED")
            provider = registry.resolve(capability)
            if provider is None: raise SkillError("No local provider is configured for this capability.", "MISSING_CAPABILITY_PROVIDER")
            if skill["trust_state"] != "trusted": raise SkillError("Skill must be trusted before it can execute a privileged provider.", "SKILL_TRUST_REQUIRED")
            provider_results.append(provider.execute(dict(request_spec)))
        result_text = str(structured.get("prompt")) if isinstance(structured, dict) and structured.get("prompt") else raw
        result = {"execution_id": execution_id, "parent_execution_id": parent_execution_id, "skill_id": skill["id"], "skill": skill["display_name"], "version": skill["version"], "operation": operation, "inputs": inputs, "target": target, "source_prompt_id": source_prompt_id, "model": model, "parameters": parameters, "resources": selected, "capabilities": capabilities, "dependencies": [{"execution_id": item["execution_id"], "skill": item["skill"], "version": item.get("version")} for item in dependent_results], "content": raw, "result_text": result_text, "structured": structured, "provider_results": provider_results, "runtime_status": status["status"]}
        conn.execute("UPDATE skill_execution_traces SET resources_json=?,capabilities_json=?,result_json=?,provider_results_json=?,status='completed',duration_ms=?,schema_path=? WHERE id=?", (json.dumps(selected), json.dumps(capabilities), json.dumps(result), json.dumps(provider_results), round((time.monotonic()-started)*1000), schema_path, execution_id))
        return result
    except Exception as exc:
        conn.execute("UPDATE skill_execution_traces SET status='failed',error=?,duration_ms=? WHERE id=?", (str(exc), round((time.monotonic()-started)*1000), execution_id))
        if isinstance(exc, SkillError): raise
        raise SkillError(str(exc), "SKILL_RUNTIME_ERROR") from exc


def _prompt_row(conn, identifier: str | int):
    text = str(identifier or "").strip()
    if not text:
        return None
    return conn.execute("SELECT * FROM prompts WHERE id=? OR sync_id=?", (int(text) if text.isdigit() else -1, text)).fetchone()


def source_prompt_input(conn, identifier: str | int) -> dict[str, str]:
    row = _prompt_row(conn, identifier)
    if not row:
        raise SkillError("Source prompt was not found.", "SOURCE_PROMPT_NOT_FOUND")
    return {"type": "text", "role": "source", "content": str(row["content"])}


def save_skill_derivative(conn, source_prompt_id: str | int, execution_id: str, *, title: str | None = None, replace_original: bool = False) -> dict[str, Any]:
    source = _prompt_row(conn, source_prompt_id)
    if not source:
        raise SkillError("Source prompt was not found.", "SOURCE_PROMPT_NOT_FOUND")
    trace = conn.execute("SELECT t.*,s.display_name,s.version FROM skill_execution_traces t JOIN skills s ON s.id=t.skill_id WHERE t.id=?", (execution_id,)).fetchone()
    if not trace or trace["status"] != "completed" or not trace["result_json"]:
        raise SkillError("Completed Skill execution was not found.", "SKILL_EXECUTION_NOT_FOUND")
    if trace["source_prompt_id"] is None:
        raise SkillError("Skill execution was not run from a source prompt.", "SKILL_EXECUTION_SOURCE_MISMATCH")
    if int(trace["source_prompt_id"]) != int(source["id"]):
        raise SkillError("Skill execution belongs to a different source prompt.", "SKILL_EXECUTION_SOURCE_MISMATCH")
    result = json.loads(trace["result_json"])
    content = str(result.get("result_text") or result.get("content") or "").strip()
    if not content:
        raise SkillError("Skill execution has no prompt result to save.", "INVALID_SKILL_RESULT")
    created = now()
    if replace_original:
        conn.execute("UPDATE prompts SET content=?,updated_at=?,revision=revision+1,source=? WHERE id=?", (content, created, "skill_runtime", source["id"]))
        derived_id, action = int(source["id"]), "replaced"
    else:
        derived_title = str(title or f"{source['title']} - {_humanise_identifier(str(trace['target'] or trace['operation']))}").strip()
        cursor = conn.execute("""INSERT INTO prompts(title,category,tool,prompt_type,content,notes,thumbnail,parent_id,group_id,sync_id,source,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (derived_title, source["category"], source["tool"], source["prompt_type"], content, source["notes"], None, source["id"], source["group_id"], str(uuid.uuid4()).upper(), "skill_runtime", created, created))
        derived_id, action = int(cursor.lastrowid), "created"
        tags = conn.execute("SELECT tag_id FROM prompt_tags WHERE prompt_id=?", (source["id"],)).fetchall()
        for tag in tags:
            conn.execute("INSERT OR IGNORE INTO prompt_tags(prompt_id,tag_id) VALUES (?,?)", (derived_id, tag["tag_id"]))
    derivation_id = str(uuid.uuid4()).upper()
    conn.execute("""INSERT INTO prompt_skill_derivations(id,source_prompt_id,derived_prompt_id,execution_id,skill_id,skill_version,operation,target,execution_model,action,metadata_json,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", (derivation_id, source["id"], derived_id, execution_id, trace["skill_id"], trace["version"], trace["operation"], trace["target"], trace["model"], action, json.dumps({"resources": json.loads(trace["resources_json"] or "[]")}), created))
    return {"derivation_id": derivation_id, "source_prompt_id": int(source["id"]), "prompt_id": derived_id, "action": action, "content": content}


def list_prompt_derivations(conn, prompt_id: str | int) -> list[dict[str, Any]]:
    row = _prompt_row(conn, prompt_id)
    if not row:
        raise SkillError("Prompt was not found.", "SOURCE_PROMPT_NOT_FOUND")
    records = conn.execute("""SELECT d.*,s.display_name AS skill_name,p.title AS source_title,dp.title AS derived_title
        FROM prompt_skill_derivations d JOIN skills s ON s.id=d.skill_id JOIN prompts p ON p.id=d.source_prompt_id
        LEFT JOIN prompts dp ON dp.id=d.derived_prompt_id
        WHERE d.source_prompt_id=? OR d.derived_prompt_id=? ORDER BY d.created_at DESC""", (row["id"], row["id"])).fetchall()
    return [{**dict(item), "metadata": json.loads(item["metadata_json"] or "{}")} for item in records]
