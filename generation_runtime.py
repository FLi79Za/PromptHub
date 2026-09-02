"""Profile-driven ComfyUI dispatch and generation persistence for PromptHub.

The module deliberately contains no model-family prompting knowledge.  A profile
maps semantic PromptHub values to a preserved API-format ComfyUI graph.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import ipaddress
import json
import mimetypes
import re
import socket
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode, urlparse

import requests


SEMANTIC_INPUT_TYPES = {
    "prompt": "text", "negative_prompt": "text", "start_image": "image",
    "end_image": "image", "reference_image": "image", "reference_images": "images",
    "guide_image": "image", "guide_images": "images", "audio": "audio",
    "reference_video": "video", "seed": "integer", "width": "integer",
    "height": "integer", "duration": "number", "frame_count": "integer",
    "fps": "number", "steps": "integer", "cfg": "number", "strength": "number",
}
MEDIA_TYPES = {"image", "images", "audio", "video"}
GENERATION_STATES = {"preparing", "queued", "running", "completed", "failed", "cancelled"}
PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,79}$")


class GenerationError(Exception):
    def __init__(self, message: str, code: str = "GENERATION_ERROR", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def json_value(row: sqlite3.Row | dict[str, Any], name: str, default: Any) -> Any:
    try:
        raw = row[name]
    except (KeyError, IndexError):
        return default
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def apply_generation_migrations(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS comfyui_servers (
        id TEXT PRIMARY KEY, display_name TEXT NOT NULL, base_url TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1, description TEXT, is_default INTEGER NOT NULL DEFAULT 0,
        health_state TEXT NOT NULL DEFAULT 'configured', health_detail_json TEXT NOT NULL DEFAULT '{}',
        last_checked_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS workflow_profiles (
        id TEXT PRIMARY KEY, display_name TEXT NOT NULL, generation_kind TEXT NOT NULL,
        model_family TEXT, mode TEXT, server_id TEXT NOT NULL, source_workflow_path TEXT NOT NULL,
        source_hash TEXT NOT NULL, profile_version INTEGER NOT NULL DEFAULT 1,
        inputs_json TEXT NOT NULL DEFAULT '{}', outputs_json TEXT NOT NULL DEFAULT '[]',
        parameters_json TEXT NOT NULL DEFAULT '{}', compatibility_json TEXT NOT NULL DEFAULT '{}',
        enabled INTEGER NOT NULL DEFAULT 1, validation_state TEXT NOT NULL DEFAULT 'unvalidated',
        validation_detail_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        FOREIGN KEY(server_id) REFERENCES comfyui_servers(id) ON DELETE RESTRICT
    );
    CREATE INDEX IF NOT EXISTS idx_workflow_profiles_server ON workflow_profiles(server_id, enabled);
    CREATE TABLE IF NOT EXISTS generations (
        id TEXT PRIMARY KEY, source_prompt_id INTEGER NOT NULL, prompt_revision INTEGER,
        prompt_content_hash TEXT NOT NULL, prompt_content_snapshot TEXT, derivation_id TEXT, profile_id TEXT NOT NULL,
        profile_version INTEGER NOT NULL, workflow_hash TEXT NOT NULL, server_id TEXT NOT NULL,
        comfy_prompt_id TEXT, client_id TEXT, status TEXT NOT NULL, progress REAL,
        supplied_inputs_json TEXT NOT NULL DEFAULT '{}', parameters_json TEXT NOT NULL DEFAULT '{}',
        result_media_json TEXT NOT NULL DEFAULT '[]', error_code TEXT, error_message TEXT,
        error_detail_json TEXT NOT NULL DEFAULT '{}', retry_of TEXT, submitted_at TEXT,
        completed_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        FOREIGN KEY(source_prompt_id) REFERENCES prompts(id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id) REFERENCES workflow_profiles(id) ON DELETE RESTRICT,
        FOREIGN KEY(server_id) REFERENCES comfyui_servers(id) ON DELETE RESTRICT,
        FOREIGN KEY(retry_of) REFERENCES generations(id) ON DELETE SET NULL
    );
    CREATE INDEX IF NOT EXISTS idx_generations_prompt ON generations(source_prompt_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_generations_job ON generations(server_id, comfy_prompt_id);
    """)
    generation_columns = {row[1] for row in conn.execute("PRAGMA table_info(generations)")}
    if "prompt_content_snapshot" not in generation_columns:
        conn.execute("ALTER TABLE generations ADD COLUMN prompt_content_snapshot TEXT")


def validate_server_url(value: str) -> str:
    value = str(value or "").strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise GenerationError("ComfyUI server URL must be an HTTP(S) URL without embedded credentials.", "INVALID_SERVER_URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise GenerationError("ComfyUI server URL must not contain a path, query, or fragment.", "INVALID_SERVER_URL")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))}
    except socket.gaierror:
        # A configured but currently unresolved private hostname is permitted; health will report it unreachable.
        addresses = set()
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not (ip.is_private or ip.is_loopback or ip.is_link_local):
            raise GenerationError("ComfyUI servers must resolve to a local or private-network address.", "UNSAFE_SERVER_URL", {"address": address})
    return value


def upsert_server(conn: sqlite3.Connection, data: dict[str, Any]) -> dict[str, Any]:
    server_id = str(data.get("id") or "").strip().lower()
    if not PROFILE_ID_RE.fullmatch(server_id):
        raise GenerationError("Server id must use lowercase letters, numbers, dots, dashes, or underscores.", "INVALID_SERVER")
    name = str(data.get("display_name") or "").strip()
    if not name:
        raise GenerationError("Server display name is required.", "INVALID_SERVER")
    base_url = validate_server_url(str(data.get("base_url") or ""))
    enabled, is_default, now = bool(data.get("enabled", True)), bool(data.get("is_default", False)), utc_now()
    if is_default:
        conn.execute("UPDATE comfyui_servers SET is_default=0, updated_at=?", (now,))
    existing = conn.execute("SELECT id FROM comfyui_servers WHERE id=?", (server_id,)).fetchone()
    if existing:
        conn.execute("""UPDATE comfyui_servers SET display_name=?,base_url=?,enabled=?,description=?,is_default=?,updated_at=? WHERE id=?""",
                     (name, base_url, int(enabled), str(data.get("description") or "").strip(), int(is_default), now, server_id))
        operation = "updated"
    else:
        conn.execute("""INSERT INTO comfyui_servers(id,display_name,base_url,enabled,description,is_default,created_at,updated_at)
                        VALUES(?,?,?,?,?,?,?,?)""", (server_id, name, base_url, int(enabled), str(data.get("description") or "").strip(), int(is_default), now, now))
        operation = "created"
    return {"operation": operation, "server": get_server(conn, server_id)}


def _server_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["enabled"], item["is_default"] = bool(item["enabled"]), bool(item["is_default"])
    item["health_detail"] = json_value(row, "health_detail_json", {})
    item.pop("health_detail_json", None)
    return item


def get_server(conn: sqlite3.Connection, server_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM comfyui_servers WHERE id=?", (server_id,)).fetchone()
    if not row:
        raise GenerationError("ComfyUI server was not found.", "SERVER_NOT_FOUND")
    return _server_dict(row)


def list_servers(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [_server_dict(row) for row in conn.execute("SELECT * FROM comfyui_servers ORDER BY is_default DESC, display_name COLLATE NOCASE")]


def validate_api_workflow(graph: Any) -> dict[str, Any]:
    if not isinstance(graph, dict) or not graph:
        raise GenerationError("Workflow must be a non-empty API-format JSON object.", "INVALID_WORKFLOW")
    errors: list[dict[str, str]] = []
    for node_id, node in graph.items():
        if not isinstance(node_id, str) or not isinstance(node, dict) or not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs", {}), dict):
            errors.append({"node_id": str(node_id), "message": "Each node requires class_type and an inputs object."})
    if errors:
        raise GenerationError("Workflow is not valid ComfyUI API format.", "INVALID_WORKFLOW", {"errors": errors})
    return graph


def inspect_workflow(graph: dict[str, Any]) -> dict[str, Any]:
    validate_api_workflow(graph)
    candidates: list[dict[str, Any]] = []
    for node_id, node in graph.items():
        class_type = node["class_type"]
        for field, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                continue
            suggestions: list[str] = []
            haystack = f"{class_type} {field}".lower()
            for role in SEMANTIC_INPUT_TYPES:
                tokens = role.split("_")
                if all(token in haystack for token in tokens) or (role == "prompt" and field.lower() in {"text", "prompt"}) or (role == "seed" and "seed" in field.lower()):
                    suggestions.append(role)
            if "loadimage" in class_type.lower() and field.lower() == "image":
                suggestions = ["start_image", "end_image", "reference_image", "guide_image"]
            elif "primitivestring" in class_type.lower() and field.lower() in {"value", "text"}:
                suggestions = ["prompt", "negative_prompt"]
            value_type = "boolean" if isinstance(value, bool) else "integer" if isinstance(value, int) else "number" if isinstance(value, float) else "text"
            if "image" in haystack:
                value_type = "image"
            elif "video" in haystack:
                value_type = "video"
            elif "audio" in haystack:
                value_type = "audio"
            candidates.append({"node_id": node_id, "class_type": class_type, "field": field, "current_value": value, "value_type": value_type, "suggestions": suggestions})
    output_candidates = [{"node_id": node_id, "class_type": node["class_type"],
                          "suggested_type": "video" if any(token in node["class_type"].lower() for token in ("video", "vhs", "combine")) else "image"}
                         for node_id, node in graph.items() if any(token in node["class_type"].lower() for token in ("saveimage", "previewimage", "video", "vhs", "combine"))]
    return {"format": "api", "node_count": len(graph), "source_hash": sha256_json(graph), "candidates": candidates,
            "output_candidates": output_candidates, "required_node_types": sorted({node["class_type"] for node in graph.values()})}


def _normalise_mapping(role: str, mapping: dict[str, Any]) -> dict[str, Any]:
    node_id, field = str(mapping.get("node_id") or mapping.get("node") or ""), str(mapping.get("field") or mapping.get("input") or "")
    if not node_id or not field:
        raise GenerationError(f"Mapping for {role} requires node_id and field.", "INVALID_MAPPING", {"role": role})
    value_type = str(mapping.get("type") or SEMANTIC_INPUT_TYPES.get(role) or "text")
    return {"node_id": node_id, "field": field, "type": value_type, "required": bool(mapping.get("required", False)),
            "multiple": bool(mapping.get("multiple", value_type == "images")), "label": str(mapping.get("label") or role.replace("_", " ").title()),
            **({"default": mapping["default"]} if "default" in mapping else {}),
            **({"enum": list(mapping["enum"])} if isinstance(mapping.get("enum"), list) else {})}


def validate_profile(profile: dict[str, Any], graph: dict[str, Any], server_ids: Iterable[str]) -> dict[str, Any]:
    errors, warnings = [], []
    try:
        validate_api_workflow(graph)
    except GenerationError as exc:
        return {"state": "invalid", "errors": [{"code": exc.code, "message": str(exc)}], "warnings": []}
    if profile.get("server_id") not in set(server_ids):
        errors.append({"code": "INVALID_SERVER", "message": "Target server does not exist."})
    if profile.get("generation_kind") not in {"image", "video"}:
        errors.append({"code": "INVALID_KIND", "message": "generation_kind must be image or video."})
    inputs = profile.get("inputs") or {}
    if "prompt" not in inputs:
        warnings.append({"code": "PROMPT_UNMAPPED", "message": "Profile does not map the current prompt."})
    for role, mapping in inputs.items():
        node_id, field = mapping.get("node_id"), mapping.get("field")
        if node_id not in graph:
            errors.append({"code": "MISSING_NODE", "role": role, "message": f"Node {node_id} does not exist."})
        elif field not in graph[node_id].get("inputs", {}):
            errors.append({"code": "MISSING_FIELD", "role": role, "message": f"Node {node_id} has no input named {field}."})
    for output in profile.get("outputs") or []:
        if str(output.get("node_id") or "") not in graph:
            errors.append({"code": "MISSING_OUTPUT_NODE", "message": f"Output node {output.get('node_id')} does not exist."})
    return {"state": "invalid" if errors else "valid", "errors": errors, "warnings": warnings}


def validate_saved_profile(conn: sqlite3.Connection, profile_id: str, *, live: bool = False,
                           session: requests.Session | None = None) -> dict[str, Any]:
    profile = get_profile(conn, profile_id)
    try:
        graph = load_profile_graph(profile)
        validation = validate_profile(profile, graph, [row[0] for row in conn.execute("SELECT id FROM comfyui_servers")])
        if live and validation["state"] == "valid":
            server = get_server(conn, profile["server_id"])
            if not server["enabled"]:
                validation["errors"].append({"code": "SERVER_DISABLED", "message": "Target server is disabled."})
            else:
                definitions = ComfyUIClient(server["base_url"], session=session).object_info()
                missing = sorted({node["class_type"] for node in graph.values()} - set(definitions))
                if missing:
                    validation["errors"].append({"code": "MISSING_NODE_CLASSES", "message": "Target server lacks required node classes.", "classes": missing})
                validation["live_schema_checked"] = True
            if validation["errors"]:
                validation["state"] = "invalid"
    except GenerationError as exc:
        validation = {"state": "invalid", "errors": [{"code": exc.code, "message": str(exc), "details": exc.details}], "warnings": []}
    now = utc_now()
    conn.execute("UPDATE workflow_profiles SET validation_state=?,validation_detail_json=?,updated_at=? WHERE id=?",
                 (validation["state"], canonical_json(validation), now, profile_id))
    return validation


def _profile_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["enabled"] = bool(item["enabled"])
    for column, target, default in (("inputs_json", "inputs", {}), ("outputs_json", "outputs", []), ("parameters_json", "parameters", {}),
                                     ("compatibility_json", "compatibility", {}), ("validation_detail_json", "validation_detail", {})):
        item[target] = json_value(row, column, default)
        item.pop(column, None)
    return item


def get_profile(conn: sqlite3.Connection, profile_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM workflow_profiles WHERE id=?", (profile_id,)).fetchone()
    if not row:
        raise GenerationError("Workflow Profile was not found.", "PROFILE_NOT_FOUND")
    return _profile_dict(row)


def list_profiles(conn: sqlite3.Connection, *, include_disabled: bool = True) -> list[dict[str, Any]]:
    where = "" if include_disabled else "WHERE p.enabled=1 AND s.enabled=1"
    rows = conn.execute(f"""SELECT p.*,s.display_name AS server_name,s.health_state AS server_health FROM workflow_profiles p
                           JOIN comfyui_servers s ON s.id=p.server_id {where} ORDER BY p.display_name COLLATE NOCASE""").fetchall()
    return [_profile_dict(row) for row in rows]


def save_profile(conn: sqlite3.Connection, data: dict[str, Any], graph: dict[str, Any], workflow_root: str | Path, *, allow_update: bool = False) -> dict[str, Any]:
    profile_id = str(data.get("id") or "").strip().lower()
    if not PROFILE_ID_RE.fullmatch(profile_id):
        raise GenerationError("Profile id must use lowercase letters, numbers, dots, dashes, or underscores.", "INVALID_PROFILE")
    existing = conn.execute("SELECT * FROM workflow_profiles WHERE id=?", (profile_id,)).fetchone()
    if existing and not allow_update:
        raise GenerationError("A Workflow Profile with this id already exists.", "DUPLICATE_PROFILE")
    graph = validate_api_workflow(graph)
    inputs = {str(role): _normalise_mapping(str(role), mapping) for role, mapping in (data.get("inputs") or {}).items() if isinstance(mapping, dict)}
    outputs = [{"node_id": str(item.get("node_id") or ""), "type": str(item.get("type") or data.get("generation_kind") or "image"),
                "label": str(item.get("label") or "Result")} for item in (data.get("outputs") or []) if isinstance(item, dict)]
    profile = {"id": profile_id, "display_name": str(data.get("display_name") or "").strip(),
               "generation_kind": str(data.get("generation_kind") or "").lower(), "model_family": str(data.get("model_family") or "").strip().lower(),
               "mode": str(data.get("mode") or "").strip().lower(), "server_id": str(data.get("server_id") or "").strip(),
               "inputs": inputs, "outputs": outputs, "parameters": data.get("parameters") or {}, "compatibility": data.get("compatibility") or {}}
    if not profile["display_name"]:
        raise GenerationError("Profile display name is required.", "INVALID_PROFILE")
    validation = validate_profile(profile, graph, [row[0] for row in conn.execute("SELECT id FROM comfyui_servers")])
    digest, now = sha256_json(graph), utc_now()
    root = Path(workflow_root).resolve() / profile_id
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / f"{digest}.json"
    if not source_path.exists():
        source_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    version = int(existing["profile_version"] + 1) if existing else 1
    values = (profile["display_name"], profile["generation_kind"], profile["model_family"], profile["mode"], profile["server_id"], str(source_path), digest, version,
              canonical_json(inputs), canonical_json(outputs), canonical_json(profile["parameters"]), canonical_json(profile["compatibility"]), int(data.get("enabled", True)),
              validation["state"], canonical_json(validation), now)
    if existing:
        conn.execute("""UPDATE workflow_profiles SET display_name=?,generation_kind=?,model_family=?,mode=?,server_id=?,source_workflow_path=?,source_hash=?,profile_version=?,inputs_json=?,outputs_json=?,parameters_json=?,compatibility_json=?,enabled=?,validation_state=?,validation_detail_json=?,updated_at=? WHERE id=?""", values + (profile_id,))
        operation = "updated"
    else:
        conn.execute("""INSERT INTO workflow_profiles(display_name,generation_kind,model_family,mode,server_id,source_workflow_path,source_hash,profile_version,inputs_json,outputs_json,parameters_json,compatibility_json,enabled,validation_state,validation_detail_json,updated_at,id,created_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values + (profile_id, now))
        operation = "created"
    return {"operation": operation, "profile": get_profile(conn, profile_id), "validation": validation}


def duplicate_profile(conn: sqlite3.Connection, profile_id: str, new_id: str, display_name: str, workflow_root: str | Path) -> dict[str, Any]:
    source = get_profile(conn, profile_id)
    graph = json.loads(Path(source["source_workflow_path"]).read_text(encoding="utf-8"))
    source.update({"id": new_id, "display_name": display_name, "enabled": False})
    return save_profile(conn, source, graph, workflow_root)


@dataclass
class ComfyUIClient:
    base_url: str
    timeout: float = 20.0
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        self.base_url = validate_server_url(self.base_url)
        self.session = self.session or requests.Session()

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.session.request(method, f"{self.base_url}{path}", timeout=kwargs.pop("timeout", self.timeout), **kwargs)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise GenerationError("ComfyUI request failed.", "COMFYUI_REQUEST_FAILED", {"path": path, "error": str(exc)}) from exc

    def health(self) -> dict[str, Any]:
        started = time.monotonic()
        try:
            payload = self._json("GET", "/system_stats", timeout=min(self.timeout, 5))
            return {"reachable": True, "state": "reachable", "latency_ms": round((time.monotonic() - started) * 1000), "system_stats": payload}
        except GenerationError as exc:
            return {"reachable": False, "state": "unreachable", "error": str(exc), "details": exc.details}

    def object_info(self) -> dict[str, Any]:
        payload = self._json("GET", "/object_info")
        return payload if isinstance(payload, dict) else {}

    def upload(self, filename: str, content: bytes, media_type: str) -> dict[str, Any]:
        endpoint = "/upload/image" if media_type in {"image", "images"} else "/upload/image"
        files = {"image": (Path(filename).name, content, mimetypes.guess_type(filename)[0] or "application/octet-stream")}
        try:
            response = self.session.post(f"{self.base_url}{endpoint}", files=files, data={"type": "input", "overwrite": "false"}, timeout=max(self.timeout, 60))
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise GenerationError("Media upload to ComfyUI failed.", "INPUT_UPLOAD_FAILED", {"filename": Path(filename).name, "error": str(exc)}) from exc
        name = payload.get("name") if isinstance(payload, dict) else None
        if not name:
            raise GenerationError("ComfyUI upload did not return an input filename.", "INPUT_UPLOAD_FAILED", {"response": payload})
        return {"name": name, "subfolder": payload.get("subfolder", ""), "type": payload.get("type", "input")}

    def submit(self, graph: dict[str, Any], client_id: str) -> str:
        payload = self._json("POST", "/prompt", json={"prompt": graph, "client_id": client_id})
        prompt_id = payload.get("prompt_id") if isinstance(payload, dict) else None
        if not prompt_id:
            raise GenerationError("ComfyUI did not return a prompt id.", "QUEUE_SUBMISSION_FAILED", {"response": payload})
        return str(prompt_id)

    def queue(self) -> dict[str, Any]:
        payload = self._json("GET", "/queue")
        return payload if isinstance(payload, dict) else {}

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        payload = self._json("GET", f"/history/{prompt_id}")
        if isinstance(payload, dict):
            return payload.get(prompt_id) or (payload if payload.get("outputs") is not None else None)
        return None

    def view(self, media: dict[str, Any]) -> tuple[bytes, str]:
        query = urlencode({"filename": media["filename"], "subfolder": media.get("subfolder", ""), "type": media.get("type", "output")})
        try:
            response = self.session.get(f"{self.base_url}/view?{query}", timeout=max(self.timeout, 120))
            response.raise_for_status()
            return response.content, response.headers.get("Content-Type") or mimetypes.guess_type(media["filename"])[0] or "application/octet-stream"
        except requests.RequestException as exc:
            raise GenerationError("ComfyUI output retrieval failed.", "OUTPUT_RETRIEVAL_FAILED", {"filename": media.get("filename"), "error": str(exc)}) from exc


def check_server(conn: sqlite3.Connection, server_id: str, *, session: requests.Session | None = None) -> dict[str, Any]:
    server = get_server(conn, server_id)
    if not server["enabled"]:
        result = {"reachable": False, "state": "disabled"}
    else:
        result = ComfyUIClient(server["base_url"], session=session).health()
    now = utc_now()
    conn.execute("UPDATE comfyui_servers SET health_state=?,health_detail_json=?,last_checked_at=?,updated_at=? WHERE id=?",
                 (result["state"], canonical_json(result), now, now, server_id))
    return {"server": get_server(conn, server_id), "health": result}


def load_profile_graph(profile: dict[str, Any]) -> dict[str, Any]:
    path = Path(profile["source_workflow_path"])
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationError("Stored source workflow cannot be loaded.", "PROFILE_INVALIDATED", {"path": str(path)}) from exc
    validate_api_workflow(graph)
    current_hash = sha256_json(graph)
    if current_hash != profile["source_hash"]:
        raise GenerationError("Stored source workflow changed after profile validation.", "PROFILE_INVALIDATED", {"expected": profile["source_hash"], "actual": current_hash})
    return graph


def _coerce(value: Any, definition: dict[str, Any], role: str) -> Any:
    value_type = definition.get("type", "text")
    if value in (None, ""):
        return value
    try:
        if value_type == "integer": value = int(value)
        elif value_type == "number": value = float(value)
        elif value_type == "boolean": value = value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes", "on"}
        elif value_type not in MEDIA_TYPES: value = str(value)
    except (TypeError, ValueError) as exc:
        raise GenerationError(f"Invalid value for {role}.", "INVALID_PARAMETER_TYPE", {"role": role, "type": value_type}) from exc
    if definition.get("enum") and value not in definition["enum"]:
        raise GenerationError(f"Invalid choice for {role}.", "INVALID_PARAMETER_VALUE", {"role": role, "allowed": definition["enum"]})
    return value


def _media_bytes(value: Any, role: str) -> tuple[str, bytes]:
    if isinstance(value, dict) and value.get("content_base64"):
        try:
            return Path(str(value.get("filename") or f"{role}.bin")).name, base64.b64decode(value["content_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise GenerationError(f"Invalid base64 media for {role}.", "INVALID_MEDIA_INPUT") from exc
    path_value = value.get("source_path") if isinstance(value, dict) else value
    if not isinstance(value, dict) or not value.get("managed_upload"):
        raise GenerationError(f"Local paths are not accepted for {role}; upload the file or send base64 media.", "UNSAFE_MEDIA_PATH", {"role": role})
    path = Path(str(path_value or ""))
    if not path.is_file():
        raise GenerationError(f"Media input for {role} was not found.", "INVALID_MEDIA_INPUT", {"role": role})
    return path.name, path.read_bytes()


def _media_source_record(value: Any, role: str) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("content_base64"):
        return {"filename": Path(str(value.get("filename") or f"{role}.bin")).name, "transport": "inline"}
    path_value = value.get("source_path") if isinstance(value, dict) else value
    path = Path(str(path_value or "")).resolve()
    return {"filename": path.name, "source_path": str(path), "transport": "managed_file", "managed_upload": True}


def materialise_workflow(profile: dict[str, Any], graph: dict[str, Any], values: dict[str, Any], client: ComfyUIClient) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = copy.deepcopy(graph)
    recorded: dict[str, Any] = {}
    for role, definition in profile["inputs"].items():
        value = values.get(role, definition.get("default"))
        if value in (None, "", []):
            if definition.get("required"):
                raise GenerationError(f"Required input is missing: {definition.get('label', role)}.", "REQUIRED_INPUT_MISSING", {"role": role})
            continue
        value = _coerce(value, definition, role)
        if definition["type"] in MEDIA_TYPES:
            media_values = value if isinstance(value, list) else [value]
            uploaded = []
            for media_value in media_values:
                name, content = _media_bytes(media_value, role)
                remote = client.upload(name, content, definition["type"])
                uploaded.append(remote)
            injected = [item["name"] for item in uploaded] if definition.get("multiple") else uploaded[0]["name"]
            recorded[role] = {"source": [_media_source_record(item, role) for item in media_values], "comfyui": uploaded}
        else:
            injected, recorded[role] = value, value
        runtime[definition["node_id"]]["inputs"][definition["field"]] = injected
    return runtime, recorded


def _generation_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for column, target, default in (("supplied_inputs_json", "supplied_inputs", {}), ("parameters_json", "parameters", {}),
                                     ("result_media_json", "result_media", []), ("error_detail_json", "error_detail", {})):
        item[target] = json_value(row, column, default)
        item.pop(column, None)
    for media in item.get("result_media") or []:
        content_type = str(media.get("content_type") or "").lower()
        suffix = Path(str(media.get("filename") or "")).suffix.lower()
        if content_type.startswith("video/") or suffix in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
            media["kind"] = "video"
        elif content_type.startswith("image/"):
            media["kind"] = "image"
    return item


def get_generation(conn: sqlite3.Connection, generation_id: str) -> dict[str, Any]:
    row = conn.execute("""SELECT g.*,p.display_name AS profile_name,p.generation_kind,p.model_family,p.mode,s.display_name AS server_name
                          FROM generations g JOIN workflow_profiles p ON p.id=g.profile_id JOIN comfyui_servers s ON s.id=g.server_id WHERE g.id=?""", (generation_id,)).fetchone()
    if not row:
        raise GenerationError("Generation was not found.", "GENERATION_NOT_FOUND")
    return _generation_dict(row)


def list_prompt_generations(conn: sqlite3.Connection, prompt_id: str | int) -> list[dict[str, Any]]:
    prompt = conn.execute("SELECT id FROM prompts WHERE id=? OR sync_id=?", (int(prompt_id) if str(prompt_id).isdigit() else -1, str(prompt_id))).fetchone()
    if not prompt:
        raise GenerationError("Prompt was not found.", "SOURCE_PROMPT_NOT_FOUND")
    rows = conn.execute("""SELECT g.*,p.display_name AS profile_name,p.generation_kind,p.model_family,p.mode,s.display_name AS server_name
                           FROM generations g JOIN workflow_profiles p ON p.id=g.profile_id JOIN comfyui_servers s ON s.id=g.server_id
                           WHERE g.source_prompt_id=? ORDER BY g.created_at DESC""", (prompt["id"],)).fetchall()
    return [_generation_dict(row) for row in rows]


def compatible_profiles(conn: sqlite3.Connection, prompt_id: str | int, *, show_all: bool = False) -> list[dict[str, Any]]:
    prompt = conn.execute("SELECT * FROM prompts WHERE id=? OR sync_id=?", (int(prompt_id) if str(prompt_id).isdigit() else -1, str(prompt_id))).fetchone()
    if not prompt:
        raise GenerationError("Prompt was not found.", "SOURCE_PROMPT_NOT_FOUND")
    derivation = conn.execute("SELECT target,metadata_json FROM prompt_skill_derivations WHERE derived_prompt_id=? ORDER BY created_at DESC LIMIT 1", (prompt["id"],)).fetchone()
    normalise = lambda value: re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    target = normalise(derivation["target"]) if derivation else ""
    derivation_metadata = json_value(derivation, "metadata_json", {}) if derivation else {}
    target_mode = normalise(derivation_metadata.get("mode") or derivation_metadata.get("target_mode"))
    category = str(prompt["category"] or "").lower()
    profiles = list_profiles(conn, include_disabled=False)
    for profile in profiles:
        score, reasons = 0, []
        declared_targets = {
            normalise(value)
            for value in (profile.get("compatibility") or {}).get("targets", [])
            if str(value or "").strip()
        }
        if category in {"image", "video"} and profile["generation_kind"] == category:
            score += 20; reasons.append("prompt category")
        if target and (target == normalise(profile.get("model_family")) or target in declared_targets):
            score += 100; reasons.append("Skill target")
        if target_mode and target_mode == normalise(profile.get("mode")):
            score += 40; reasons.append("Skill mode")
        profile["compatibility_score"], profile["compatibility_reasons"] = score, reasons
        profile["recommended"] = "Skill target" in reasons or "Skill mode" in reasons
        block_reason = ""
        if profile.get("validation_state") != "valid":
            block_reason = "Profile validation is required"
        elif profile.get("server_health") == "unreachable":
            block_reason = "ComfyUI server is unreachable"
        profile["selectable"] = not block_reason
        profile["selection_block_reason"] = block_reason
    profiles.sort(key=lambda item: (-item["compatibility_score"], item["display_name"].lower()))
    # Compatibility is advisory. Enabled profiles remain available so one prompt can
    # deliberately be compared across model families; execution validation remains
    # authoritative for genuine technical incompatibilities.
    return profiles


def submit_generation(conn: sqlite3.Connection, prompt_id: str | int, profile_id: str, values: dict[str, Any], *, retry_of: str | None = None,
                      prompt_override: str | None = None, prompt_revision_override: int | None = None,
                      session: requests.Session | None = None) -> dict[str, Any]:
    prompt = conn.execute("SELECT * FROM prompts WHERE id=? OR sync_id=?", (int(prompt_id) if str(prompt_id).isdigit() else -1, str(prompt_id))).fetchone()
    if not prompt:
        raise GenerationError("Prompt was not found.", "SOURCE_PROMPT_NOT_FOUND")
    profile, generation_id, now = get_profile(conn, profile_id), str(uuid.uuid4()), utc_now()
    derivation = conn.execute("SELECT id FROM prompt_skill_derivations WHERE derived_prompt_id=? ORDER BY created_at DESC LIMIT 1", (prompt["id"],)).fetchone()
    server = get_server(conn, profile["server_id"])
    if not profile["enabled"] or profile["validation_state"] != "valid":
        raise GenerationError("Workflow Profile is disabled or invalid.", "PROFILE_INVALIDATED", profile.get("validation_detail"))
    if not server["enabled"]:
        raise GenerationError("Target ComfyUI server is disabled.", "SERVER_DISABLED")
    graph = load_profile_graph(profile)
    validation = validate_profile(profile, graph, [server["id"]])
    if validation["state"] != "valid":
        raise GenerationError("Workflow Profile is no longer valid.", "PROFILE_INVALIDATED", validation)
    effective_prompt = str(prompt_override if prompt_override is not None else prompt["content"])
    effective_revision = prompt_revision_override if prompt_revision_override is not None else (prompt["revision"] if "revision" in prompt.keys() else None)
    merged = {"prompt": effective_prompt, **(values or {})}
    parameters = {key: value for key, value in merged.items() if key not in {"prompt"} and profile["inputs"].get(key, {}).get("type") not in MEDIA_TYPES}
    conn.execute("""INSERT INTO generations(id,source_prompt_id,prompt_revision,prompt_content_hash,prompt_content_snapshot,derivation_id,profile_id,profile_version,workflow_hash,server_id,status,supplied_inputs_json,parameters_json,retry_of,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (generation_id, prompt["id"], effective_revision,
                    hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest(), effective_prompt, derivation["id"] if derivation else None, profile_id, profile["profile_version"], profile["source_hash"], server["id"], "preparing", "{}", canonical_json(parameters), retry_of, now, now))
    client = ComfyUIClient(server["base_url"], session=session)
    try:
        runtime_graph, recorded = materialise_workflow(profile, graph, merged, client)
        parameters = {key: value for key, value in recorded.items() if key != "prompt" and profile["inputs"].get(key, {}).get("type") not in MEDIA_TYPES}
        client_id = str(uuid.uuid4())
        comfy_prompt_id = client.submit(runtime_graph, client_id)
        now = utc_now()
        conn.execute("""UPDATE generations SET comfy_prompt_id=?,client_id=?,status='queued',supplied_inputs_json=?,parameters_json=?,submitted_at=?,updated_at=? WHERE id=?""",
                     (comfy_prompt_id, client_id, canonical_json(recorded), canonical_json(parameters), now, now, generation_id))
    except GenerationError as exc:
        now = utc_now()
        conn.execute("UPDATE generations SET status='failed',error_code=?,error_message=?,error_detail_json=?,completed_at=?,updated_at=? WHERE id=?",
                     (exc.code, str(exc), canonical_json(exc.details), now, now, generation_id))
        raise
    return get_generation(conn, generation_id)


def discover_outputs(history: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = history.get("outputs") or {}
    selected_nodes = {str(item["node_id"]) for item in profile.get("outputs") or [] if item.get("node_id")}
    semantic_types = {
        str(item["node_id"]): str(item.get("type") or profile.get("generation_kind") or "image")
        for item in profile.get("outputs") or []
        if item.get("node_id")
    }
    media: list[dict[str, Any]] = []
    for node_id, node_output in outputs.items():
        if selected_nodes and str(node_id) not in selected_nodes:
            continue
        if not isinstance(node_output, dict):
            continue
        for collection, kind in (("images", "image"), ("gifs", "video"), ("videos", "video"), ("audio", "audio")):
            for item in node_output.get(collection) or []:
                if isinstance(item, dict) and item.get("filename"):
                    media.append({"node_id": str(node_id), "kind": semantic_types.get(str(node_id), kind), "filename": Path(str(item["filename"])).name,
                                  "subfolder": str(item.get("subfolder") or ""), "type": str(item.get("type") or "output")})
    return media


def _queue_ids(queue: dict[str, Any], key: str) -> set[str]:
    result = set()
    for item in queue.get(key) or []:
        if isinstance(item, (list, tuple)) and len(item) > 1:
            result.add(str(item[1]))
        elif isinstance(item, dict) and item.get("prompt_id"):
            result.add(str(item["prompt_id"]))
    return result


def refresh_generation(conn: sqlite3.Connection, generation_id: str, media_root: str | Path, *, session: requests.Session | None = None) -> dict[str, Any]:
    generation = get_generation(conn, generation_id)
    if generation["status"] in {"completed", "failed", "cancelled"}:
        return generation
    server, profile = get_server(conn, generation["server_id"]), get_profile(conn, generation["profile_id"])
    client = ComfyUIClient(server["base_url"], session=session)
    history = client.history(generation["comfy_prompt_id"])
    now = utc_now()
    if history:
        status = history.get("status") or {}
        status_text = str(status.get("status_str") or "").lower()
        messages = status.get("messages") or []
        if status_text in {"error", "failed"} or not bool(status.get("completed", True)):
            conn.execute("UPDATE generations SET status='failed',error_code='EXECUTION_FAILED',error_message=?,error_detail_json=?,completed_at=?,updated_at=? WHERE id=?",
                         ("ComfyUI execution failed.", canonical_json({"status": status, "messages": messages}), now, now, generation_id))
        else:
            outputs = discover_outputs(history, profile)
            if not outputs:
                conn.execute("UPDATE generations SET status='failed',error_code='MISSING_OUTPUT',error_message='ComfyUI completed without a mapped output.',error_detail_json=?,completed_at=?,updated_at=? WHERE id=?",
                             (canonical_json({"available_output_nodes": list((history.get("outputs") or {}).keys())}), now, now, generation_id))
            else:
                destination = Path(media_root).resolve() / generation_id
                destination.mkdir(parents=True, exist_ok=True)
                stored = []
                for index, output in enumerate(outputs):
                    content, content_type = client.view(output)
                    safe_name = f"{index + 1:02d}_{Path(output['filename']).name}"
                    path = destination / safe_name
                    path.write_bytes(content)
                    stored.append({**output, "stored_path": str(path), "url": f"/generations/{generation_id}/media/{index}", "content_type": content_type, "size": len(content)})
                conn.execute("UPDATE generations SET status='completed',progress=100,result_media_json=?,completed_at=?,updated_at=? WHERE id=?",
                             (canonical_json(stored), now, now, generation_id))
    else:
        queue = client.queue()
        prompt_id = str(generation["comfy_prompt_id"])
        state = "running" if prompt_id in _queue_ids(queue, "queue_running") else "queued" if prompt_id in _queue_ids(queue, "queue_pending") else generation["status"]
        submitted = generation.get("submitted_at")
        timed_out = False
        if submitted and state in {"preparing", "queued", "running"}:
            try:
                timed_out = (datetime.now(timezone.utc) - datetime.fromisoformat(submitted)).total_seconds() > 21600
            except ValueError:
                timed_out = False
        if timed_out:
            conn.execute("UPDATE generations SET status='failed',error_code='TIMEOUT',error_message='Generation did not reach terminal history within six hours.',completed_at=?,updated_at=? WHERE id=?", (now, now, generation_id))
        else:
            conn.execute("UPDATE generations SET status=?,updated_at=? WHERE id=?", (state, now, generation_id))
    return get_generation(conn, generation_id)


def regenerate(conn: sqlite3.Connection, generation_id: str, *, new_seed: bool = False, current_prompt: bool = False,
               session: requests.Session | None = None) -> dict[str, Any]:
    previous = get_generation(conn, generation_id)
    values = dict(previous["parameters"])
    supplied = previous["supplied_inputs"]
    for role, item in supplied.items():
        if role == "prompt":
            continue
        if isinstance(item, dict) and item.get("source"):
            reusable = [source for source in item["source"] if source.get("transport") == "managed_file" and source.get("source_path")]
            if reusable:
                values[role] = reusable if len(reusable) > 1 else reusable[0]
            continue
        values[role] = item
    if new_seed and "seed" in values:
        values["seed"] = int.from_bytes(uuid.uuid4().bytes[:8], "big") & 0x7FFFFFFFFFFFFFFF
    return submit_generation(conn, previous["source_prompt_id"], previous["profile_id"], values, retry_of=generation_id,
                             prompt_override=None if current_prompt else previous.get("prompt_content_snapshot"),
                             prompt_revision_override=None if current_prompt else previous.get("prompt_revision"), session=session)
