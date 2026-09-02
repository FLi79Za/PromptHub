"""Reusable prompt-domain services for PromptHub's Integration API."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable


PROMPT_TYPES = ("Generation", "Edit", "Instruction")
MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 25
INTEGRATION_SCHEMA_VERSION = 1


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="microseconds")


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})"))


def integration_schema_ready(conn: sqlite3.Connection) -> bool:
    required_tables = {"integration_prompt_history", "schema_migrations"}
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return (
        required_tables.issubset(tables)
        and column_exists(conn, "prompts", "revision")
        and column_exists(conn, "prompts", "source")
    )


def apply_integration_migrations(conn: sqlite3.Connection) -> list[str]:
    """Apply additive, idempotent Integration API schema migrations."""
    applied: list[str] = []
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )

    existing = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE version = ?", (INTEGRATION_SCHEMA_VERSION,)
    ).fetchone()
    if existing and integration_schema_ready(conn):
        return applied

    if not column_exists(conn, "prompts", "revision"):
        conn.execute("ALTER TABLE prompts ADD COLUMN revision INTEGER NOT NULL DEFAULT 1")
    conn.execute("UPDATE prompts SET revision = 1 WHERE revision IS NULL OR revision < 1")

    if not column_exists(conn, "prompts", "source"):
        conn.execute("ALTER TABLE prompts ADD COLUMN source TEXT")

    if not column_exists(conn, "prompts", "sync_id"):
        conn.execute("ALTER TABLE prompts ADD COLUMN sync_id TEXT")
    missing = conn.execute(
        "SELECT id FROM prompts WHERE sync_id IS NULL OR TRIM(sync_id) = ''"
    ).fetchall()
    for row in missing:
        conn.execute("UPDATE prompts SET sync_id = ? WHERE id = ?", (str(uuid.uuid4()).upper(), row[0]))

    duplicate = conn.execute(
        """
        SELECT sync_id, COUNT(*) AS count
        FROM prompts
        WHERE sync_id IS NOT NULL AND TRIM(sync_id) <> ''
        GROUP BY sync_id
        HAVING COUNT(*) > 1
        LIMIT 1
        """
    ).fetchone()
    if duplicate:
        raise ServiceError(
            "DUPLICATE_SYNC_ID",
            "Integration migration cannot continue because duplicate sync identifiers exist.",
            409,
        )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_prompts_sync_id ON prompts(sync_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_revision ON prompts(revision)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS integration_prompt_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_id INTEGER NOT NULL,
            prompt_sync_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            source TEXT NOT NULL,
            change_summary TEXT,
            previous_json TEXT,
            new_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_integration_history_prompt_created
        ON integration_prompt_history(prompt_id, created_at DESC)
        """
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO schema_migrations(version, name, applied_at)
        VALUES (?, ?, ?)
        """,
        (INTEGRATION_SCHEMA_VERSION, "integration_api_v1", utc_now()),
    )
    applied.append("001_integration_api_v1")
    return applied


def _clean_text(
    value: Any,
    field: str,
    *,
    required: bool = False,
    max_length: int = 10000,
    allow_none: bool = False,
) -> str | None:
    if value is None and allow_none:
        return None
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ServiceError("INVALID_FIELD_TYPE", f"{field} must be a string.", details={"field": field})
    cleaned = value.strip()
    if required and not cleaned:
        raise ServiceError("VALIDATION_ERROR", f"{field} is required.", details={"field": field})
    if len(cleaned) > max_length:
        raise ServiceError(
            "FIELD_TOO_LONG",
            f"{field} exceeds the maximum length.",
            details={"field": field, "max_length": max_length},
        )
    return cleaned


def _parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ServiceError("INVALID_FIELD_TYPE", f"{field} must be a boolean.", details={"field": field})


def _parse_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ServiceError("INVALID_FIELD_TYPE", f"{field} must be an integer.", details={"field": field})
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ServiceError("INVALID_FIELD_TYPE", f"{field} must be an integer.", details={"field": field})
    if parsed < 1:
        raise ServiceError("VALIDATION_ERROR", f"{field} must be positive.", details={"field": field})
    return parsed


def _metadata_row(
    conn: sqlite3.Connection,
    table: str,
    id_value: Any,
    name_value: Any,
    field: str,
    *,
    required: bool,
    create_missing: bool,
    dry_run: bool,
) -> dict[str, Any] | None:
    if id_value is not None:
        metadata_id = _parse_positive_int(id_value, f"{field}_id")
        row = conn.execute(f"SELECT id, name FROM {table} WHERE id = ?", (metadata_id,)).fetchone()
        if not row:
            raise ServiceError(
                "INVALID_METADATA",
                f"Unknown {field}_id.",
                details={"field": f"{field}_id", "value": metadata_id},
            )
        return {"id": int(row["id"]), "name": row["name"]}

    if name_value is None:
        if required:
            raise ServiceError("VALIDATION_ERROR", f"{field} is required.", details={"field": field})
        return None
    name = _clean_text(name_value, field, required=required, max_length=200)
    if not name:
        return None
    row = conn.execute(
        f"SELECT id, name FROM {table} WHERE name = ? COLLATE NOCASE", (name,)
    ).fetchone()
    if row:
        return {"id": int(row["id"]), "name": row["name"]}
    if not create_missing:
        raise ServiceError(
            "INVALID_METADATA",
            f"Unknown {field}. Set create_missing_metadata=true to create it explicitly.",
            details={"field": field, "value": name},
        )
    if dry_run:
        return {"id": None, "name": name, "would_create": True}

    if table == "prompt_groups":
        conn.execute(
            "INSERT INTO prompt_groups(name, description, created_at) VALUES (?, NULL, ?)",
            (name, utc_now()),
        )
    else:
        conn.execute(f"INSERT INTO {table}(name) VALUES (?)", (name,))
    row = conn.execute(f"SELECT id, name FROM {table} WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
    return {"id": int(row["id"]), "name": row["name"]}


def _prompt_type(id_value: Any, name_value: Any, *, required: bool) -> dict[str, Any] | None:
    if id_value is not None:
        prompt_type_id = _parse_positive_int(id_value, "prompt_type_id")
        if prompt_type_id > len(PROMPT_TYPES):
            raise ServiceError("INVALID_METADATA", "Unknown prompt_type_id.", details={"field": "prompt_type_id"})
        return {"id": prompt_type_id, "name": PROMPT_TYPES[prompt_type_id - 1]}
    if name_value is None:
        if required:
            raise ServiceError("VALIDATION_ERROR", "prompt_type is required.", details={"field": "prompt_type"})
        return None
    name = _clean_text(name_value, "prompt_type", required=required, max_length=120)
    for index, known in enumerate(PROMPT_TYPES, 1):
        if known.casefold() == (name or "").casefold():
            return {"id": index, "name": known}
    raise ServiceError(
        "INVALID_METADATA",
        "Unknown prompt_type.",
        details={"field": "prompt_type", "allowed": list(PROMPT_TYPES)},
    )


def _normalise_tags(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[Any] = value.split(",")
    elif isinstance(value, list):
        values = value
    else:
        raise ServiceError("INVALID_FIELD_TYPE", "tags must be a list or comma-separated string.")
    output: list[Any] = []
    seen: set[str] = set()
    for item in values:
        key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _resolve_tags(
    conn: sqlite3.Connection,
    value: Any,
    *,
    create_missing: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for item in _normalise_tags(value):
        tag_id = None
        tag_name = None
        if isinstance(item, dict):
            tag_id = item.get("id")
            tag_name = item.get("name")
        elif isinstance(item, int) and not isinstance(item, bool):
            tag_id = item
        else:
            tag_name = item
        tag = _metadata_row(
            conn,
            "tags",
            tag_id,
            tag_name,
            "tag",
            required=True,
            create_missing=create_missing,
            dry_run=dry_run,
        )
        if tag:
            resolved.append(tag)
    return resolved


def _set_prompt_tags(conn: sqlite3.Connection, prompt_id: int, tags: list[dict[str, Any]]) -> None:
    conn.execute("DELETE FROM prompt_tags WHERE prompt_id = ?", (prompt_id,))
    for tag in tags:
        if tag.get("id") is None:
            raise ServiceError("INVALID_METADATA", "A tag could not be persisted.")
        conn.execute(
            "INSERT OR IGNORE INTO prompt_tags(prompt_id, tag_id) VALUES (?, ?)",
            (prompt_id, tag["id"]),
        )


def _row_tags(conn: sqlite3.Connection, prompt_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT t.id, t.name
        FROM tags t
        JOIN prompt_tags pt ON pt.tag_id = t.id
        WHERE pt.prompt_id = ?
        ORDER BY t.name COLLATE NOCASE
        """,
        (prompt_id,),
    ).fetchall()
    return [{"id": int(row["id"]), "name": row["name"]} for row in rows]


def _lookup_clause(identifier: Any) -> tuple[str, Any]:
    if isinstance(identifier, int) and not isinstance(identifier, bool):
        return "p.id = ?", identifier
    text = str(identifier or "").strip()
    if not text:
        raise ServiceError("INVALID_IDENTIFIER", "Prompt identifier is required.")
    return "p.sync_id = ?", text


def fetch_prompt(conn: sqlite3.Connection, identifier: Any, *, allow_integer_fallback: bool = True) -> dict[str, Any]:
    clause, value = _lookup_clause(identifier)
    row = conn.execute(
        f"""
        SELECT p.*, c.id AS category_id, tl.id AS tool_id,
               pg.name AS group_name, parent.sync_id AS parent_sync_id,
               (SELECT COUNT(*) FROM prompts child WHERE child.parent_id = p.id) AS child_count
        FROM prompts p
        LEFT JOIN categories c ON c.name = p.category
        LEFT JOIN tools tl ON tl.name = p.tool
        LEFT JOIN prompt_groups pg ON pg.id = p.group_id
        LEFT JOIN prompts parent ON parent.id = p.parent_id
        WHERE {clause}
        LIMIT 1
        """,
        (value,),
    ).fetchone()
    if not row and allow_integer_fallback and isinstance(identifier, str) and identifier.isdigit():
        return fetch_prompt(conn, int(identifier), allow_integer_fallback=False)
    if not row:
        raise ServiceError("PROMPT_NOT_FOUND", "Prompt not found.", 404)
    prompt = dict(row)
    prompt["id"] = int(prompt["id"])
    prompt["category_id"] = int(prompt["category_id"]) if prompt.get("category_id") is not None else None
    prompt["tool_id"] = int(prompt["tool_id"]) if prompt.get("tool_id") is not None else None
    prompt["group_id"] = int(prompt["group_id"]) if prompt.get("group_id") is not None else None
    prompt["parent_id"] = int(prompt["parent_id"]) if prompt.get("parent_id") is not None else None
    prompt["pinned"] = bool(prompt.pop("pinned_at", None))
    prompt["revision"] = int(prompt.get("revision") or 1)
    prompt["prompt_type_id"] = next(
        (index for index, name in enumerate(PROMPT_TYPES, 1) if name.casefold() == str(prompt.get("prompt_type") or "").casefold()),
        None,
    )
    prompt["child_count"] = int(prompt.get("child_count") or 0)
    prompt["group"] = prompt.pop("group_name", None)
    prompt["tags"] = _row_tags(conn, prompt["id"])
    prompt.pop("thumbnail", None)
    return prompt


def _record_audit(
    conn: sqlite3.Connection,
    prompt: dict[str, Any],
    action_type: str,
    source: str,
    change_summary: str | None,
    previous: dict[str, Any] | None,
    new: dict[str, Any] | None,
) -> None:
    conn.execute(
        """
        INSERT INTO integration_prompt_history
            (prompt_id, prompt_sync_id, action_type, source, change_summary,
             previous_json, new_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prompt["id"],
            prompt["sync_id"],
            action_type,
            source,
            change_summary,
            json.dumps(previous, ensure_ascii=False, sort_keys=True) if previous is not None else None,
            json.dumps(new, ensure_ascii=False, sort_keys=True) if new is not None else None,
            utc_now(),
        ),
    )


def _source_and_summary(payload: dict[str, Any]) -> tuple[str, str | None]:
    source = _clean_text(payload.get("source", "integration_api"), "source", required=True, max_length=120)
    summary = _clean_text(payload.get("change_summary"), "change_summary", max_length=1000, allow_none=True)
    return source or "integration_api", summary or None


def _resolve_parent(conn: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("parent_sync_id") is not None:
        return fetch_prompt(conn, payload["parent_sync_id"])
    if payload.get("parent_id") is not None:
        return fetch_prompt(conn, _parse_positive_int(payload["parent_id"], "parent_id"))
    return None


def create_prompt(conn: sqlite3.Connection, payload: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ServiceError("INVALID_JSON", "Request body must be a JSON object.")
    create_missing = _parse_bool(payload.get("create_missing_metadata", False), "create_missing_metadata")
    title = _clean_text(payload.get("title"), "title", required=True, max_length=300)
    content = _clean_text(payload.get("content"), "content", required=True, max_length=500_000)
    notes = _clean_text(payload.get("notes"), "notes", max_length=100_000, allow_none=True)
    category = _metadata_row(
        conn, "categories", payload.get("category_id"), payload.get("category", "Other"), "category",
        required=True, create_missing=create_missing, dry_run=dry_run,
    )
    tool = _metadata_row(
        conn, "tools", payload.get("tool_id"), payload.get("tool", "Generic"), "tool",
        required=True, create_missing=create_missing, dry_run=dry_run,
    )
    prompt_type = _prompt_type(payload.get("prompt_type_id"), payload.get("prompt_type", "Instruction"), required=True)
    group = _metadata_row(
        conn, "prompt_groups", payload.get("group_id"), payload.get("group"), "group",
        required=False, create_missing=create_missing, dry_run=dry_run,
    )
    tags = _resolve_tags(conn, payload.get("tags", []), create_missing=create_missing, dry_run=dry_run)
    parent = _resolve_parent(conn, payload)
    pinned = _parse_bool(payload.get("pinned", False), "pinned")
    source, summary = _source_and_summary(payload)

    sync_id = _clean_text(payload.get("sync_id") or str(uuid.uuid4()).upper(), "sync_id", required=True, max_length=100)
    if conn.execute("SELECT 1 FROM prompts WHERE sync_id = ?", (sync_id,)).fetchone():
        raise ServiceError("DUPLICATE_SYNC_ID", "sync_id already exists.", 409, {"sync_id": sync_id})

    now = utc_now()
    proposed = {
        "id": None,
        "sync_id": sync_id,
        "title": title,
        "content": content,
        "notes": notes or None,
        "category": category["name"],
        "category_id": category.get("id"),
        "tool": tool["name"],
        "tool_id": tool.get("id"),
        "prompt_type": prompt_type["name"],
        "prompt_type_id": prompt_type["id"],
        "tags": tags,
        "group": group["name"] if group else None,
        "group_id": group.get("id") if group else None,
        "parent_id": parent["id"] if parent else None,
        "parent_sync_id": parent["sync_id"] if parent else None,
        "pinned": pinned,
        "created_at": now,
        "updated_at": now,
        "revision": 1,
        "child_count": 0,
        "source": source,
    }
    if dry_run:
        return {"dry_run": True, "applied": False, "before": None, "after": proposed}

    cur = conn.execute(
        """
        INSERT INTO prompts
            (title, category, tool, prompt_type, content, notes, thumbnail, parent_id,
             group_id, sync_id, pinned_at, created_at, updated_at, revision, source)
        VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, 1, ?)
        """,
        (
            title, category["name"], tool["name"], prompt_type["name"], content, notes or None,
            parent["id"] if parent else None, group.get("id") if group else None, sync_id,
            now if pinned else None, now, now, source,
        ),
    )
    _set_prompt_tags(conn, int(cur.lastrowid), tags)
    created = fetch_prompt(conn, int(cur.lastrowid))
    _record_audit(conn, created, "create", source, summary, None, created)
    return {"dry_run": False, "applied": True, "prompt": created}


def _check_conflict(current: dict[str, Any], payload: dict[str, Any]) -> None:
    expected_revision = payload.get("expected_revision")
    expected_updated_at = payload.get("expected_updated_at")
    if expected_revision is None and expected_updated_at is None:
        raise ServiceError(
            "PRECONDITION_REQUIRED",
            "expected_revision or expected_updated_at is required.",
            428,
        )
    if expected_revision is not None:
        parsed = _parse_positive_int(expected_revision, "expected_revision")
        if parsed != current["revision"]:
            raise ServiceError(
                "STALE_UPDATE",
                "The prompt has changed since it was retrieved.",
                409,
                {"current_revision": current["revision"], "current_updated_at": current["updated_at"]},
            )
    if expected_updated_at is not None:
        expected = _clean_text(expected_updated_at, "expected_updated_at", required=True, max_length=100)
        if expected != current["updated_at"]:
            raise ServiceError(
                "STALE_UPDATE",
                "The prompt has changed since it was retrieved.",
                409,
                {"current_revision": current["revision"], "current_updated_at": current["updated_at"]},
            )


def update_prompt(
    conn: sqlite3.Connection,
    identifier: Any,
    payload: dict[str, Any],
    *,
    dry_run: bool = False,
    action_type: str = "update",
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ServiceError("INVALID_JSON", "Request body must be a JSON object.")
    current = fetch_prompt(conn, identifier)
    _check_conflict(current, payload)
    create_missing = _parse_bool(payload.get("create_missing_metadata", False), "create_missing_metadata")
    source, summary = _source_and_summary(payload)
    after = dict(current)

    text_fields = {
        "title": (True, 300),
        "content": (True, 500_000),
        "notes": (False, 100_000),
    }
    for field, (required, limit) in text_fields.items():
        if field in payload:
            value = _clean_text(payload[field], field, required=required, max_length=limit, allow_none=not required)
            after[field] = value or (None if not required else value)

    if "category" in payload or "category_id" in payload:
        category = _metadata_row(
            conn, "categories", payload.get("category_id"), payload.get("category"), "category",
            required=True, create_missing=create_missing, dry_run=dry_run,
        )
        after["category"], after["category_id"] = category["name"], category.get("id")
    if "tool" in payload or "tool_id" in payload:
        tool = _metadata_row(
            conn, "tools", payload.get("tool_id"), payload.get("tool"), "tool",
            required=True, create_missing=create_missing, dry_run=dry_run,
        )
        after["tool"], after["tool_id"] = tool["name"], tool.get("id")
    if "prompt_type" in payload or "prompt_type_id" in payload:
        prompt_type = _prompt_type(payload.get("prompt_type_id"), payload.get("prompt_type"), required=True)
        after["prompt_type"], after["prompt_type_id"] = prompt_type["name"], prompt_type["id"]
    if "group" in payload or "group_id" in payload:
        if payload.get("group") is None and payload.get("group_id") is None:
            after["group"], after["group_id"] = None, None
        else:
            group = _metadata_row(
                conn, "prompt_groups", payload.get("group_id"), payload.get("group"), "group",
                required=False, create_missing=create_missing, dry_run=dry_run,
            )
            after["group"], after["group_id"] = (group["name"], group.get("id")) if group else (None, None)
    if "tags" in payload:
        after["tags"] = _resolve_tags(
            conn, payload.get("tags"), create_missing=create_missing, dry_run=dry_run
        )
    if "pinned" in payload:
        after["pinned"] = _parse_bool(payload["pinned"], "pinned")

    ignored = {
        "expected_revision", "expected_updated_at", "dry_run", "source", "change_summary",
        "create_missing_metadata", "category_id", "tool_id", "prompt_type_id", "group_id",
    }
    allowed = set(text_fields) | {"category", "tool", "prompt_type", "group", "tags", "pinned"} | ignored
    unsupported = sorted(set(payload) - allowed)
    if unsupported:
        raise ServiceError("UNSUPPORTED_FIELD", "One or more fields cannot be updated.", details={"fields": unsupported})

    comparable_fields = (
        "title", "content", "notes", "category", "category_id", "tool", "tool_id",
        "prompt_type", "prompt_type_id", "group", "group_id", "tags", "pinned",
    )
    changed = any(current.get(field) != after.get(field) for field in comparable_fields)
    if changed:
        after["revision"] = current["revision"] + 1
        after["updated_at"] = utc_now()
        after["source"] = source

    if dry_run:
        return {"dry_run": True, "applied": False, "changed": changed, "before": current, "after": after}
    if not changed:
        return {"dry_run": False, "applied": True, "changed": False, "prompt": current}

    conn.execute(
        """
        UPDATE prompts
        SET title = ?, content = ?, notes = ?, category = ?, tool = ?, prompt_type = ?,
            group_id = ?, pinned_at = CASE WHEN ? THEN COALESCE(pinned_at, ?) ELSE NULL END,
            updated_at = ?, revision = ?, source = ?
        WHERE id = ? AND revision = ?
        """,
        (
            after["title"], after["content"], after.get("notes"), after["category"], after["tool"],
            after["prompt_type"], after.get("group_id"), 1 if after["pinned"] else 0, after["updated_at"],
            after["updated_at"], after["revision"], source, current["id"], current["revision"],
        ),
    )
    if conn.execute("SELECT changes()").fetchone()[0] != 1:
        raise ServiceError("STALE_UPDATE", "The prompt changed while the update was being applied.", 409)
    _set_prompt_tags(conn, current["id"], after["tags"])
    updated = fetch_prompt(conn, current["id"])
    _record_audit(conn, updated, action_type, source, summary, current, updated)
    return {"dry_run": False, "applied": True, "changed": True, "prompt": updated}


def create_version(conn: sqlite3.Connection, identifier: Any, payload: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    parent = fetch_prompt(conn, identifier)
    if payload.get("expected_revision") is not None or payload.get("expected_updated_at") is not None:
        _check_conflict(parent, payload)
    version_payload = {
        "title": payload.get("title", f"{parent['title']} (Version)"),
        "content": payload.get("content", parent["content"]),
        "notes": payload.get("notes", parent.get("notes")),
        "category_id": payload.get("category_id", parent.get("category_id")),
        "tool_id": payload.get("tool_id", parent.get("tool_id")),
        "prompt_type_id": payload.get("prompt_type_id", parent.get("prompt_type_id")),
        "group_id": payload.get("group_id", parent.get("group_id")),
        "tags": payload.get("tags", parent.get("tags", [])),
        "pinned": payload.get("pinned", False),
        "parent_id": parent["parent_id"] or parent["id"],
        "source": payload.get("source", "integration_api"),
        "change_summary": payload.get("change_summary"),
        "create_missing_metadata": payload.get("create_missing_metadata", False),
    }
    for name_field in ("category", "tool", "prompt_type", "group"):
        if name_field in payload:
            version_payload[name_field] = payload[name_field]
            version_payload.pop(f"{name_field}_id", None)
    result = create_prompt(conn, version_payload, dry_run=dry_run)
    if dry_run:
        result["parent"] = {"id": parent["id"], "sync_id": parent["sync_id"], "revision": parent["revision"]}
        return result
    created = result["prompt"]
    source, summary = _source_and_summary(payload)
    conn.execute(
        "UPDATE integration_prompt_history SET action_type = 'version_create', source = ?, change_summary = ? "
        "WHERE id = (SELECT MAX(id) FROM integration_prompt_history WHERE prompt_id = ?)",
        (source, summary, created["id"]),
    )
    return {
        "dry_run": False,
        "applied": True,
        "prompt": created,
        "parent": {"id": parent["id"], "sync_id": parent["sync_id"], "revision": parent["revision"]},
    }


def organise_prompt(conn: sqlite3.Connection, identifier: Any, payload: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ServiceError("INVALID_JSON", "Request body must be a JSON object.")
    current = fetch_prompt(conn, identifier)
    allowed = {
        "expected_revision", "expected_updated_at", "source", "change_summary", "dry_run",
        "create_missing_metadata", "category", "category_id", "tool", "tool_id",
        "prompt_type", "prompt_type_id", "group", "group_id", "pinned", "pin", "unpin",
        "add_tags", "remove_tags", "replace_tags",
    }
    unsupported = sorted(set(payload) - allowed)
    if unsupported:
        raise ServiceError("UNSUPPORTED_FIELD", "One or more organisation actions are unsupported.", details={"fields": unsupported})
    if "pin" in payload and "unpin" in payload:
        raise ServiceError("VALIDATION_ERROR", "Use pin or unpin, not both.")
    update_payload = {
        key: payload[key]
        for key in (
            "expected_revision", "expected_updated_at", "source", "change_summary",
            "create_missing_metadata", "category", "category_id", "tool", "tool_id",
            "prompt_type", "prompt_type_id", "group", "group_id", "pinned",
        )
        if key in payload
    }
    tags = current["tags"]
    tag_action_fields = [key for key in ("add_tags", "remove_tags", "replace_tags") if key in payload]
    if len(tag_action_fields) > 1:
        raise ServiceError("VALIDATION_ERROR", "Use only one tag action per request.")
    if "replace_tags" in payload:
        update_payload["tags"] = payload["replace_tags"]
    elif "add_tags" in payload:
        update_payload["tags"] = tags + _normalise_tags(payload["add_tags"])
    elif "remove_tags" in payload:
        removals = _normalise_tags(payload["remove_tags"])
        removal_names = {
            str(item.get("name") if isinstance(item, dict) else item).strip().casefold()
            for item in removals
            if not isinstance(item, int)
        }
        removal_ids = {
            int(item.get("id") if isinstance(item, dict) else item)
            for item in removals
            if isinstance(item, int) or (isinstance(item, dict) and item.get("id") is not None)
        }
        update_payload["tags"] = [
            tag for tag in tags
            if tag["id"] not in removal_ids and tag["name"].casefold() not in removal_names
        ]
    if payload.get("pin") is not None:
        update_payload["pinned"] = _parse_bool(payload["pin"], "pin")
    if payload.get("unpin") is not None and _parse_bool(payload["unpin"], "unpin"):
        update_payload["pinned"] = False
    return update_prompt(conn, identifier, update_payload, dry_run=dry_run, action_type="organise")


def search_prompts(conn: sqlite3.Connection, filters: dict[str, Any]) -> dict[str, Any]:
    try:
        page = max(1, int(filters.get("page", 1)))
        page_size = max(1, min(MAX_PAGE_SIZE, int(filters.get("page_size", DEFAULT_PAGE_SIZE))))
    except (TypeError, ValueError):
        raise ServiceError("INVALID_PAGINATION", "page and page_size must be integers.")

    include_content = _parse_bool(filters.get("include_content", False), "include_content")
    where = ["1=1"]
    params: list[Any] = []

    query = _clean_text(filters.get("query"), "query", max_length=500, allow_none=True)
    if query:
        for term in [part.strip() for part in query.split(",") if part.strip()]:
            like = f"%{term.lower()}%"
            where.append(
                "(LOWER(p.title) LIKE ? OR LOWER(p.content) LIKE ? OR LOWER(COALESCE(p.notes,'')) LIKE ? "
                "OR EXISTS (SELECT 1 FROM prompt_tags qpt JOIN tags qt ON qt.id=qpt.tag_id "
                "WHERE qpt.prompt_id=p.id AND LOWER(qt.name) LIKE ?))"
            )
            params.extend([like, like, like, like])

    exact_filters = (
        ("category", "p.category"),
        ("tool", "p.tool"),
        ("prompt_type", "p.prompt_type"),
    )
    for key, column in exact_filters:
        if filters.get(key) not in (None, ""):
            value = _clean_text(filters[key], key, required=True, max_length=200)
            where.append(f"{column} = ? COLLATE NOCASE")
            params.append(value)

    id_filters = (
        ("category_id", "categories", "p.category"),
        ("tool_id", "tools", "p.tool"),
    )
    for key, table, column in id_filters:
        if filters.get(key) not in (None, ""):
            metadata_id = _parse_positive_int(filters[key], key)
            where.append(f"{column} = (SELECT name FROM {table} WHERE id = ?)")
            params.append(metadata_id)
    if filters.get("prompt_type_id") not in (None, ""):
        prompt_type = _prompt_type(filters["prompt_type_id"], None, required=True)
        where.append("p.prompt_type = ?")
        params.append(prompt_type["name"])

    if filters.get("tag") not in (None, ""):
        tag = _clean_text(filters["tag"], "tag", required=True, max_length=200)
        where.append(
            "EXISTS (SELECT 1 FROM prompt_tags pt JOIN tags t ON t.id=pt.tag_id "
            "WHERE pt.prompt_id=p.id AND t.name = ? COLLATE NOCASE)"
        )
        params.append(tag)
    if filters.get("tag_id") not in (None, ""):
        where.append("EXISTS (SELECT 1 FROM prompt_tags pt WHERE pt.prompt_id=p.id AND pt.tag_id=?)")
        params.append(_parse_positive_int(filters["tag_id"], "tag_id"))
    if filters.get("group") not in (None, ""):
        group = _clean_text(filters["group"], "group", required=True, max_length=200)
        where.append("p.group_id = (SELECT id FROM prompt_groups WHERE name = ? COLLATE NOCASE)")
        params.append(group)
    if filters.get("group_id") not in (None, ""):
        where.append("p.group_id = ?")
        params.append(_parse_positive_int(filters["group_id"], "group_id"))
    if filters.get("parent_id") not in (None, ""):
        where.append("p.parent_id = ?")
        params.append(_parse_positive_int(filters["parent_id"], "parent_id"))
    if filters.get("pinned") not in (None, ""):
        pinned = _parse_bool(filters["pinned"], "pinned")
        where.append("p.pinned_at IS NOT NULL" if pinned else "p.pinned_at IS NULL")
    for key, operator in (("updated_after", ">="), ("updated_before", "<=")):
        if filters.get(key) not in (None, ""):
            value = _clean_text(filters[key], key, required=True, max_length=100)
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                raise ServiceError("INVALID_DATE", f"{key} must be an ISO-8601 timestamp.")
            where.append(f"p.updated_at {operator} ?")
            params.append(value)

    sort = str(filters.get("sort") or "updated_at")
    allowed_sort = {"updated_at", "created_at", "title", "category", "tool", "prompt_type", "revision"}
    if sort not in allowed_sort:
        raise ServiceError("INVALID_SORT", "Unsupported sort field.", details={"allowed": sorted(allowed_sort)})
    order = str(filters.get("order") or "desc").lower()
    if order not in {"asc", "desc"}:
        raise ServiceError("INVALID_SORT", "order must be asc or desc.")
    direction = order.upper()
    where_sql = " AND ".join(where)
    total = conn.execute(f"SELECT COUNT(*) FROM prompts p WHERE {where_sql}", params).fetchone()[0]
    content_column = "p.content" if include_content else "NULL AS content"
    rows = conn.execute(
        f"""
        SELECT p.id, p.sync_id, p.title, {content_column}, p.notes, p.category, p.tool,
               p.prompt_type, p.group_id, p.parent_id, p.pinned_at, p.created_at,
               p.updated_at, p.revision, p.source, c.id AS category_id, tl.id AS tool_id,
               pg.name AS group_name, parent.sync_id AS parent_sync_id,
               (SELECT COUNT(*) FROM prompts child WHERE child.parent_id=p.id) AS child_count
        FROM prompts p
        LEFT JOIN categories c ON c.name=p.category
        LEFT JOIN tools tl ON tl.name=p.tool
        LEFT JOIN prompt_groups pg ON pg.id=p.group_id
        LEFT JOIN prompts parent ON parent.id=p.parent_id
        WHERE {where_sql}
        ORDER BY CASE WHEN p.pinned_at IS NULL THEN 1 ELSE 0 END ASC,
                 p.pinned_at DESC, p.{sort} {direction}, p.id {direction}
        LIMIT ? OFFSET ?
        """,
        params + [page_size, (page - 1) * page_size],
    ).fetchall()
    prompts: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["pinned"] = bool(item.pop("pinned_at", None))
        item["group"] = item.pop("group_name", None)
        item["revision"] = int(item.get("revision") or 1)
        item["child_count"] = int(item.get("child_count") or 0)
        item["tags"] = _row_tags(conn, int(item["id"]))
        if not include_content:
            item.pop("content", None)
        prompts.append(item)
    pages = (total + page_size - 1) // page_size if total else 0
    return {
        "prompts": prompts,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": int(total),
            "total_pages": pages,
            "has_next": page < pages,
            "has_previous": page > 1 and pages > 0,
        },
    }


def metadata(conn: sqlite3.Connection) -> dict[str, Any]:
    def named_counts(table: str, prompt_column: str | None = None) -> list[dict[str, Any]]:
        if prompt_column:
            rows = conn.execute(
                f"""
                SELECT m.id, m.name, COUNT(p.id) AS prompt_count
                FROM {table} m
                LEFT JOIN prompts p ON p.{prompt_column}=m.name
                GROUP BY m.id, m.name ORDER BY m.name COLLATE NOCASE
                """
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT id, name, 0 AS prompt_count FROM {table} ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return [dict(row) for row in rows]

    tags = conn.execute(
        """
        SELECT t.id, t.name, COUNT(pt.prompt_id) AS prompt_count
        FROM tags t LEFT JOIN prompt_tags pt ON pt.tag_id=t.id
        GROUP BY t.id, t.name ORDER BY t.name COLLATE NOCASE
        """
    ).fetchall()
    groups = conn.execute(
        """
        SELECT g.id, g.name, g.description, COUNT(p.id) AS prompt_count
        FROM prompt_groups g LEFT JOIN prompts p ON p.group_id=g.id
        GROUP BY g.id, g.name, g.description ORDER BY g.name COLLATE NOCASE
        """
    ).fetchall()
    type_counts = {
        row["prompt_type"]: int(row["count"])
        for row in conn.execute("SELECT prompt_type, COUNT(*) count FROM prompts GROUP BY prompt_type")
    }
    return {
        "categories": named_counts("categories", "category"),
        "tools": named_counts("tools", "tool"),
        "prompt_types": [
            {"id": index, "name": name, "prompt_count": type_counts.get(name, 0)}
            for index, name in enumerate(PROMPT_TYPES, 1)
        ],
        "tags": [dict(row) for row in tags],
        "groups": [dict(row) for row in groups],
    }
