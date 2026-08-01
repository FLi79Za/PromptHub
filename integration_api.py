"""Versioned JSON-only Integration API blueprint for PromptHub."""

from __future__ import annotations

import hmac
import logging
import sqlite3
import time
import uuid
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, Response, g, jsonify, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from integration_config import get_or_create_token, load_config
from prompt_service import (
    INTEGRATION_SCHEMA_VERSION,
    ServiceError,
    create_prompt,
    create_version,
    fetch_prompt,
    integration_schema_ready,
    metadata,
    organise_prompt,
    search_prompts,
    update_prompt,
)


API_VERSION = "1.0.0"
BASE_PATH = "/api/integration/v1"
logger = logging.getLogger("prompthub.integration")


def _meta(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    value = {
        "api_version": API_VERSION,
        "request_id": getattr(g, "integration_request_id", None),
    }
    if extra:
        value.update(extra)
    return value


def success(data: Any, status: int = 200, meta: dict[str, Any] | None = None) -> tuple[Response, int]:
    return jsonify({"success": True, "data": data, "error": None, "meta": _meta(meta)}), status


def failure(
    code: str,
    message: str,
    status: int,
    details: dict[str, Any] | None = None,
) -> tuple[Response, int]:
    g.integration_result_code = code
    return jsonify(
        {
            "success": False,
            "data": None,
            "error": {"code": code, "message": message, "details": details or {}},
            "meta": _meta(),
        }
    ), status


def _json_body() -> dict[str, Any]:
    try:
        value = request.get_json(silent=False)
    except BadRequest as exc:
        raise ServiceError("INVALID_JSON", "Request body contains invalid JSON.") from exc
    if not isinstance(value, dict):
        raise ServiceError("INVALID_JSON", "Request body must be a JSON object.")
    return value


def _dry_run(payload: dict[str, Any]) -> bool:
    value = payload.get("dry_run", False)
    if isinstance(value, bool):
        return value
    raise ServiceError("INVALID_FIELD_TYPE", "dry_run must be a boolean.", details={"field": "dry_run"})


def create_integration_blueprint(get_db: Callable[[], sqlite3.Connection]) -> Blueprint:
    api = Blueprint("integration_api_v1", __name__, url_prefix=BASE_PATH)

    @api.before_request
    def integration_before_request():
        g.integration_started = time.perf_counter()
        g.integration_request_id = uuid.uuid4().hex
        g.integration_result_code = "OK"
        g.integration_action = request.endpoint or "unknown"
        config = load_config()

        if not config["api_enabled"]:
            return failure("API_DISABLED", "The PromptHub Integration API is disabled.", 503)

        content_length = request.content_length or 0
        if content_length > config["max_json_bytes"]:
            return failure(
                "REQUEST_TOO_LARGE",
                "JSON request exceeds the configured size limit.",
                413,
                {"max_json_bytes": config["max_json_bytes"]},
            )

        if request.method in {"POST", "PATCH", "PUT"} and not request.is_json:
            return failure("UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/json.", 415)

        if request.method == "OPTIONS" or request.endpoint == "integration_api_v1.health":
            return None

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            g.integration_result_code = "AUTH_REQUIRED"
            return failure("AUTH_REQUIRED", "Bearer token authentication is required.", 401)
        supplied = auth_header[7:].strip()
        expected = get_or_create_token()
        if not supplied or not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
            g.integration_result_code = "INVALID_TOKEN"
            return failure("INVALID_TOKEN", "The bearer token is invalid.", 401)

        is_write = request.method in {"POST", "PATCH", "PUT", "DELETE"}
        if is_write and not config["write_enabled"]:
            return failure("WRITE_ACCESS_DISABLED", "Integration API write access is disabled.", 403)
        if not is_write and not config["read_enabled"]:
            return failure("READ_ACCESS_DISABLED", "Integration API read access is disabled.", 403)

        with get_db() as conn:
            if not integration_schema_ready(conn):
                return failure(
                    "SCHEMA_NOT_READY",
                    "The Integration API database migration has not been applied.",
                    503,
                    {"required_schema_version": INTEGRATION_SCHEMA_VERSION},
                )
        return None

    @api.after_request
    def integration_after_request(response: Response):
        config = load_config()
        origin = request.headers.get("Origin")
        if origin and origin in config["allowed_origins"]:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, OPTIONS"

        duration_ms = (time.perf_counter() - getattr(g, "integration_started", time.perf_counter())) * 1000
        prompt_identifier = None
        if request.view_args:
            prompt_identifier = request.view_args.get("identifier")
        logger.info(
            "integration_request method=%s route=%s status=%s prompt_identifier=%s action=%s result=%s duration_ms=%.2f",
            request.method,
            request.path,
            response.status_code,
            prompt_identifier or "-",
            getattr(g, "integration_action", "unknown"),
            getattr(g, "integration_result_code", "UNKNOWN"),
            duration_ms,
        )
        return response

    @api.errorhandler(ServiceError)
    def handle_service_error(exc: ServiceError):
        return failure(exc.code, exc.message, exc.status, exc.details)

    @api.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(_exc: RequestEntityTooLarge):
        return failure("REQUEST_TOO_LARGE", "JSON request exceeds the configured size limit.", 413)

    @api.errorhandler(sqlite3.DatabaseError)
    def handle_database_error(_exc: sqlite3.DatabaseError):
        logger.exception("Integration API database operation failed")
        return failure("DATABASE_ERROR", "The database operation could not be completed.", 500)

    @api.errorhandler(Exception)
    def handle_unexpected_error(_exc: Exception):
        logger.exception("Unexpected Integration API failure")
        return failure("INTERNAL_ERROR", "The request could not be completed.", 500)

    @api.route("/health", methods=["GET"])
    def health():
        config = load_config()
        database_connected = False
        schema_ready = False
        try:
            with get_db() as conn:
                database_connected = conn.execute("SELECT 1").fetchone()[0] == 1
                schema_ready = integration_schema_ready(conn)
        except sqlite3.DatabaseError:
            pass
        data = {
            "api_version": API_VERSION,
            "application_version": None,
            "database": {
                "connected": database_connected,
                "schema_ready": schema_ready,
                "schema_version": INTEGRATION_SCHEMA_VERSION if schema_ready else None,
            },
            "availability": {
                "read": bool(config["api_enabled"] and config["read_enabled"] and database_connected and schema_ready),
                "write": bool(config["api_enabled"] and config["write_enabled"] and database_connected and schema_ready),
            },
            "authentication": {"required": True, "scheme": "Bearer", "configured": True},
            "network": {"host": config["host"], "local_only": config["host"] in {"127.0.0.1", "localhost", "::1"}},
            "capabilities": [
                "prompt_search", "prompt_retrieval", "prompt_create", "prompt_update",
                "optimistic_concurrency", "related_versions", "prompt_organisation",
                "metadata", "dry_run", "integration_history",
            ],
        }
        status = 200 if database_connected and schema_ready else 503
        return success(data, status=status)

    @api.route("/prompts", methods=["GET"])
    def list_prompts():
        with get_db() as conn:
            result = search_prompts(conn, request.args.to_dict(flat=True))
        pagination = result.pop("pagination")
        return success(result, meta={"pagination": pagination})

    @api.route("/prompts/<identifier>", methods=["GET"])
    def get_one_prompt(identifier: str):
        with get_db() as conn:
            prompt = fetch_prompt(conn, identifier)
        return success({"prompt": prompt})

    @api.route("/prompts", methods=["POST"])
    def post_prompt():
        payload = _json_body()
        with get_db() as conn:
            result = create_prompt(conn, payload, dry_run=_dry_run(payload))
        return success(result, status=200 if result.get("dry_run") else 201)

    @api.route("/prompts/<identifier>", methods=["PATCH"])
    def patch_prompt(identifier: str):
        payload = _json_body()
        with get_db() as conn:
            result = update_prompt(conn, identifier, payload, dry_run=_dry_run(payload))
        return success(result)

    @api.route("/prompts/<identifier>/versions", methods=["POST"])
    def post_version(identifier: str):
        payload = _json_body()
        with get_db() as conn:
            result = create_version(conn, identifier, payload, dry_run=_dry_run(payload))
        return success(result, status=200 if result.get("dry_run") else 201)

    @api.route("/prompts/<identifier>/organise", methods=["POST"])
    def post_organise(identifier: str):
        payload = _json_body()
        with get_db() as conn:
            result = organise_prompt(conn, identifier, payload, dry_run=_dry_run(payload))
        return success(result)

    @api.route("/prompts/<identifier>/history", methods=["GET"])
    def get_integration_history(identifier: str):
        with get_db() as conn:
            prompt = fetch_prompt(conn, identifier)
            rows = conn.execute(
                """
                SELECT id, prompt_id, prompt_sync_id, action_type, source, change_summary,
                       previous_json, new_json, created_at
                FROM integration_prompt_history
                WHERE prompt_id = ? ORDER BY created_at DESC, id DESC
                LIMIT 100
                """,
                (prompt["id"],),
            ).fetchall()
        history = []
        import json
        for row in rows:
            item = dict(row)
            item["previous"] = json.loads(item.pop("previous_json")) if item.get("previous_json") else None
            item["new"] = json.loads(item.pop("new_json")) if item.get("new_json") else None
            history.append(item)
        return success({"prompt": {"id": prompt["id"], "sync_id": prompt["sync_id"]}, "history": history})

    @api.route("/metadata", methods=["GET"])
    def get_metadata():
        with get_db() as conn:
            result = metadata(conn)
        return success(result)

    return api
