"""Versioned JSON-only Integration API blueprint for PromptHub."""

from __future__ import annotations

import hmac
import base64
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
from ai_library import (
    AILibraryError,
    audit_collection,
    collection_snapshot,
    create_action,
    create_collection,
    create_resource,
    duplicate_collection,
    list_collection_snapshots,
    prepare_document,
    preview_document_import,
    rebuild_collection,
    rebuild_document,
    retrieve_knowledge,
    save_prepared_document,
    suggested_action_resources,
    update_collection,
)
from skill_runtime import SkillError, compare_skill, configure_provider, export_skill, import_skill, inspect_skill, list_prompt_derivations, list_skills, run_skill, save_skill_derivative, source_prompt_input, update_skill
from generation_runtime import (
    GenerationError, check_server, compatible_profiles, duplicate_profile, get_generation,
    inspect_workflow, list_profiles, list_prompt_generations, list_servers, refresh_generation,
    regenerate, save_profile, submit_generation, upsert_server, validate_saved_profile,
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


def create_integration_blueprint(
    get_db: Callable[[], sqlite3.Connection],
    *,
    embedder: Callable[[str, str], list[float]] | None = None,
    list_models: Callable[[], list[str]] | None = None,
    skill_generate: Callable[..., str] | None = None,
    skill_base_dir: str | None = None,
    workflow_root: str | None = None,
    generation_media_root: str | None = None,
) -> Blueprint:
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
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, DELETE, OPTIONS"

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

    @api.errorhandler(AILibraryError)
    def handle_ai_library_error(exc: AILibraryError):
        return failure("KNOWLEDGE_ERROR", str(exc), 400)

    @api.errorhandler(GenerationError)
    def handle_generation_error(exc: GenerationError):
        status = 404 if exc.code.endswith("NOT_FOUND") else 409 if exc.code in {"DUPLICATE_PROFILE", "PROFILE_INVALIDATED"} else 400
        return failure(exc.code, str(exc), status, exc.details)

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
                "knowledge_collections", "knowledge_source_import", "knowledge_retrieval_test",
                "knowledge_audit", "knowledge_rebuild", "knowledge_action_bundle",
                "skills", "skill_import", "skill_inspection", "skill_runtime", "skill_export",
                "skill_operations", "prompt_skill_derivations",
                "comfyui_servers", "workflow_profiles", "workflow_import", "generation_dispatch",
                "generation_status", "generation_history", "generation_regenerate",
            ],
        }
        status = 200 if database_connected and schema_ready else 503
        return success(data, status=status)

    @api.route("/comfyui/servers", methods=["GET", "POST"])
    def api_comfyui_servers():
        with get_db() as conn:
            if request.method == "GET":
                return success({"servers": list_servers(conn)})
            return success(upsert_server(conn, _json_body()))

    @api.route("/comfyui/servers/<server_id>/health", methods=["POST"])
    def api_comfyui_server_health(server_id: str):
        with get_db() as conn:
            return success(check_server(conn, server_id))

    @api.route("/comfyui/workflows/inspect", methods=["POST"])
    def api_comfyui_workflow_inspect():
        graph = _json_body().get("workflow")
        return success({"inspection": inspect_workflow(graph)})

    @api.route("/comfyui/profiles", methods=["GET", "POST"])
    def api_comfyui_profiles():
        with get_db() as conn:
            if request.method == "GET":
                return success({"profiles": list_profiles(conn)})
            data = _json_body()
            return success(save_profile(conn, data, data.get("workflow"), workflow_root or "provider_workflows/comfyui_profiles", allow_update=bool(data.get("allow_update"))))

    @api.route("/comfyui/profiles/<profile_id>/duplicate", methods=["POST"])
    def api_comfyui_profile_duplicate(profile_id: str):
        data = _json_body()
        with get_db() as conn:
            return success(duplicate_profile(conn, profile_id, str(data.get("id") or ""), str(data.get("display_name") or ""), workflow_root or "provider_workflows/comfyui_profiles"))

    @api.route("/comfyui/profiles/<profile_id>/validate", methods=["POST"])
    def api_comfyui_profile_validate(profile_id: str):
        data = _json_body()
        with get_db() as conn:
            return success({"validation": validate_saved_profile(conn, profile_id, live=bool(data.get("live")))})

    @api.route("/prompts/<identifier>/compatible-workflows", methods=["GET"])
    def api_compatible_workflows(identifier: str):
        with get_db() as conn:
            return success({"profiles": compatible_profiles(conn, identifier, show_all=request.args.get("show_all") == "1")})

    @api.route("/prompts/<identifier>/generations", methods=["GET", "POST"])
    def api_prompt_generations(identifier: str):
        with get_db() as conn:
            if request.method == "GET":
                return success({"generations": list_prompt_generations(conn, identifier)})
            data = _json_body()
            generation = submit_generation(
                conn,
                identifier,
                str(data.get("profile_id") or ""),
                data.get("inputs") or {},
                prompt_override=data.get("prompt_override"),
            )
            return success({"generation": generation}, status=202)

    @api.route("/generations/<generation_id>", methods=["GET"])
    def api_generation_status(generation_id: str):
        with get_db() as conn:
            generation = refresh_generation(conn, generation_id, generation_media_root or "static/uploads/generations")
            return success({"generation": generation})

    @api.route("/generations/<generation_id>/regenerate", methods=["POST"])
    def api_generation_regenerate(generation_id: str):
        data = _json_body()
        with get_db() as conn:
            generation = regenerate(conn, generation_id, new_seed=bool(data.get("new_seed")), current_prompt=bool(data.get("current_prompt")))
            return success({"generation": generation}, status=202)

    @api.route("/skills", methods=["GET"])
    def api_list_skills():
        with get_db() as conn:
            return success({"skills": list_skills(conn, str(request.args.get("q") or "").strip())})

    @api.route("/skills/inspect", methods=["POST"])
    def api_inspect_skill():
        try:
            data = request.get_json(silent=True) or {}
            return success({"inspection": inspect_skill(str(data.get("source") or ""))})
        except SkillError as exc:
            return failure(exc.code, str(exc), 400, exc.details)

    @api.route("/skills/import", methods=["POST"])
    def api_import_skill():
        data = request.get_json(silent=True) or {}
        try:
            with get_db() as conn:
                result = import_skill(conn, str(data.get("source") or ""), base_dir=skill_base_dir or ".", source_type=str(data.get("source_type") or "portable"), source_platform=str(data.get("source_platform") or "unknown"), tags=data.get("tags") or [])
            return success(result, status=409 if result.get("operation") == "conflict" else 200)
        except SkillError as exc:
            return failure(exc.code, str(exc), 400, exc.details)

    @api.route("/skills/<skill_id>", methods=["GET"])
    def api_get_skill(skill_id: str):
        with get_db() as conn:
            items = [item for item in list_skills(conn) if item["id"] == skill_id or item["name"].lower() == skill_id.lower()]
        if not items:
            return failure("SKILL_NOT_FOUND", "Skill was not found.", 404)
        return success(items[0])

    @api.route("/skills/<skill_id>/run", methods=["POST"])
    def api_run_skill(skill_id: str):
        if skill_generate is None:
            return failure("SKILL_RUNTIME_UNAVAILABLE", "Skill runtime is not configured.", 503)
        data = request.get_json(silent=True) or {}
        request_text, model = str(data.get("request") or "").strip(), str(data.get("model") or "").strip()
        if not model:
            return failure("INVALID_REQUEST", "execution model is required.", 400)
        if list_models is not None and model not in list_models():
            return failure("MODEL_NOT_INSTALLED", f"The selected Ollama model is not installed: {model}", 400)
        try:
            with get_db() as conn:
                inputs = list(data.get("inputs") or [])
                source_prompt_id = data.get("source_prompt_id")
                operation = str(data.get("operation") or "create").lower()
                if source_prompt_id and operation != "create" and not any(isinstance(item, dict) and item.get("role") in {"source", "draft"} for item in inputs):
                    inputs.insert(0, source_prompt_input(conn, source_prompt_id))
                source_row = conn.execute("SELECT id FROM prompts WHERE id=? OR sync_id=?", (int(source_prompt_id) if str(source_prompt_id or "").isdigit() else -1, str(source_prompt_id or ""))).fetchone() if source_prompt_id else None
                if data.get("freeform_instruction") and not any(isinstance(item, dict) and item.get("role") == "instruction" for item in inputs):
                    inputs.append({"type": "text", "role": "instruction", "content": str(data["freeform_instruction"]).strip()})
                result = run_skill(conn, skill_id, request_text, model, data.get("parameters") or {}, skill_generate, operation=operation, inputs=inputs or None, target=data.get("target"), source_prompt_id=int(source_row["id"]) if source_row else None, capability_registry={"structured_output": {"status": "PRESERVED", "provider": "Ollama"}, "image_generation": {"status": "SUBSTITUTED", "provider": "ComfyUI"}})
            return success(result)
        except SkillError as exc:
            return failure(exc.code, str(exc), 400, exc.details)

    @api.route("/prompts/<identifier>/skill-derivations", methods=["GET", "POST"])
    def api_prompt_skill_derivations(identifier: str):
        try:
            with get_db() as conn:
                if request.method == "GET":
                    return success({"derivations": list_prompt_derivations(conn, identifier)})
                data = request.get_json(silent=True) or {}
                result = save_skill_derivative(conn, identifier, str(data.get("execution_id") or ""), title=data.get("title"), replace_original=bool(data.get("replace_original", False)))
            return success(result)
        except SkillError as exc:
            return failure(exc.code, str(exc), 400, exc.details)

    @api.route("/skills/<skill_id>/compare", methods=["POST"])
    def api_compare_skill(skill_id: str):
        data = request.get_json(silent=True) or {}
        try:
            with get_db() as conn:
                return success({"comparison": compare_skill(conn, skill_id, str(data.get("source") or ""))})
        except SkillError as exc:
            return failure(exc.code, str(exc), 409 if exc.code == "SKILL_UPDATE_CONFLICT" else 400, exc.details)

    @api.route("/skills/<skill_id>/update", methods=["POST"])
    def api_update_skill(skill_id: str):
        data = request.get_json(silent=True) or {}
        try:
            with get_db() as conn:
                return success(update_skill(conn, skill_id, str(data.get("source") or ""), base_dir=skill_base_dir or "."))
        except SkillError as exc:
            return failure(exc.code, str(exc), 409 if exc.code == "SKILL_UPDATE_CONFLICT" else 400, exc.details)

    @api.route("/skills/providers", methods=["POST"])
    def api_configure_skill_provider():
        data = request.get_json(silent=True) or {}
        try:
            with get_db() as conn:
                return success(configure_provider(conn, str(data.get("provider_id") or ""), str(data.get("capability") or ""), data.get("config") or {}, enabled=bool(data.get("enabled")), trust_requirement=str(data.get("trust_requirement") or "trusted")))
        except Exception as exc:
            return failure("PROVIDER_CONFIGURATION_ERROR", str(exc), 400)

    @api.route("/skills/<skill_id>/export", methods=["POST"])
    def api_export_skill(skill_id: str):
        data = request.get_json(silent=True) or {}
        destination = str(data.get("destination") or "").strip()
        if not destination:
            return failure("INVALID_REQUEST", "destination is required.", 400)
        try:
            with get_db() as conn:
                path = export_skill(conn, skill_id, destination, include_local_metadata=bool(data.get("include_local_metadata")))
            return success({"path": path})
        except SkillError as exc:
            return failure(exc.code, str(exc), 400, exc.details)

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

    def require_embedder() -> Callable[[str, str], list[float]]:
        if embedder is None:
            raise ServiceError("EMBEDDING_UNAVAILABLE", "Knowledge embedding is not configured.", status=503)
        return embedder

    def source_bytes(payload: dict[str, Any]) -> bytes:
        if isinstance(payload.get("content_text"), str):
            return payload["content_text"].encode("utf-8")
        encoded = payload.get("content_base64")
        if not isinstance(encoded, str) or not encoded:
            raise ServiceError("SOURCE_REQUIRED", "content_text or content_base64 is required.")
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ServiceError("INVALID_BASE64", "content_base64 is not valid base64.") from exc

    @api.route("/knowledge/collections", methods=["GET"])
    def list_knowledge_collections():
        with get_db() as conn:
            return success({"collections": list_collection_snapshots(conn)})

    @api.route("/knowledge/collections/<collection_id>", methods=["GET"])
    def inspect_knowledge_collection(collection_id: str):
        with get_db() as conn:
            return success({"collection": collection_snapshot(conn, collection_id)})

    @api.route("/knowledge/collections", methods=["POST"])
    def post_knowledge_collection():
        payload = _json_body()
        preview = {
            "name": str(payload.get("name") or "").strip(),
            "description": str(payload.get("description") or "").strip(),
            "embedding_model": str(payload.get("embedding_model") or "nomic-embed-text").strip(),
            "chunk_size": int(payload.get("chunk_size") or 1800),
            "chunk_overlap": int(payload.get("chunk_overlap") or 200),
            "knowledge_domain": str(payload.get("knowledge_domain") or "").strip(),
            "version_label": str(payload.get("version_label") or "").strip(),
        }
        if not preview["name"]:
            raise ServiceError("VALIDATION_ERROR", "Collection name is required.")
        if _dry_run(payload):
            return success({"dry_run": True, "operation": "create", "collection": preview})
        with get_db() as conn:
            collection_id = create_collection(conn, **preview)
            result = collection_snapshot(conn, collection_id)
        return success({"dry_run": False, "collection": result}, status=201)

    @api.route("/knowledge/collections/<collection_id>", methods=["PATCH"])
    def patch_knowledge_collection(collection_id: str):
        payload = _json_body()
        with get_db() as conn:
            current = collection_snapshot(conn, collection_id)
            expected = payload.get("expected_revision")
            if expected is None:
                raise ServiceError("REVISION_REQUIRED", "expected_revision is required.")
            proposed = {
                "name": payload.get("name", current["name"]),
                "description": payload.get("description", current.get("description") or ""),
                "embedding_model": payload.get("embedding_model", current["embedding_model"]),
                "chunk_size": payload.get("chunk_size", current["chunk_size"]),
                "chunk_overlap": payload.get("chunk_overlap", current["chunk_overlap"]),
                "knowledge_domain": payload.get("knowledge_domain", current.get("knowledge_domain") or ""),
                "version_label": payload.get("version_label", current.get("version_label") or ""),
                "expected_revision": int(expected),
            }
            if _dry_run(payload):
                if int(expected) != int(current["revision"]):
                    raise ServiceError("STALE_KNOWLEDGE_UPDATE", "Knowledge collection revision is stale.", status=409,
                                       details={"current_revision": current["revision"]})
                return success({"dry_run": True, "operation": "update", "before": current, "proposed": proposed})
            needs_rebuild = update_collection(conn, collection_id, **proposed)
            result = collection_snapshot(conn, collection_id)
        return success({"dry_run": False, "needs_rebuild": needs_rebuild, "collection": result})

    @api.route("/knowledge/collections/<collection_id>/sources", methods=["POST"])
    def post_knowledge_source(collection_id: str):
        payload = _json_body()
        filename = str(payload.get("filename") or "").strip()
        if not filename:
            raise ServiceError("VALIDATION_ERROR", "filename is required.")
        data = source_bytes(payload)
        if len(data) > 10 * 1024 * 1024:
            raise ServiceError("SOURCE_TOO_LARGE", "Knowledge sources are limited to 10 MB.", status=413)
        mode = str(payload.get("ingestion_mode") or "optimised").lower()
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
        with get_db() as conn:
            preview = preview_document_import(conn, collection_id, filename, data, ingestion_mode=mode, provenance=provenance)
            if _dry_run(payload):
                return success({"dry_run": True, "preview": preview})
            expected = payload.get("expected_revision")
            if expected is None:
                raise ServiceError("REVISION_REQUIRED", "expected_revision is required.")
            if int(expected) != int(preview["collection_revision"]):
                raise ServiceError("STALE_KNOWLEDGE_UPDATE", "Knowledge collection revision is stale.", status=409,
                                   details={"current_revision": preview["collection_revision"]})
            if any(item["code"] == "VERSION_CONFLICT" for item in preview["warnings"]) and not payload.get("accept_version_conflict"):
                raise ServiceError("VERSION_CONFLICT", "Incoming source version conflicts with the collection. Review or explicitly accept it.", status=409,
                                   details={"warnings": preview["warnings"]})
        prepared = prepare_document(filename, data, _collection_row(get_db, collection_id), require_embedder(), ingestion_mode=mode, provenance=provenance)
        with get_db() as conn:
            document_id = save_prepared_document(conn, collection_id, prepared)
            result = collection_snapshot(conn, collection_id)
        return success({"dry_run": False, "operation": preview["operation"], "document_id": document_id,
                        "chunk_count": len(prepared["chunks"]), "collection": result})

    def _collection_row(factory: Callable[[], sqlite3.Connection], collection_id: str):
        with factory() as conn:
            row = conn.execute("SELECT * FROM ai_knowledge_collections WHERE id=?", (collection_id,)).fetchone()
        if not row:
            raise AILibraryError("Knowledge collection was not found.")
        return row

    @api.route("/knowledge/collections/<collection_id>/search", methods=["POST"])
    def test_knowledge_retrieval(collection_id: str):
        payload = _json_body()
        query = str(payload.get("query") or "").strip()
        if not query:
            raise ServiceError("VALIDATION_ERROR", "Retrieval query is required.")
        with get_db() as conn:
            passages = retrieve_knowledge(conn, collection_id, query, require_embedder(), limit=int(payload.get("limit") or 5))
        return success({"query": query, "passages": passages})

    @api.route("/knowledge/collections/<collection_id>/audit", methods=["GET"])
    def get_knowledge_audit(collection_id: str):
        models = list_models() if list_models else None
        with get_db() as conn:
            return success(audit_collection(conn, collection_id, models))

    @api.route("/knowledge/collections/<collection_id>/rebuild", methods=["POST"])
    def post_knowledge_rebuild(collection_id: str):
        payload = _json_body()
        if not payload.get("confirm"):
            with get_db() as conn:
                current = collection_snapshot(conn, collection_id)
            return success({"dry_run": True, "operation": "rebuild", "collection": current})
        with get_db() as conn:
            count = rebuild_collection(conn, collection_id, require_embedder())
            result = collection_snapshot(conn, collection_id)
        return success({"dry_run": False, "chunk_count": count, "collection": result})

    @api.route("/knowledge/documents/<document_id>/rebuild", methods=["POST"])
    def post_document_rebuild(document_id: str):
        payload = _json_body()
        if not payload.get("confirm"):
            return success({"dry_run": True, "operation": "rebuild_document", "document_id": document_id})
        with get_db() as conn:
            count = rebuild_document(conn, document_id, require_embedder())
        return success({"dry_run": False, "document_id": document_id, "chunk_count": count})

    @api.route("/knowledge/collections/<collection_id>/duplicate", methods=["POST"])
    def post_collection_duplicate(collection_id: str):
        payload = _json_body()
        name = str(payload.get("name") or "").strip() or None
        if _dry_run(payload):
            with get_db() as conn:
                current = collection_snapshot(conn, collection_id)
            return success({"dry_run": True, "operation": "duplicate", "source": current,
                            "new_name": name or f"{current['name']} (Copy)"})
        with get_db() as conn:
            new_id = duplicate_collection(conn, collection_id, new_name=name)
            result = collection_snapshot(conn, new_id)
        return success({"dry_run": False, "collection": result}, status=201)

    @api.route("/knowledge/collections/<collection_id>/suggestions", methods=["GET"])
    def get_collection_suggestions(collection_id: str):
        with get_db() as conn:
            collection = collection_snapshot(conn, collection_id)
        domain = collection.get("knowledge_domain") or collection["name"]
        return success({"collection_id": collection_id, **suggested_action_resources(domain)})

    @api.route("/knowledge/documents/<document_id>", methods=["DELETE"])
    def delete_knowledge_document(document_id: str):
        payload = request.get_json(silent=True) or {}
        with get_db() as conn:
            document = conn.execute("SELECT * FROM ai_knowledge_documents WHERE id=?", (document_id,)).fetchone()
            if not document:
                raise AILibraryError("Knowledge document was not found.")
            collection = collection_snapshot(conn, document["collection_id"])
            if not payload.get("confirm"):
                return success({"dry_run": True, "operation": "delete_document", "document": dict(document),
                                "collection_revision": collection["revision"]})
            expected = payload.get("expected_revision")
            if expected is None or int(expected) != int(collection["revision"]):
                raise ServiceError("STALE_KNOWLEDGE_UPDATE", "Knowledge collection revision is stale or missing.", status=409,
                                   details={"current_revision": collection["revision"]})
            conn.execute("DELETE FROM ai_knowledge_documents WHERE id=?", (document_id,))
            conn.execute("UPDATE ai_knowledge_collections SET revision=revision+1, updated_at=datetime('now') WHERE id=?", (document["collection_id"],))
            updated = collection_snapshot(conn, document["collection_id"])
        return success({"dry_run": False, "deleted_document_id": document_id, "collection": updated})

    @api.route("/knowledge/collections/<collection_id>", methods=["DELETE"])
    def delete_knowledge_collection(collection_id: str):
        payload = request.get_json(silent=True) or {}
        with get_db() as conn:
            collection = collection_snapshot(conn, collection_id)
            if not payload.get("confirm"):
                return success({"dry_run": True, "operation": "delete_collection", "collection": collection})
            expected = payload.get("expected_revision")
            if expected is None or int(expected) != int(collection["revision"]):
                raise ServiceError("STALE_KNOWLEDGE_UPDATE", "Knowledge collection revision is stale or missing.", status=409,
                                   details={"current_revision": collection["revision"]})
            conn.execute("DELETE FROM ai_knowledge_collections WHERE id=?", (collection_id,))
        return success({"dry_run": False, "deleted_collection_id": collection_id})

    @api.route("/knowledge/collections/<collection_id>/action-bundle", methods=["POST"])
    def post_action_bundle(collection_id: str):
        payload = _json_body()
        required = ["system_name", "system_content", "template_name", "template_content", "action_name"]
        missing = [key for key in required if not str(payload.get(key) or "").strip()]
        if missing:
            raise ServiceError("VALIDATION_ERROR", "Action bundle fields are required.", details={"missing": missing})
        preview = {key: payload.get(key) for key in required + ["action_description", "model"]}
        if _dry_run(payload):
            return success({"dry_run": True, "operation": "create_action_bundle", "proposed": preview})
        with get_db() as conn:
            collection_snapshot(conn, collection_id)
            system_id = create_resource(conn, kind="system", name=payload["system_name"], content=payload["system_content"], description=payload.get("action_description") or "")
            template_id = create_resource(conn, kind="template", name=payload["template_name"], content=payload["template_content"], description=payload.get("action_description") or "")
            action_id = create_action(conn, {"name": payload["action_name"], "description": payload.get("action_description") or "",
                "model": payload.get("model"), "system_instruction_id": system_id, "prompt_template_id": template_id,
                "knowledge_collection_id": collection_id, "allow_runtime_instruction": True, "enabled": True})
        return success({"dry_run": False, "system_instruction_id": system_id, "prompt_template_id": template_id, "action_id": action_id}, status=201)

    return api
