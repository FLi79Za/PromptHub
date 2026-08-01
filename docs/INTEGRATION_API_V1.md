# PromptHub Integration API v1

## Purpose

The PromptHub Integration API v1 prepares the existing desktop application for a future ChatGPT app or plugin. It adds a stable, authenticated, versioned JSON contract without replacing the Flask application, SQLite database, browser-extension routes, UI, or existing parent/child prompt model.

This phase does not include a ChatGPT plugin manifest, hosted connector, public tunnel, cloud service, or account system.

## Architecture

- `integration_api.py` defines a dedicated Flask blueprint under `/api/integration/v1`.
- `prompt_service.py` contains reusable validation, metadata resolution, search, create, update, version, organisation, serialization, audit, and migration logic.
- `integration_config.py` stores local configuration, token, Flask secret, and logging outside the repository.
- `tools/migrate_integration_api.py` makes a consistent pre-migration database backup and applies additive, repeatable migrations.
- `tools/manage_integration_api.py` shows status, changes non-secret settings, and explicitly displays or regenerates the bearer token.
- `integration_prompt_history` records complete before/after JSON for Integration API changes. The existing `prompt_uses` table remains usage history and is not repurposed.

All SQL values are parameterized. Dynamic sort and table identifiers are selected from fixed internal allow-lists.

## Base path and response format

Base path:

```text
/api/integration/v1
```

Success:

```json
{
  "success": true,
  "data": {},
  "error": null,
  "meta": {
    "api_version": "1.0.0",
    "request_id": "..."
  }
}
```

Failure:

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "content is required.",
    "details": {"field": "content"}
  },
  "meta": {
    "api_version": "1.0.0",
    "request_id": "..."
  }
}
```

Responses never include stack traces, database paths, bearer tokens, secret keys, or raw configuration.

## Installation and migration

From the PromptHub project directory:

```powershell
.\env\Scripts\python.exe .\tools\migrate_integration_api.py
```

The command first creates `A:\AI\Prompt library\Plib_Backups\Plib_PreIntegrationMigration_YYYYMMDD_HHMMSS`, uses SQLite's online backup API, verifies `PRAGMA integrity_check`, preserves WAL/SHM sidecars if present, records SHA-256 values, and only then applies the migration. It is safe to rerun; already-applied migrations report `none_already_current`.

The API health endpoint returns `SCHEMA_NOT_READY`/HTTP 503 for protected routes until the migration is present.

## Authentication

All endpoints except health require:

```http
Authorization: Bearer <token>
```

The token is generated with Python `secrets.token_urlsafe(48)` and compared with `hmac.compare_digest`. It is stored in:

```text
%LOCALAPPDATA%\PromptHub\integration_api.token
```

Explicitly view it:

```powershell
.\env\Scripts\python.exe .\tools\manage_integration_api.py show-token
```

Regenerate it, immediately revoking the old token:

```powershell
.\env\Scripts\python.exe .\tools\manage_integration_api.py regenerate-token
```

Do not paste tokens into source code, documentation, shell history shared with others, screenshots, or logs. HTTP 401 is returned for a missing or invalid token. HTTP 403 is returned when a valid token is used while read or write access is disabled.

## Configuration

Non-secret configuration is stored in:

```text
%LOCALAPPDATA%\PromptHub\integration_api.json
```

Defaults:

```json
{
  "api_enabled": true,
  "read_enabled": true,
  "write_enabled": true,
  "allowed_origins": [],
  "host": "127.0.0.1",
  "port": null,
  "max_json_bytes": 1048576,
  "debug": false
}
```

- `api_enabled`: enables the namespace. Disabled returns HTTP 503.
- `read_enabled`: controls authenticated GET access other than health.
- `write_enabled`: controls authenticated POST/PATCH access.
- `allowed_origins`: exact origins allowed to receive Integration API CORS headers. Empty by default.
- `host`: restricted to `127.0.0.1`, `localhost`, or `::1`; unsafe values fall back to `127.0.0.1`.
- `port`: fixed local port, or `null` to preserve PromptHub's existing dynamic-port selection.
- `max_json_bytes`: Integration API JSON limit, clamped between 16 KiB and 4 MiB.
- `debug`: enables more detailed integration-only logging. It does not enable Flask debug mode.

Update settings with the CLI, for example:

```powershell
.\env\Scripts\python.exe .\tools\manage_integration_api.py set write_enabled false
.\env\Scripts\python.exe .\tools\manage_integration_api.py set allowed_origins "http://127.0.0.1:3000,http://localhost:3000"
.\env\Scripts\python.exe .\tools\manage_integration_api.py set port 8080
```

Restart PromptHub after host, port, CORS, or debug changes. Set `PROMPTHUB_CONFIG_DIR` only when an alternative application-data directory is required. `PROMPTHUB_HOST` and `PROMPTHUB_PORT` remain available as launch-time overrides; non-loopback hosts are rejected.

The Flask session key is generated separately in `%LOCALAPPDATA%\PromptHub\flask_secret.key`. Moving away from the former hard-coded placeholder can invalidate old local sessions once; PromptHub data is unaffected.

## Endpoints

### Health

```http
GET /api/integration/v1/health
```

No token is required. Returns API version, database connectivity/schema state, read/write availability, authentication scheme, local-only network status, and capabilities. It never returns secrets.

### Search and list prompts

```http
GET /api/integration/v1/prompts
```

Query parameters:

- `query`: comma-separated terms; each term must match title, content, notes, or tags.
- `category` / `category_id`
- `tool` / `tool_id`
- `prompt_type` / `prompt_type_id`
- `tag` / `tag_id`
- `group` / `group_id`
- `pinned`: boolean
- `parent_id`: integer parent ID
- `updated_after` / `updated_before`: ISO-8601 timestamps
- `sort`: `updated_at`, `created_at`, `title`, `category`, `tool`, `prompt_type`, or `revision`
- `order`: `asc` or `desc`
- `page`: defaults to 1
- `page_size`: defaults to 25; maximum 100
- `include_content`: defaults to false

The default response contains summaries and excludes full prompt content. Pagination is returned in `meta.pagination`.

Example:

```powershell
$token = .\env\Scripts\python.exe .\tools\manage_integration_api.py show-token
$headers = @{ Authorization = "Bearer $token" }
Invoke-RestMethod -Headers $headers -Uri 'http://127.0.0.1:8080/api/integration/v1/prompts?query=portrait&tag=cinematic&page=1&page_size=25'
```

### Retrieve one prompt

```http
GET /api/integration/v1/prompts/<sync_id-or-integer-id>
```

`sync_id` is preferred. Numeric fallback is retained for local compatibility. The response includes structured metadata IDs/names, tags, parent identifiers, pin state, timestamps, revision, child count, and source.

### Create a prompt

```http
POST /api/integration/v1/prompts
Content-Type: application/json
Authorization: Bearer <token>
```

```json
{
  "title": "Product photo prompt",
  "content": "Create a studio product photograph...",
  "notes": "For campaign concepts",
  "category": "Image",
  "tool": "Flux",
  "prompt_type": "Generation",
  "tags": ["product", "studio"],
  "group": "Campaign 2026",
  "pinned": false,
  "source": "chatgpt",
  "change_summary": "Initial version",
  "create_missing_metadata": false
}
```

`title` and `content` are required. A UUID-based `sync_id` is generated when omitted. Duplicate `sync_id` returns HTTP 409. Categories, tools, groups, and tags are not silently created; set `create_missing_metadata: true` explicitly. Omitting `parent_id`/`parent_sync_id` creates an independent prompt, which is also the way to save an independent variation.

### Update a prompt

```http
PATCH /api/integration/v1/prompts/<identifier>
```

```json
{
  "expected_revision": 3,
  "content": "Proposed replacement content",
  "tags": ["product", "studio", "approved"],
  "source": "chatgpt",
  "change_summary": "Tighten composition instructions"
}
```

Supply `expected_revision` or exact `expected_updated_at`. A mismatch returns HTTP 409 `STALE_UPDATE`; newer work is never overwritten automatically. Successful changes increment `revision`, update `updated_at`, and write a full before/after integration history entry.

### Create a related version

```http
POST /api/integration/v1/prompts/<identifier>/versions
```

```json
{
  "title": "Product photo prompt v2",
  "content": "Create a high-key studio product photograph...",
  "source": "chatgpt",
  "change_summary": "High-key lighting version"
}
```

Metadata and tags are copied by default and can be overridden. The new prompt receives its own `sync_id`. Existing family behavior is preserved: a child created from an existing child points to the same family root.

### Organise a prompt

```http
POST /api/integration/v1/prompts/<identifier>/organise
```

```json
{
  "expected_revision": 4,
  "category": "Image",
  "group": "Campaign 2026",
  "add_tags": ["approved"],
  "pin": true,
  "source": "chatgpt",
  "change_summary": "Move to approved campaign set"
}
```

Supported actions are category/tool/prompt-type/group changes, `add_tags`, `remove_tags`, `replace_tags`, `pin`, `unpin`, and direct `pinned`. Use only one tag action in a request.

### Metadata

```http
GET /api/integration/v1/metadata
```

Returns categories, tools, prompt types, tags, and groups with IDs, names, descriptions where available, and prompt counts. Prompt types remain the existing fixed PromptHub values rather than introducing a parallel database table.

### Integration history

```http
GET /api/integration/v1/prompts/<identifier>/history
```

Returns up to 100 recent Integration API change records with action type, source, summary, timestamp, and before/after values. This endpoint is authenticated because history may include prompt content.

## Dry run

Create, update, version, and organisation requests accept:

```json
{"dry_run": true}
```

Validation, metadata resolution, duplicate/conflict checks, and proposed before/after construction run normally. No prompt, metadata, history record, timestamp, or revision is changed. The response contains `dry_run: true` and `applied: false`.

## Error codes

- `AUTH_REQUIRED`, `INVALID_TOKEN` — HTTP 401
- `READ_ACCESS_DISABLED`, `WRITE_ACCESS_DISABLED` — HTTP 403
- `PROMPT_NOT_FOUND` — HTTP 404
- `DUPLICATE_SYNC_ID`, `STALE_UPDATE` — HTTP 409
- `PRECONDITION_REQUIRED` — HTTP 428
- `REQUEST_TOO_LARGE` — HTTP 413
- `UNSUPPORTED_MEDIA_TYPE` — HTTP 415
- `VALIDATION_ERROR`, `INVALID_FIELD_TYPE`, `FIELD_TOO_LONG`, `INVALID_METADATA`, `INVALID_JSON`, `INVALID_DATE`, `INVALID_SORT`, `INVALID_PAGINATION`, `UNSUPPORTED_FIELD` — HTTP 400
- `SCHEMA_NOT_READY`, `API_DISABLED` — HTTP 503
- `DATABASE_ERROR`, `INTERNAL_ERROR` — HTTP 500 with internal details suppressed

## Security assumptions and local-only limitations

- The API is intended only for the same Windows machine and binds to loopback.
- It is plain local HTTP; do not expose it over a LAN, public tunnel, reverse proxy, or the internet in this phase.
- Bearer tokens protect the API but do not create multiple users or permissions.
- CORS is not authentication. Integration origins are exact allow-list entries; an empty list is safest for server-to-server clients.
- Existing browser-extension routes retain their prior broad CORS behavior for compatibility. That existing risk is not extended to `/api/integration/v1`.
- Integration logs contain method, route, status, prompt identifier, action, result code, and duration. Prompt bodies, notes, tokens, and secrets are excluded.
- Configuration files are local user files. File permissions are restricted on a best-effort basis; Windows account security remains part of the trust boundary.

## Running tests

The suite uses only `unittest` and temporary SQLite databases:

```powershell
.\env\Scripts\python.exe -B -m unittest discover -s tests -v
```

Tests never point at the live `prompts.db`.

## Future ChatGPT app consumption

A future local ChatGPT bridge can discover capabilities through health, authenticate with the local bearer token, search summaries first, retrieve full content only when needed, preview changes with `dry_run`, and then apply them with `expected_revision`. It should persist and use `sync_id` as the external identifier and treat HTTP 409 as a prompt to refetch and show the user a conflict.

The future bridge still needs a suitable ChatGPT app/plugin surface, tool schema, user-consent flow, secure local connectivity mechanism, and end-to-end threat review. Those are deliberately outside this phase.

## Known limitations and future work

- No multi-user authorization or per-action scopes.
- No TLS because the service is loopback-only.
- No public/cloud hosting or remote synchronization.
- No final ChatGPT plugin/app manifest or connector.
- Existing UI and extension changes increment revisions but only Integration API writes create `integration_prompt_history` records.
- Application version metadata is not currently defined, so health returns `application_version: null`.
- Prompt types are a fixed in-code enumeration, matching existing PromptHub behavior.
- Integration history is capped at 100 records per API response; archival/export can be added later.
