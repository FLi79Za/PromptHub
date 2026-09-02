# PromptHub Integration API v1 Change Log

## Safety checkpoints

- Created and verified `A:\AI\Prompt library\Plib_Backups\Plib_PreIntegrationAPI_20260801_082317` before live changes.
- Verified the consistent backup database with `PRAGMA integrity_check`.
- Verified all 28 SHA-256 manifest entries.
- Created branch `feature/chatgpt-integration-api-v1`.
- Created pre-change checkpoint commit `c99d949`.
- Established a passing isolated baseline for the existing UI and browser-extension workflows.

## Added files

- `integration_api.py` — Flask blueprint, authentication, JSON envelope, CORS allow-list, request limits, error handling, endpoints, and safe request logging.
- `integration_config.py` — external configuration, bearer-token lifecycle, generated Flask secret, and rotating integration log setup.
- `prompt_service.py` — additive migrations and reusable prompt search/create/update/version/organisation/metadata/audit services.
- `tests/__init__.py` — test package marker.
- `tests/test_integration_api.py` — isolated standard-library API and regression tests.
- `tools/manage_integration_api.py` — status, token view/regeneration, and non-secret setting management.
- `tools/migrate_integration_api.py` — mandatory consistent pre-migration backup and repeatable schema migration.
- `tools/sqlite_backup.py` — small SQLite online-backup helper used by rollback.
- `tools/rollback_integration_api.ps1` — confirmed, process-aware, safety-backup-first restore tool.
- `docs/INTEGRATION_API_V1.md` — complete API, configuration, security, testing, limitation, and future-integration documentation.
- `ROLLBACK_INTEGRATION_API.md` — exact scripted and manual restoration instructions.
- `CHANGELOG_INTEGRATION_API.md` — this file.

## Changed files

- `app.py`
  - Registers the versioned Integration API blueprint.
  - Replaces the hard-coded Flask secret with generated external configuration.
  - Excludes the Integration API namespace from legacy wildcard CORS while preserving existing route behavior.
  - Honors safe loopback host and optional port configuration.
  - Assigns new UI/import/extension prompts a stable `sync_id` and source immediately.
  - Increments revisions for existing UI/import changes so API optimistic concurrency detects non-API edits.
- `.gitignore`
  - Ignores Integration API token, secret, config, and log filenames if they are ever placed beside the app.
- `README.md`
  - Adds a concise Integration API setup/security section and link to detailed documentation.

No templates, static assets, browser-extension response structures, existing route paths, dependencies, or requirements entries were changed.

## Database schema migration 001

All changes are additive and repeatable:

- `prompts.revision INTEGER NOT NULL DEFAULT 1`
- `prompts.source TEXT`
- backfill missing/invalid revisions to 1
- preserve existing populated `sync_id` values
- generate UUID values only for missing `sync_id` values
- confirm no duplicate populated `sync_id` values
- retain/create unique index `idx_prompts_sync_id`
- add index `idx_prompts_revision`
- add `schema_migrations(version, name, applied_at)`
- add focused `integration_prompt_history` table
- add `idx_integration_history_prompt_created`

No table recreation, column deletion, prompt deletion, destructive conversion, or parallel version model is introduced.

## Dependencies

No new dependency was added. The implementation uses Flask, Flask-CORS, and Python standard-library modules already available to PromptHub.

## Applied migration and verification

- Created and verified immediate pre-migration database backup `A:\AI\Prompt library\Plib_Backups\Plib_PreIntegrationMigration_20260801_085130`.
- Applied `001_integration_api_v1`; post-migration `PRAGMA integrity_check` returned `ok`.
- Confirmed 832 prompts before and after migration, 832 distinct populated `sync_id` values, revisions initialised to 1, and zero initial integration-audit rows.
- Final automated result: 14 tests passed with temporary databases and temporary output directories.
- Live read-only smoke result: PromptHub launched and restarted; 832 prompts remained; UI/search/edit/history/management/JSON sync/extension routes and Integration API health/search/retrieval passed; unauthenticated writes returned 401; untrusted CORS origins were denied.
- Isolated write smoke result: Integration create/retrieve/dry-run/update/conflict/version/organisation/history plus existing UI create/edit/pin/history/saved-view/search/export/management and browser-extension create all passed without touching the live database.
