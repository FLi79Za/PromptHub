# Roll Back PromptHub Integration API v1

Original verified backup:

```text
A:\AI\Prompt library\Plib_Backups\Plib_PreIntegrationAPI_20260801_082317
```

Original project:

```text
A:\AI\Prompt library\Plib
```

The original backup was created through SQLite's online backup API before any live source/schema change, contains source, templates, static assets, configuration/documentation, extension-related files, `prompts.db`, WAL/SHM sidecars, `BACKUP_MANIFEST.md`, and `SHA256SUMS.txt`, and passed `PRAGMA integrity_check` plus checksum verification.

## Preferred scripted rollback

1. Close PromptHub and any browser-extension operation that may be writing to it.
2. Open PowerShell in `A:\AI\Prompt library\Plib`.
3. Run without execution-policy changes if local policy permits:

   ```powershell
   .\tools\rollback_integration_api.ps1
   ```

4. Review the displayed source and destination paths.
5. Type the exact confirmation `RESTORE`.

The script:

- refuses unexpected paths;
- refuses to continue when a Python process appears to be listening on PromptHub's known ports;
- creates `A:\AI\Prompt library\Plib_Backups\Plib_PreRollbackSafety_YYYYMMDD_HHMMSS` first;
- copies the current changed source and makes a consistent SQLite safety backup;
- removes only the explicitly listed Integration API files;
- restores original source files and the consistent original `prompts.db`;
- removes current WAL/SHM sidecars so SQLite recreates sidecars for the restored database;
- verifies restored files against the original SHA-256 manifest; and
- writes `A:\AI\Prompt library\Plib_Backups\rollback_integration_api_YYYYMMDD_HHMMSS.log`.

The script does not delete unrelated or unknown files and is not run automatically.

## Manual rollback

Use this only if the script cannot run.

1. Stop PromptHub. Confirm no Python/Flask PromptHub process is listening on ports 8080, 5000, 5173, or 3000.
2. Create a new safety copy of the current project and use SQLite's online backup API for the current `prompts.db`.
3. From the original backup, copy all application files back to `A:\AI\Prompt library\Plib`, excluding `BACKUP_MANIFEST.md`, `SHA256SUMS.txt`, and database sidecars.
4. Copy the original backup's consistent `prompts.db` to `A:\AI\Prompt library\Plib\prompts.db`.
5. Delete only `A:\AI\Prompt library\Plib\prompts.db-wal` and `A:\AI\Prompt library\Plib\prompts.db-shm`; SQLite recreates them.
6. Remove only these Integration API additions:

   ```text
   integration_api.py
   integration_config.py
   prompt_service.py
   ROLLBACK_INTEGRATION_API.md
   CHANGELOG_INTEGRATION_API.md
   docs\INTEGRATION_API_V1.md
   tests\__init__.py
   tests\test_integration_api.py
   tools\manage_integration_api.py
   tools\migrate_integration_api.py
   tools\rollback_integration_api.ps1
   tools\sqlite_backup.py
   ```

7. Verify every entry in the backup's `SHA256SUMS.txt` against its restored project file. The original backup database checksum is authoritative; the captured sidecars are evidence only.
8. Launch PromptHub with:

   ```powershell
   .\env\Scripts\python.exe .\app.py
   ```

9. Verify the library, search, create/edit, history, categories/tools/tags/groups, pinning, saved views, JSON sync, and browser-extension endpoints.

## Git rollback reference

- Feature branch: `feature/chatgpt-integration-api-v1`
- Pre-change checkpoint: `c99d949 checkpoint: pre integration API v1`

The physical backup is the authoritative rollback mechanism because it includes the untracked SQLite database and uploaded assets. Do not use `git reset --hard` as a database rollback.

## Local configuration after rollback

Rollback does not automatically delete `%LOCALAPPDATA%\PromptHub`. Those files are outside the application and harmless to the restored version. If removal is desired, first preserve them with the rollback safety backup, then manually remove only:

```text
integration_api.json
integration_api.token
flask_secret.key
integration_api.log*
```
