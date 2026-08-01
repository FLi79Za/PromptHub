from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


TEST_CONFIG_DIR = tempfile.TemporaryDirectory(prefix="prompthub-test-config-")
os.environ["PROMPTHUB_CONFIG_DIR"] = TEST_CONFIG_DIR.name
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app as prompthub  # noqa: E402
from integration_config import get_or_create_token  # noqa: E402
from prompt_service import apply_integration_migrations, utc_now  # noqa: E402


class IntegrationApiV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="prompthub-api-test-")
        self.db_path = Path(self.temp_dir.name) / "prompts.db"
        prompthub.DB_PATH = self.db_path
        prompthub.TEMP_DIR = Path(self.temp_dir.name) / "app-temp"
        prompthub.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        prompthub.app.config.update(TESTING=True)
        prompthub.ollama_models_cached = lambda ttl_seconds=10.0: []
        prompthub.ollama_is_ready_cached = lambda ttl_seconds=10.0: False
        prompthub.ollama_list_models = lambda: []
        prompthub.init_db()
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            apply_integration_migrations(conn)
            now = utc_now()
            conn.execute("INSERT OR IGNORE INTO categories(name) VALUES ('Writing')")
            conn.execute("INSERT OR IGNORE INTO tools(name) VALUES ('ChatGPT')")
            conn.execute(
                "INSERT INTO prompt_groups(name, description, created_at) VALUES ('Work', 'Work prompts', ?)",
                (now,),
            )
            conn.execute("INSERT INTO tags(name) VALUES ('fixture')")
            conn.execute(
                """
                INSERT INTO prompts
                    (title, category, tool, prompt_type, content, notes, parent_id, group_id,
                     sync_id, pinned_at, created_at, updated_at, revision, source)
                VALUES ('Fixture Prompt', 'Writing', 'ChatGPT', 'Instruction',
                        'Fixture content searchable needle', 'Fixture notes', NULL, 1,
                        'SYNC-FIXTURE-001', NULL, ?, ?, 1, 'test_fixture')
                """,
                (now, now),
            )
            self.fixture_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            fixture_tag_id = conn.execute("SELECT id FROM tags WHERE name='fixture'").fetchone()[0]
            conn.execute(
                "INSERT INTO prompt_tags(prompt_id, tag_id) VALUES (?, ?)",
                (self.fixture_id, fixture_tag_id),
            )
            conn.commit()
        self.client = prompthub.app.test_client()
        self.token = get_or_create_token()
        self.auth = {"Authorization": f"Bearer {self.token}"}

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def api_json(self, method: str, path: str, payload=None, *, authenticated: bool = True):
        kwargs = {}
        if authenticated:
            kwargs["headers"] = self.auth
        if payload is not None:
            kwargs["json"] = payload
        return self.client.open(path, method=method, **kwargs)

    def create_prompt(self, **overrides):
        payload = {
            "title": "Created through API",
            "content": "Created content",
            "category": "Writing",
            "tool": "ChatGPT",
            "prompt_type": "Instruction",
            "tags": ["fixture"],
            "group": "Work",
            "source": "chatgpt",
            "change_summary": "Created in test",
        }
        payload.update(overrides)
        response = self.api_json("POST", "/api/integration/v1/prompts", payload)
        return response

    def test_authentication_required_and_valid_authentication_accepted(self):
        unauthenticated = self.api_json("GET", "/api/integration/v1/prompts", authenticated=False)
        self.assertEqual(401, unauthenticated.status_code)
        self.assertEqual("AUTH_REQUIRED", unauthenticated.get_json()["error"]["code"])
        authenticated = self.api_json("GET", "/api/integration/v1/prompts")
        self.assertEqual(200, authenticated.status_code)

        unauthenticated_write = self.api_json(
            "POST",
            "/api/integration/v1/prompts",
            {"title": "Blocked", "content": "Blocked"},
            authenticated=False,
        )
        self.assertEqual(401, unauthenticated_write.status_code)

    def test_integration_cors_does_not_inherit_legacy_wildcard(self):
        response = self.client.get(
            "/api/integration/v1/health",
            headers={"Origin": "https://untrusted.example"},
        )
        self.assertNotIn("Access-Control-Allow-Origin", response.headers)
        legacy = self.client.get("/api/ext/categories", headers={"Origin": "https://extension.example"})
        self.assertEqual("https://extension.example", legacy.headers.get("Access-Control-Allow-Origin"))

    def test_health_is_non_sensitive_and_available_without_token(self):
        response = self.api_json("GET", "/api/integration/v1/health", authenticated=False)
        self.assertEqual(200, response.status_code)
        data = response.get_json()["data"]
        self.assertTrue(data["database"]["connected"])
        self.assertTrue(data["database"]["schema_ready"])
        self.assertNotIn("token", str(data).lower())

    def test_search_prompt_content_tags_and_pagination(self):
        for number in range(3):
            self.create_prompt(title=f"Paged {number}")
        response = self.api_json(
            "GET",
            "/api/integration/v1/prompts?query=needle&include_content=true&page=1&page_size=1",
        )
        self.assertEqual(200, response.status_code)
        body = response.get_json()
        self.assertEqual(1, len(body["data"]["prompts"]))
        self.assertIn("content", body["data"]["prompts"][0])
        self.assertEqual(1, body["meta"]["pagination"]["total_items"])

        tag_response = self.api_json("GET", "/api/integration/v1/prompts?tag=fixture&page_size=2")
        self.assertEqual(2, len(tag_response.get_json()["data"]["prompts"]))
        self.assertTrue(tag_response.get_json()["meta"]["pagination"]["has_next"])

    def test_retrieve_by_sync_id_and_integer_id(self):
        by_sync = self.api_json("GET", "/api/integration/v1/prompts/SYNC-FIXTURE-001")
        self.assertEqual(200, by_sync.status_code)
        self.assertEqual(self.fixture_id, by_sync.get_json()["data"]["prompt"]["id"])
        by_id = self.api_json("GET", f"/api/integration/v1/prompts/{self.fixture_id}")
        self.assertEqual(200, by_id.status_code)
        self.assertEqual("SYNC-FIXTURE-001", by_id.get_json()["data"]["prompt"]["sync_id"])

    def test_metadata_returns_ids_names_and_counts(self):
        response = self.api_json("GET", "/api/integration/v1/metadata")
        self.assertEqual(200, response.status_code)
        data = response.get_json()["data"]
        self.assertEqual({"categories", "tools", "prompt_types", "tags", "groups"}, set(data))
        writing = next(item for item in data["categories"] if item["name"] == "Writing")
        self.assertIsInstance(writing["id"], int)
        self.assertEqual(1, writing["prompt_count"])

    def test_create_prompt_and_reject_invalid_creation(self):
        created = self.create_prompt(sync_id="SYNC-CREATED-001")
        self.assertEqual(201, created.status_code)
        prompt = created.get_json()["data"]["prompt"]
        self.assertEqual("SYNC-CREATED-001", prompt["sync_id"])
        self.assertEqual("chatgpt", prompt["source"])
        invalid = self.api_json("POST", "/api/integration/v1/prompts", {"title": "Missing content"})
        self.assertEqual(400, invalid.status_code)
        self.assertEqual("VALIDATION_ERROR", invalid.get_json()["error"]["code"])

    def test_duplicate_sync_id_is_rejected(self):
        response = self.create_prompt(sync_id="SYNC-FIXTURE-001")
        self.assertEqual(409, response.status_code)
        self.assertEqual("DUPLICATE_SYNC_ID", response.get_json()["error"]["code"])

    def test_update_creates_history_and_stale_update_conflicts(self):
        response = self.api_json(
            "PATCH",
            "/api/integration/v1/prompts/SYNC-FIXTURE-001",
            {
                "expected_revision": 1,
                "content": "Updated content",
                "source": "chatgpt",
                "change_summary": "Improve wording",
            },
        )
        self.assertEqual(200, response.status_code)
        updated = response.get_json()["data"]["prompt"]
        self.assertEqual(2, updated["revision"])
        history = self.api_json("GET", "/api/integration/v1/prompts/SYNC-FIXTURE-001/history")
        self.assertEqual(1, len(history.get_json()["data"]["history"]))
        self.assertEqual("update", history.get_json()["data"]["history"][0]["action_type"])

        stale = self.api_json(
            "PATCH",
            "/api/integration/v1/prompts/SYNC-FIXTURE-001",
            {"expected_revision": 1, "title": "Stale overwrite"},
        )
        self.assertEqual(409, stale.status_code)
        self.assertEqual("STALE_UPDATE", stale.get_json()["error"]["code"])

    def test_create_related_version_preserves_original(self):
        response = self.api_json(
            "POST",
            "/api/integration/v1/prompts/SYNC-FIXTURE-001/versions",
            {"content": "Version content", "title": "Fixture v2", "change_summary": "New version"},
        )
        self.assertEqual(201, response.status_code)
        version = response.get_json()["data"]["prompt"]
        self.assertNotEqual("SYNC-FIXTURE-001", version["sync_id"])
        self.assertEqual(self.fixture_id, version["parent_id"])
        original = self.api_json("GET", "/api/integration/v1/prompts/SYNC-FIXTURE-001")
        self.assertEqual("Fixture content searchable needle", original.get_json()["data"]["prompt"]["content"])

    def test_organise_add_remove_tags_and_change_metadata(self):
        add = self.api_json(
            "POST",
            "/api/integration/v1/prompts/SYNC-FIXTURE-001/organise",
            {
                "expected_revision": 1,
                "category": "Other",
                "tool": "Generic",
                "group": None,
                "add_tags": ["added"],
                "create_missing_metadata": True,
                "pin": True,
            },
        )
        self.assertEqual(200, add.status_code)
        prompt = add.get_json()["data"]["prompt"]
        self.assertEqual("Other", prompt["category"])
        self.assertEqual("Generic", prompt["tool"])
        self.assertIsNone(prompt["group"])
        self.assertTrue(prompt["pinned"])
        self.assertIn("added", [tag["name"] for tag in prompt["tags"]])

        remove = self.api_json(
            "POST",
            "/api/integration/v1/prompts/SYNC-FIXTURE-001/organise",
            {"expected_revision": 2, "remove_tags": ["fixture"], "unpin": True},
        )
        self.assertEqual(200, remove.status_code)
        prompt = remove.get_json()["data"]["prompt"]
        self.assertNotIn("fixture", [tag["name"] for tag in prompt["tags"]])
        self.assertFalse(prompt["pinned"])

    def test_dry_run_creates_no_prompt_history_or_timestamp_change(self):
        before = self.api_json("GET", "/api/integration/v1/prompts/SYNC-FIXTURE-001").get_json()["data"]["prompt"]
        response = self.api_json(
            "PATCH",
            "/api/integration/v1/prompts/SYNC-FIXTURE-001",
            {"expected_revision": 1, "content": "Dry content", "dry_run": True},
        )
        self.assertEqual(200, response.status_code)
        self.assertFalse(response.get_json()["data"]["applied"])
        after = self.api_json("GET", "/api/integration/v1/prompts/SYNC-FIXTURE-001").get_json()["data"]["prompt"]
        self.assertEqual(before["content"], after["content"])
        self.assertEqual(before["updated_at"], after["updated_at"])
        history = self.api_json("GET", "/api/integration/v1/prompts/SYNC-FIXTURE-001/history")
        self.assertEqual([], history.get_json()["data"]["history"])

        count_before = self.api_json("GET", "/api/integration/v1/prompts?page_size=100").get_json()["meta"]["pagination"]["total_items"]
        dry_create = self.create_prompt(title="Dry create", dry_run=True)
        self.assertEqual(200, dry_create.status_code)
        count_after = self.api_json("GET", "/api/integration/v1/prompts?page_size=100").get_json()["meta"]["pagination"]["total_items"]
        self.assertEqual(count_before, count_after)

    def test_missing_prompt_and_invalid_metadata(self):
        missing = self.api_json("GET", "/api/integration/v1/prompts/DOES-NOT-EXIST")
        self.assertEqual(404, missing.status_code)
        invalid = self.create_prompt(category="Unknown Category")
        self.assertEqual(400, invalid.status_code)
        self.assertEqual("INVALID_METADATA", invalid.get_json()["error"]["code"])

    def test_existing_extension_and_ui_routes_still_work(self):
        self.assertEqual(200, self.client.get("/").status_code)
        self.assertEqual(200, self.client.get("/prompt/new").status_code)
        self.assertEqual(200, self.client.get(f"/prompt/{self.fixture_id}/edit").status_code)
        self.assertEqual(200, self.client.get(f"/prompt/{self.fixture_id}/history").status_code)
        self.assertEqual(200, self.client.get("/api/ext/categories").status_code)
        self.assertEqual(200, self.client.get("/api/ext/prompts?limit=1").status_code)
        self.assertEqual(200, self.client.post(f"/api/prompt/{self.fixture_id}/pin").status_code)
        exported = self.client.post("/export/selected", data={"prompt_ids": str(self.fixture_id)})
        self.assertEqual(200, exported.status_code)
        self.assertEqual("application/zip", exported.mimetype)

        created = self.client.post(
            "/prompt/new",
            data={
                "title": "UI regression prompt",
                "content": "UI content",
                "category": "Other",
                "tool": "Generic",
                "prompt_type": "Instruction",
            },
        )
        self.assertEqual(302, created.status_code)
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute("SELECT id, sync_id FROM prompts WHERE title='UI regression prompt'").fetchone()
        self.assertIsNotNone(row[1])
        edited = self.client.post(
            f"/prompt/{row[0]}/edit",
            data={
                "title": "UI regression edited",
                "content": "UI edited content",
                "category": "Other",
                "tool": "Generic",
                "prompt_type": "Instruction",
            },
        )
        self.assertEqual(302, edited.status_code)

        extension = self.client.post(
            "/api/ext/prompt",
            json={"title": "Extension regression", "content": "Captured", "category": "Other", "tool": "Generic"},
        )
        self.assertEqual(200, extension.status_code)
        self.assertEqual({"success", "id", "title", "category", "tool", "prompt_type", "group", "tags"}, set(extension.get_json()))


if __name__ == "__main__":
    unittest.main()
