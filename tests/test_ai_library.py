from __future__ import annotations

import io
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch


TEST_CONFIG_DIR = Path(tempfile.gettempdir()) / "prompthub-ai-tests-config"
TEST_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PROMPTHUB_CONFIG_DIR"] = str(TEST_CONFIG_DIR)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app as prompthub  # noqa: E402
from ai_library import (  # noqa: E402
    AILibraryError,
    apply_ai_migrations,
    build_action_request,
    create_action,
    create_collection,
    create_resource,
    get_action_for_execution,
    prepare_document,
    retrieve_knowledge,
    save_prepared_document,
)


def keyword_embed(text: str, model: str) -> list[float]:
    lower = text.lower()
    return [1.0, 0.0] if "minimax" in lower else [0.0, 1.0]


class AILibraryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        apply_ai_migrations(self.connection)

    def tearDown(self) -> None:
        self.connection.close()

    def test_migration_is_idempotent_and_uses_stable_text_ids(self):
        apply_ai_migrations(self.connection)
        resource_id = create_resource(
            self.connection, kind="system", name="Prompt specialist", content="Be precise."
        )
        self.assertRegex(resource_id, r"^[0-9A-F-]{36}$")
        self.assertEqual(
            2,
            self.connection.execute("SELECT version FROM schema_migrations WHERE version=2").fetchone()[0],
        )

    def test_referenced_deletions_set_action_links_to_null(self):
        system_id = create_resource(self.connection, kind="system", name="System", content="System text")
        template_id = create_resource(self.connection, kind="template", name="Task", content="Task text")
        collection_id = create_collection(self.connection, name="Docs")
        action_id = create_action(
            self.connection,
            {
                "name": "Optimise",
                "system_instruction_id": system_id,
                "prompt_template_id": template_id,
                "knowledge_collection_id": collection_id,
                "enabled": True,
                "allow_runtime_instruction": True,
            },
        )
        self.connection.execute("DELETE FROM ai_resources WHERE id IN (?, ?)", (system_id, template_id))
        self.connection.execute("DELETE FROM ai_knowledge_collections WHERE id=?", (collection_id,))
        row = self.connection.execute("SELECT * FROM ai_actions WHERE id=?", (action_id,)).fetchone()
        self.assertIsNone(row["system_instruction_id"])
        self.assertIsNone(row["prompt_template_id"])
        self.assertIsNone(row["knowledge_collection_id"])

    def test_document_is_chunked_embedded_and_retrieved_by_similarity(self):
        collection_id = create_collection(
            self.connection, name="MiniMax Docs", embedding_model="test-embed", chunk_size=400, chunk_overlap=20
        )
        collection = self.connection.execute(
            "SELECT * FROM ai_knowledge_collections WHERE id=?", (collection_id,)
        ).fetchone()
        text = (
            "MiniMax prompting uses a structured subject and camera description. " * 12
            + "\n\nUnrelated gardening notes discuss tomatoes and soil. " * 12
        )
        prepared = prepare_document("guide.md", text.encode(), collection, keyword_embed)
        save_prepared_document(self.connection, collection_id, prepared)
        passages = retrieve_knowledge(
            self.connection, collection_id, "How should I prompt MiniMax?", keyword_embed, limit=2
        )
        self.assertTrue(passages)
        self.assertIn("MiniMax", passages[0]["content"])
        self.assertEqual("guide.md", passages[0]["filename"])

    def test_prompt_construction_separates_all_components(self):
        action = {
            "system_content": "You are a model specialist.",
            "template_content": "Rewrite for MiniMax.",
        }
        system, prompt = build_action_request(
            action,
            "A person walks through fog.",
            "Keep it concise.",
            [{"filename": "docs.md", "score": 0.9, "content": "Reference syntax."}],
        )
        self.assertIn("untrusted documentation", system)
        self.assertIn("<REFERENCE_CONTEXT>", prompt)
        self.assertIn("<TASK_INSTRUCTION>", prompt)
        self.assertIn("<ONE_OFF_INSTRUCTION>", prompt)
        self.assertIn("<CURRENT_PROMPT>", prompt)
        self.assertLess(prompt.index("<REFERENCE_CONTEXT>"), prompt.index("<CURRENT_PROMPT>"))

    def test_reimporting_same_filename_updates_document_without_duplicates(self):
        collection_id = create_collection(self.connection, name="Update Docs", embedding_model="test")
        collection = self.connection.execute(
            "SELECT * FROM ai_knowledge_collections WHERE id=?", (collection_id,)
        ).fetchone()
        first = prepare_document("guide.md", b"Initial guidance about gardens.", collection, keyword_embed)
        document_id = save_prepared_document(self.connection, collection_id, first)
        second = prepare_document("guide.md", b"Updated MiniMax camera guidance.", collection, keyword_embed)
        updated_id = save_prepared_document(self.connection, collection_id, second)
        self.assertEqual(document_id, updated_id)
        self.assertEqual(
            1,
            self.connection.execute(
                "SELECT COUNT(*) FROM ai_knowledge_documents WHERE collection_id=?", (collection_id,)
            ).fetchone()[0],
        )
        chunk = self.connection.execute(
            "SELECT content FROM ai_knowledge_chunks WHERE document_id=?", (document_id,)
        ).fetchone()[0]
        self.assertIn("Updated MiniMax", chunk)

    def test_disabled_referenced_resource_blocks_execution(self):
        system_id = create_resource(
            self.connection, kind="system", name="Disabled", content="Text", enabled=False
        )
        action_id = create_action(
            self.connection,
            {"name": "Blocked", "system_instruction_id": system_id, "enabled": True},
        )
        with self.assertRaisesRegex(AILibraryError, "disabled"):
            get_action_for_execution(self.connection, action_id)


class AIWorkflowRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="prompthub-ai-route-")
        self.db_path = Path(self.temp_dir.name) / "prompts.db"
        prompthub.DB_PATH = self.db_path
        prompthub.TEMP_DIR = Path(self.temp_dir.name) / "temp"
        prompthub.TEMP_DIR.mkdir()
        prompthub.app.config.update(TESTING=True)
        prompthub.ollama_models_cached = lambda ttl_seconds=10.0: ["writer:latest"]
        prompthub.ollama_is_ready_cached = lambda ttl_seconds=10.0: True
        prompthub.ollama_list_models = lambda: ["writer:latest", "embed:test"]
        prompthub.init_db()
        with prompthub.get_db() as conn:
            now = "2026-08-13T00:00:00"
            conn.execute(
                """
                INSERT INTO prompts(title, category, tool, prompt_type, content, notes,
                    sync_id, source, created_at, updated_at)
                VALUES ('Route Fixture', 'Other', 'Generic', 'Instruction',
                    'Original prompt remains', '', 'AI-ROUTE-1', 'test', ?, ?)
                """,
                (now, now),
            )
            self.prompt_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            system_id = create_resource(conn, kind="system", name="Writer", content="Write well")
            template_id = create_resource(conn, kind="template", name="Improve", content="Improve the prompt")
            self.action_id = create_action(
                conn,
                {
                    "name": "Improve locally",
                    "model": "writer:latest",
                    "system_instruction_id": system_id,
                    "prompt_template_id": template_id,
                    "enabled": True,
                    "allow_runtime_instruction": True,
                },
            )
        self.client = prompthub.app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_management_and_use_pages_expose_new_workflow(self):
        management = self.client.get("/ai")
        self.assertEqual(200, management.status_code)
        self.assertIn(b"AI Actions & Knowledge", management.data)
        use_page = self.client.get(f"/prompt/{self.prompt_id}/render")
        self.assertEqual(200, use_page.status_code)
        self.assertIn(b"Use AI", use_page.data)
        self.assertNotIn(b"Refine with Ollama", use_page.data)

    def test_execution_returns_review_text_without_mutating_prompt(self):
        with patch.object(prompthub, "ollama_generate", return_value="Generated review result") as generate:
            response = self.client.post(
                f"/api/ai/actions/{self.action_id}/execute",
                json={"content": "Working final prompt", "instruction": "Be vivid"},
            )
        self.assertEqual(200, response.status_code)
        self.assertEqual("Generated review result", response.get_json()["content"])
        self.assertIn("<CURRENT_PROMPT>\nWorking final prompt", generate.call_args.kwargs["prompt"])
        with closing(sqlite3.connect(self.db_path)) as conn:
            stored = conn.execute("SELECT content FROM prompts WHERE id=?", (self.prompt_id,)).fetchone()[0]
        self.assertEqual("Original prompt remains", stored)

    def test_embedding_failure_does_not_create_document(self):
        with prompthub.get_db() as conn:
            collection_id = create_collection(conn, name="Failure Docs", embedding_model="embed:test")
        with patch.object(prompthub, "ollama_embed", side_effect=AILibraryError("Embedding failed cleanly")):
            response = self.client.post(
                f"/ai/collections/{collection_id}/documents",
                data={"document": (io.BytesIO(b"some useful documentation"), "notes.md")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )
        self.assertEqual(200, response.status_code)
        self.assertIn(b"Embedding failed cleanly", response.data)
        with closing(sqlite3.connect(self.db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM ai_knowledge_documents").fetchone()[0]
        self.assertEqual(0, count)


if __name__ == "__main__":
    unittest.main()
