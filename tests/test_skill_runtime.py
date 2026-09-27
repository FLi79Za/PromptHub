import json
import sqlite3
import tempfile
import unittest
import zipfile
import shutil
from pathlib import Path

from skill_runtime import SkillError, apply_skill_migrations, compare_skill, configure_provider, discover_skill_targets, import_skill, inspect_skill, list_prompt_derivations, normalise_skill_execution, run_skill, save_skill_derivative, skill_runtime_status, update_skill


class SkillRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "video-prompt-craft"
        (self.root / "references").mkdir(parents=True)
        (self.root / "SKILL.md").write_text("---\nname: video-prompt-craft\ndisplay_name: Video Prompt Craft\nversion: 2.0\n---\nFollow this skill. Use model-specific guidance and action choreography.", encoding="utf-8")
        (self.root / "references" / "minimax-h3.md").write_text("MINIMAX H3 guidance", encoding="utf-8")
        (self.root / "references" / "ltx-25.md").write_text("LTX guidance", encoding="utf-8")
        (self.root / "references" / "action-choreography.md").write_text("ACTION choreography", encoding="utf-8")
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("CREATE TABLE prompts(id INTEGER PRIMARY KEY AUTOINCREMENT,title TEXT,category TEXT,tool TEXT,prompt_type TEXT,content TEXT,notes TEXT,thumbnail TEXT,parent_id INTEGER,group_id INTEGER,sync_id TEXT,source TEXT,revision INTEGER DEFAULT 1,created_at TEXT,updated_at TEXT)")
        self.conn.execute("CREATE TABLE prompt_tags(prompt_id INTEGER,tag_id INTEGER,PRIMARY KEY(prompt_id,tag_id))")
        apply_skill_migrations(self.conn)

    def tearDown(self):
        self.conn.close()
        self.temp.cleanup()

    def test_import_is_idempotent_and_hashes_content(self):
        first = import_skill(self.conn, self.root, base_dir=self.temp.name, source_platform="codex")
        second = import_skill(self.conn, self.root, base_dir=self.temp.name, source_platform="codex")
        self.assertEqual("imported", first["operation"])
        self.assertEqual("identical", second["operation"])
        self.assertTrue(first["inspection"]["content_hash"])
        self.assertEqual("codex", json.loads(self.conn.execute("select provenance_json from skills").fetchone()[0])["source_platform"])

    def test_progressive_resource_routing_and_model_choice(self):
        imported = import_skill(self.conn, self.root, base_dir=self.temp.name)
        calls = []
        def fake_generate(**kwargs):
            calls.append(kwargs)
            return "finished minimax prompt"
        result = run_skill(self.conn, imported["skill_id"], "Create a 12-second MiniMax H3 fight", "qwen3.5:9b", {"target_model": "MiniMax H3"}, fake_generate)
        self.assertEqual("qwen3.5:9b", result["model"])
        self.assertIn("references/minimax-h3.md", result["resources"])
        self.assertIn("references/action-choreography.md", result["resources"])
        self.assertNotIn("references/ltx-25.md", result["resources"])
        self.assertIn("MINIMAX H3 guidance", calls[0]["prompt"])

    def test_all_execution_operations_use_general_text_inputs(self):
        imported = import_skill(self.conn, self.root, base_dir=self.temp.name)
        self.conn.execute("UPDATE skills SET runtime_config_json=? WHERE id=?", (json.dumps({"supported_operations": ["create", "transform", "convert", "refine", "diagnose"]}), imported["skill_id"]))
        for operation in ("create", "transform", "convert", "refine", "diagnose"):
            role = "brief" if operation == "create" else "source"
            seen = []
            result = run_skill(self.conn, imported["skill_id"], "", "qwen", {}, lambda **kwargs: seen.append(kwargs) or "done", operation=operation, inputs=[{"type":"text","role":role,"content":"source text"}], target="minimax_h3")
            self.assertEqual(operation, result["operation"])
            self.assertIn(f"OPERATION: {operation.upper()}", seen[0]["prompt"])

    def test_legacy_request_maps_to_create_brief(self):
        imported = import_skill(self.conn, self.root, base_dir=self.temp.name)
        result = run_skill(self.conn, imported["skill_id"], "new brief", "qwen", {}, lambda **_: "done")
        self.assertEqual("create", result["operation"])
        self.assertEqual("brief", result["inputs"][0]["role"])

    def test_transient_draft_executes_without_prompt_id_and_preserves_input(self):
        imported = import_skill(self.conn, self.root, base_dir=self.temp.name)
        self.conn.execute("UPDATE skills SET runtime_config_json=? WHERE id=?", (json.dumps({"supported_operations": ["refine"]}), imported["skill_id"]))
        seen = []
        result = run_skill(self.conn, imported["skill_id"], "", "qwen", {}, lambda **kwargs: seen.append(kwargs) or "refined draft", operation="refine", inputs=[{"type":"text", "role":"draft", "content":"current unsaved draft"}], target="minimax_h3")
        self.assertIsNone(result["source_prompt_id"])
        self.assertEqual("draft", result["inputs"][0]["role"])
        self.assertIn("current unsaved draft", seen[0]["prompt"])
        self.assertEqual("completed", self.conn.execute("SELECT status FROM skill_execution_traces WHERE id=?", (result["execution_id"],)).fetchone()[0])

    def test_draft_input_overrides_saved_source_in_runtime_contract(self):
        imported = import_skill(self.conn, self.root, base_dir=self.temp.name)
        self.conn.execute("UPDATE skills SET runtime_config_json=? WHERE id=?", (json.dumps({"supported_operations": ["refine"]}), imported["skill_id"]))
        result = run_skill(self.conn, imported["skill_id"], "", "qwen", {}, lambda **kwargs: kwargs["prompt"], operation="refine", inputs=[{"type":"text", "role":"draft", "content":"unsaved version"}], source_prompt_id=1)
        self.assertIn("unsaved version", result["content"])
        self.assertNotIn("persisted version", result["content"])

    def test_invalid_operation_and_future_input_type_fail_cleanly(self):
        with self.assertRaises(SkillError) as error:
            normalise_skill_execution(operation="erase", request="x")
        self.assertEqual("UNSUPPORTED_SKILL_OPERATION", error.exception.code)
        with self.assertRaises(SkillError) as error:
            normalise_skill_execution(operation="transform", inputs=[{"type":"image","role":"source","content":"x.png"}])
        self.assertEqual("UNSUPPORTED_SKILL_INPUT_TYPE", error.exception.code)

    def test_target_discovery_uses_profile_resources_not_generic_docs(self):
        (self.root / "references" / "common-prompt-craft.md").write_text("# Common Prompt Craft", encoding="utf-8")
        imported = import_skill(self.conn, self.root, base_dir=self.temp.name)
        skill = dict(self.conn.execute("SELECT * FROM skills WHERE id=?", (imported["skill_id"],)).fetchone())
        targets = discover_skill_targets(skill)
        self.assertIn("minimax_h3", [item["id"] for item in targets])
        self.assertNotIn("common_prompt_craft", [item["id"] for item in targets])

    def test_lyrics_context_routes_lyrics_reference(self):
        lyrics = self.root / "references" / "lyrics-structuring.md"
        lyrics.write_text("LYRICS guidance", encoding="utf-8")
        imported = import_skill(self.conn, self.root, base_dir=self.temp.name)
        self.conn.execute("UPDATE skills SET runtime_config_json=? WHERE id=?", (json.dumps({"supported_operations": ["refine"]}), imported["skill_id"]))
        result = run_skill(self.conn, imported["skill_id"], "", "qwen", {}, lambda **_: "done", operation="refine", inputs=[{"type":"text","role":"source","content":"[Verse] line\n[Chorus] hook"}], target="suno")
        self.assertIn("references/lyrics-structuring.md", result["resources"])

    def test_skill_result_saves_as_non_destructive_derivative_with_provenance(self):
        imported = import_skill(self.conn, self.root, base_dir=self.temp.name)
        self.conn.execute("UPDATE skills SET runtime_config_json=? WHERE id=?", (json.dumps({"supported_operations": ["convert"]}), imported["skill_id"]))
        cursor = self.conn.execute("INSERT INTO prompts(title,category,tool,prompt_type,content,sync_id,source,created_at,updated_at) VALUES ('Original','Video','Generic','Generation','original content','SOURCE-1','test','now','now')")
        source_id = cursor.lastrowid
        result = run_skill(self.conn, imported["skill_id"], "", "qwen", {}, lambda **_: "converted content", operation="convert", inputs=[{"type":"text","role":"source","content":"original content"}], target="minimax_h3", source_prompt_id=source_id)
        saved = save_skill_derivative(self.conn, source_id, result["execution_id"])
        self.assertNotEqual(source_id, saved["prompt_id"])
        self.assertEqual("original content", self.conn.execute("SELECT content FROM prompts WHERE id=?", (source_id,)).fetchone()[0])
        derived = self.conn.execute("SELECT content,parent_id FROM prompts WHERE id=?", (saved["prompt_id"],)).fetchone()
        self.assertEqual(("converted content", source_id), tuple(derived))
        lineage = list_prompt_derivations(self.conn, source_id)
        self.assertEqual("convert", lineage[0]["operation"])
        self.assertEqual("minimax_h3", lineage[0]["target"])

    def test_replace_original_requires_explicit_save_flag(self):
        imported = import_skill(self.conn, self.root, base_dir=self.temp.name)
        self.conn.execute("UPDATE skills SET runtime_config_json=? WHERE id=?", (json.dumps({"supported_operations": ["refine"]}), imported["skill_id"]))
        source_id = self.conn.execute("INSERT INTO prompts(title,category,tool,prompt_type,content,sync_id,source,created_at,updated_at) VALUES ('Original','Video','Generic','Generation','before','SOURCE-2','test','now','now')").lastrowid
        result = run_skill(self.conn, imported["skill_id"], "", "qwen", {}, lambda **_: "after", operation="refine", inputs=[{"type":"text","role":"source","content":"before"}], source_prompt_id=source_id)
        save_skill_derivative(self.conn, source_id, result["execution_id"], replace_original=True)
        self.assertEqual("after", self.conn.execute("SELECT content FROM prompts WHERE id=?", (source_id,)).fetchone()[0])

    def test_zip_path_traversal_is_rejected(self):
        archive = Path(self.temp.name) / "bad.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("../escape.txt", "bad")
        with self.assertRaises(SkillError) as error:
            inspect_skill(archive)
        self.assertEqual("UNSAFE_PATH", error.exception.code)

    def test_nested_dependency_execution_has_hierarchical_result(self):
        child = import_skill(self.conn, self.root, base_dir=self.temp.name)
        parent_root = Path(self.temp.name) / "director"
        parent_root.mkdir(); (parent_root / "SKILL.md").write_text("---\nname: director\n---\nUse $video-prompt-craft.", encoding="utf-8")
        parent = import_skill(self.conn, parent_root, base_dir=self.temp.name)
        self.conn.execute("UPDATE skills SET dependencies_json=? WHERE id=?", (json.dumps([{"skill": "video-prompt-craft", "required": True}]), parent["skill_id"]))
        seen = []
        def fake_generate(**kwargs):
            seen.append(kwargs["system"])
            return json.dumps({"prompt": "specialist"}) if "format" in kwargs else "director result"
        self.conn.execute("UPDATE skills SET runtime_config_json=? WHERE id=?", (json.dumps({"result_schema": {"type": "object"}}), child["skill_id"]))
        result = run_skill(self.conn, parent["skill_id"], "Create an H3 shot", "qwen", {"target_model":"MiniMax H3"}, fake_generate)
        self.assertEqual(1, len(result["dependencies"]))
        self.assertEqual("Video Prompt Craft", result["dependencies"][0]["skill"])
        traces = self.conn.execute("select parent_execution_id from skill_execution_traces where skill_id=?", (child["skill_id"],)).fetchall()
        self.assertTrue(any(trace[0] == result["execution_id"] for trace in traces))

    def test_provider_and_runtime_status_are_gated_by_trust(self):
        imported = import_skill(self.conn, self.root, base_dir=self.temp.name)
        self.conn.execute("UPDATE skills SET capabilities_json=? WHERE id=?", (json.dumps(["image_generation"]), imported["skill_id"]))
        configure_provider(self.conn, "comfyui", "image_generation", {"server_url":"http://127.0.0.1:9", "workflow_path":"missing.json", "input_mappings":{"prompt":{"node_id":"1","input":"text"}}}, enabled=True)
        self.assertEqual("DEGRADED", skill_runtime_status(self.conn, imported["skill_id"])["status"])
        with self.assertRaises(SkillError) as error:
            run_skill(self.conn, imported["skill_id"], "make image", "qwen", {"capability_requests":[{"capability":"image_generation", "prompt":"test"}]}, lambda **_: "ok")
        self.assertEqual("SKILL_TRUST_REQUIRED", error.exception.code)

    def test_safe_update_preserves_local_adapter_and_reports_changed_file(self):
        imported = import_skill(self.conn, self.root, base_dir=self.temp.name, tags=["reviewed", "prompt-craft"])
        self.conn.execute("UPDATE skills SET runtime_config_json=? WHERE id=?", (json.dumps({"instruction_override":"local Qwen adapter"}), imported["skill_id"]))
        incoming = Path(self.temp.name) / "incoming-video"; shutil.copytree(self.root, incoming)
        (incoming / "references" / "minimax-h3.md").write_text("updated MiniMax guidance", encoding="utf-8")
        comparison = compare_skill(self.conn, imported["skill_id"], incoming)
        self.assertEqual("NEWER", comparison["state"])
        self.assertIn("references/minimax-h3.md", comparison["files"]["modified"])
        update_skill(self.conn, imported["skill_id"], incoming, base_dir=self.temp.name)
        adapter = json.loads(self.conn.execute("select runtime_config_json from skills where id=?", (imported["skill_id"],)).fetchone()[0])
        self.assertEqual("local Qwen adapter", adapter["instruction_override"])
        tags = json.loads(self.conn.execute("select tags_json from skills where id=?", (imported["skill_id"],)).fetchone()[0])
        self.assertEqual(["reviewed", "prompt-craft"], tags)

    def test_conflicting_local_portable_source_refuses_update(self):
        imported = import_skill(self.conn, self.root, base_dir=self.temp.name)
        incoming = Path(self.temp.name) / "incoming-conflict"; shutil.copytree(self.root, incoming)
        (incoming / "SKILL.md").write_text((incoming / "SKILL.md").read_text(encoding="utf-8") + "\nIncoming change", encoding="utf-8")
        self.conn.execute("UPDATE skills SET locally_modified=1 WHERE id=?", (imported["skill_id"],))
        with self.assertRaises(SkillError) as error:
            update_skill(self.conn, imported["skill_id"], incoming, base_dir=self.temp.name)
        self.assertEqual("SKILL_UPDATE_CONFLICT", error.exception.code)


if __name__ == "__main__":
    unittest.main()
