import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from skill_runtime import apply_skill_migrations, import_skill, list_skills, run_skill
from style_compiler import StyleCompilerError, compile_style_request, load_style_signatures


COMPILER = """# Obscure-style compiler

## Curated signature library

### Alternative photographic processes
- **Mordançage**: silver-gelatin black-and-white print with acid-bleached highlights, lifted emulsion, branching chemical fissures and cracked relief.
- **Gum bichromate**: soft-focus photograph built from translucent pigment-and-gum layers, muted colour, paper tooth and selectively lost detail.
- **Cyanotype**: contact-print logic, strong Prussian blue field, pale blue-white silhouettes, matte paper and crisp exposure boundaries.

### Printmaking
- **Aquatint**: granular etched tonal fields, smoky washes, restrained linework and plate-driven transitions.
"""


class StyleCompilerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "image-prompt-craft"
        (self.root / "references").mkdir(parents=True)
        (self.root / "SKILL.md").write_text("---\nname: image-prompt-craft\ndescription: Create, convert, refine and diagnose image prompts.\n---\nUse model-specific guidance.", encoding="utf-8")
        (self.root / "references" / "obscure-style-compiler.md").write_text(COMPILER, encoding="utf-8")
        (self.root / "references" / "core-patterns.md").write_text("CORE PATTERNS", encoding="utf-8")
        for name in ("krea-2", "ideogram-4", "flux-2", "gpt-image", "z-image"):
            (self.root / "references" / f"{name}.md").write_text(name.upper() + " GUIDANCE", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_curated_style_classifies_and_builds_canonical_profile(self):
        result = compile_style_request(self.root, "Create a mordançage portrait", {})
        self.assertEqual("obscure_or_technical", result["classification"])
        self.assertEqual("Mordançage", result["canonical_name"])
        self.assertIn("silver-gelatin", result["observable_signature"])
        self.assertTrue(result["attributes"]["surface_and_material_response"])

    def test_unknown_and_hybrid_styles_decompose_without_false_certainty(self):
        unknown = compile_style_request(self.root, "portrait", {"style": "dust archive", "medium": "worn paper", "region": "southern Africa", "visible_traits": "faded violet stamps"})
        self.assertEqual("partially_understood", unknown["classification"])
        self.assertEqual("uncertain", unknown["confidence"])
        self.assertEqual("worn paper", unknown["attributes"]["medium_and_substrate"])
        hybrid = compile_style_request(self.root, "Use this invented style: rusted neon archive", {})
        self.assertEqual("hybrid_or_invented", hybrid["classification"])

    def test_malformed_style_is_rejected_and_plain_request_is_unchanged(self):
        self.assertIsNone(compile_style_request(self.root, "A cinematic portrait in rain", {}))
        with self.assertRaises(StyleCompilerError):
            compile_style_request(self.root, "portrait", {"style": "x" * 121})

    def test_model_adapter_routing_is_progressive_for_three_targets(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE prompts(id INTEGER PRIMARY KEY AUTOINCREMENT,title TEXT,category TEXT,tool TEXT,prompt_type TEXT,content TEXT,notes TEXT,thumbnail TEXT,parent_id INTEGER,group_id INTEGER,sync_id TEXT,source TEXT,revision INTEGER DEFAULT 1,created_at TEXT,updated_at TEXT)")
        conn.execute("CREATE TABLE prompt_tags(prompt_id INTEGER,tag_id INTEGER,PRIMARY KEY(prompt_id,tag_id))")
        apply_skill_migrations(conn)
        imported = import_skill(conn, self.root, base_dir=self.temp.name)
        conn.execute("UPDATE skills SET runtime_config_json=? WHERE id=?", (json.dumps({"supported_operations": ["convert"]}), imported["skill_id"]))
        catalogue = list_skills(conn)
        self.assertIn("style_compiler", catalogue[0]["features"])
        self.assertNotIn("obscure_style_compiler", [item["id"] for item in catalogue[0]["targets"]])
        for target, expected, excluded in (
            ("krea_2", "references/krea-2.md", "references/flux-2.md"),
            ("ideogram_4", "references/ideogram-4.md", "references/krea-2.md"),
            ("flux_2_klein", "references/flux-2.md", "references/ideogram-4.md"),
        ):
            seen = []
            result = run_skill(conn, imported["skill_id"], "", "qwen", {"style": "mordançage"}, lambda **kwargs: seen.append(kwargs) or "ready prompt", operation="convert", inputs=[{"type": "text", "role": "draft", "content": "portrait, fixed pose"}], target=target)
            self.assertIn("references/obscure-style-compiler.md", result["resources"])
            self.assertIn("references/core-patterns.md", result["resources"])
            self.assertIn(expected, result["resources"])
            self.assertNotIn(excluded, result["resources"])
            if target == "flux_2_klein":
                self.assertNotIn("references/gpt-image.md", result["resources"])
                self.assertNotIn("references/z-image.md", result["resources"])
            self.assertIn("style_compilation", seen[0]["prompt"])
            self.assertTrue(seen[0]["prompt"].endswith("portable Skill's existing output format."))
            self.assertGreaterEqual(seen[0]["prompt"].count("portrait, fixed pose"), 2)
        conn.close()

    def test_edit_invariants_and_no_prompt_clutter_are_in_compiler_contract(self):
        result = compile_style_request(self.root, "Convert this photo into an aquatint print", {"invariants": ["identity", "pose", "framing"]})
        self.assertEqual(["identity", "pose", "framing"], result["invariants"])
        joined = " ".join(result["translation_guidance"]).lower()
        self.assertIn("do not add generic negative prompts", joined)
        self.assertNotIn("masterpiece", joined)


if __name__ == "__main__":
    unittest.main()
