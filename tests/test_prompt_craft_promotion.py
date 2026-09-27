"""Smoke test for the manual promotion boundary; no live PromptHub needed."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.prompt_craft_promotion import promote, stage


class PromotionSmokeTest(unittest.TestCase):
    def test_unsafe_staging_destinations_leave_installed_package_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            installed = root / "installed"
            installed.mkdir()
            original = installed / "SKILL.md"
            original.write_text("Original guidance", encoding="utf-8")
            finding = root / "finding.json"
            finding.write_text(json.dumps({"discovery_id": "real-123", "revision": 2,
                "release": "v2", "primary_source": "https://example.org/release",
                "target_reference": "references/example-v2.md", "approved_text": "Reviewed guidance."}), encoding="utf-8")
            manifest = root / "stage.promotion.json"
            manifest.write_text("Existing reviewed manifest", encoding="utf-8")

            def fake_request(base, token, method, path, payload=None):
                if path == "/health":
                    return {"availability": {"write": True}}
                self.assertEqual((method, path), ("GET", "/skills/skill"))
                return {"package_path": str(installed), "content_hash": "old-hash"}

            with patch("tools.prompt_craft_promotion.request", side_effect=fake_request), \
                    patch("tools.prompt_craft_promotion.shutil.copytree") as copy:
                for destination, error in ((installed, ValueError),
                        (installed / "stage", ValueError), (root / "stage", FileExistsError)):
                    with self.subTest(destination=destination), self.assertRaises(error):
                        stage("local", "token", "skill", finding, destination)
                copy.assert_not_called()
            self.assertEqual(list(installed.iterdir()), [original])
            self.assertEqual(original.read_text(encoding="utf-8"), "Original guidance")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "Existing reviewed manifest")

    def test_stage_then_explicit_promotion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            installed = root / "installed"
            installed.mkdir()
            (installed / "SKILL.md").write_text("Original guidance", encoding="utf-8")
            finding = root / "finding.json"
            finding.write_text(json.dumps({"discovery_id": "real-123", "revision": 2,
                "release": "v2", "primary_source": "https://example.org/release",
                "target_reference": "references/example-v2.md", "approved_text": "Use a reference image."}), encoding="utf-8")
            staged = root / "stage"
            calls = []

            def fake_request(base, token, method, path, payload=None):
                calls.append((method, path))
                if path == "/health":
                    return {"availability": {"write": True}}
                if path.endswith("/compare"):
                    return {"comparison": {"state": "NEWER", "files": {"added": ["references/example-v2.md"], "removed": [], "modified": []},
                        "incoming": {"content_hash": "new-hash"}}}
                if path.endswith("/update"):
                    return {"result": {"operation": "updated"}}
                return {"package_path": str(installed), "content_hash": "new-hash" if ("POST", "/skills/image-prompt-craft/update") in calls else "old-hash"}

            with patch("tools.prompt_craft_promotion.request", side_effect=fake_request):
                manifest = stage("local", "token", "image-prompt-craft", finding, staged)
                self.assertEqual((staged / "SKILL.md").read_text(), "Original guidance")
                self.assertEqual(calls.count(("POST", "/skills/image-prompt-craft/update")), 0)
                with self.assertRaises(ValueError):
                    promote("local", "token", root / "stage.promotion.json", "unapproved")
                result = promote("local", "token", root / "stage.promotion.json", manifest["sha256"])
                self.assertEqual(result["content_hash"], "new-hash")
                self.assertEqual(calls.count(("POST", "/skills/image-prompt-craft/update")), 1)

    def test_synthetic_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            record = Path(temp) / "finding.json"
            record.write_text(json.dumps({"discovery_id": "d2fc33b564d5d72ac03f80ab", "revision": 1,
                "release": "setup-test-v1", "primary_source": "https://example.org",
                "target_reference": "references/test.md", "approved_text": "test"}))
            with self.assertRaises(ValueError):
                stage("local", "token", "skill", record, Path(temp) / "stage")


if __name__ == "__main__":
    unittest.main()
