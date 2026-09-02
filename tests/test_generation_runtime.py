import base64
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import requests

from generation_runtime import (
    ComfyUIClient, GenerationError, apply_generation_migrations, check_server,
    compatible_profiles, get_generation, inspect_workflow, list_prompt_generations,
    list_servers, materialise_workflow, refresh_generation, regenerate, save_profile,
    sha256_json, submit_generation, upsert_server, validate_profile,
    validate_saved_profile,
)


class FakeResponse:
    def __init__(self, payload=None, content=b"", content_type="application/json", status=200):
        self.payload, self.content, self.status_code = payload, content, status
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.requests = []
        self.history_payload = None
        self.queue_payload = {"queue_running": [], "queue_pending": []}
        self.fail_health = False

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        if url.endswith("/system_stats"):
            if self.fail_health:
                raise requests.ConnectionError("offline")
            return FakeResponse({"system": {"comfyui_version": "test"}})
        if url.endswith("/prompt"):
            return FakeResponse({"prompt_id": "job-123"})
        if url.endswith("/object_info"):
            return FakeResponse({"CLIPTextEncode": {"input": {}}})
        if "/history/" in url:
            return FakeResponse(self.history_payload or {})
        if url.endswith("/queue"):
            return FakeResponse(self.queue_payload)
        raise AssertionError(url)

    def post(self, url, **kwargs):
        self.requests.append(("POST_UPLOAD", url, kwargs))
        filename = kwargs["files"]["image"][0]
        return FakeResponse({"name": f"uploaded_{filename}", "subfolder": "", "type": "input"})

    def get(self, url, **kwargs):
        self.requests.append(("GET_VIEW", url, kwargs))
        return FakeResponse(content=b"result-bytes", content_type="image/png")


def graph():
    return {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "original"}},
        "2": {"class_type": "KSampler", "inputs": {"seed": 7, "steps": 20, "positive": ["1", 0]}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "example.png"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
    }


class GenerationRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("""CREATE TABLE prompts(id INTEGER PRIMARY KEY, sync_id TEXT, title TEXT, category TEXT,
                          tool TEXT, prompt_type TEXT, content TEXT, revision INTEGER DEFAULT 1)""")
        self.conn.execute("""CREATE TABLE prompt_skill_derivations(id TEXT, derived_prompt_id INTEGER, target TEXT,
                          metadata_json TEXT, created_at TEXT)""")
        apply_generation_migrations(self.conn)
        self.conn.execute("INSERT INTO prompts VALUES(1,'PROMPT-1','Test','Image','Generic','Generation','a silver bird',3)")
        upsert_server(self.conn, {"id": "image-local", "display_name": "Image", "base_url": "http://127.0.0.1:8188", "enabled": True, "is_default": True})

    def tearDown(self):
        self.conn.close()
        self.temp.cleanup()

    def profile(self, profile_id="flux-test", kind="image", inputs=None, outputs=None, compatibility=None):
        return save_profile(self.conn, {
            "id": profile_id, "display_name": profile_id, "generation_kind": kind,
            "model_family": "flux_2_klein", "mode": "t2i", "server_id": "image-local",
            "inputs": inputs or {
                "prompt": {"node_id": "1", "field": "text", "type": "text", "required": True},
                "seed": {"node_id": "2", "field": "seed", "type": "integer", "default": 7},
            },
            "outputs": outputs if outputs is not None else [{"node_id": "9", "type": kind}],
            "compatibility": compatibility or {},
        }, graph(), self.root / "workflows")["profile"]

    def test_server_configuration_default_disabled_and_health(self):
        upsert_server(self.conn, {"id": "video-local", "display_name": "Video", "base_url": "http://127.0.0.1:8189", "enabled": False, "is_default": True})
        servers = list_servers(self.conn)
        self.assertTrue(servers[0]["is_default"])
        self.assertEqual("video-local", servers[0]["id"])
        self.assertFalse(check_server(self.conn, "video-local", session=FakeSession())["health"]["reachable"])
        fake = FakeSession(); fake.fail_health = True
        result = check_server(self.conn, "image-local", session=fake)
        self.assertEqual("unreachable", result["health"]["state"])

    def test_public_server_url_is_rejected(self):
        with self.assertRaises(GenerationError) as caught:
            upsert_server(self.conn, {"id": "public-host", "display_name": "No", "base_url": "http://8.8.8.8:8188"})
        self.assertEqual("UNSAFE_SERVER_URL", caught.exception.code)

    def test_workflow_inspection_hash_and_duplicate_profile(self):
        inspection = inspect_workflow(graph())
        self.assertEqual(4, inspection["node_count"])
        self.assertIn("prompt", next(item for item in inspection["candidates"] if item["field"] == "text")["suggestions"])
        profile = self.profile()
        self.assertEqual(sha256_json(graph()), profile["source_hash"])
        with self.assertRaises(GenerationError) as caught:
            self.profile()
        self.assertEqual("DUPLICATE_PROFILE", caught.exception.code)

    def test_live_profile_validation_checks_target_node_classes(self):
        self.profile()
        validation = validate_saved_profile(self.conn, "flux-test", live=True, session=FakeSession())
        self.assertEqual("invalid", validation["state"])
        self.assertIn("MISSING_NODE_CLASSES", {item["code"] for item in validation["errors"]})

    def test_mapping_validation_detects_node_field_and_required_inputs(self):
        invalid = {"generation_kind": "image", "server_id": "image-local", "inputs": {
            "prompt": {"node_id": "missing", "field": "text", "required": True},
            "seed": {"node_id": "2", "field": "missing", "required": False},
        }, "outputs": []}
        result = validate_profile(invalid, graph(), ["image-local"])
        self.assertEqual("invalid", result["state"])
        self.assertEqual({"MISSING_NODE", "MISSING_FIELD"}, {item["code"] for item in result["errors"]})
        self.profile(inputs={"prompt": {"node_id": "1", "field": "text", "required": True}, "start_image": {"node_id": "3", "field": "image", "type": "image", "required": True}})
        with self.assertRaises(GenerationError) as caught:
            submit_generation(self.conn, 1, "flux-test", {}, session=FakeSession())
        self.assertEqual("REQUIRED_INPUT_MISSING", caught.exception.code)

    def test_prompt_and_media_injection_clone_source(self):
        profile = self.profile(inputs={"prompt": {"node_id": "1", "field": "text", "required": True}, "start_image": {"node_id": "3", "field": "image", "type": "image", "required": True}})
        original = graph(); fake = FakeSession()
        values = {"prompt": "replacement", "start_image": {"filename": "start.png", "content_base64": base64.b64encode(b"pixels").decode()}}
        runtime, recorded = materialise_workflow(profile, original, values, ComfyUIClient("http://127.0.0.1:8188", session=fake))
        self.assertEqual("original", original["1"]["inputs"]["text"])
        self.assertEqual("replacement", runtime["1"]["inputs"]["text"])
        self.assertEqual("uploaded_start.png", runtime["3"]["inputs"]["image"])
        self.assertEqual("uploaded_start.png", recorded["start_image"]["comfyui"][0]["name"])

    def test_successful_image_generation_and_remote_result_storage(self):
        self.profile(); fake = FakeSession()
        generation = submit_generation(self.conn, "PROMPT-1", "flux-test", {"seed": "182771"}, session=fake)
        self.assertEqual("queued", generation["status"])
        self.assertEqual(182771, generation["parameters"]["seed"])
        fake.history_payload = {"job-123": {"status": {"completed": True, "status_str": "success"}, "outputs": {"9": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}}}}
        completed = refresh_generation(self.conn, generation["id"], self.root / "media", session=fake)
        self.assertEqual("completed", completed["status"])
        self.assertEqual(b"result-bytes", Path(completed["result_media"][0]["stored_path"]).read_bytes())
        self.assertEqual(1, len(list_prompt_generations(self.conn, 1)))

    def test_video_output_and_missing_output_failure(self):
        self.profile("h3-test", "video")
        fake = FakeSession(); first = submit_generation(self.conn, 1, "h3-test", {}, session=fake)
        # SaveVideo may report an MP4 under ComfyUI's historical `images` bucket;
        # the profile's semantic output type remains authoritative.
        fake.history_payload = {"job-123": {"status": {"completed": True}, "outputs": {"9": {"images": [{"filename": "clip.mp4", "type": "output"}]}}}}
        done = refresh_generation(self.conn, first["id"], self.root / "media", session=fake)
        self.assertEqual("video", done["result_media"][0]["kind"])
        legacy_media = [{"filename": "clip.mp4", "content_type": "video/mp4", "kind": "image"}]
        self.conn.execute("UPDATE generations SET result_media_json=? WHERE id=?", (json.dumps(legacy_media), first["id"]))
        self.assertEqual("video", get_generation(self.conn, first["id"])["result_media"][0]["kind"])
        second = submit_generation(self.conn, 1, "h3-test", {}, session=fake)
        fake.history_payload = {"job-123": {"status": {"completed": True}, "outputs": {}}}
        failed = refresh_generation(self.conn, second["id"], self.root / "media", session=fake)
        self.assertEqual("MISSING_OUTPUT", failed["error_code"])

    def test_execution_failure_and_queue_state(self):
        self.profile(); fake = FakeSession(); generation = submit_generation(self.conn, 1, "flux-test", {}, session=fake)
        fake.queue_payload = {"queue_running": [[1, "job-123", {}, {}]], "queue_pending": []}
        self.assertEqual("running", refresh_generation(self.conn, generation["id"], self.root / "media", session=fake)["status"])
        fake.history_payload = {"job-123": {"status": {"completed": False, "status_str": "error", "messages": [["execution_error", {"exception_message": "boom"}]]}, "outputs": {}}}
        failed = refresh_generation(self.conn, generation["id"], self.root / "media", session=fake)
        self.assertEqual("EXECUTION_FAILED", failed["error_code"])

    def test_skill_target_compatibility(self):
        saved = self.profile(compatibility={"targets": ["flux_2_klein"]})
        self.profile("krea-test", compatibility={"targets": ["krea_2"]})
        self.profile("ideogram-test", compatibility={"targets": ["ideogram_4"]})
        self.conn.execute("UPDATE workflow_profiles SET model_family='flux' WHERE id=?", (saved["id"],))
        self.conn.execute("INSERT INTO prompt_skill_derivations VALUES('d1',1,'flux_2_klein','{}','2026-01-01')")
        profiles = compatible_profiles(self.conn, 1)
        self.assertEqual("flux-test", profiles[0]["id"])
        self.assertIn("Skill target", profiles[0]["compatibility_reasons"])
        self.assertTrue(profiles[0]["recommended"])
        self.assertEqual({"flux-test", "krea-test", "ideogram-test"}, {profile["id"] for profile in profiles})
        self.assertTrue(all(profile["selectable"] for profile in profiles))

    def test_generic_prompt_can_use_multiple_profiles_without_skill_metadata(self):
        self.profile("flux-test")
        self.profile("krea-test")
        self.profile("ideogram-test")
        profiles = compatible_profiles(self.conn, 1)
        self.assertEqual({"flux-test", "krea-test", "ideogram-test"}, {profile["id"] for profile in profiles})
        self.assertFalse(any(profile["recommended"] for profile in profiles))
        for profile_id in ("flux-test", "krea-test", "ideogram-test"):
            submit_generation(self.conn, 1, profile_id, {}, session=FakeSession())
        history = list_prompt_generations(self.conn, 1)
        self.assertEqual(3, len(history))
        self.assertEqual({"flux-test", "krea-test", "ideogram-test"}, {item["profile_id"] for item in history})

    def test_generation_prompt_override_is_snapshotted_without_editing_saved_prompt(self):
        self.profile(); fake = FakeSession()
        generation = submit_generation(self.conn, 1, "flux-test", {}, prompt_override="generation-only wording", session=fake)
        submitted = next(call for call in fake.requests if call[1].endswith("/prompt"))[2]["json"]["prompt"]
        self.assertEqual("generation-only wording", submitted["1"]["inputs"]["text"])
        self.assertEqual("generation-only wording", generation["prompt_content_snapshot"])
        self.assertEqual("a silver bird", self.conn.execute("SELECT content FROM prompts WHERE id=1").fetchone()[0])

    def test_invalid_and_disabled_profiles_cannot_submit(self):
        self.profile()
        self.conn.execute("UPDATE workflow_profiles SET enabled=0 WHERE id='flux-test'")
        with self.assertRaises(GenerationError) as caught:
            submit_generation(self.conn, 1, "flux-test", {}, session=FakeSession())
        self.assertEqual("PROFILE_INVALIDATED", caught.exception.code)

    def test_regenerate_can_retain_snapshot_or_use_current_prompt(self):
        self.profile(); first_session = FakeSession()
        first = submit_generation(self.conn, 1, "flux-test", {}, session=first_session)
        self.conn.execute("UPDATE prompts SET content='a gold bird', revision=4 WHERE id=1")
        old_session = FakeSession()
        old = regenerate(self.conn, first["id"], current_prompt=False, session=old_session)
        submitted_old = next(call for call in old_session.requests if call[1].endswith("/prompt"))[2]["json"]["prompt"]
        self.assertEqual("a silver bird", submitted_old["1"]["inputs"]["text"])
        self.assertEqual(3, old["prompt_revision"])
        current_session = FakeSession()
        current = regenerate(self.conn, first["id"], current_prompt=True, session=current_session)
        submitted_current = next(call for call in current_session.requests if call[1].endswith("/prompt"))[2]["json"]["prompt"]
        self.assertEqual("a gold bird", submitted_current["1"]["inputs"]["text"])
        self.assertEqual(4, current["prompt_revision"])


if __name__ == "__main__":
    unittest.main()
