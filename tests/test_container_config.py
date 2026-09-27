"""Exercise startup settings with temporary data and no listening server."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
STARTUP = r'''
import json
import logging
import runpy
from unittest.mock import patch
import requests

settings = {}
def capture_run(self, **kwargs):
    settings.update(kwargs)

with patch("flask.Flask.run", new=capture_run):
    namespace = runpy.run_path("app.py", run_name="__main__")
with patch("requests.get", side_effect=requests.ConnectionError("offline test")):
    response = namespace["app"].test_client().get("/")
settings["status"] = response.status_code
settings["database"] = str(namespace["DB_PATH"])
logging.shutdown()
print("STARTUP_RESULT=" + json.dumps(settings))
'''


class ContainerConfigTests(unittest.TestCase):
    def start(self, root, container, host):
        environment = dict(os.environ, PROMPTHUB_CONFIG_DIR=str(root / "config"),
            PROMPTHUB_DB_PATH=str(root / "nested" / "data" / "prompts.db"),
            PROMPTHUB_PORT="5000", PROMPTHUB_CONTAINER=container, PROMPTHUB_HOST=host)
        result = subprocess.run([sys.executable, "-c", STARTUP], cwd=ROOT,
            env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith("STARTUP_RESULT="))
        return json.loads(line.split("=", 1)[1])

    def test_container_startup_and_database_survive_second_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.start(root, "1", "0.0.0.0")
            self.assertEqual((first["host"], first["port"], first["status"]), ("0.0.0.0", 5000, 200))
            database = root / "nested" / "data" / "prompts.db"
            self.assertEqual(Path(first["database"]), database)
            with sqlite3.connect(database) as conn:
                conn.execute("CREATE TABLE persistence_probe(value TEXT)")
                conn.execute("INSERT INTO persistence_probe VALUES ('keep me')")
            conn.close()
            second = self.start(root, "1", "0.0.0.0")
            self.assertEqual(second["status"], 200)
            with sqlite3.connect(database) as conn:
                self.assertEqual(conn.execute("SELECT value FROM persistence_probe").fetchone(), ("keep me",))
            conn.close()

    def test_native_startup_rejects_wildcard_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.start(Path(temp), "", "0.0.0.0")
            self.assertEqual(result["host"], "127.0.0.1")
            self.assertEqual(result["status"], 200)


if __name__ == "__main__":
    unittest.main()
