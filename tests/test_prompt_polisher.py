import json
import shutil
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from skill_runtime import (apply_skill_migrations, import_skill, inspect_skill,
                           list_skills, run_skill, skill_supported_operations)

PACKAGE = Path(__file__).resolve().parents[1] / 'skill_packages' / 'prompt-polisher'

class PromptPolisherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.conn = sqlite3.connect(':memory:')
        self.conn.row_factory = sqlite3.Row
        apply_skill_migrations(self.conn)

    def tearDown(self):
        self.conn.close()
        self.temp.cleanup()

    def test_bundled_import_preserves_package_and_is_idempotent(self):
        bundled = self.base / 'skill_packages' / 'prompt-polisher'
        shutil.copytree(PACKAGE, bundled)
        before = inspect_skill(bundled)['content_hash']
        first = import_skill(self.conn, bundled, base_dir=self.base)
        self.assertEqual('imported', first['operation'])
        self.assertEqual(before, inspect_skill(bundled)['content_hash'])
        self.assertEqual('identical', import_skill(self.conn, bundled, base_dir=self.base)['operation'])

    def test_wrapped_zip_import_uses_actual_skill_root(self):
        archive = self.base / 'wrapped.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            for p in PACKAGE.rglob('*'):
                if p.is_file():
                    z.write(p, 'wrapper/prompt-polisher/' + p.relative_to(PACKAGE).as_posix())
        imported = import_skill(self.conn, archive, base_dir=self.base)
        self.assertEqual(inspect_skill(PACKAGE)['content_hash'], inspect_skill(imported['package_path'])['content_hash'])

    def test_operations_targets_and_copy_ready_result(self):
        imported = import_skill(self.conn, PACKAGE, base_dir=self.base)
        skill = list_skills(self.conn)[0]
        self.assertEqual(['create','transform','convert','refine','diagnose'], skill['operations'])
        self.assertEqual(['qwen'], [t['id'] for t in skill['targets']])
        for operation in skill['operations']:
            with self.subTest(operation=operation):
                seen = []
                def writer(**kw):
                    seen.append(kw)
                    return json.dumps({'prompt':'Draft a brief apology email.', 'clarifying_questions':[], 'rationale':['Removed repetition.']})
                target = 'qwen' if operation == 'convert' else None
                result = run_skill(self.conn, imported['skill_id'], '', 'gemma4:latest', {}, writer,
                    operation=operation, inputs=[{'type':'text','role':'brief' if operation=='create' else 'draft','content':'help write a short sorry email'}], target=target)
                self.assertEqual('Draft a brief apology email.', result['result_text'])
                self.assertEqual('gemma4:latest', result['model'])
                self.assertEqual(target, result['target'])
                self.assertEqual(['references/qwen.md'] if target else [], result['resources'])
                self.assertIn('format', seen[0])
                self.assertEqual('completed', self.conn.execute('SELECT status FROM skill_execution_traces WHERE id=?',(result['execution_id'],)).fetchone()[0])

    def test_local_operation_override_wins(self):
        imported = import_skill(self.conn, PACKAGE, base_dir=self.base)
        skill = dict(self.conn.execute('SELECT * FROM skills WHERE id=?',(imported['skill_id'],)).fetchone())
        skill['runtime_config_json'] = json.dumps({'supported_operations':['refine']})
        self.assertEqual(['refine'], skill_supported_operations(skill))

    def test_legacy_skill_still_uses_description(self):
        self.assertEqual(['create','refine'], skill_supported_operations({'name':'legacy','description':'Refine prose'}))
