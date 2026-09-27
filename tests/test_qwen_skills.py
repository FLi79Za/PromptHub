"""Exercise the bundled Qwen packages through the actual PromptHub runtime."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from skill_runtime import apply_skill_migrations, import_skill, list_skills, run_skill

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('qwen-2-1-txt2img', 'qwen-2-1-image-edit')

class QwenSkillTests(unittest.TestCase):
    def test_import_discovery_and_all_operations_load_reference_without_model_in_brief(self):
        with tempfile.TemporaryDirectory() as temp:
            conn = sqlite3.connect(':memory:')
            conn.row_factory = sqlite3.Row
            apply_skill_migrations(conn)
            for name in NAMES:
                with self.subTest(skill=name):
                    package = ROOT / 'skill_packages' / name
                    imported = import_skill(conn, package, base_dir=temp)
                    again = import_skill(conn, package, base_dir=temp)
                    self.assertEqual('identical', again['operation'])
                    item = next(s for s in list_skills(conn) if s['name'] == name)
                    self.assertEqual('FULLY_OPERATIONAL', item['runtime']['status'])
                    self.assertEqual(1, len(item['targets']))
                    self.assertEqual([], item['capabilities'])
                    self.assertEqual({'create','transform','convert','refine','diagnose'}, set(item['operations']))
                    for operation in item['operations']:
                        captured = []
                        def generate(**kwargs):
                            captured.append(kwargs)
                            return 'A finished prompt.'
                        role = 'brief' if operation == 'create' else 'draft'
                        brief = 'A blue teapot labelled "TEA".' if name.endswith('txt2img') else 'Change the hat to blue; preserve the person and background.'
                        result = run_skill(conn, imported['skill_id'], '', 'test-writer', {}, generate,
                            operation=operation, inputs=[{'type':'text','role':role,'content':brief}])
                        self.assertEqual(['references/qwen-image-2.1.md'], result['resources'])
                        self.assertEqual('A finished prompt.', result['result_text'])
                        self.assertEqual([], result['provider_results'])
                        self.assertIn(brief, captured[0]['prompt'])
                        self.assertIn((package/'references/qwen-image-2.1.md').read_text(encoding='utf-8'), captured[0]['prompt'])
                        self.assertEqual('completed', conn.execute('SELECT status FROM skill_execution_traces WHERE id=?',(result['execution_id'],)).fetchone()[0])
            conn.close()

if __name__ == '__main__':
    unittest.main()
