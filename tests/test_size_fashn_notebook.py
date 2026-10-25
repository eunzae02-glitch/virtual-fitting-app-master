"""Colab 설치 셀을 다운로드 없이 실행해 제약과 오류 출력, import 검사를 확인한다."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent


class TestSizeFashnNotebook(unittest.TestCase):
    def install_cell(self):
        notebook = json.loads((ROOT/'notebooks/debug_size_fashn.ipynb').read_text(encoding='utf-8'))
        return next(''.join(cell['source']) for cell in notebook['cells']
                    if cell['cell_type'] == 'code' and ''.join(cell['source']).startswith('# 3.'))

    def test_install_and_weight_free_import_check_use_same_python(self):
        results = [subprocess.CompletedProcess([], 0, stdout='install log'),
                   subprocess.CompletedProcess([], 0, stdout='import log')]
        output = StringIO()
        with patch('subprocess.run', side_effect=results) as runner, redirect_stdout(output):
            exec(compile(self.install_cell(), 'colab_install_cell', 'exec'), {'FASHN': '/test/fashn'})
        install, imports = [call.args[0] for call in runner.call_args_list]
        self.assertEqual(install[:4], [sys.executable, '-m', 'pip', 'install'])
        self.assertIn('gradio==6.30.0', install)
        self.assertIn('transformers==5.20.2', install)
        self.assertIn('huggingface-hub>=1.31,<3', install)
        self.assertNotIn('--no-deps', install)
        self.assertNotIn('--dry-run', install)
        self.assertEqual(imports[:2], [sys.executable, '-c'])
        self.assertIn('from fashn_vton import TryOnPipeline', imports[2])
        self.assertIn('from transformers import SegformerForSemanticSegmentation', imports[2])
        self.assertNotIn('from_pretrained(', imports[2])
        self.assertNotIn('TryOnPipeline(', imports[2])
        self.assertIn('install log', output.getvalue())
        self.assertIn('import log', output.getvalue())
        compile(imports[2], 'import_check', 'exec')

    def test_failed_install_prints_captured_error_and_stops_before_import(self):
        output = StringIO()
        failure = subprocess.CompletedProcess([], 1, stdout='ERROR: dependency conflict')
        with patch('subprocess.run', return_value=failure) as runner, redirect_stdout(output):
            with self.assertRaises(subprocess.CalledProcessError):
                exec(compile(self.install_cell(), 'colab_install_cell', 'exec'), {'FASHN': '/test/fashn'})
        runner.assert_called_once()
        self.assertIn('ERROR: dependency conflict', output.getvalue())


if __name__ == '__main__':
    unittest.main()
