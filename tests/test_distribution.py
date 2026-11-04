"""Scientific inputs stay protected while project text and source stay editable."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from williams_diffraction.delivery_io import verify_bundle

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('project_bootstrap', ROOT / 'scripts/bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class DistributionTests(unittest.TestCase):
    def make_input(self, root):
        (root / 'data').mkdir()
        (root / 'provenance').mkdir()
        data = root / 'data/observation.csv'
        data.write_text('x,y\n1,2\n')
        item = dict(path='data/observation.csv', bytes=data.stat().st_size,
                    sha256=hashlib.sha256(data.read_bytes()).hexdigest())
        (root / 'provenance/input_manifest.json').write_text(json.dumps(dict(files=[item])))
        return data

    def test_readme_and_source_edits_do_not_block_input_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_input(root)
            (root / 'README.md').write_text('My research introduction')
            (root / 'src').mkdir()
            (root / 'src/example.py').write_text('value = 1\n')
            verify_bundle(root, quiet=True)
            (root / 'README.md').write_text('My revised introduction')
            (root / 'src/example.py').write_text('value = 2\n')
            self.assertEqual(len(verify_bundle(root, quiet=True)['files']), 1)
            self.assertFalse((root / 'runtime-bundles').exists())

    def test_missing_or_changed_scientific_input_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = self.make_input(root)
            data.write_text('x,y\n1,3\n')
            with self.assertRaisesRegex(ValueError, 'Scientific input changed'):
                verify_bundle(root, quiet=True)
            data.unlink()
            with self.assertRaisesRegex(ValueError, 'Scientific input changed'):
                verify_bundle(root, quiet=True)

    def test_reading_pdf_replacement_does_not_block_calculation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_input(root)
            (root / 'paper').mkdir()
            pdf = root / 'paper/manuscript.pdf'
            original = (ROOT / 'paper/manuscript.pdf').read_bytes()
            pdf.write_bytes(original)
            verify_bundle(root, quiet=True)
            pdf.write_bytes(original + b'\n% A revised reading copy\n')
            verify_bundle(root, quiet=True)

    def test_invalid_download_is_not_installed_or_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'dependency.whl'
            with patch.object(bootstrap.urllib.request, 'urlopen', return_value=io.BytesIO(b'invalid download')):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    bootstrap.download('https://files.pythonhosted.org/example.whl', '0' * 64, target)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_verified_cache_does_not_need_network(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'dependency.whl'
            target.write_bytes(b'verified dependency')
            expected = hashlib.sha256(target.read_bytes()).hexdigest()
            with patch.object(bootstrap.urllib.request, 'urlopen', side_effect=AssertionError('Network was used')):
                self.assertEqual(bootstrap.download('https://files.pythonhosted.org/example.whl', expected, target), target)


if __name__ == '__main__':
    unittest.main()
