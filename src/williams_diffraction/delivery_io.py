"""IO for this independent delivery; no historical path resolver or hidden input."""
import csv
import hashlib
import html
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bundle(root, quiet=False):
    root = Path(root).resolve()
    manifest = read_json(root / 'provenance/input_manifest.json')
    for row in manifest['files']:
        p = root / row['path']
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError('Input link escapes or aliases a scientific input: ' + row['path'])
        if not p.is_file() or p.stat().st_size != row['bytes'] or digest(p) != row['sha256']:
            raise ValueError('Scientific input changed or missing: ' + row['path'])
    if not quiet:
        print(f"PASS: {len(manifest['files'])} scientific input files; no RAW required.")
    return manifest


def new_output(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or any(out.is_relative_to(root / d) for d in ('app','src','data','paper','docs','results','provenance','scripts','tests','examples','configs','runtime-bundles','.runtime')):
        raise ValueError('Use a new work/ directory or a separate external output directory')
    out.mkdir(parents=True, exist_ok=False)
    return out


def parity_svg(path, pairs, title, x_label='Observed', y_label='Recomputed'):
    """Small dependency-free diagnostic; values come from this run, not saved images."""
    pairs = list(pairs)
    if not pairs:
        raise ValueError('No points to plot')
    values = [float(x) for pair in pairs for x in pair]
    lo, hi = min(values), max(values)
    margin = max((hi-lo)*.05, 1e-8)
    lo, hi = lo-margin, hi+margin
    def px(x): return 70 + (float(x)-lo)/(hi-lo)*480
    def py(y): return 550 - (float(y)-lo)/(hi-lo)*480
    circles = ''.join(f'<circle cx="{px(x):.3f}" cy="{py(y):.3f}" r="2.4" fill="#167d9a" opacity=".65"/>' for x,y in pairs)
    Path(path).write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="650" height="630" viewBox="0 0 650 630">
<rect width="650" height="630" fill="white"/><g font-family="sans-serif" fill="#152c3c">
<text x="70" y="32" font-size="18">{html.escape(title)}</text>
<path d="M70 70 V550 H550" fill="none" stroke="#152c3c"/><path d="M70 550 L550 70" stroke="#b05a2a" stroke-dasharray="5 4"/>
{circles}<text x="210" y="605">{html.escape(x_label)}</text><text x="10" y="50">{html.escape(y_label)}</text>
<text x="70" y="575">{lo:.5g}</text><text x="515" y="575">{hi:.5g}</text>
<text x="70" y="60">n={len(pairs)}; diagonal is equality</text></g></svg>''', encoding='utf-8')
