"""Prepare locked dependencies inside the project; never install globally."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import ssl
import subprocess
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def download(url, expected, destination):
    """Publish a cached download only after its exact upstream hash matches."""
    destination = Path(destination)
    if destination.is_file() and digest(destination) == expected:
        return destination
    if not url.startswith('https://'):
        raise ValueError('Dependency downloads require HTTPS')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f'.part.{os.getpid()}')
    try:
        # The standalone interpreter ships pip and its CA bundle on every target.
        from pip._vendor import certifi
        context = ssl.create_default_context(cafile=certifi.where())
        request = urllib.request.Request(url, headers={'User-Agent': 'Williams-Diffraction-Setup/0.8'})
        with urllib.request.urlopen(request, context=context, timeout=90) as response, temporary.open('wb') as output:
            copied = 0
            next_notice = 10 * 1024 * 1024
            while block := response.read(1024 * 1024):
                output.write(block)
                copied += len(block)
                if copied >= next_notice:
                    print(f'  {copied / 1024**2:.0f} MB', flush=True)
                    next_notice += 10 * 1024 * 1024
        if digest(temporary) != expected:
            raise ValueError('Downloaded file checksum mismatch: ' + destination.name)
        temporary.replace(destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(lock, target):
    spec = {'python': lock['python_version'], 'platform': lock['platforms'][target]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prepare(target):
    prefix = ROOT / '.runtime' / target
    if not Path(sys.executable).resolve().is_relative_to(prefix):
        raise RuntimeError('Setup must use the Python interpreter inside this project')
    lock = json.loads((ROOT / 'configs/environment_lock.json').read_text(encoding='utf-8'))
    if platform.python_version() != lock['python_version']:
        raise RuntimeError('Project Python version differs from environment_lock.json')
    marker = prefix / 'environment.json'
    expected = fingerprint(lock, target)
    if marker.exists():
        try:
            current = json.loads(marker.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            current = {}
        versions_ok = True
        for item in lock['platforms'][target]['wheels']:
            try:
                versions_ok &= importlib.metadata.version(item['name']) == item['version']
            except importlib.metadata.PackageNotFoundError:
                versions_ok = False
        if current.get('lock_sha256') == expected and versions_ok:
            return
    print('正在准备项目环境：从官方源获取锁定版本，仅安装到本项目。', flush=True)
    cache = ROOT / '.runtime/downloads' / target
    wheels = []
    for item in lock['platforms'][target]['wheels']:
        print(f"下载或核对 {item['name']} {item['version']}…", flush=True)
        wheels.append(download(item['url'], item['sha256'], cache / item['filename']))
    print('正在安装依赖；完成后将打开研究界面…', flush=True)
    marker.unlink(missing_ok=True)
    environment = {key: value for key, value in os.environ.items() if not key.startswith('PIP_')}
    environment.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull)
    subprocess.run([
        sys.executable, '-I', '-B', '-m', 'pip', '--isolated', 'install', '--no-index', '--no-deps',
        '--no-compile', '--no-cache-dir', '--disable-pip-version-check', '--force-reinstall',
        '--prefix', str(prefix / 'python'),
        *map(str, wheels),
    ], check=True, env=environment)
    temporary = marker.with_name(f'environment.writing.{os.getpid()}')
    temporary.write_text(json.dumps({'platform': target, 'lock_sha256': expected}, indent=2) + '\n', encoding='utf-8')
    temporary.replace(marker)
    print('环境准备完成。之后启动可离线运行。', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', required=True, choices=['macos-arm64', 'macos-x86_64', 'windows-x86_64'])
    args = parser.parse_args()
    try:
        prepare(args.platform)
    except Exception as error:
        print('环境准备未完成：' + str(error), file=sys.stderr)
        print('请检查网络与项目目录写入权限，再次双击启动器会重试。', file=sys.stderr)
        raise SystemExit(2)
