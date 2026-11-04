#!/usr/bin/env python3
"""Check the bundled runtime, then enter the menu or the explicit CLI."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def check_environment():
    if not Path(sys.executable).resolve().is_relative_to(ROOT/'.runtime'):
        raise RuntimeError('请用根目录的run_reproduction.command（Mac）或run_reproduction.bat（Windows）启动。')
    lock=json.loads((ROOT/'configs/environment_lock.json').read_text())
    if platform.python_version()!=lock['python_version']:
        raise RuntimeError('项目Python版本与环境记录不符，请重新启动以检查项目内环境。')
    modules={};names={'numpy':'numpy','scipy':'scipy','Pillow':'PIL','opencv-python-headless':'cv2'}
    for distribution,name in names.items():
        module=importlib.import_module(name);version=importlib.metadata.version(distribution)
        if version!=lock['dependencies'][distribution]:raise RuntimeError('包内依赖版本不符：'+distribution)
        path=Path(module.__file__).resolve()
        if not path.is_relative_to(ROOT/'.runtime'):raise RuntimeError('依赖来自项目外部：'+name)
        modules[name]=dict(version=version,file=path.relative_to(ROOT).as_posix())
    return dict(status='PASS',python=platform.python_version(),platform=platform.system(),machine=platform.machine(),
                executable=Path(sys.executable).resolve().relative_to(ROOT).as_posix(),modules=modules,uses_system_python=False)


def main():
    os.environ['PYTHONNOUSERSITE']='1'
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    try:result=check_environment()
    except Exception as e:
        print('项目环境检查未通过：'+str(e),file=sys.stderr)
        print('请保留完整项目目录和上方错误；重新启动会检查项目内环境。',file=sys.stderr)
        return 2
    args=sys.argv[1:]
    if args==['--check']:
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    print('项目专用环境已就绪，可离线运行。',flush=True)
    if args and args[0]=='--cli':script=ROOT/'scripts/project.py';args=args[1:]
    elif not args:script=ROOT/'scripts/start.py'
    else:raise ValueError('Supported: no args, --check, or --cli <project arguments>')
    return subprocess.run([sys.executable,'-I','-B',str(script),*args],cwd=ROOT,check=False).returncode


if __name__=='__main__':raise SystemExit(main())
