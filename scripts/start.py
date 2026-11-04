#!/usr/bin/env python3
"""Open the local research guide using the shared package entry."""
from pathlib import Path
import os
import secrets
import subprocess
import sys
import time


def main(no_browser=False):
    root=Path(__file__).resolve().parents[1]
    folder=root/'work/sessions'/(time.strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(3))
    folder.mkdir(parents=True)
    session=folder/'session.json'
    args=[sys.executable,'-I','-B','-X','utf8',str(root/'scripts/project.py'),'serve','--session-file',str(session)]
    if no_browser:args.append('--no-browser')
    options=dict(start_new_session=True) if os.name!='nt' else dict(creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    with (folder/'server.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(args,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,**options)
    deadline=time.monotonic()+30
    while not session.exists():
        if process.poll() is not None:
            raise RuntimeError('界面启动失败：'+(folder/'server.log').read_text(encoding='utf-8'))
        if time.monotonic()>deadline:
            process.terminate()
            raise RuntimeError('界面启动超时，请查看 '+str(folder/'server.log'))
        time.sleep(.1)
    print('HTML 运行界面已打开。点击运行后，会另开命令行窗口显示计算过程。',flush=True)
    print('可以关闭这个启动窗口；计算完成后请回到 HTML 查看结果。',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
