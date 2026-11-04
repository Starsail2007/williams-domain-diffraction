"""A visible console follows the real worker state, independently of the page."""
import json
import codecs
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def open_console(root, folder, metadata, headless=False):
    write_json(folder/'console.json', metadata)
    args = [sys.executable, '-I', '-B', '-X', 'utf8',
            str(root/'scripts/project.py'), 'console', '--job', str(folder)]
    if headless:
        with (folder/'console.log').open('w', encoding='utf-8') as output:
            process = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT, cwd=root)
    elif sys.platform == 'darwin':
        command = folder/'查看计算过程.command'
        command.write_text('#!/bin/zsh\ncd ' + shlex.quote(str(root)) +
                           '\n' + shlex.join(args) + '\n', encoding='utf-8')
        command.chmod(0o755)
        subprocess.run(['/usr/bin/open', '-a', 'Terminal', str(command)], check=True, timeout=15)
        process = None
    elif sys.platform == 'win32':
        command = folder/'查看计算过程.bat'
        quote = lambda value: '"' + str(value).replace('%', '%%') + '"'
        text = '@echo off\r\nchcp 65001 >nul\r\nsetlocal DisableDelayedExpansion\r\n'
        text += ' '.join(quote(value) for value in args) + ' <CON >CON 2>&1\r\n'
        command.write_bytes(text.encode('utf-8'))
        process = subprocess.Popen(
            [os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/c', str(command)],
            creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=root)
    else:
        raise ValueError('本界面支持 macOS 和 Windows 的命令行窗口')
    deadline = time.monotonic() + 20
    while not (folder/'console_ready.json').exists():
        if process is not None and process.poll() is not None:
            raise RuntimeError('命令行窗口未能启动，请查看本次日志')
        if time.monotonic() > deadline:
            raise RuntimeError('没有收到命令行窗口的启动确认；本次尚未开始计算，请重新运行')
        time.sleep(.1)
    ready=json.loads((folder/'console_ready.json').read_text(encoding='utf-8'))
    if not headless and not ready.get('tty'):
        raise RuntimeError('命令行未连接到可见终端，计算尚未启动')
    return process


def run_console(root, folder):
    root, folder = Path(root).resolve(), Path(folder).resolve()
    if not folder.is_relative_to(root/'work/lab_runs'):
        raise ValueError('Console job must belong to this project')
    metadata = json.loads((folder/'console.json').read_text(encoding='utf-8'))
    def say(message):
        print(message,flush=True)
        with (folder/'console_transcript.log').open('a',encoding='utf-8') as transcript:
            transcript.write(message+'\n')
    say('\n' + metadata['title'])
    say('这里实时显示所选算法的中间变量与实际输出。请保留此窗口。\n')
    write_json(folder/'console_ready.json', dict(pid=os.getpid(),tty=sys.stdin.isatty()))
    offset = 0
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    def stream_worker(drain=False):
        nonlocal offset
        log=folder/'worker.log'
        if not log.exists():return
        with log.open('rb') as source:
            source.seek(offset)
            while True:
                chunk=source.read(256*1024)
                if not chunk:break
                offset=source.tell()
                text=decoder.decode(chunk)
                sys.stdout.write(text);sys.stdout.flush()
                with (folder/'console_transcript.log').open('a',encoding='utf-8') as transcript:
                    transcript.write(text)
                if not drain:break
    previous_wait = None
    while True:
        stream_worker()
        state_file = folder/'result/state.json'
        if state_file.exists():
            state = json.loads(state_file.read_text(encoding='utf-8'))
            waiting = state.get('current_step') if state.get('status') == 'waiting' else None
            if waiting and waiting != previous_wait:
                say('计算已暂停，请查看运行记录。')
            previous_wait = waiting
        final = folder/'console_status.json'
        if final.exists():
            stream_worker(drain=True)
            last=decoder.decode(b'',final=True)
            if last:
                sys.stdout.write(last);sys.stdout.flush()
            status = json.loads(final.read_text(encoding='utf-8'))
            if status['status'] == 'complete':
                say('\n已完成。HTML 页面中已经有本次计算结果。')
                say('查看结果：' + metadata['url'])
                say('结果目录：' + metadata['output'])
            elif status['status'] == 'cancelled':
                say('\n本次计算已停止。已完成的步骤保留在 HTML 页面中。')
            else:
                say('\n计算未完成：' + status.get('message', '请查看日志'))
                say('日志：' + metadata['log'])
            break
        time.sleep(.15)
    if sys.stdin.isatty():
        try:
            input('\n按回车结束此窗口；HTML 页面仍可查看结果。')
        except (EOFError, KeyboardInterrupt):
            pass
