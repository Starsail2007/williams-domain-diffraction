"""Offline loopback UI with one managed scientific worker at a time."""
import base64
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import quote, unquote, urlsplit
import webbrowser
from .delivery_io import read_json, verify_bundle


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False)+'\n',encoding='utf-8');temp.replace(path)


def decode_upload(request):
    try:
        scale=float(request.get('scale'))
        if not math.isfinite(scale) or scale<=0:raise ValueError()
    except (ValueError,TypeError):raise ValueError('请输入有限的正数标尺（µm/px）')
    try:data=base64.b64decode(request['image'],validate=True)
    except ValueError:raise ValueError('图片数据无效，请重新选择PNG或JPG')
    if len(data)>32*1024**2:raise ValueError('图片不能超过32 MB')
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ('PNG','JPEG') or im.mode not in ('L','RGB','RGBA') or min(im.size)<64 or im.width*im.height>24_000_000:raise ValueError()
            image_format=im.format;im.verify()
    except Exception as exc:raise ValueError('请使用每边至少64像素、总计不超过2400万像素的8位PNG或JPG') from exc
    return data,scale,image_format


class Lab:
    def __init__(self,root,headless_console=False):
        self.root=root;self.token=secrets.token_urlsafe(24);self.jobs={};self.lock=threading.Lock();self.closed=False
        self.headless_console=headless_console;self.url=''
        self.cases={q['id']:q for q in read_json(root/'configs/experiments.json')['tools']}

    def follow_worker(self,job):
        while job['process'].poll() is None:
            time.sleep(.15)
        state=self.state(job)
        save(job['folder']/'console_status.json',dict(status=state['status'],message=state.get('message','')))
        if state['status']=='complete':
            print('已完成：'+(job.get('title') or self.cases[job['case']]['title'])+'。HTML 页面中已经有本次计算结果。',flush=True)

    def state(self,job):
        p=job['folder']/'result/state.json'
        state=read_json(p) if p.exists() else dict(status='starting',events=[],message='正在准备本次计算',version=0)
        if job.get('batch'):state.setdefault('tasks',job['tasks'])
        code=job['process'].poll()
        if job.get('cancelled'):
            state.update(status='cancelled',message='已停止')
            for task in state.get('tasks',[]):
                if task['status'] not in ('complete','failed'):task.update(status='cancelled',message='已停止')
        if code is not None and code!=0 and state['status']=='complete':
            state.update(status='failed',message=f'计算进程异常退出（{code}），请查看日志')
        if code is None and state['status']=='complete':
            state.update(status='running',message='正在结束计算并保存记录')
        if code is not None and state['status'] not in ('complete','failed','cancelled'):
            state.update(status='cancelled' if job.get('cancelled') else 'failed',message='计算已停止' if job.get('cancelled') else f'计算进程退出（{code}），请查看日志')
        if code is not None and state['status']=='failed':
            for task in state.get('tasks',[]):
                if task['status'] in ('running','starting'):task.update(status='failed',message='计算进程中断')
                elif task['status']=='pending':task.update(status='cancelled',message='未运行')
        return {**state,'job':job['id'],'case':job['case'],'algorithm':job.get('algorithm',job['case']),
                'sample':job.get('sample'),'log':(job['folder']/'worker.log').relative_to(self.root).as_posix()}

    def start(self,request):
        if 'tasks' in request:return self.start_batch(request)
        with self.lock:
            if self.closed:raise ValueError('演示已经关闭，请重新启动项目')
            if any(j['process'].poll() is None for j in self.jobs.values()):raise ValueError('已有计算正在进行。请等待完成或停止本次计算。')
            from .independent_lab import select_definition
            case=request.get('algorithm');sample_id=request.get('sample')
            tool,sample=select_definition(self.root,case,sample_id)
            if 'mode' in request:raise ValueError('各算法独立运行，无需分步或连续模式')
            data=None;scale=None
            if 'image' in request:
                if tool['type']!='texture' or sample_id!='own_image':raise ValueError('图片上传只用于自己的显微图像')
                data,scale,image_format=decode_upload(request)
            elif sample_id=='own_image':raise ValueError('请先选择显微图片并填写标尺')
            job_id=time.strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(4)
            folder=self.root/'work/lab_runs'/job_id;folder.mkdir(parents=True,exist_ok=False)
            args=[sys.executable,'-u','-I','-B','-X','utf8',str(self.root/'scripts/project.py'),'experiment','--algorithm',case,'--sample',sample_id,'--out',str(folder/'result')]
            if data is not None:
                image=folder/('input.png' if image_format=='PNG' else 'input.jpg');image.write_bytes(data)
                args+=['--image',str(image),'--pixel-size-um',str(scale)]
            save(folder/'request.json',dict(algorithm=case,sample=sample_id,image_name=Path(str(request.get('filename',''))).name,scale=scale))
            from .terminal_console import open_console
            try:
                console=open_console(self.root,folder,dict(
                    title=tool['title']+' · '+sample['label'],steps=[dict(id=case,title=tool['title'])],
                    url=self.url+'#algorithm/'+quote(case,safe='')+'/'+quote(sample_id,safe=''),
                    output=(folder/'result').relative_to(self.root).as_posix(),
                    log=(folder/'worker.log').relative_to(self.root).as_posix()),self.headless_console)
            except Exception as error:
                save(folder/'console_status.json',dict(status='failed',message=str(error)))
                raise ValueError('命令行窗口未打开，计算尚未启动：'+str(error)) from error
            with (folder/'worker.log').open('w',encoding='utf-8') as log:process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=self.root)
            job=dict(id=job_id,folder=folder,case=case,algorithm=case,sample=sample_id,process=process,console=console);self.jobs[job_id]=job
            watcher=threading.Thread(target=self.follow_worker,args=(job,),daemon=True);job['watcher']=watcher;watcher.start()
            return self.state(job)

    def start_batch(self,request):
        with self.lock:
            if self.closed:raise ValueError('演示已经关闭，请重新启动项目')
            if any(j['process'].poll() is None for j in self.jobs.values()):raise ValueError('已有计算正在进行。请等待完成或停止本次计算。')
            from .batch_lab import validate_tasks
            tasks=validate_tasks(self.root,request.get('tasks'))
            if 'mode' in request:raise ValueError('批任务自动完成，无需分步模式')
            own=any(task['sample']=='own_image' for task in tasks)
            data=scale=image_format=None
            if own:
                if 'image' not in request:raise ValueError('请先选择显微图片并填写标尺')
                data,scale,image_format=decode_upload(request)
            elif 'image' in request:raise ValueError('所选算法未使用自选图像')
            job_id=time.strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(4)
            folder=self.root/'work/lab_runs'/job_id;folder.mkdir(parents=True,exist_ok=False)
            if own:
                image=folder/('input.png' if image_format=='PNG' else 'input.jpg');image.write_bytes(data)
                for task in tasks:
                    if task['sample']=='own_image':task.update(image=str(image),pixel_size_um=scale)
            save(folder/'tasks.json',tasks)
            save(folder/'request.json',dict(tasks=[dict(algorithm=q['algorithm'],sample=q['sample']) for q in tasks],image_name=Path(str(request.get('filename',''))).name,scale=scale))
            first=tasks[0];title='所选算法'
            from .terminal_console import open_console
            try:
                console=open_console(self.root,folder,dict(title=title,
                    steps=[dict(id=q['algorithm'],title=self.cases[q['algorithm']]['title']) for q in tasks],
                    url=self.url+'#algorithm/'+quote(first['algorithm'],safe='')+'/'+quote(first['sample'],safe=''),
                    output=(folder/'result').relative_to(self.root).as_posix(),log=(folder/'worker.log').relative_to(self.root).as_posix()),self.headless_console)
            except Exception as error:
                save(folder/'console_status.json',dict(status='failed',message=str(error)))
                raise ValueError('命令行窗口未打开，计算尚未启动：'+str(error)) from error
            args=[sys.executable,'-u','-I','-B','-X','utf8',str(self.root/'scripts/project.py'),'batch-experiments','--tasks',str(folder/'tasks.json'),'--out',str(folder/'result')]
            with (folder/'worker.log').open('w',encoding='utf-8') as log:process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=self.root)
            pending=[dict(task_id=f'task_{i+1:03d}',algorithm=q['algorithm'],sample=q['sample'],status='pending',events=[]) for i,q in enumerate(tasks)]
            job=dict(id=job_id,folder=folder,case='batch',algorithm=None,sample=None,batch=True,title=title,tasks=pending,process=process,console=console)
            self.jobs[job_id]=job
            watcher=threading.Thread(target=self.follow_worker,args=(job,),daemon=True);job['watcher']=watcher;watcher.start()
            return self.state(job)

    def cancel(self,job_id):
        with self.lock:
            job=self.jobs.get(job_id)
            if job is None:raise ValueError('未找到本次计算')
            job['cancelled']=True
            if job['process'].poll() is None:
                job['process'].terminate()
                try:job['process'].wait(timeout=5)
                except subprocess.TimeoutExpired:job['process'].kill();job['process'].wait(timeout=5)
            path=job['folder']/'result/state.json'
            if path.exists():
                state=read_json(path);state.update(status='cancelled',message='已停止',version=state.get('version',0)+1)
                for task in state.get('tasks',[]):
                    if task['status'] not in ('complete','failed'):task.update(status='cancelled',message='未运行' if task['status']=='pending' else '已停止')
                save(path,state)
            return self.state(job)

    def close(self):
        self.closed=True
        for job in list(self.jobs.values()):
            if job['process'].poll() is None:self.cancel(job['id'])
        for job in self.jobs.values():
            if job.get('watcher'):job['watcher'].join(timeout=3)


def serve(root,port=0,no_browser=False,session_file=None,headless_console=False):
    root=Path(root).resolve();verify_bundle(root,quiet=True);lab=Lab(root,headless_console)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(root),**kwargs)
        def log_message(self,*args):pass
        def end_headers(self):
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store');super().end_headers()
        def allowed_host(self):return self.headers.get('Host')==f'127.0.0.1:{self.server.server_port}'
        def response(self,value,status=200):
            payload=(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n').encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            if not self.allowed_host():self.send_error(403);return
            path=unquote(urlsplit(self.path).path)
            if path=='/api/session':self.response(dict(token=lab.token,jobs=[lab.state(j) for j in lab.jobs.values()]));return
            if path.startswith('/api/job/'):
                job=lab.jobs.get(path.rsplit('/',1)[-1]);self.response(lab.state(job) if job else dict(error='没有这个运行'),200 if job else 404);return
            if path=='/':self.path='/START_HERE.html';path=self.path
            target=(root/path.lstrip('/')).resolve();relative=target.relative_to(root) if target.is_relative_to(root) else None
            allowed={'START_HERE.html','app','src','scripts','configs','data','results','paper','docs','provenance','references','work','README.md'}
            if relative is None or not relative.parts or relative.parts[0] not in allowed or not target.is_file():self.send_error(404);return
            if relative.parts[0]=='work' and (len(relative.parts)<2 or relative.parts[1]!='lab_runs'):self.send_error(404);return
            if target.suffix=='.py':self.extensions_map={**self.extensions_map,'.py':'text/plain; charset=utf-8'}
            super().do_GET()
        def do_POST(self):
            if not self.allowed_host() or self.headers.get('X-Lab-Token')!=lab.token:self.send_error(403);return
            origin=self.headers.get('Origin')
            if origin is not None and origin!=f'http://127.0.0.1:{self.server.server_port}':self.send_error(403);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=46*1024**2:raise ValueError('请求大小无效')
                request=json.loads(self.rfile.read(length))
                if not isinstance(request,dict):raise ValueError('请求格式无效')
                path=urlsplit(self.path).path
                if path=='/api/run':answer=lab.start(request)
                elif path=='/api/cancel':answer=lab.cancel(request.get('job'))
                elif path=='/api/shutdown':
                    lab.close();answer=dict(status='closed');threading.Thread(target=self.server.shutdown,daemon=True).start()
                else:self.send_error(404);return
                self.response(answer)
            except (ValueError,KeyError,TypeError) as exc:self.response(dict(error=str(exc)),400)
            except Exception as exc:self.response(dict(error='操作未完成：'+str(exc)),500)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    url=f'http://127.0.0.1:{server.server_port}/'
    lab.url=url
    if session_file:save(Path(session_file),dict(url=url,pid=os.getpid()))
    print('研究界面已就绪：'+url,flush=True)
    print('在页面中选择案例。结束时点击页面右上角“结束演示”。',flush=True)
    print('点击运行会打开独立命令行窗口；计算完成后，HTML 页面会显示本次结果。',flush=True)
    if not no_browser:
        try:
            if not webbrowser.open(url):print('请在浏览器打开上面的本地地址。',flush=True)
        except Exception as exc:print('自动打开浏览器未完成，请打开上面的地址：'+str(exc),flush=True)
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:print('\n演示已结束。',flush=True)
    finally:lab.close();server.server_close()
