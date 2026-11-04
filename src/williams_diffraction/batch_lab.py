"""Run a fixed selection independently, with one live state and stdout stream."""
from pathlib import Path
import time
import traceback
from .delivery_io import new_output,read_json,write_json
from .guided_lab import atomic_json
from .independent_lab import run_experiment,select_definition


def validate_tasks(root,tasks):
    if not isinstance(tasks,list) or not 1<=len(tasks)<=30:
        raise ValueError('请选择1到30项算法')
    ids=set();clean=[]
    for task in tasks:
        if not isinstance(task,dict):raise ValueError('任务格式无效')
        algorithm,sample=task.get('algorithm'),task.get('sample')
        select_definition(root,algorithm,sample)
        if algorithm in ids:raise ValueError('同一算法在一批中只能选择一次')
        ids.add(algorithm);clean.append(dict(algorithm=algorithm,sample=sample))
    return clean


def run_batch(root,output,tasks):
    root=Path(root).resolve();clean=validate_tasks(root,tasks)
    out=new_output(root,output);(out/'tasks').mkdir()
    entries=[dict(task_id=f'task_{i+1:03d}',**task,status='pending',events=[],version=0) for i,task in enumerate(clean)]
    state=dict(status='running',events=[],tasks=entries,version=0,current_task=None,message='运行中')
    started=time.monotonic()
    def publish():
        state['version']+=1;atomic_json(out/'state.json',state)
    publish()
    for index,(task,entry) in enumerate(zip(tasks,entries)):
        tool,sample=select_definition(root,entry['algorithm'],entry['sample'])
        print(f"\n=== {index+1}/{len(entries)} · {tool['title']} · {sample['label']} ===",flush=True)
        entry['status']='running';state['current_task']=entry['task_id'];publish()
        child=out/'tasks'/entry['task_id']
        def update(child_state):
            entry.update(child_state);publish()
        try:
            run_experiment(root,child,entry['algorithm'],entry['sample'],task.get('image'),task.get('pixel_size_um'),on_state=update)
        except Exception as error:
            traceback.print_exc()
            entry.update(status='failed',message=str(error));publish()
    failures=[entry for entry in entries if entry['status']!='complete']
    state.update(status='failed' if failures else 'complete',message='部分任务失败' if failures else '已完成',elapsed_s=time.monotonic()-started)
    publish();write_json(out/'run.json',state)
    return state
