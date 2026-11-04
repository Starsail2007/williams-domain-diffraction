"""One reading route shared by saved results and newly calculated chapters."""
import html
import os
from pathlib import Path
from .delivery_io import read_json

STYLE='''body{max-width:1080px;margin:40px auto;padding:0 24px;font:17px/1.8 system-ui,sans-serif;color:#243b46;background:#fafbf9}h1{font-size:32px;line-height:1.45}h2{font-size:23px;margin-top:38px}h3{font-size:19px}a{color:#086b79;text-underline-offset:4px}nav{display:flex;gap:20px;flex-wrap:wrap;font-size:15px}figure,details,.panel{margin:24px 0;padding:20px;background:white;border:1px solid #dce4de;border-radius:8px}figure img{display:block;max-width:100%;height:auto;margin:auto}figcaption{font-size:15px;color:#50646c;margin-top:12px}summary{cursor:pointer;font-weight:600}.muted{color:#60747b;font-size:15px}.buttons{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}.buttons a{display:inline-block;padding:8px 18px;border:1px solid #397b7d;border-radius:6px;text-decoration:none;background:white}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:9px 12px;border-bottom:1px solid #e0e6e2;text-align:left}.scroll{overflow-x:auto}code{background:#eef2ef;padding:2px 5px}pre{padding:16px;background:#eef2ef;overflow-x:auto;font-size:14px}.route{display:grid;grid-template-columns:46px 1fr;gap:4px 14px;margin:18px 0;padding:18px 0;border-bottom:1px solid #dce4de}.route b{font-size:25px;font-weight:500;color:#337878}.route h2{margin:0;font-size:21px}.route p{margin:3px 0;font-size:16px}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}.gallery figure{margin:0}.next{margin-top:38px;padding-top:20px;border-top:1px solid #cad8ce}footer{margin-top:36px;font-size:14px;color:#60747b}li{margin:6px 0}'''

def href(folder,target):
    try:return os.path.relpath(target,folder).replace(os.sep,'/')
    except ValueError:
        # Windows CLI output may be on another drive. Saved public pages use
        # same-root relative links; this applies only to that local run.
        return Path(target).resolve().as_uri()
def a(folder,target,label):return '<a href="'+html.escape(href(folder,target),quote=True)+'">'+html.escape(label)+'</a>'
def figure(folder,target,caption):
    link=html.escape(href(folder,target),quote=True)
    return f'<figure><a href="{link}"><img loading="lazy" src="{link}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'
def table(headers,rows):
    from .algorithm_graphics import table as original_table
    return original_table(headers,rows)
def details(title,body):return '<details><summary>'+html.escape(title)+'</summary>'+body+'</details>'
def write_page(path,title,body):
    Path(path).write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+STYLE+'</style>'+body+'</html>',encoding='utf-8')

def chapter_page(root,out,chapter,body,computed,run_root,selected):
    chapters=read_json(root/'configs/chapters.json');index=next(i for i,q in enumerate(chapters) if q['id']==chapter)
    item=chapters[index]
    nav='<nav>'+a(out,root/'START_HERE.html','研究首页')+' '+a(out,root/'paper/manuscript.pdf','论文PDF')+' '+a(out,root/'README.md','项目说明')+'</nav>'
    intro=f'<h1>{index+1}. {html.escape(item["title"])}</h1><p>{html.escape(item["question"])}</p>'
    command='--cli research --chapter '+chapter+' --out work/'+chapter+'_01'
    rerun=details('重新计算本节','<p>双击Mac或Windows启动器，在研究界面选择本节的案例，再点击分步计算或计算整个案例。</p><p>命令行参数：</p><pre>'+command+'</pre><p class="muted">每次写入新目录。下方数据记录说明了本节实际重算范围。</p>')
    links=[]
    for j,label in ((index-1,'上一环节：'),(index+1,'继续：')):
        if not 0<=j<len(chapters):continue
        q=chapters[j];target=(run_root if q['id'] in selected else root/'results/research')/q['folder']/'index.html'
        suffix='' if q['id'] in selected else '（随包结果）'
        links.append(a(out,target,label+q['title']+suffix))
    bottom='<div class="next"><nav>'+' '.join(links)+'</nav></div>'
    footer='<footer>'+a(out,out/'chapter.json','本次计算记录')+' · '+a(out,root/'src/williams_diffraction/research_route.py','研究流程代码')+'</footer>'
    write_page(out/'index.html',item['title'],nav+intro+body+rerun+bottom+footer)

def run_index(root,out,chapters,results):
    cards=''
    for q,result in zip(chapters,results):cards+='<div class="route"><b>'+str(q['number'])+'</b><div><h2>'+a(out,out/q['folder']/'index.html',q['title'])+'</h2><p>'+html.escape(result['reader_result'])+'</p></div></div>'
    write_page(out/'index.html','本次研究复现','<nav>'+a(out,root/'START_HERE.html','研究首页')+'</nav><h1>本次研究复现</h1><p>以下是本次计算生成的结果。各环节保留所用实验与输入来源。</p>'+cards+'<footer>'+a(out,out/'research_run.json','完整运行记录')+'</footer>')
