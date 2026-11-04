"""Recompute the four technical sections."""
from contextlib import redirect_stdout
from pathlib import Path
import time
from .delivery_io import read_json,write_json,new_output,verify_bundle
from . import research_pages as pages

def connect_case_pages(root,chapter_out):
    for p in chapter_out.rglob('index.html'):
        if p==chapter_out/'index.html':continue
        text=p.read_text();nav='<nav>'+pages.a(p.parent,chapter_out/'index.html','返回本研究环节')+' · '+pages.a(p.parent,root/'START_HERE.html','研究首页')+'</nav>'
        if '<h1>' not in text:raise ValueError('Missing case page title: '+p.name)
        p.write_text(text.replace('<h1>',nav+'<h1>',1),encoding='utf-8')

def run_research(root,output,chapter='all'):
    verify_bundle(root,quiet=True);out=new_output(root,output);started=time.perf_counter()
    registry=read_json(root/'configs/chapters.json')
    chosen=[dict(q,number=i+1) for i,q in enumerate(registry) if chapter=='all' or q['id']==chapter]
    if not chosen:raise ValueError('No such research chapter')
    from .research_measurement import chapter as measurement
    from .research_analysis import periods,intensity,discussion
    functions=dict(measurement=measurement,periods=periods,intensity=intensity,discussion=discussion)
    results=[];selected=[q['id'] for q in chosen]
    for q in chosen:
        target=out/q['folder'];target.mkdir()
        print(f'{q["number"]}/{len(registry)}  {q["title"]}……',flush=True)
        with (target/'calculation.log').open('w',encoding='utf-8') as log:
            try:
                with redirect_stdout(log):body,result=functions[q['id']](root,target)
            except Exception:
                print('本节未完成，计算日志：'+str(target/'calculation.log'),flush=True)
                raise
        result.update(chapter=q['id'],inputs_and_limits='Each chapter declares its experiment and input stage; illustrative pixel processing is not a synthetic end-to-end source for the historical model datasets.')
        write_json(target/'chapter.json',result)
        pages.chapter_page(root,target,q['id'],body,result,out,selected)
        connect_case_pages(root,target);results.append(result)
        print('完成：'+q['title'],flush=True)
    report=dict(status='PASS',chapters=results,elapsed_s=time.perf_counter()-started,network_required=False,full_raw_required=False,
                scope='Fresh calculations use pixels, accepted integrals, or labelled coordinates as explicitly declared; mechanism diagrams illustrate hypotheses.')
    write_json(out/'research_run.json',report);pages.run_index(root,out,chosen,results)
    print('结果：'+str(out/'index.html'),flush=True)
    return report

def run_component(root,output,component):
    """Direct RG/photometry commands for an independently runnable method."""
    verify_bundle(root,quiet=True);out=new_output(root,output)
    from .research_measurement import run_rg,run_photometry,native_profiles
    if component=='rg':
        result=run_rg(root,out/'rg')
        from .delivery_io import read_json
        import numpy as np
        info=read_json(root/'data/research/photometry125.json')
        with np.load(root/'data/research'/info['array'],allow_pickle=False) as archive:native_profiles(archive['mosaic'],info,out/'rg')
        body=pages.figure(out,out/'rg/01_clipping.png','原生R/G剖面，不插值、不缩放增益。')+pages.figure(out,out/'rg/02_rg_and_residual.png','重新拟合既定未饱和积分，并查看残差。')+pages.a(out,out/'rg/fits.csv','三轮、两孔径与留出结果')
        title='RG通道拟合'
    else:
        result=run_photometry(root,out/'photometry')
        body=pages.figure(out,out/'photometry/125_audit_p007_measurement.png','原生CFA测光与实际积分区域。')+pages.a(out,out/'photometry/net_integrals.csv','全部22个光斑的净积分')
        title='RAW相对测光'
    write_json(out/'component.json',result)
    pages.write_page(out/'index.html',title,'<nav>'+pages.a(out,root/'START_HERE.html','研究首页')+'</nav><h1>'+title+'</h1>'+body+'<footer>'+pages.a(out,out/'component.json','本次计算记录')+'</footer>')
    return result

def run_own_image(root,output,image,pixel_size_um):
    import io
    from .delivery_algorithms import run_algorithms
    out=Path(output).resolve();log=io.StringIO()
    print('正在分析显微图像……',flush=True)
    try:
        with redirect_stdout(log):report=run_algorithms(root,out,image=image,pixel_size_um=pixel_size_um)
    except Exception:
        if out.is_dir():(out/'calculation.log').write_text(log.getvalue(),encoding='utf-8')
        raise
    (out/'calculation.log').write_text(log.getvalue(),encoding='utf-8')
    q=report['cases'][0]
    body='<nav>'+pages.a(out,root/'START_HERE.html','研究首页')+'</nav><h1>自己的显微图像</h1><p>按输入的标定尺度重新计算频谱、频率族、裁剪和分块。</p>'
    body+=pages.table(['最强峰周期 µm','主频率族周期 µm','频率族歧义'],[[q['strongest_period_um'],q['family_period_um'],q['family_ambiguous']]])
    body+='<p>'+pages.a(out,out/'own_image/index.html','查看原图、频谱与检查过程')+'</p><p>主频率族未判定时保留空值，不强行选择周期。纹理频率不唯一确定取向基频。</p>'
    pages.write_page(out/'index.html','自己的显微图像',body)
    child=out/'own_image/index.html';text=child.read_text();nav='<nav>'+pages.a(child.parent,out/'index.html','本次结果')+' · '+pages.a(child.parent,root/'START_HERE.html','研究首页')+'</nav>'
    child.write_text(text.replace('<h1>',nav+'<h1>',1),encoding='utf-8')
    print('结果：'+str(out/'index.html'),flush=True)
    return report
