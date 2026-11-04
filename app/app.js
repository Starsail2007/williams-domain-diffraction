/* Independently select an algorithm and its input; no stage unlocking. */
(function(){
  'use strict';
  const catalog=window.WILLIAMS_CATALOG,G=window.GuideState,main=document.getElementById('main');
  const tools=new Map(catalog.tools.map(t=>[t.id,t])),runs=new Map(),lastSamples=new Map(),selections=new Map();
  let session=null,job=null,busy=false,closed=false,noticeTimer,ownFile=null,ownURL=null,ownScale='';
  const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const url=p=>String(p).split('/').map(encodeURIComponent).join('/');
  const fmt=x=>x===null||x===undefined?'—':typeof x==='boolean'?(x?'是':'否'):Array.isArray(x)?x.map(fmt).join('、'):typeof x==='number'?(Number.isInteger(x)?String(x):Math.abs(x)>1e6||Math.abs(x)<.0001?x.toExponential(4):Number(x.toPrecision(6)).toString()):String(x);
  const key=(algorithm,sample)=>JSON.stringify([algorithm,sample]);
  const runKey=run=>key(run.algorithm,run.sample);
  function route(){
    const parts=String(location.hash||'').replace(/^#/,'').split('/').map(x=>{try{return decodeURIComponent(x);}catch{return '';}});
    const tool=tools.get(parts[1])||catalog.tools[0];
    const sample=tool.samples.find(q=>q.id===parts[2])||tool.samples.find(q=>q.id===lastSamples.get(tool.id))||tool.samples[0];
    return {tool,sample};
  }
  const currentRun=(t,s)=>runs.get(key(t.id,s.id))||null;
  const phaseTools=t=>catalog.tools.filter(q=>q.chapter===t.chapter);
  const sampleFor=t=>t.samples.find(s=>s.id===lastSamples.get(t.id))||t.samples[0];
  function selectedFor(t){if(!selections.has(t.chapter))selections.set(t.chapter,new Set([t.id]));return selections.get(t.chapter);}
  function absorb(next){
    const items=next.tasks||[next];
    for(const item of items){
      if(!item.algorithm||!item.sample)continue;
      const status=next.status==='cancelled'&&['pending','running','starting'].includes(item.status)?'cancelled':item.status;
      runs.set(key(item.algorithm,item.sample),{...item,status,job:next.job,log:next.log});
    }
  }
  function clearOwnResults(){for(const [k] of runs){if(JSON.parse(k)[1]==='own_image')runs.delete(k);}}
  const link=(p,label)=>`<a href="${url(p)}" target="_blank" rel="noopener">${esc(label)} ↗</a>`;
  function go(algorithm,sample){const t=tools.get(algorithm),s=sample||lastSamples.get(algorithm)||t.samples[0].id;lastSamples.set(algorithm,s);const hash='#algorithm/'+encodeURIComponent(algorithm)+'/'+encodeURIComponent(s);if(location.hash===hash)render();else location.hash=hash;}
  function notice(message){const n=document.getElementById('notice');n.textContent=message;n.hidden=false;clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>n.hidden=true,9000);}
  async function api(path,body){const response=await fetch('/api/'+path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json','X-Lab-Token':session?.token||''}:{},body:body?JSON.stringify(body):undefined});const q=await response.json();if(!response.ok)throw Error(q.error||'请求未完成');return q;}
  function image(p,label='计算图像'){return `<figure class="image-card"><button data-image="${esc(p)}" aria-label="放大${esc(label)}"><img src="${url(p)}" alt="${esc(label)}" loading="lazy"></button></figure>`;}
  function table(t){if(!t||!t.rows?.length)return '';return `<div class="table-wrap"><table><thead><tr>${t.headers.map(h=>`<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${t.rows.map(r=>`<tr>${r.map(v=>`<td>${esc(fmt(v))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
  function metrics(m){return m?.length?`<dl class="values">${m.map(([k,v])=>`<div><dt>${esc(k)}</dt><dd>${esc(fmt(v))}</dd></div>`).join('')}</dl>`:'';}
  function flow(t){return `<nav class="research-flow" aria-label="研究环节">${catalog.chapters.map(ch=>`<button class="flow-tab ${t.chapter===ch.id?'selected':''}" data-chapter="${ch.id}">${esc(ch.title)}</button>`).join('')}</nav>`;}
  function payload(event,label){return `<div class="gallery ${event.images?.length===2?'two':''}">${(event.images||[]).map((p,i)=>image(p,label+' '+(i+1))).join('')}</div>${metrics(event.metrics)}${table(event.table)}${event.evidence?.length?`<div id="evidence-slot">${evidenceHTML(event.evidence)}</div>`:''}`;}
  function coreResult(t,event){
    const allowed={
      'photometry.integral':['G1净积分 / DN·RAW px²'],
      'texture.checks':['二维共同支持'],
      'texture.profile':['所选谱峰周期 / µm','附近ACF峰 / µm'],
      'texture.final':['最强谱峰周期 / µm','通过质量检查的频率族 / µm'],
      'period.horizontal':['首峰周期 / µm','三峰共同周期 / µm'],
      'discussion.doubling':['输入频率族周期 / µm','二次响应示意周期 / µm']
    };
    const raw=event.metrics||[],values=raw.filter(([name])=>(allowed[t.id]||[]).includes(name));
    const labels={'所选谱峰周期 / µm':'周期 / µm','附近ACF峰 / µm':'自相关周期 / µm','通过质量检查的频率族 / µm':'频率族周期 / µm','输入频率族周期 / µm':'输入周期 / µm','二次响应示意周期 / µm':'示意周期 / µm'};
    let resultTable=event.table;
    if(['intensity.single_sine','intensity.double_harmonic'].includes(t.id)&&resultTable)resultTable={headers:[resultTable.headers[1]],rows:resultTable.rows.map(row=>[row[1]])};
    if(t.id==='detection.models'&&resultTable)resultTable={headers:[resultTable.headers[0],resultTable.headers[2]],rows:resultTable.rows.map(row=>[row[0],row[2]])};
    if(t.id==='texture.checks'&&resultTable)resultTable={headers:[resultTable.headers[0],resultTable.headers[1],resultTable.headers[3]],rows:resultTable.rows.map(row=>[row[0],row[1],row[3]])};
    let message='';
    if(raw.some(([name])=>name==='判断'))message=t.id==='discussion.doubling'?'该样本未得到有效主周期，无法生成倍频示意。':t.id==='discussion.combination'?'该样本未得到合格的两组频率，无法计算和差频。':t.id==='texture.profile'?'未得到有效剖面':'未得到有效周期';
    if(t.id==='texture.family'&&raw.some(([name,value])=>name==='质量检查'&&value!=='RESOLVED_CANDIDATES')){message='未得到有效频率族';resultTable=null;}
    if(raw.some(([name,value])=>name==='频率族有歧义'&&value===true))message='频率族有歧义';
    const view={...event,metrics:values.map(([name,value])=>[labels[name]||name,value]),table:resultTable};
    return payload(view,'结果')+(message?`<p class="outcome">${message}</p>`:'');
  }
  function statusHTML(run){
    if(busy)return '<span class="run-status" role="status">运行中</span>';
    if(!run)return '';
    const text={complete:'完成',running:'运行中',starting:'运行中',cancelled:'已停止',failed:'失败'};
    const progress=run.tasks?' '+run.tasks.filter(q=>q.status==='complete').length+'/'+run.tasks.length:'';
    return `<span class="run-status ${run.status==='failed'?'failed':''}" role="status">${esc(text[run.status]||'运行中')}${progress}</span>${!G.terminal(run.status)?'<button class="quiet" data-action="cancel">停止</button>':''}${run.status==='failed'?`<span class="failure">${esc(run.message)}</span>`:''}`;
  }
  function ownForm(){const locked=job&&!G.terminal(job.status);return `<form id="own-form"><label class="upload-label">${ownFile?esc(ownFile.name):'选择图像'}<input aria-label="图像" type="file" id="own-file" accept="image/png,image/jpeg" ${locked?'disabled':''}></label><label>标尺（µm/px）<input type="number" id="own-scale" step="any" min="0" value="${esc(ownScale)}" ${locked?'disabled':''} required></label></form>`;}
  function taskList(t){
    const active=busy||job&&!G.terminal(job.status),selected=selectedFor(t);
    return `<div class="task-grid">${phaseTools(t).map(q=>{
      const s=sampleFor(q),run=currentRun(q,s),labels={pending:'排队',starting:'运行',running:'运行',complete:'完成',failed:'失败',cancelled:'已停止'};
      return `<div class="task-row ${q.id===t.id?'viewing':''}"><input type="checkbox" data-task-check="${q.id}" aria-label="选择${esc(q.title)}" ${selected.has(q.id)?'checked':''} ${active?'disabled':''}><button class="task-view" data-view="${q.id}" ${q.id===t.id?'aria-current="true"':''}>${esc(q.title)}</button><select data-task-sample="${q.id}" aria-label="${esc(q.title)}样本" ${active?'disabled':''}>${q.samples.map(v=>`<option value="${esc(v.id)}" ${v.id===s.id?'selected':''}>${esc(v.label)}</option>`).join('')}</select><span class="task-status ${run?.status==='failed'?'failed':''}">${esc(labels[run?.status]||'')}</span></div>`;
    }).join('')}</div>`;
  }
  function inputContext(t,input){
    const labels={
      'intensity.single_sine':['已有测量数据','运行后拟合模型，生成曲线和参数。'],
      'intensity.double_harmonic':['已有测量数据','运行后拟合模型，生成曲线和参数。'],
      'intensity.comparison':['已有测量数据','运行后拟合两种模型，并进行留出检验。'],
      'intensity.two_d':['已有测量数据','表中为光斑积分，运行后拟合二维强度谱。'],
      'rg.selection':['已有通道积分','运行后按质量条件筛选数据。'],
      'rg.fitted':['已有通道积分','运行后拟合红绿通道响应，并进行整帧留出。'],
      'rg.final':['已有通道积分','运行后拟合响应，计算残差与重采样区间。'],
      'period.horizontal':['已有峰距与标定值','运行后换算衍射周期。'],
      'period.geometry':['已有节点坐标与索引','运行后拟合点阵几何，并进行留一检验。'],
      'period.comparison':['已有周期数据','运行后按既定对应倍率汇总差异。'],
      'discussion.doubling':['显微图像','以图中周期为尺度，演示二次响应如何产生倍频（周期减半）。'],
      'discussion.combination':['显微图像','从图中提取两组空间频率向量，计算它们的和、差及对应周期。']
    };
    return labels[t.id]||[input.images?.length?'原图':'输入',''];
  }
  function workspace(t,s){
    const run=currentRun(t,s),event=G.snapshot(run?.events,t.id),active=job&&!G.terminal(job.status),selected=selectedFor(t);
    const input=s.id==='own_image'?{images:ownURL?[ownURL]:[],metrics:[],table:null,evidence:[]}:s.input;
    const inputHTML=s.id==='own_image'&&ownURL?`<figure class="image-card"><img src="${esc(ownURL)}" alt="原图"></figure>`:payload({...input,metrics:[],images:input.images},'原图');
    const [inputTitle,inputNote]=inputContext(t,input);
    const resultTitle=t.id==='discussion.doubling'?'倍频示意':t.id==='discussion.combination'?'频率组合':t.type==='intensity'||['rg.fitted','rg.final','period.geometry'].includes(t.id)?'拟合结果':'结果';
    const own=s.id==='own_image'||phaseTools(t).some(q=>selected.has(q.id)&&sampleFor(q).id==='own_image');
    return `${flow(t)}<article class="independent-workspace"><div class="selection-bar"><button class="button primary" data-action="run" ${!session||active||busy||!selected.size?'disabled':''}>运行所选</button><button class="button" data-action="run-all" ${!session||active||busy?'disabled':''}>全部运行</button><button class="quiet" data-action="clear-selection" ${active||busy?'disabled':''}>清空</button><span class="selection-count">已选 ${selected.size} 项</span>${statusHTML(job)}</div>${taskList(t)}${own?ownForm():''}<div class="view-heading"><h2>${esc(t.title)} · ${esc(s.label)}</h2><button class="source-button" id="algorithm-source">源码</button></div><div class="comparison ${event?'has-result':''}"><section class="input-section"><h2>${esc(inputTitle)}</h2>${inputNote?`<p class="input-note">${esc(inputNote)}</p>`:''}${inputHTML}</section>${event?`<section class="result-section"><h2>${resultTitle}</h2>${coreResult(t,event)}</section>`:''}</div></article>`;
  }
  function render(){const {tool,sample}=route();main.innerHTML=closed?'<p class="closed">已结束</p>':workspace(tool,sample);renderBanner();bind();}
  function renderBanner(){
    const b=document.getElementById('active-run'),{tool,sample}=route();
    const current=job?.tasks?.find(q=>q.task_id===job.current_task)||job;
    b.hidden=!job||G.terminal(job.status)||current?.algorithm===tool.id&&current?.sample===sample.id;
    if(!b.hidden&&current?.algorithm){b.innerHTML='<button id="return-run">查看正在运行的算法</button>';document.getElementById('return-run').onclick=()=>go(current.algorithm,current.sample);}else b.hidden=true;
  }
  function bind(){
    main.querySelectorAll('[data-chapter]').forEach(b=>b.onclick=()=>go(catalog.tools.find(t=>t.chapter===b.dataset.chapter).id));
    main.querySelectorAll('[data-task-check]').forEach(b=>b.onchange=()=>{const t=tools.get(b.dataset.taskCheck),selected=selectedFor(t);if(b.checked)selected.add(t.id);else selected.delete(t.id);render();});
    main.querySelectorAll('[data-task-sample]').forEach(b=>b.onchange=()=>{lastSamples.set(b.dataset.taskSample,b.value);if(route().tool.id===b.dataset.taskSample)go(b.dataset.taskSample,b.value);else render();});
    main.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>go(b.dataset.view,sampleFor(tools.get(b.dataset.view)).id));
    const source=document.getElementById('algorithm-source');if(source)source.onclick=()=>showSource(route().tool);
    main.querySelectorAll('[data-image]').forEach(b=>b.onclick=()=>{document.getElementById('large-image').src=url(b.dataset.image);document.getElementById('image-dialog').showModal();});
    main.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>action(b.dataset.action));bindEvidence();
    const form=document.getElementById('own-form');if(form){document.getElementById('own-scale').oninput=e=>{ownScale=e.target.value;clearOwnResults();const result=main.querySelector?.('.result-section');if(result)result.remove();};document.getElementById('own-file').onchange=e=>{ownFile=e.target.files[0];if(ownURL)URL.revokeObjectURL(ownURL);ownURL=ownFile?URL.createObjectURL(ownFile):null;clearOwnResults();render();};form.onsubmit=submitOwn;}
  }
  function bindEvidence(){main.querySelectorAll('[data-evidence]').forEach(b=>b.onclick=()=>{const {tool,sample}=route(),event=G.snapshot(currentRun(tool,sample)?.events,tool.id);document.getElementById('evidence-slot').innerHTML=evidenceHTML(event.evidence,Number(b.dataset.evidence));bindEvidence();});}
  function evidenceHTML(items,selected=0){if(!items?.length)return '';const item=items[Math.min(selected,items.length-1)];return `<section class="evidence"><div class="evidence-buttons">${items.map((v,i)=>`<button class="${i===selected?'selected':''}" data-evidence="${i}">${v.record.status==='DETECTED'?'检出':'未检出'}</button>`).join('')}</div><img src="${url(item.image)}" alt="局部结果"></section>`;}
  function showSource(step,tab=0,full=false){const key=step.sources[tab],q=catalog.sources[key],panel=document.getElementById('code-panel');const lines=full?q.full_lines:q.lines,start=full?q.function_start:q.start;panel.hidden=false;document.body.classList.add('code-open');panel.innerHTML=`<div class="code-head"><h2>源码</h2><button class="close" id="close-code" aria-label="关闭">×</button></div><p class="filename">${esc(q.file)}</p>${step.sources.length>1?`<div class="code-tabs">${step.sources.map((k,i)=>`<button data-code-tab="${i}" class="${i===tab?'selected':''}">${esc(catalog.sources[k].symbol)}</button>`).join('')}</div>`:`<p class="filename">${esc(q.symbol)}</p>`}<pre class="code-list">${lines.map((line,i)=>`<span class="code-line ${q.focus&&line.includes(q.focus)?'highlight':''}"><span class="line-no">${start+i}</span>${esc(line)}</span>`).join('')}</pre><div class="run-actions"><button class="button" id="full-function">${full?'收起':'展开'}</button>${link(q.file,'文件')}</div>`;document.getElementById('close-code').onclick=closeCode;document.getElementById('full-function').onclick=()=>showSource(step,tab,!full);panel.querySelectorAll('[data-code-tab]').forEach(b=>b.onclick=()=>showSource(step,Number(b.dataset.codeTab)));document.getElementById('close-code').focus();}
  function closeCode(){document.getElementById('code-panel').hidden=true;document.body.classList.remove('code-open');render();}
  async function action(name){
    if(busy)return;
    const {tool}=route(),selected=selectedFor(tool);
    if(name==='clear-selection'){selected.clear();render();return;}
    if(name==='cancel'){try{job=await api('cancel',{job:job.job});absorb(job);render();}catch(e){notice(e.message);}return;}
    if(!['run','run-all'].includes(name))return;
    if(name==='run-all'){selected.clear();phaseTools(tool).forEach(t=>selected.add(t.id));}
    const tasks=phaseTools(tool).filter(t=>selected.has(t.id)).map(t=>({algorithm:t.id,sample:sampleFor(t).id}));
    if(!tasks.length){notice('请选择算法');return;}
    const own=tasks.some(t=>t.sample==='own_image');
    if(own&&(!ownFile||!Number.isFinite(Number(ownScale))||Number(ownScale)<=0)){notice('请选择图像并填写有效标尺');render();return;}
    if(own&&ownFile.size>32*1024**2){notice('图片不能超过32 MB');render();return;}
    busy=true;tasks.forEach(t=>runs.delete(key(t.algorithm,t.sample)));render();
    try{
      const body={tasks};
      if(own){body.image=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.onerror=()=>reject(Error('读取图片失败'));reader.readAsDataURL(ownFile);});body.scale=Number(ownScale);body.filename=ownFile.name;}
      job=await api('run',body);absorb(job);
      if(!selected.has(route().tool.id))go(tasks[0].algorithm,tasks[0].sample);
    }catch(e){notice(e.message);}finally{busy=false;render();}
  }
  async function submitOwn(event){event.preventDefault();await action('run');}
  async function poll(){
    if(!session||closed||!job||G.terminal(job.status))return;
    try{const next=await api('job/'+job.job),previous=job;job=next;absorb(job);renderBanner();if(next.version!==previous.version||next.status!==previous.status){if(document.getElementById('code-panel').hidden)render();}}
    catch(e){if(!closed){document.getElementById('connection').hidden=false;document.getElementById('connection').textContent='连接中断，请重新通过启动器打开项目。';session=null;render();}}
  }
  async function connect(){
    if(location.protocol==='file:'){document.getElementById('connection').hidden=false;document.getElementById('connection').textContent='请通过启动器打开项目，才能运行。';return;}
    try{session=await api('session');document.getElementById('connection').hidden=true;document.getElementById('end-session').hidden=false;session.jobs.forEach(absorb);job=session.jobs.filter(j=>!G.terminal(j.status)).at(-1)||null;if(job?.tasks){const phases=new Map();for(const task of job.tasks){const t=tools.get(task.algorithm);if(!t)continue;if(!phases.has(t.chapter))phases.set(t.chapter,new Set());phases.get(t.chapter).add(t.id);lastSamples.set(t.id,task.sample);}for(const [chapter,ids] of phases)selections.set(chapter,ids);}render();}
    catch{document.getElementById('connection').hidden=false;document.getElementById('connection').textContent='请通过启动器打开项目，才能运行。';}
  }
  document.getElementById('close-image').onclick=()=>document.getElementById('image-dialog').close();
  document.getElementById('end-session').onclick=async()=>{try{await api('shutdown',{});closed=true;session=null;if(job&&!G.terminal(job.status))job.status='cancelled';document.getElementById('connection').hidden=true;document.getElementById('end-session').hidden=true;render();}catch(e){notice(e.message);}};
  document.addEventListener('keydown',e=>{if(e.key==='Escape')closeCode();});
  window.addEventListener('hashchange',()=>{closeCode();render();main.focus({preventScroll:true});window.scrollTo(0,0);});
  render();connect();setInterval(poll,800);
})();
