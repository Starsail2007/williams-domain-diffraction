(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.GuideState=api;})(typeof window==='undefined'?globalThis:window,function(){
  'use strict';
  function parseRoute(hash){const parts=String(hash||'#home').replace(/^#/,'').split('/').map(x=>{try{return decodeURIComponent(x);}catch{return '';}});return {page:parts[0]||'home',id:parts[1]||'',step:Math.max(0,Number.parseInt(parts[2]||'0',10)||0)};}
  function snapshot(events,step){return (events||[]).filter(e=>e.step===step&&e.status!=='superseded').at(-1)||null;}
  function completedCount(events,steps){return steps.filter(s=>snapshot(events,s.id)).length;}
  function terminal(status){return ['complete','failed','cancelled'].includes(status);}
  function visibleEvent(mode,live,recorded,step){return mode==='live'?snapshot(live?.events||[],step):null;}
  function clampStep(value,count){return Math.max(0,Math.min(value,count-1));}
  return {parseRoute,snapshot,completedCount,terminal,visibleEvent,clampStep};
});
