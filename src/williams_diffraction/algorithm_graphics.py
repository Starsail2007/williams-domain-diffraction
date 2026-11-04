"""Views of measured algorithm evidence; no measurements are taken from views."""
import html
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont

COLORS={'INDEPENDENT_STRONG':'#32ddf4','INDEPENDENT_WEAK':'#ffbd42','GUIDED':'#fd68df'}


def font(size=17):return ImageFont.load_default(size=size)
def tone(a):
    return np.uint8(np.clip(np.arcsinh(np.maximum(a,0)/8)/np.arcsinh((16383-512)/8),0,1)*255)


def label(im,lines):
    im=im.convert('RGB');d=ImageDraw.Draw(im);h=26*len(lines)+10
    d.rectangle((0,0,im.width,h),fill='#142734')
    for i,line in enumerate(lines):d.text((10,6+26*i),line,font=font(17),fill='white')
    return im


def screen_overlay(image,seed,corners,edges,H,red_shape):
    im=image.convert('RGB');d=ImageDraw.Draw(im)
    for i,(a,b) in enumerate(zip(corners,np.roll(corners,-1,axis=0))):
        d.line([tuple(a),tuple(b)],fill='#4ce4bd' if edges[i]['supported'] else '#ffbd42',width=4)
    return im


def marked_image(base,peaks,T,exclusions=(),only_independent=False,proposals=()):
    from .geometry import transform
    im=base.convert('RGB');d=ImageDraw.Draw(im)
    for q in proposals:
        if 'predicted_raw_xy' not in q:continue
        x,y=transform([q['predicted_raw_xy']],T)[0]
        color='#c6cbd0' if q.get('status')!='DETECTED' else '#fd68df'
        d.rectangle((x-4,y-4,x+4,y+4),outline=color,width=1)
    for p in peaks:
        origin=p['detection_origin']
        if only_independent and origin.startswith('GUIDED'):continue
        color=COLORS['GUIDED'] if origin.startswith('GUIDED') else COLORS[origin]
        x,y=transform([[p['u_raw_px'],p['v_raw_px']]],T)[0]
        d.ellipse((x-6,y-6,x+6,y+6),outline=color,width=2)
        if p['review_flags']:d.rectangle((x-9,y-9,x+9,y+9),outline='#ffffff',width=1)
        d.text((x+7,y-10),p.get('label',''),font=font(13),fill=color,stroke_width=1,stroke_fill='black')
    for e in exclusions:
        x,y=transform([[e['u_raw_px'],e['v_raw_px']]],T)[0]
        d.line((x-9,y-9,x+9,y+9),fill='#ff6464',width=3);d.line((x-9,y+9,x+9,y-9),fill='#ff6464',width=3)
        if 'segment_start_raw_xy' in e:
            a,b=transform([e['segment_start_raw_xy']],T)[0];d.line((a,b,x,y),fill='#ff6464',width=2)
    return im


def contrast_view(field,valid,title):
    values=field[valid.astype(bool)];hi=max(12.,float(np.percentile(values,99.7)))
    a=np.uint8(np.clip(np.arcsinh(np.maximum(field,0)/2)/np.arcsinh(hi/2),0,1)*255)
    im=Image.fromarray(cv2.cvtColor(cv2.applyColorMap(a,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB))
    im.thumbnail((1200,1000))
    return im


def patches(path,red,audit,contrast=None,offset=(0,0)):
    """Show an actual guided success and rejected prediction windows together."""
    chosen=[]
    for wanted in ('DETECTED','NO_ACCEPTED_LOCAL_MAXIMUM','SCREEN_EDGE_OR_OUTSIDE','MEASURED_BROADBAND_BACKGROUND'):
        found=next((q for q in audit if q.get('status')==wanted and ('predicted_raw_xy' in q if wanted=='DETECTED' else ('predicted_raw_xy' in q or 'raw_xy' in q))),None)
        if found:chosen.append(found)
    if not chosen:
        label(Image.new('RGB',(780,120),'#142734'),['No supported guided search window for this case.',
             'See model gates and all_search_evidence.json; no grid was populated.']).save(path);return []
    im=Image.new('RGB',(1100,240*len(chosen)),'#142734');d=ImageDraw.Draw(im)
    for i,q in enumerate(chosen):
        position=np.asarray(q.get('predicted_raw_xy',q.get('raw_xy')),float)
        xr,yr=position/2;x,y=np.rint([xr,yr]).astype(int)
        x0,x1=max(0,x-45),min(red.shape[1],x+46);y0,y1=max(0,y-45),min(red.shape[0],y+46)
        patch=red[y0:y1,x0:x1];lo,hi=np.percentile(patch,[5,99.8]);a=np.uint8(np.clip((patch-lo)/max(hi-lo,1),0,1)*255)
        tile=Image.fromarray(a).convert('RGB').resize((190,190),Image.Resampling.NEAREST)
        filtered=Image.new('RGB',(190,190),'black')
        if contrast is not None:
            fx0,fy0=max(x0,int(offset[0])),max(y0,int(offset[1]));fx1,fy1=min(x1,int(offset[0])+contrast.shape[1]),min(y1,int(offset[1])+contrast.shape[0])
            fp=np.zeros(patch.shape,float)
            if fx1>fx0 and fy1>fy0:fp[fy0-y0:fy1-y0,fx0-x0:fx1-x0]=contrast[fy0-int(offset[1]):fy1-int(offset[1]),fx0-int(offset[0]):fx1-int(offset[0])]
            fp=np.uint8(np.clip((fp+2)/14,0,1)*255)
            filtered=Image.fromarray(cv2.cvtColor(cv2.applyColorMap(fp,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)).resize((190,190))
        for view in (tile,filtered):
            draw=ImageDraw.Draw(view);cx=(xr-x0+.5)/patch.shape[1]*190-.5;cy=(yr-y0+.5)/patch.shape[0]*190-.5
            draw.line((cx-8,cy,cx+8,cy),fill='#fd68df',width=1);draw.line((cx,cy-8,cx,cy+8),fill='#fd68df',width=1)
            if 'measured_raw_xy' in q:
                a,b=np.asarray(q['measured_raw_xy'])/2;a=(a-x0+.5)/patch.shape[1]*190-.5;b=(b-y0+.5)/patch.shape[0]*190-.5
                draw.ellipse((a-6,b-6,a+6,b+6),outline='#57ffbf',width=2)
        base=i*240;im.paste(tile,(5,base+28));im.paste(filtered,(205,base+28))
        d.text((8,base+5),'native R',font=font(14),fill='white');d.text((208,base+5),'contrast: -2..12 DN',font=font(14),fill='white')
        d.text((415,base+18),q['status'],font=font(18),fill='white')
        d.text((415,base+48),str(q.get('origin','')),font=font(14),fill='white')
        role='Prediction' if 'predicted_raw_xy' in q else 'Inspection'
        d.text((415,base+76),f'{role} RAW: ({position[0]:.3f}, {position[1]:.3f}) px',font=font(15),fill='white')
        d.text((415,base+102),f'Native display range: {lo:.2f} to {hi:.2f} DN',font=font(15),fill='white')
        d.text((415,base+128),'Magenta: '+role.lower()+'; green: final centroid',font=font(14),fill='white')
        if 'contrast_DN' in q:d.text((415,base+157),f'contrast {q["contrast_DN"]:.3g} DN; NCC {q.get("diffraction_shape_NCC",0):.3f}',font=font(15),fill='white')
        if 'final_prediction_error_raw_px' in q:d.text((415,base+184),f'FINAL centroid offset: {q["final_prediction_error_raw_px"]:.3f} RAW px',font=font(15),fill='white')
    im.save(path)
    return chosen


def line_svg(path,panels):
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{360*len(panels)}" viewBox="0 0 1100 {360*len(panels)}"><rect width="100%" height="100%" fill="white"/>']
    canvas=Image.new('RGB',(2200,720*len(panels)),'white');draw=ImageDraw.Draw(canvas)
    colors=['#167d9a','#db7632','#8d56ac']
    for i,p in enumerate(panels):
        points=[(float(x),float(y)) for _,xx,yy in p['series'] for x,y in zip(xx,yy) if np.isfinite([x,y]).all()]
        if not points:continue
        xx,yy=np.array(points).T;x0,x1=float(xx.min()),float(xx.max());y0,y1=float(yy.min()),float(yy.max())
        if y1-y0<1e-12:y0,y1=min(0.,y0),max(1.,y1*1.1)
        dx=max(x1-x0,1e-12);dy=max(y1-y0,1e-12)
        px=lambda x:80+(float(x)-x0)/dx*940;py=lambda y:280-(float(y)-y0)/dy*210
        base=i*360
        draw.text((160,2*(base+8)),p['title'],font=font(34),fill='#182732')
        draw.line([(160,2*(base+60)),(160,2*(base+280)),(2040,2*(base+280))],fill='#182732',width=2)
        for fraction in (0,.25,.5,.75,1):
            x=x0+fraction*dx;y=y0+fraction*dy
            draw.line((2*px(x),2*(base+60),2*px(x),2*(base+280)),fill='#e0e8e7',width=1)
            draw.text((2*px(x)-25,2*(base+286)),f'{x:.4g}',font=font(25),fill='#182732')
            draw.text((12,2*(base+py(y)-7)),f'{y:.4g}',font=font(24),fill='#182732')
        draw.text((960,2*(base+309)),p['xlabel'],font=font(28),fill='#182732')
        draw.text((20,2*(base+38)),p['ylabel'],font=font(25),fill='#182732')
        parts.append(f'<g transform="translate(0,{i*360})" font-family="sans-serif" fill="#182732"><text x="80" y="25" font-size="17">{html.escape(p["title"])}</text><path d="M80 60 V280 H1020" fill="none" stroke="#182732"/>')
        for j,(name,x,y) in enumerate(p['series']):
            xy=' '.join(f'{px(a):.3f},{py(b):.3f}' for a,b in zip(x,y) if np.isfinite([a,b]).all())
            parts.append(f'<polyline points="{xy}" fill="none" stroke="{colors[j%3]}" stroke-width="1.2"/><text x="{80+310*j}" y="330" font-size="13" fill="{colors[j%3]}">{html.escape(name)}</text>')
            curve=[(2*px(a),2*(base+py(b))) for a,b in zip(x,y) if np.isfinite([a,b]).all()]
            if len(curve)>1:draw.line(curve,fill=colors[j%3],width=3)
            draw.text((2*(80+310*j),2*(base+337)),name,font=font(25),fill=colors[j%3])
        parts.append(f'<text x="80" y="302">{x0:.4g}</text><text x="980" y="302">{x1:.4g}</text><text x="480" y="307">{html.escape(p["xlabel"])}</text><text x="10" y="55">{html.escape(p["ylabel"])}</text><text x="10" y="78">{y1:.4g}</text><text x="10" y="280">{y0:.4g}</text></g>')
    parts.append('</svg>');Path(path).write_text(''.join(parts),encoding='utf-8')
    canvas.save(Path(path).with_suffix('.png'))


def fft_view(power,fx,fy,peaks,selected):
    extent=.25;xs=np.flatnonzero(abs(fx)<=extent);ys=np.flatnonzero(abs(fy)<=extent)
    if len(xs)<2 or len(ys)<2:
        return label(Image.new('RGB',(900,80),'#142734'),['No resolved spectrum'])
    log=np.log1p(power[np.ix_(ys,xs)]);lo,hi=np.percentile(log,[3,99.8]);z=np.uint8(np.clip((log-lo)/max(hi-lo,1e-12),0,1)*255)
    heat=Image.fromarray(cv2.cvtColor(cv2.applyColorMap(z,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)).resize((500,500))
    im=Image.new('RGB',(690,650),'white');im.paste(heat,(100,55));d=ImageDraw.Draw(im)
    px=lambda x:100+(x-fx[xs[0]])/(fx[xs[-1]]-fx[xs[0]])*500
    py=lambda y:55+(fy[ys[0]]-y)/(fy[ys[0]]-fy[ys[-1]])*500
    for tick in (-.2,-.1,0,.1,.2):
        d.text((px(tick)-17,565),f'{tick:.1f}',font=font(15),fill='black');d.text((45,py(tick)-8),f'{tick:.1f}',font=font(15),fill='black')
    for p in peaks:
        color='#76ffbb' if p['rank'] in selected else '#fafafa'
        for sign in (-1,1):
            x,y=px(sign*p['fx_cycles_um']),py(sign*p['fy_cycles_um'])
            if 100<=x<=600 and 55<=y<=555:
                d.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
                if sign==1:d.text((x+6,y-7),str(p['rank']),font=font(13),fill=color,stroke_width=1,stroke_fill='black')
    d.text((8,30),'fy / um^-1',font=font(13),fill='black')
    d.text((220,600),'fx / um^-1',font=font(16),fill='black')
    return im


STYLE='body{max-width:1180px;margin:36px auto;padding:0 22px;font:17px/1.75 system-ui,sans-serif;color:#172b38;background:#fafbf8}h1{line-height:1.4}h2{margin-top:36px}a{color:#075e85}img{max-width:100%;height:auto}figure,section,.box{background:white;border:1px solid #dae1dd;border-radius:10px;padding:18px;margin:20px 0}figcaption{margin-top:12px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:9px;text-align:left;border-bottom:1px solid #dde3df}code{background:#edf1ef;padding:2px 5px}small{color:#526975}.scroll{overflow-x:auto}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}.gallery figure{margin:0}summary{cursor:pointer;font-weight:bold}'


def page(path,title,body):
    Path(path).write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+STYLE+'</style><h1>'+html.escape(title)+'</h1>'+body+'</html>',encoding='utf-8')


def figure(file,caption):return f'<figure><a href="{file}"><img src="{file}" loading="lazy" alt="{html.escape(caption)}"></a><figcaption>{caption}</figcaption></figure>'
def table(headers,rows):
    def human(value):
        if value is None:return '—'
        if isinstance(value,bool):return '是' if value else '否'
        if isinstance(value,float):return f'{value:.6g}'
        return str(value)
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(human(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
