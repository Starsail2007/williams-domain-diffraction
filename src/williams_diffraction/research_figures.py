"""Small scientific plots; calculations remain in the research modules."""
from pathlib import Path
import math
import numpy as np
from PIL import Image,ImageDraw,ImageFont

COLORS=['#087f8c','#cf6336','#7655a5','#337b50','#687985']

def font(size=22):return ImageFont.load_default(size=size)

def plot(path,panels,columns=1):
    width,height=900,560
    canvas=Image.new('RGB',(width*columns,height*math.ceil(len(panels)/columns)),'white')
    d=ImageDraw.Draw(canvas)
    for i,p in enumerate(panels):
        ox,oy=(i%columns)*width,(i//columns)*height
        all_points=[(float(x),float(y)) for s in p['series'] for x,y in zip(s['x'],s['y']) if np.isfinite([x,y]).all()]
        if not all_points:raise ValueError('Plot has no finite observations: '+p['title'])
        a=np.asarray(all_points);xmin,xmax=a[:,0].min(),a[:,0].max();ymin,ymax=a[:,1].min(),a[:,1].max()
        if p.get('zero_y'):ymin=min(0,ymin)
        dx=max(float(xmax-xmin),max(abs(float(xmax)),1)*.01);dy=max(float(ymax-ymin),max(abs(float(ymax)),1)*.01)
        xmin-=dx*.03;xmax+=dx*.03;ymin-=dy*.06;ymax+=dy*.08
        if 'xlim' in p:xmin,xmax=p['xlim']
        if 'ylim' in p:ymin,ymax=p['ylim']
        if p.get('equal_aspect'):
            units=max((xmax-xmin)/735,(ymax-ymin)/305);cx,cy=(xmin+xmax)/2,(ymin+ymax)/2
            xmin,xmax=cx-units*735/2,cx+units*735/2;ymin,ymax=cy-units*305/2,cy+units*305/2
        px=lambda x:ox+115+(float(x)-xmin)/(xmax-xmin)*735
        py=lambda y:oy+412-(float(y)-ymin)/(ymax-ymin)*305
        title_size=25
        while title_size>16 and d.textlength(p['title'],font=font(title_size))>width-140:title_size-=1
        d.text((ox+115,oy+12),p['title'],font=font(title_size),fill='#152c37')
        d.text((ox+16,oy+67),p['ylabel'],font=font(19),fill='#152c37')
        for fraction in np.linspace(0,1,5):
            x=xmin+fraction*(xmax-xmin);y=ymin+fraction*(ymax-ymin)
            d.line((px(x),oy+107,px(x),oy+412),fill='#e1e8e6',width=1)
            d.line((ox+115,py(y),ox+850,py(y)),fill='#e1e8e6',width=1)
            d.text((px(x)-25,oy+424),f'{x:.3g}',font=font(19),fill='#30434c')
            d.text((ox+10,py(y)-10),f'{y:.3g}',font=font(19),fill='#30434c')
        d.line([(ox+115,oy+107),(ox+115,oy+412),(ox+850,oy+412)],fill='#30434c',width=2)
        if ymin<0<ymax:d.line((ox+115,py(0),ox+850,py(0)),fill='#94a6a6',width=1)
        for j,s in enumerate(p['series']):
            color=s.get('color',COLORS[j%len(COLORS)])
            points=[(px(x),py(y)) for x,y in zip(s['x'],s['y']) if np.isfinite([x,y]).all() and xmin<=x<=xmax and ymin<=y<=ymax]
            if s.get('style')=='scatter':
                radius=s.get('radius',2)
                for x,y in points:d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)
            elif len(points)>1:d.line(points,fill=color,width=s.get('width',3))
            # One short legend entry per row avoids overlap on two-curve plots.
            d.line((ox+115+(j%2)*365,oy+490+(j//2)*27,ox+140+(j%2)*365,oy+490+(j//2)*27),fill=color,width=3)
            d.text((ox+150+(j%2)*365,oy+478+(j//2)*27),s['name'],font=font(18),fill=color)
        d.text((ox+330,oy+454),p['xlabel'],font=font(20),fill='#152c37')
    canvas.save(path)

def series(name,x,y,style='line',**kwargs):return dict(name=name,x=np.asarray(x),y=np.asarray(y),style=style,**kwargs)

def montage(path,items,columns=4,cell=(400,340)):
    width,height=cell;canvas=Image.new('RGB',(width*columns,height*math.ceil(len(items)/columns)),'#eff3f1');d=ImageDraw.Draw(canvas)
    for i,(source,title) in enumerate(items):
        with Image.open(source) as source:
            im=source.convert('RGB');im.thumbnail((width-12,height-55))
        x,y=(i%columns)*width,(i//columns)*height
        canvas.paste(im,(x+(width-im.width)//2,y+47))
        d.text((x+12,y+10),title,font=font(21),fill='#17303d')
    canvas.save(path)

def photometry_views(folder,mosaic,info,measurements,channels):
    """Draw the actual background/partition used by the unchanged measurer."""
    from .green_photometry_trial import channel_patch,support_mask
    from .weak_tail_trial import region_cell
    from .green_channel_study import aperture_weights
    from .delivery_algorithms import save_csv
    peaks=info['peaks'];centers=np.array([[p['u_raw_px'],p['v_raw_px']] for p in peaks])
    shapes=[dict(center=c,shape={k.removeprefix('background_exclusion_shape_'):v for k,v in m.items() if k.startswith('background_exclusion_shape_')}) for c,m in zip(centers,measurements)]
    cfg=info['photometry_config'];refs=np.array([[r['u_raw_px'],r['v_raw_px']] for r in info['reflections']]).reshape(-1,2)
    cumulative=[]
    partition=min(measurements,key=lambda q:q['aperture_coverage_min'])['peak_uid']
    for uid in dict.fromkeys((info['target_uid'],info['comparison_uid'],partition)):
        peak=next(p for p in peaks if p['peak_uid']==uid);center=np.array([peak['u_raw_px'],peak['v_raw_px']]);panels=[];growth={}
        for name in ('G1','G2'):
            row=next(q for q in channels if q['peak_uid']==uid and q['channel']==name)
            half=600 if row['background_method']=='EXPANDED_PLANE_180_580' else 360
            z,xx,yy,ci=channel_patch(mosaic,info['black_level'],center,half,name)
            bg=row['background_c0_DN']+row['background_cx_per300_DN']*xx/300+row['background_cy_per300_DN']*yy/300
            mask,inside=support_mask(xx,yy,center,shapes,refs,np.asarray(info['corners_raw_px']),cfg)
            mask &= np.hypot(xx,yy)>=180
            cell=region_cell(xx,yy,center,centers,inside)
            radii=np.arange(10,181,5);growth[name]=[float(4*np.sum(aperture_weights(xx,yy,r)*cell*(z-bg))) for r in radii]
            for radius in (50,125):np.testing.assert_allclose(4*np.sum(aperture_weights(xx,yy,radius)*cell*(z-bg)),row[f'Q_r{radius}'],rtol=1e-12,atol=1e-8)
            if name=='G1':
                def gray(a):
                    lo,hi=np.percentile(a,[1,99.8]);return Image.fromarray(np.uint8(np.clip((a-lo)/max(float(hi-lo),1e-9),0,1)*255)).convert('RGB')
                im=gray(z);d=ImageDraw.Draw(im);sx=sy=.5
                cx,cy=-xx[0,0]/2,-yy[0,0]/2
                for radius,color in ((50,'#f6bd59'),(125,'#48e3cb')):d.ellipse((cx-radius*sx,cy-radius*sy,cx+radius*sx,cy+radius*sy),outline=color,width=2)
                panels.append((im,'G1: measured aperture (50 / 125 RAW px)'))
                im=gray(z);a=np.asarray(im).copy();a[mask]=a[mask]*.2+np.array([30,180,150])*.8
                panels.append((Image.fromarray(a.astype('uint8')),'Eligible background samples'))
                residual=z-bg;limit=max(1.,float(np.percentile(abs(residual),99)));f=np.clip(residual/limit,-1,1)
                rgb=np.full((*f.shape,3),245.,float);rgb[:,:,0]-=np.maximum(-f,0)*180;rgb[:,:,1]-=abs(f)*130;rgb[:,:,2]-=np.maximum(f,0)*180
                panels.append((Image.fromarray(rgb.astype('uint8')),'Signed signal: blue negative, orange positive'))
                weights=aperture_weights(xx,yy,125)*cell
                panels.append((gray(weights),'Actual weights: aperture / neighbour / screen'))
                sheet=Image.new('RGB',(1200,1260),'white');draw=ImageDraw.Draw(sheet)
                for i,(view,title) in enumerate(panels):
                    view=view.resize((560,560),Image.Resampling.NEAREST);x,y=(i%2)*600,(i//2)*630
                    sheet.paste(view,(x+(600-view.width)//2,y+45));draw.text((x+12,y+12),title,font=font(19),fill='#17303d')
                sheet.save(folder/(uid+'_measurement.png'))
        mean=(np.asarray(growth['G1'])+growth['G2'])/2
        cumulative += [dict(peak_uid=uid,radius_raw_px=int(r),Q_G1=float(a),Q_G2=float(b),Q_G=float(c)) for r,a,b,c in zip(radii,growth['G1'],growth['G2'],mean)]
        plot(folder/(uid+'_aperture.png'),[dict(title='Net signal versus aperture radius',xlabel='radius (RAW px)',ylabel='Q / million DN RAW px^2',series=[series('G1',radii,np.asarray(growth['G1'])/1e6),series('G2',radii,np.asarray(growth['G2'])/1e6),series('G mean',radii,mean/1e6)])])
    save_csv(folder/'aperture_growth.csv',cumulative)
