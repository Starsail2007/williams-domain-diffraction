"""Candidate geometry only. No photometric integration or voltage assignment."""

from .paths import FROZEN_INPUTS, CONFIG_DIR, PACKAGE_DIR, PROJECT_ROOT, source_label
import argparse
import csv
import hashlib
import itertools
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import scipy
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
from .inventory import BASE, RAW, write_csv, write_json, sha256
from .raw_check import display

DISPLAY_SIZE=np.array([1406,938],dtype=float)
FONT_PATH='/System/Library/Fonts/Menlo.ttc'


def display_to_raw(points, red_shape, offset=(0,0)):
    return 2*((np.asarray(points,dtype=float)+.5)*np.array(red_shape[::-1])/DISPLAY_SIZE-.5)+np.asarray(offset)


def raw_to_display(points, red_shape, offset=(0,0)):
    return ((np.asarray(points)-np.asarray(offset))/2+.5)*DISPLAY_SIZE/np.array(red_shape[::-1])-.5


def transform(points,H):
    points=np.asarray(points,dtype=float)
    h=np.c_[points,np.ones(len(points))]@H.T
    if np.any(np.abs(h[:,2])<1e-12):
        raise ValueError('Degenerate projection')
    return h[:,:2]/h[:,2:]


def homography(corners,width,height):
    c=np.asarray(corners,dtype=np.float32)
    if c.shape!=(4,2) or not np.isfinite(c).all() or width<=0 or height<=0:
        raise ValueError('Invalid corners or dimensions')
    if not cv2.isContourConvex(c) or abs(cv2.contourArea(c))<1:
        raise ValueError('Degenerate/nonconvex corners')
    target=np.array([[0,0],[width,0],[width,height],[0,height]],np.float32)
    return cv2.getPerspectiveTransform(c,target)


def centroid(red, center, radius):
    x,y=center; pad=int(np.ceil(radius*2.1))
    x0,x1=max(0,int(x)-pad),min(red.shape[1],int(x)+pad+1)
    y0,y1=max(0,int(y)-pad),min(red.shape[0],int(y)+pad+1)
    yy,xx=np.mgrid[y0:y1,x0:x1]; d=np.hypot(xx-x,yy-y)
    values=red[y0:y1,x0:x1]
    ann=(d>=radius*1.45)&(d<=radius*2.07)
    bg=float(np.median(values[ann]))
    w=np.maximum(values-bg,0)*(d<=radius)
    if w.sum()<=0:
        return np.array([x,y]),bg,0.,0.
    cx,cy=float((w*xx).sum()/w.sum()),float((w*yy).sum()/w.sum())
    cov=np.array([[(w*(xx-cx)**2).sum(),(w*(xx-cx)*(yy-cy)).sum()],
                  [(w*(xx-cx)*(yy-cy)).sum(),(w*(yy-cy)**2).sum()]])/w.sum()
    eig=np.maximum(np.linalg.eigvalsh(cov),0)
    return np.array([cx,cy]),bg,float(np.sqrt(eig[1])*2),float(np.sqrt(eig[1]/max(eig[0],1e-10)))


def detect(red,box,cfg):
    # Filtering only locates candidates. Coordinates refined on unfiltered R samples.
    x0,y0,x1,y1=map(int,box)
    sub=red[y0:y1,x0:x1]
    smooth=ndi.gaussian_filter(sub,cfg['gaussian_sigma_red_samples'])
    contrast=smooth-ndi.gaussian_filter(sub,cfg['background_sigma_red_samples'])
    size=int(cfg['min_separation_raw_px']/2)*2+1
    maxima=(contrast==ndi.maximum_filter(contrast,size=size))&(contrast>=cfg['minimum_contrast_DN'])
    yy,xx=np.where(maxima)
    seeds=sorted([(float(contrast[y,x]),x+x0,y+y0) for y,x in zip(yy,xx)],reverse=True)
    selected=[]
    for amp,x,y in seeds:
        if all(np.hypot(x-q[1],y-q[2])*2>=cfg['min_separation_raw_px'] for q in selected):
            selected.append((amp,x,y))
    return selected


def circle(draw,xy,r,color,width=2):
    x,y=xy;draw.ellipse((x-r,y-r,x+r,y+r),outline=color,width=width)


def run(config_path,out,mapping_path=None):
    import rawpy
    from .paths import require_writable_run
    out=require_writable_run(out)
    config=json.loads(config_path.read_text())
    out.mkdir(parents=True,exist_ok=False)
    for d in ('overlays','rectified','crops'):(out/d).mkdir()
    write_json(out/'run_config.json',config)
    config_hash=sha256(config_path)
    frame_map={f['frame_id']:f for f in csv.DictReader(mapping_path.open(encoding='utf-8-sig'))} if mapping_path else {}
    font=ImageFont.truetype(FONT_PATH,18);small=ImageFont.truetype(FONT_PATH,15)
    spots=[];pairs=[];calibrations=[];summaries=[];all_annotations=[]
    for sample in config['samples']:
        fid=sample['frame_id']; cfg=config['detection'];path=RAW/f'{fid}.ARW'
        meta=frame_map.get(fid,{})
        voltage=meta.get('voltage_recorded_V','')
        direction=meta.get('scan_direction','unknown')
        label=f'{direction.upper()} {float(voltage):.2f} V' if voltage else 'voltage pending'
        with rawpy.imread(str(path)) as raw:
            if raw.sizes.flip!=0 or raw.sizes.top_margin!=0 or raw.sizes.left_margin!=0:
                raise ValueError('Rotation/crop offset not supported for this sensor configuration')
            if raw.raw_image_visible.ndim!=2 or raw.raw_pattern.tolist()!=[[0,1],[3,2]] or raw.color_desc!=b'RGBG':
                raise ValueError('Unverified sensor layout')
            a=raw.raw_image_visible
            black=raw.black_level_per_channel[0]
            red=a[::2,::2].astype(np.float32)-black
            white=raw.white_level;cw=raw.camera_white_level_per_channel[0]
            corners=display_to_raw(sample['corners_display'],red.shape)
            H=homography(corners,config['screen_width_mm'],config['screen_height_mm'])
            box_display=np.array(sample['detection_box_display']).reshape(2,2)
            box_red=display_to_raw(box_display,red.shape).reshape(-1)/2
            seeds=detect(red,box_red,cfg)
            these=[]
            for amp,x,y in seeds:
                center,bg,major,ratio=centroid(red,(x,y),cfg['centroid_radius_raw_px']/2)
                other,_,_,_=centroid(red,(x,y),50/2)
                u,v=center*2
                mm=transform([[u,v]],H)[0]
                if not (0<=mm[0]<=config['screen_width_mm'] and 0<=mm[1]<=config['screen_height_mm']):continue
                radius=cfg['centroid_radius_raw_px']/2
                lo=np.maximum(np.floor(center-radius).astype(int),0)
                hi=np.minimum(np.ceil(center+radius).astype(int)+1,np.array(red.shape[::-1]))
                yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
                mask=(xx-center[0])**2+(yy-center[1])**2<=radius**2
                vals=a[::2,::2][lo[1]:hi[1],lo[0]:hi[0]][mask]
                nclip=int(np.sum(vals>=white));nnear=int(np.sum(vals>=cw))
                flags=['REVIEW_REQUIRED','GEOMETRY_UNVERIFIED','ZERO_ORDER_UNCERTAIN']
                if nclip:flags.append('SATURATED')
                elif nnear:flags.append('NEAR_SATURATION')
                if amp<cfg['low_contrast_DN']:flags.append('LOW_CONTRAST_CANDIDATE')
                if ratio>2.5:flags.append('ELONGATED')
                if major>25:flags.append('BROAD_OR_DIFFUSE')
                edge_dist=cv2.pointPolygonTest(corners.astype(np.float32),(float(u),float(v)),True)
                if edge_dist<135:flags.append('EDGE_RISK')
                if fid in ('018','064','077'):flags.append('OVERLAP_REVIEW')
                these.append(dict(frame_id=fid,voltage_recorded_V=voltage,scan_direction=direction,u_raw_px=float(u),v_raw_px=float(v),
                    X_mm_candidate=float(mm[0]),Y_mm_candidate=float(mm[1]),
                    screen_calibration_status='CANDIDATE_UNCONFIRMED',
                    coordinate_origin='RAW_visible_pixel_center_0_0',channel='R_CFA_index0',
                    localization='positive_background_subtracted_centroid; NOT_flux_integral',
                    localization_radius_raw_px=cfg['centroid_radius_raw_px'],
                    radius_sensitivity_raw_px=float(np.linalg.norm(center-other)*2),
                    detection_contrast_DN=amp,local_background_DN_after_black=bg,
                    shape_major_rms_raw_px=major,shape_axis_ratio=ratio,
                    sample_count=len(vals),saturated_count=nclip,near_white_count=nnear,
                    saturated_fraction=nclip/len(vals),raw_max=int(vals.max()),
                    zero_order_confirmed=False,zero_order_distance_mm='',
                    integrated_signal='',normalized_signal='',
                    geometry_group_id=f'G{fid}',photometry_valid=False,
                    quality_flags=';'.join(flags)))
            # Neutral IDs in screen row bands then x; never identifies order or mode.
            these.sort(key=lambda s:(round(s['Y_mm_candidate']/5),s['X_mm_candidate']))
            for i,s in enumerate(these,1):s['spot_id']=f'spot_{i:03d}'
            for s,t in itertools.combinations(these,2):
                pd=float(np.hypot(s['u_raw_px']-t['u_raw_px'],s['v_raw_px']-t['v_raw_px']))
                md=float(np.hypot(s['X_mm_candidate']-t['X_mm_candidate'],s['Y_mm_candidate']-t['Y_mm_candidate']))
                pairs.append(dict(frame_id=fid,voltage_recorded_V=voltage,scan_direction=direction,spot_id_1=s['spot_id'],spot_id_2=t['spot_id'],
                    measurement_type='candidate_pair_distance_not_assigned_order',distance_raw_px=pd,
                    distance_mm_candidate=md,uncertainty_mm='',
                    quality_flags=';'.join(sorted(set((s['quality_flags']+';'+t['quality_flags']).split(';'))))))
            im=Image.fromarray(display(red)).resize(tuple(DISPLAY_SIZE.astype(int)),Image.Resampling.LANCZOS).convert('RGB')
            overlay=im.copy();draw=ImageDraw.Draw(overlay)
            draw.polygon([tuple(c) for c in sample['corners_display']],outline='#ffce56',width=3)
            for i,c in enumerate(sample['corners_display']):
                circle(draw,c,7,'#ffce56',3);draw.text((c[0]+9,c[1]-23),['TL','TR','BR','BL'][i],font=font,fill='#ffce56')
            draw.rectangle(sample['detection_box_display'],outline='#8498b7',width=1)
            for s in these:
                pos=raw_to_display([[s['u_raw_px'],s['v_raw_px']]],red.shape)[0]
                color='#ff7070' if s['saturated_count'] else '#70e0ff'
                circle(draw,pos,65/2*DISPLAY_SIZE[0]/red.shape[1],color)
                # Proposed background ring; not used as a finalized flux aperture.
                circle(draw,pos,95/2*DISPLAY_SIZE[0]/red.shape[1],'#74a891',1)
                circle(draw,pos,135/2*DISPLAY_SIZE[0]/red.shape[1],'#74a891',1)
                draw.line((pos[0]-4,pos[1],pos[0]+4,pos[1]),fill=color,width=2)
                draw.line((pos[0],pos[1]-4,pos[0],pos[1]+4),fill=color,width=2)
                draw.text((pos[0]+10,pos[1]-23),s['spot_id'][5:],font=small,fill=color,stroke_width=2,stroke_fill='black')
            draw.rectangle((0,0,1406,64),fill='#141b25')
            draw.text((14,8),f'{fid}.ARW | {label} | R CFA | candidate corners + spots',font=font,fill='white')
            draw.text((14,34),'cyan/red: localization ROI; green: proposed BG ring; red: RAW saturation',font=small,fill='#cbd3df')
            overlay.save(out/'overlays'/f'{fid}.png')
            # RAW R grid -> screen mm -> output pixel grid; endpoints at 0 and W*scale.
            scale=6;R=np.diag([2.,2.,1.]);S=np.diag([scale,scale,1.])
            rect=cv2.warpPerspective(display(red),S@H@R,(601,721),flags=cv2.INTER_LINEAR)
            rect=Image.fromarray(rect).convert('RGB');rd=ImageDraw.Draw(rect)
            for s in these:
                p=(s['X_mm_candidate']*scale,s['Y_mm_candidate']*scale)
                circle(rd,p,5,'#ff7070' if s['saturated_count'] else '#70e0ff')
                rd.text((min(p[0]+6,563),p[1]-18),s['spot_id'][5:],font=small,fill='#70e0ff',stroke_width=2,stroke_fill='black')
            rd.rectangle((10,10,590,54),fill='#141b25')
            rd.text((18,14),f'{fid} | {label} | CANDIDATE',font=font,fill='#ffce56')
            rd.line((30,690,150,690),fill='white',width=4);rd.text((30,660),'20 mm',font=font,fill='white')
            rect.save(out/'rectified'/f'{fid}.png')
            # Larger region view for weak spots, same global display range.
            overlay.crop(tuple(sample['detection_box_display'])).resize((1000,int(1000*(box_display[1,1]-box_display[0,1])/(box_display[1,0]-box_display[0,0]))),Image.Resampling.LANCZOS).save(out/'crops'/f'{fid}.png')
            calibrations.append(dict(frame_id=fid,corners_raw_px=corners.tolist(),corners_display=sample['corners_display'],
                H_raw_to_screen_mm=H.tolist(),red_plane_to_raw_matrix=R.tolist(),screen_mm=[100,120],
                status='CANDIDATE_UNCONFIRMED',independent_validation=False,lens_distortion_corrected=False,
                zero_order_confirmed=False,raw_sha256=sha256(path),config_sha256=config_hash))
            all_annotations.append(dict(frame_id=fid,corners=calibrations[-1],candidates=these,
                review_status='pending_user',zero_order_spot_id=None,confirmed_spot_ids=[],rejected_spot_ids=[]))
            summaries.append(dict(frame_id=fid,description=sample['description'],candidate_count=len(these),
                saturated_candidates=sum(s['saturated_count']>0 for s in these),geometry_group_id=f'G{fid}',
                scan_direction=direction,voltage_recorded_V=voltage,quality_flags='REVIEW_REQUIRED;GEOMETRY_UNVERIFIED'+('' if voltage else ';MAPPING_MISSING')))
            spots.extend(these)
            print(fid,len(these),'candidates',flush=True)
    write_csv(out/'spots.csv',spots);write_csv(out/'pairs.csv',pairs);write_csv(out/'samples.csv',summaries)
    write_json(out/'calibration.json',calibrations);write_json(out/'candidate_annotations.json',all_annotations)
    write_json(out/'software_versions.json',dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        rawpy=rawpy.__version__,libraw=rawpy.libraw_version,opencv=cv2.__version__,source_hashes={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}))
    canvas=Image.new('RGB',(1803,2884),'#141b25')
    for i,s in enumerate(config['samples']):
        canvas.paste(Image.open(out/'rectified'/f"{s['frame_id']}.png"),((i%3)*601,(i//3)*721))
    canvas.save(out/'sample_overview.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=BASE/'annotations/sample_config.json')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--mapping',type=Path);args=p.parse_args()
    run(args.config,args.out,args.mapping)
