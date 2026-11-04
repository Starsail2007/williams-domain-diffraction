"""Decode every RAW; save channel diagnostics and RAW-derived sample displays."""

from .paths import FROZEN_INPUTS, CONFIG_DIR, PACKAGE_DIR, PROJECT_ROOT, source_label
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .inventory import BASE, RAW, write_csv, write_json

SAMPLES = ['001','011','012','014','018','026','046','064','070','077']


def display(a):
    # Fixed display-only asinh transform. Black already subtracted.
    return np.uint8(np.clip(np.arcsinh(np.maximum(a,0)/8)/np.arcsinh((16383-512)/8),0,1)*255)


def run(out):
    import rawpy
    from .paths import require_writable_run
    out=require_writable_run(out)
    out.mkdir(parents=True,exist_ok=False)
    (out/'raw_displays').mkdir()
    results=[]
    for p in sorted(RAW.glob('*.ARW')):
        try:
            with rawpy.imread(str(p)) as r:
                a=r.raw_image_visible
                rec=dict(frame_id=p.stem,decode_status='OK',sizes=r.sizes._asdict(),
                         raw_type=str(r.raw_type),pattern=r.raw_pattern.tolist(),
                         color_desc=r.color_desc.decode(),black=r.black_level_per_channel,
                         white_level=r.white_level,camera_white=r.camera_white_level_per_channel)
                stats=[]
                if a.ndim!=2 or r.raw_pattern.shape!=(2,2):
                    raise ValueError('This analysis requires a verified 2x2 CFA')
                for dy in range(2):
                    for dx in range(2):
                        ci=int(r.raw_pattern[dy,dx]); v=a[dy::2,dx::2]
                        threshold=(r.camera_white_level_per_channel or [r.white_level]*4)[ci]
                        stats.append(dict(channel_index=ci,channel=r.color_desc.decode()[ci],dx=dx,dy=dy,
                                          count=v.size,min=int(v.min()),max=int(v.max()),
                                          median=float(np.median(v)),p999=float(np.percentile(v,99.9)),
                                          n_at_white=int(np.sum(v>=r.white_level)),
                                          n_at_camera_white=int(np.sum(v>=threshold))))
                rec['channel_stats']=stats
                if p.stem in SAMPLES:
                    rpos=np.argwhere(r.raw_pattern==0)
                    if len(rpos)!=1 or r.color_desc[0]!=ord('R'):
                        raise ValueError('Red CFA channel is not unique')
                    dy,dx=map(int,rpos[0])
                    red=a[dy::2,dx::2].astype(np.float32)-r.black_level_per_channel[0]
                    im=Image.fromarray(display(red))
                    # Sample grid q=(u-dx)/2; resize center convention saved below.
                    im=im.resize((1406,938),Image.Resampling.LANCZOS).convert('RGB')
                    im.save(out/'raw_displays'/f'{p.stem}.png')
                    rec['raw_display']={'channel':0,'red_offset_xy':[dx,dy],
                        'sample_plane_size':[red.shape[1],red.shape[0]],'display_size':list(im.size),
                        'display_to_raw_formula':'u=2*((x+0.5)*sample_width/display_width-0.5)+dx; same for v',
                        'transform':'clip(asinh(max(raw-black,0)/8)/asinh((16383-512)/8),0,1)'}
                results.append(rec)
        except Exception as e:
            results.append(dict(frame_id=p.stem,decode_status='FAILED',error=str(e)))
        if len(results)%10==0:
            print(f'Decoded {len(results)}/80',flush=True)
    write_json(out/'raw_metadata.json',results)
    rows=[]
    for rec in results:
        if rec['decode_status']!='OK':
            rows.append(dict(frame_id=rec['frame_id'],decode_status='FAILED')); continue
        for s in rec['channel_stats']:
            rows.append(dict(frame_id=rec['frame_id'],decode_status='OK',**s,
                             black_level=rec['black'][s['channel_index']],white_level=rec['white_level'],
                             camera_white_level=rec['camera_white'][s['channel_index']] if rec['camera_white'] else ''))
    write_csv(out/'raw_channel_diagnostics.csv',rows)
    print('Failures:',[r for r in results if r['decode_status']!='OK'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
