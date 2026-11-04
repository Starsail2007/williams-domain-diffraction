"""Stage 1 inventory. Never changes RAW bytes; explicit --rename changes names only."""

from .paths import FROZEN_INPUTS, CONFIG_DIR, PACKAGE_DIR, PROJECT_ROOT, source_label
import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .paths import LEGACY_STAGE1 as BASE
from .paths import RAW_DIR as RAW


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, fields=None):
    from .paths import require_writable_run
    path=require_writable_run(path)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path, data):
    from .paths import require_writable_run
    path=require_writable_run(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def read_metadata(path):
    import exifread
    with path.open('rb') as f:
        tags = exifread.process_file(f, details=False)
    return {k: str(v) for k, v in tags.items()
            if k not in ('JPEGThumbnail', 'TIFFThumbnail')}


def capture_key(meta):
    # Preserve camera-local wall time; no guessed time zone conversion.
    t = meta.get('EXIF DateTimeOriginal')
    if not t:
        raise ValueError('Missing DateTimeOriginal: chronological renaming blocked')
    dt = datetime.strptime(t, '%Y:%m:%d %H:%M:%S')
    sub = meta.get('EXIF SubSecTimeOriginal', '')
    return dt, float('0.' + sub) if sub else 0.0


def load_voltage_records():
    return json.loads((BASE / 'inputs/voltage_records.json').read_text())


def contact_sheet(rows, out, page_size=20):
    from .paths import require_writable_run
    out=require_writable_run(out)
    font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 17)
    for start in range(0, len(rows), page_size):
        canvas = Image.new('RGB', (1600, 1280), '#171b23')
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 8), 'Camera JPEG previews | display only | voltage mapping pending', font=font)
        for i, row in enumerate(rows[start:start+page_size]):
            x, y = (i % 4)*400, 40+(i//4)*248
            preview = out / 'previews' / (row['frame_id']+'.jpg')
            if preview.exists():
                im = Image.open(preview).convert('RGB')
                im.thumbnail((394, 210))
                canvas.paste(im, (x+(400-im.width)//2, y))
            draw.text((x+8, y+212), f"{row['frame_id']}  {row['datetime_original'][11:]}", font=font)
        canvas.save(out / f'contact_{start//page_size+1:02d}.jpg', quality=94)


def audit(out, rename=False):
    import rawpy
    from .paths import require_writable_run,RAW_DIR
    out=require_writable_run(out)
    if rename and RAW.resolve()==RAW_DIR.resolve():
        raise ValueError('Original RAW renaming is disabled; preserve the current frame identities')
    out.mkdir(parents=True, exist_ok=False)
    (out/'previews').mkdir()
    paths = sorted(RAW.glob('*.ARW'))
    if not paths:
        raise ValueError('No RAW input')
    meta, preflight, failures = {}, [], []
    for p in paths:
        row=dict(original_name=p.name,size_bytes=p.stat().st_size,sha256=sha256(p))
        try:
            meta[p.name]=read_metadata(p)
            capture_key(meta[p.name])
            row['metadata_status']='OK'
        except Exception as e:
            row['metadata_status']='FAILED'
            row['error']=str(e)
            failures.append(dict(file=p.name,error=str(e)))
        preflight.append(row)
    write_csv(out/'input_preflight.csv',preflight)
    write_json(out/'metadata_raw.json',meta)
    if failures:
        write_json(out/'input_failures.json',failures)
        raise ValueError('Metadata/time failure; input list saved; no RAW renamed')
    ordered = sorted(paths, key=lambda p: capture_key(meta[p.name]))
    keys = [capture_key(meta[p.name]) for p in ordered]
    if len(set(keys)) != len(keys):
        raise ValueError('Ambiguous capture-time ties: no renaming performed')
    plan = []
    for i, p in enumerate(ordered, 1):
        plan.append(dict(frame_id=f'{i:03d}', original_name=p.name,
                         new_name=f'{i:03d}.ARW' if rename else p.name,
                         size_bytes=p.stat().st_size, sha256=sha256(p),
                         datetime_original=meta[p.name]['EXIF DateTimeOriginal']))
    # Durable full plan and original metadata precede the first rename.
    write_csv(out/'rename_plan.csv', plan)
    for row in plan:
        src, dst = RAW/row['original_name'], RAW/row['new_name']
        if src != dst and dst.exists():
            raise FileExistsError(dst)
    if rename:
        with (out/'rename_journal.jsonl').open('x', encoding='utf-8') as journal:
            for row in plan:
                src, dst = RAW/row['original_name'], RAW/row['new_name']
                if src != dst:
                    src.rename(dst)
                # Verify bytes after rename, not only file size.
                ok = sha256(dst) == row['sha256']
                journal.write(json.dumps(dict(**row, hash_verified=ok))+'\n')
                journal.flush()
                if not ok:
                    raise RuntimeError(f'Hash mismatch {dst}')
    frames, errors = [], []
    exif_fields = {
        'exposure_time_s':'EXIF ExposureTime', 'iso':'EXIF ISOSpeedRatings',
        'f_number':'EXIF FNumber','focal_length_mm':'EXIF FocalLength',
        'camera_make':'Image Make','camera_model':'Image Model',
        'lens':'EXIF LensModel','orientation':'Image Orientation',
        'subsecond':'EXIF SubSecTimeOriginal','offset_time_original':'EXIF OffsetTimeOriginal',
        'exif_width':'EXIF ExifImageWidth','exif_height':'EXIF ExifImageLength',
    }
    for row in plan:
        tags = meta[row['original_name']]
        frame = dict(row, file_key='10.10 raw/'+row['new_name'],
                     **{k:tags.get(v,'') for k,v in exif_fields.items()},
                     voltage_recorded_V='', voltage_kind='unknown', voltage_source='unknown',
                     scan_direction='unknown', cycle_id='', step_index='', wait_time_s='',
                     frequency_Hz='', geometry_group_id='unverified',
                     quality_flags='MAPPING_MISSING;GEOMETRY_UNVERIFIED;ZERO_ORDER_UNCERTAIN')
        try:
            with rawpy.RawPy() as r:
                r.open_file(str(RAW/row['new_name']))
                thumb = r.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    im = Image.open(io.BytesIO(thumb.data))
                    im = ImageOps.exif_transpose(im).convert('RGB')
                else:
                    im = Image.fromarray(thumb.data)
                frame['preview_width'], frame['preview_height'] = im.size
                im.save(out/'previews'/f"{row['frame_id']}.jpg", quality=95)
                frame['preview_status'] = 'OK_CAMERA_RENDERING_NOT_LINEAR'
        except Exception as e:
            frame['preview_status'] = 'FAILED'
            errors.append({'file':row['new_name'],'stage':'preview','error':str(e)})
        frames.append(frame)
    group_keys = sorted(set(tuple(f[k] for k in ['exposure_time_s','iso','f_number','focal_length_mm']) for f in frames))
    for f in frames:
        key=tuple(f[k] for k in ['exposure_time_s','iso','f_number','focal_length_mm'])
        f['photometry_settings_group_id'] = f'S{group_keys.index(key)+1:02d}'
    write_csv(out/'manifest.csv', plan)
    write_csv(out/'frames.csv', frames)
    records = load_voltage_records()
    vr=[]
    for direction in ('up','down'):
        for step,v in enumerate(records[direction],1):
            vr.append(dict(record_id=f'V{len(vr)+1:03d}', scan_direction=direction,
                           step_index=step, voltage_recorded=v, unit='V_assumed_pending_confirmation',
                           voltage_kind='unknown',voltage_source='user_chat',file_key='',
                           matching_status=f'UNRESOLVED_{len(plan)}_RAW_{sum(len(records[d]) for d in ("up","down"))}_RECORDS'))
    write_csv(out/'voltage_records_unmatched.csv',vr)
    counts=Counter(r['sha256'] for r in plan)
    summary=dict(raw_count=len(plan),voltage_count=len(vr),up=len(records['up']),down=len(records['down']),
                 duplicate_hashes=[h for h,n in counts.items() if n>1],errors=errors,
                 rename_hashes_verified=rename, time_basis='DateTimeOriginal; no timezone inferred',
                 settings_groups=[list(x) for x in group_keys],
                 mapping_status='pending_user_resolution; no frame-voltage assignment made')
    write_json(out/'inventory_summary.json',summary)
    contact_sheet(frames,out)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rename',action='store_true')
    args=parser.parse_args()
    audit(args.out,args.rename)
