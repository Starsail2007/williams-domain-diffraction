"""Shared paths and lossless, explicit dataset IO."""

from .paths import FROZEN_INPUTS, CONFIG_DIR, PACKAGE_DIR, PROJECT_ROOT, source_label
import csv
import hashlib
import json
import sys
from pathlib import Path

from .paths import PROJECT_ROOT as ROOT, LEGACY_STAGE1 as STAGE1, LEGACY_STAGE2 as STAGE2, RAW_DIR as RAW

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def save_json(path,data):
    from .paths import require_writable_run
    path=require_writable_run(path)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def save_csv(path,rows,fields=None):
    from .paths import require_writable_run
    path=require_writable_run(path)
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
