"""One project root and explicit locations. Historical bundles retain their schema."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / 'configs'
LEGACY_ROOT = PROJECT_ROOT / 'archive/legacy_workspace'
LEGACY_IMAGE = LEGACY_ROOT / '图像分析'
LEGACY_STAGE1 = LEGACY_IMAGE / 'stage1'
LEGACY_STAGE2 = LEGACY_IMAGE / 'stage2'
FROZEN_INPUTS = PROJECT_ROOT / 'data/annotations/scan_2026-10-10'
CALIBRATION_DIR = PROJECT_ROOT / 'data/calibration/scan_2026-10-10'
RAW_DIR = Path(os.environ.get('WILLIAMS_RAW_DIR', PROJECT_ROOT / 'data/raw/scan_2026-10-10')).expanduser().resolve()
MEASUREMENT_DIR = PROJECT_ROOT / 'data/processed/scan_2026-10-10'
ANALYSIS_BASELINE = PROJECT_ROOT / 'results/analysis/scan_2026-10-10'
WORK_DIR = PROJECT_ROOT / 'work'

def source_label(path):
    """A project-relative provenance label; never a host-specific absolute path."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name

def require_writable_run(path):
    """Keep generation outside every preserved archive and frozen dataset."""
    path = Path(path).expanduser().resolve()
    protected = (PROJECT_ROOT / 'data', PROJECT_ROOT / 'archive', ANALYSIS_BASELINE, RAW_DIR)
    for base in protected:
        if path == base.resolve() or path.is_relative_to(base.resolve()):
            raise ValueError(f'Frozen input/history cannot be an output: {path}')
    return path
