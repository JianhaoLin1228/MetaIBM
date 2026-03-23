from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
METAIBM_DIR = PROJECT_ROOT / 'metaibm'


def ensure_metaibm_on_path() -> Path:
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return METAIBM_DIR


ensure_metaibm_on_path()