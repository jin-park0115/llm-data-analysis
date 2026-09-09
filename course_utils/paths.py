from pathlib import Path

def get_project_root(start_path=None):
    """현재 위치 또는 지정한 위치에서 프로젝트 루트를 찾는다."""
    current = Path(
        start_path or Path.cwd()
    ).resolve()
    while True:
        if (current / "data").is_dir():
            return current
        if current.parent == current:
            raise FileNotFoundError(
                "data 폴더가 있는 프로젝트 루트를 찾을 수 없습니다."
            )
        current = current.parent

def get_data_dir():
    """data/raw 폴더를 반환한다."""
    return get_project_root() / "data" / "raw"

def get_report_dir():
    """reports 폴더를 반환한다."""
    return get_project_root() / "reports"