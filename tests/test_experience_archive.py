import json
import subprocess
import sys


def test_experience_archive_recomputes(project_root):
    completed = subprocess.run(
        [sys.executable, str(project_root / "tools" / "verify_experience_archive.py")],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["ok"] is True
    assert report["metrics"]["clean_v4"]["n"] == 100
    assert report["metrics"]["clean_v4"]["wrong_stop_rate"] == 0.07
    assert report["metrics"]["shifted_v5"]["wrong_stop_rate"] == 0.02
    assert report["metrics"]["noise_selective"]["n"] == 2
