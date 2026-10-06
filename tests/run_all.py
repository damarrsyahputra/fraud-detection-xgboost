"""Menjalankan seluruh suite test aplikasi fraud detection."""

import subprocess
import sys
from pathlib import Path

TESTS = ["test_detector.py", "test_integration.py", "test_app.py"]


def main() -> int:
    here = Path(__file__).resolve().parent
    failed = []
    for name in TESTS:
        print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
        proc = subprocess.run(
            [sys.executable, str(here / name)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout.splitlines():
            if "FAIL" in line or "pemeriksaan lolos" in line or "Error" in line:
                print(line)
        if proc.returncode != 0:
            failed.append(name)
            print(proc.stdout[-1500:])
            print(proc.stderr[-1500:])

    print(f"\n{'=' * 60}")
    if failed:
        print(f"  GAGAL: {', '.join(failed)}")
        return 1
    print(f"  Semua {len(TESTS)} file test lolos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
