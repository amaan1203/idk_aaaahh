"""
scripts/00_download_data.py — Download All Required Datasets
============================================================
Downloads all 6 NASDAQ-100 CSVs from HuggingFace Hub (benstaf/nasdaq_2013_2023).
Verifies each file exists and is non-empty after download.

Run: python scripts/00_download_data.py

Outputs: dataset/*.csv (6 files)
"""

import sys
from pathlib import Path

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.data.download_datasets import download_all, REQUIRED_FILES, DEST_DIR


def verify_datasets() -> bool:
    """Check all files exist and are non-empty."""
    all_ok = True
    print("\nVerifying downloads:")
    for filename in REQUIRED_FILES:
        path = DEST_DIR / filename
        if not path.exists():
            print(f"  [MISSING] {filename}")
            all_ok = False
        elif path.stat().st_size == 0:
            print(f"  [EMPTY]   {filename}")
            all_ok = False
        else:
            import pandas as pd
            try:
                df = pd.read_csv(path, nrows=5)
                size_mb = path.stat().st_size / 1e6
                print(f"  [OK]      {filename} ({size_mb:.1f} MB, {df.shape[1]} columns)")
            except Exception as e:
                print(f"  [ERROR]   {filename}: {e}")
                all_ok = False
    return all_ok


if __name__ == "__main__":
    print("=== Script 00: Download Datasets ===\n")
    download_all()
    ok = verify_datasets()
    if not ok:
        print("\n[FAILED] Some downloads failed. Check your HF_TOKEN and network.")
        sys.exit(1)
    print("\n[DONE] All datasets ready. Run: python scripts/01_reproduce_baseline.py")
