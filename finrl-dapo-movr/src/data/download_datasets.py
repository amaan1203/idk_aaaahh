"""
src/data/download_datasets.py — Dataset Downloader
==================================================
Downloads all 6 required CSVs from HuggingFace Hub (benstaf/nasdaq_2013_2023).
Skips files that already exist. Idempotent — safe to run multiple times.

Inputs: HF_TOKEN env var (optional for public datasets)
Outputs: ./dataset/*.csv — 6 CSV files for train/test
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "benstaf/nasdaq_2013_2023"
REPO_TYPE = "dataset"
DEST_DIR = Path("dataset")

REQUIRED_FILES = [
    "train_data_2013_2018.csv",
    "train_data_deepseek_risk_2013_2018.csv",
    "train_data_deepseek_sentiment_2013_2018.csv",
    "trade_data_2019_2023.csv",
    "trade_data_deepseek_risk_2019_2023.csv",
    "trade_data_deepseek_sentiment_2019_2023.csv",
]


def download_all(dest_dir: Path = DEST_DIR, force: bool = False) -> None:
    """
    Download all required NASDAQ-100 CSV files from HuggingFace Hub.

    Parameters
    ----------
    dest_dir : local destination directory
    force    : if True, re-download even if file exists
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN", None)

    for filename in REQUIRED_FILES:
        dest_path = dest_dir / filename
        if dest_path.exists() and not force:
            size_mb = dest_path.stat().st_size / 1e6
            print(f"  already exists: {filename} ({size_mb:.1f} MB)")
            continue

        print(f"  downloading: {filename} ... ", end="", flush=True)
        try:
            cached = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type=REPO_TYPE,
                token=hf_token,
            )
            shutil.copy(cached, dest_path)
            size_mb = dest_path.stat().st_size / 1e6
            print(f"done ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"\n[ERROR] Failed to download {filename}: {e}")
            raise

    print(f"\nAll datasets saved to ./{dest_dir}/")


if __name__ == "__main__":
    download_all()
