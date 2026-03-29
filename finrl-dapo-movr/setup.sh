#!/bin/bash
set -e

echo "=== FinRL-DAPO-MOVR setup ==="

# 1. System packages
sudo apt-get update -q
sudo apt-get install -y git wget curl tmux htop

# 2. Python dependencies
pip install --upgrade pip
echo "Installing torch stack (2.4.0)..."
pip install torch>=2.4.0 torchvision>=0.19.0

echo "Installing remaining requirements..."
pip install -r requirements.txt

# 3. Create output directories
mkdir -p plots results checkpoints logs

# 4. Copy env template if .env does not exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from template. Fill in HF_TOKEN and GROQ_API_KEY before running scripts."
fi

echo "=== Setup complete. Run: python scripts/00_download_data.py ==="
