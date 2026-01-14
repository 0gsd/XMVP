#!/bin/bash
# setup_rvc_env.sh
# Creates a Conda environment 'rvc_env' with Python 3.10 compatible with rvc-python

ENV_NAME="rvc_env"

echo "[*] checking for conda..."
if ! command -v conda &> /dev/null; then
    echo "[-] conda could not be found. Please install Miniconda or Anaconda."
    exit 1
fi

echo "[*] Creating conda environment '$ENV_NAME' with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y

# Activate env
# Need to find conda.sh
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "[*] Installing dependencies..."
# Upgrade pip into env
pip install --upgrade pip

# Install PyTorch (MPS support is standard in recent torch, but 3.10 is stable)
# rvc-python will likely pull torch, but explicit is good.
# We need faiss-cpu for index usage.
pip install torch torchvision torchaudio
pip install faiss-cpu
pip install rvc-python

echo "[*] Installing project requirements..."
# We need shared libs like google-genai, pydub, etc.
# Ideally we'd valid pip install -r requirements.txt if it existed.
# For now, manually install what podcast_animator needs:
pip install google-genai pydub pillow tqdm pyyaml

echo "[*] Setup Complete!"
echo "    Run: ./run_animator.sh"
