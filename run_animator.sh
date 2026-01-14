#!/bin/bash
# run_animator.sh
# Wrapper to run podcast_animator.py inside the 'rvc_env' conda environment

ENV_NAME="rvc_env"

# Locate Conda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate
conda activate $ENV_NAME

# Run
python podcast_animator.py "$@"
