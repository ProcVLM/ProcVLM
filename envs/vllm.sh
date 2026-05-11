conda create -n vllm python=3.10 -y
conda activate vllm
conda install -c conda-forge cuda-compat=12.9 -y

# Set up environment variables for the conda environment
CONDA_ENV_PREFIX="$CONDA_PREFIX"
mkdir -p "$CONDA_ENV_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_ENV_PREFIX/etc/conda/deactivate.d"

ACTIVATE_FILE="$CONDA_ENV_PREFIX/etc/conda/activate.d/env_vars.sh"
DEACTIVATE_FILE="$CONDA_ENV_PREFIX/etc/conda/deactivate.d/env_vars.sh"
TARGET_PATH="\$CONDA_PREFIX/cuda-compat"

# activate envs
cat <<EOF > "$ACTIVATE_FILE"
#!/bin/sh
export PYTHONNOUSERSITE=1
export OLD_LD_PATH="\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$TARGET_PATH:\$LD_LIBRARY_PATH"
echo "[vllm env] LD_LIBRARY_PATH updated: Added $TARGET_PATH"
EOF

# deactivate envs
cat <<EOF > "$DEACTIVATE_FILE"
#!/bin/sh
unset PYTHONNOUSERSITE
export LD_LIBRARY_PATH="\$OLD_LD_PATH"
unset OLD_LD_PATH
echo "[vllm env] LD_LIBRARY_PATH restored."
EOF

# pip package installation
pip install uv
uv pip install -r envs/others_pip.txt
uv pip install -r envs/vllm_pip.txt