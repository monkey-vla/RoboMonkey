#!/bin/bash

# Exit on error
set -e

echo "Starting setup process..."

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 completed successfully"
    else
        echo "✗ Error during $1"
        exit 1
    fi
}

# Initialize conda in the current shell
echo "Initializing conda..."
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

# Create and activate environment
if ! conda env list | grep -qE "^\s*simpler_env\s"; then
    echo "Creating simpler_env environment..."
    $HOME/miniconda3/bin/conda create -n simpler_env python=3.10 -y
else
    echo "Conda environment 'simpler_env' already exists. Skipping creation."
fi

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate simpler_env

cd "$dir_path/../SimplerEnv"
sudo apt-get install -y libvulkan1 cudnn9-cuda-12 libx11-6
pip install --upgrade pip
pip install numpy==1.24.4
pip install -r requirements_full_install.txt --use-deprecated=legacy-resolver
pip install torch torchvision torchaudio
pip install tensorflow-probability==0.24.0
check_status "SimplerEnv base setup"

# Additional SimplerEnv dependencies
pip install -e ./ManiSkill2_real2sim/
pip install -e .
pip install pandas openpyxl matplotlib mediapy
pip install tensorflow[and-cuda]==2.18.0
pip install "git+https://github.com/nathanrooy/simulated-annealing"
pip install flax==0.8.1
pip install dlimp "git+https://github.com/kvablack/dlimp@d08da3852c149548aaa8551186d619d87375df08"
pip install distrax==0.1.5 flax==0.8.1 transformers wandb mediapy==1.2.0 tf_keras einops
pip install chex==0.1.2
check_status "SimplerEnv additional dependencies"

# OpenVLA-mini setup
echo "Setting up OpenVLA-mini..."
cd "$dir_path/../openvla-mini"
pip install -e .
pip install -r requirements-min.txt
pip install draccus rich accelerate tensorflow_graphics jsonlines robosuite==1.4.0 bddl easydict gym PyOpenGL_accelerate scikit-learn
pip install flash-attn -U --no-build-isolation
check_status "OpenVLA-mini setup"

# Environment variables
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Additional packages
pip install fastapi uvicorn json_numpy flax==0.8.1
check_status "Additional packages installation"