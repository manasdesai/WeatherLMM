#!/bin/bash
# Setup script for WeatherLMM conda environment on Derecho
# Usage: bash setup_environment_derecho.sh

set -e  # Exit on error

echo "=========================================="
echo "WeatherLMM Environment Setup for Derecho"
echo "=========================================="

# Load necessary modules
echo "Loading modules..."
module purge
module load ncarenv-basic/25.10
module load conda
module load cuda/12.2.0

# Check if we're on Derecho
if [[ ! "$HOSTNAME" == *"derecho"* ]]; then
    echo "Warning: This script is designed for Derecho. Continuing anyway..."
fi

# Check if environment already exists
if conda env list | grep -q "^weatherlmm "; then
    echo ""
    echo "Environment 'weatherlmm' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n weatherlmm -y
    else
        echo "Updating existing environment..."
        conda env update -n weatherlmm -f environment.yml
        echo ""
        echo "Environment updated. Activate with: conda activate weatherlmm"
        exit 0
    fi
fi

# Create environment from environment.yml
echo ""
echo "Creating conda environment from environment.yml..."
echo "This may take 10-15 minutes..."
conda env create -f environment.yml

# Activate environment
echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate weatherlmm

# Install PyTorch with CUDA support via pip (since CUDA is provided by system module)
echo ""
echo "Installing PyTorch with CUDA 12.2 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || echo "Warning: PyTorch CUDA check failed (may be normal on login node)"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import peft; print(f'PEFT version: {peft.__version__}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Environment 'weatherlmm' is ready."
echo ""
echo "To activate the environment, run:"
echo "  conda activate weatherlmm"
echo ""
echo "Note: CUDA may not be available on login nodes."
echo "      It will be available when running on compute nodes via PBS."
echo ""
