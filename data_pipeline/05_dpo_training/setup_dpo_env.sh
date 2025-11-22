#!/bin/bash
# Automated DPO Training Environment Setup Script
# For Lambda Labs A100 instances
# This script creates a working environment avoiding all known issues

set -e  # Exit on error

echo "=========================================="
echo "DPO Training Environment Setup"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo "⚠️  Warning: This script is tested with Python 3.10"
    echo "   Current version: $PYTHON_VERSION"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CUDA availability
echo ""
echo "Checking CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Is this a GPU instance?"
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Create virtual environment
echo "=========================================="
echo "Step 1: Creating virtual environment"
echo "=========================================="

if [ -d "dpo_venv" ]; then
    echo "⚠️  Virtual environment already exists at ./dpo_venv"
    read -p "Delete and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf dpo_venv
    else
        echo "Skipping virtual environment creation"
        source dpo_venv/bin/activate
    fi
fi

if [ ! -d "dpo_venv" ]; then
    python3 -m venv dpo_venv
    source dpo_venv/bin/activate

    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip -q

    echo "✅ Virtual environment created"
else
    echo "✅ Using existing virtual environment"
fi

# Install PyTorch with CUDA support
echo ""
echo "=========================================="
echo "Step 2: Installing PyTorch with CUDA 12.1"
echo "=========================================="

pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# Install HuggingFace stack
echo ""
echo "=========================================="
echo "Step 3: Installing HuggingFace ecosystem"
echo "=========================================="

echo "Installing transformers..."
pip install transformers==4.57.1 -q

echo "Installing accelerate..."
pip install accelerate==1.12.0 -q

echo "Installing PEFT..."
pip install peft==0.18.0 -q

echo "Installing TRL..."
pip install trl==0.25.1 -q

echo "Installing bitsandbytes..."
pip install bitsandbytes==0.45.0 -q

echo "Installing datasets..."
pip install datasets==3.2.0 -q

echo "Installing utilities..."
pip install pyyaml==6.0.2 -q

echo "✅ All packages installed"

# Verify installation
echo ""
echo "=========================================="
echo "Step 4: Verifying installation"
echo "=========================================="

python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
import accelerate
import datasets

print('✅ All imports successful!')
print(f'')
print(f'Package versions:')
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {AutoTokenizer.__version__}')
print(f'  Accelerate: {accelerate.__version__}')
print(f'  Datasets: {datasets.__version__}')
print(f'')
print(f'CUDA Configuration:')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA Version: {torch.version.cuda}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Save requirements
echo ""
echo "Saving package list to requirements_dpo_lambda.txt..."
pip freeze > requirements_dpo_lambda.txt

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source dpo_venv/bin/activate"
echo "  2. Login to HuggingFace: huggingface-cli login"
echo "  3. Start training: ./train_all_personas.sh"
echo ""
echo "Environment details saved to:"
echo "  - requirements_dpo_lambda.txt"
echo ""
