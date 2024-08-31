#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored output
print_green() {
    echo -e "${GREEN}$1${NC}"
}

print_yellow() {
    echo -e "${YELLOW}$1${NC}"
}



# Create a virtual environment
print_green "Creating virtual environment..."
python3 -m venv venv

# Activate the virtual environment
print_green "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_green "Upgrading pip..."
pip install --upgrade pip

# Install GPU-enabled JAX
print_green "Installing GPU-enabled JAX..."
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# Install other requirements
print_green "Installing other requirements..."
pip install transformers rich flax

# Deactivate the virtual environment
deactivate

print_green "Setup complete!"
print_yellow "To activate the virtual environment, run:"
echo "source venv/bin/activate"
print_yellow "Then, to verify JAX is using the GPU, run:"
echo "python -c 'import jax; print(jax.devices())'"
print_yellow "If you see GPU devices listed, you're good to go. If not, you may need to set up CUDA manually."
print_yellow "To run the program:"
echo "python main.py"