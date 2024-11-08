## Installation

To install and run follow these steps:

1. Clone the repository:
   ```
   https://github.com/hallucinomeny/autoloom.git
   cd autoloom
   ```

2. Run the setup script:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```
   This script will:
   - Create a virtual environment
   - Activate the virtual environment
   - Upgrade pip
   - Install GPU-enabled JAX
   - Install other required packages (transformers, rich, flax, PyQt6, psutil)

3. Activate the virtual environment:
   ```
   source .venv/bin/activate
   ```

4. Verify JAX is using the GPU:
   ```
   python -c 'import jax; print(jax.devices())'
   ```
   If you see GPU devices listed, you're good to go. If not, you may need to set up CUDA manually.

5. Run the program:
   ```
   python main.py
   ```

   
### Requirements

This project has the following requirements:
- Python dependencies: All listed in and installed by `setup.sh`
- Pre-trained Language Model and Cache Directory:
  - The project uses a pre-trained language model from Hugging Face
  - Default model: "distilgpt2"
  - Model can be changed by setting the `MODEL_NAME` variable in `main.py`
  - Model weights are automatically downloaded to a cache directory
  - Default cache directory: "path/to/your/cache/directory"
  - Both model and cache location can be changed by setting the `MODEL_NAME` and `MODEL_CACHE_DIR` variables in `main.py`
  - 
## License

This project is licensed under the MIT License or whatever.

## Contributing

An older unstable version of the project had an extra ui feature which plotted the logits for each token.

That version is not with us anymore, but if you feel like adding this feature back in, please do! 

:bing_smiley:
 
