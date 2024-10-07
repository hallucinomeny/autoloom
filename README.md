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

### Disclaimer

THE CODE IS A MESS. 
DO NOT EXPECT IT TO WORK.
ATTEMPT AT YOUR OWN RISK.

If it does happen to work consider yourself lucky. 
If not, please open an issue so I at the very least we can learn from the experience.

### Requirements

- Dependencies listed in `setup.sh`


The project requires weights of a pre-trained language model with hugging face name `MODEL_NAME` (default is "distilgpt2") which it downloads automatically to a directory specified by the `MODEL_CACHE_DIR` variable (default is "path/to/your/cache/directory"). Both variables can be set in the `main.py` file.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

An older unstable version of the project had an extra ui feature which plotted the logits for each token.

That version is unfortunately lost, but if you feel like adding this feature back in, please do! 

:bing_smiley:
 