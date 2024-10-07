import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtCore import QThread, pyqtSignal
from ui import MCTSUI
from mcts import MCTSWorker
from model import LanguageModel
import time
import functools
import queue
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "distilgpt2"
MODEL_CACHE_DIR = "path/to/your/cache/directory"


# Decorator to measure and print the execution time of functions
def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Thread to handle UI updates asynchronously
class UIUpdateThread(QThread):
    update_signal = pyqtSignal(list, int)

    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.running = True

    def run(self):
        while self.running:
            try:
                paths, total_nodes = self.ui.update_queue.get(timeout=0.1)
                self.update_signal.emit(paths, total_nodes)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in UIUpdateThread: {e}")

# Main application window class
class MainWindow(MCTSUI):
    def __init__(self):
        super().__init__()
        try:
            logger.info("Initializing MainWindow...")
            
            logger.info("Initializing language model...")

            self.language_model = LanguageModel.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
            
            logger.info("Creating MCTS worker...")
            self.worker = MCTSWorker(self.language_model, num_workers=16)
            
            logger.info("Setting language model in UI...")
            self.set_language_model(self.language_model)
            
            logger.info("Connecting signals...")
            self.connect_signals()

            logger.info("Initializing UI update thread...")
            self.init_ui_update_thread()

            logger.info("MainWindow initialization complete.")

        except Exception as e:
            logger.error(f"Error initializing MainWindow: {e}", exc_info=True)
            raise

    def connect_signals(self):
        """Connect various signals to their respective slots"""
        self.worker.finished.connect(self.reset_ui)
        self.worker.update_ui_signal.connect(self.update_ui)
        self.start_mcts_signal.connect(self.start_mcts_worker)
        self.reset_mcts_signal.connect(self.stop_mcts)
        self.pause_mcts_signal.connect(self.pause_mcts_worker)
        self.worker.performance_signal.connect(self.log_performance)

    def init_ui_update_thread(self):
        """Initialize and start the UI update thread"""
        self.ui_update_thread = UIUpdateThread(self)
        self.ui_update_thread.update_signal.connect(self.update_ui)
        self.ui_update_thread.start()

    @timing_decorator
    def start_mcts_worker(self, prompt, temperature, min_prob, entropy_factor):
        """Start the MCTS worker with given parameters"""
        prompt_token_ids = self.language_model.tokenizer.encode(prompt)
        # Removed setting eps in the language model
        self.worker.set_params(prompt_token_ids, temperature, min_prob, entropy_factor)
        self.worker.start()

    def log_performance(self, performance_data):
        """Log performance data from the MCTS worker"""
        print(f"Performance: {performance_data}")

    def stop_mcts(self):
        """Stop the MCTS process"""
        print("Stopping MCTS")
        self.worker.running = False
        self.worker.wait()
        self.reset_ui()

    def pause_mcts_worker(self):
        """Pause or resume the MCTS worker"""
        if self.worker.paused:
            self.worker.resume()
        else:
            self.worker.pause()

    def start_mcts(self):
        """Start the MCTS process"""
        print("Starting MCTS")
        super().start_mcts()

    def reset_mcts(self):
        """Reset the MCTS process"""
        print("Resetting MCTS")
        super().reset_mcts()

    def update_ui(self, paths, total_nodes):
        """Update the UI with new MCTS data"""
        print(f"MainWindow update_ui called: {len(paths)} paths, {total_nodes} total nodes")
        super().update_ui(paths, total_nodes)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        logger.info("Starting application...")
        main_window = MainWindow()
        main_window.show()
        logger.info("Entering main event loop...")
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        sys.exit(1)