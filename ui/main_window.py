import queue
import math
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                             QLabel, QLineEdit, QPushButton, QFrame, QFormLayout,
                             QSplitter, QTableWidgetItem, QTreeWidgetItem)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThreadPool, pyqtSlot
from PyQt6.QtGui import QDoubleValidator
from .styles import get_styles
from .tree_widget import MCTSTreeWidget, SortableTreeWidgetItem, LeafTableWidget

class MCTSUI(QMainWindow):
    update_ui_signal = pyqtSignal(list, int)
    start_mcts_signal = pyqtSignal(str, float, float, float)
    pause_mcts_signal = pyqtSignal()
    reset_mcts_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCTS Path Probabilities")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet(get_styles())

        self.thread_pool = QThreadPool()
        self.last_update_time = 0
        self.update_interval = 0.1

        self.setup_ui()

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.process_update_queue)
        self.update_timer.start(100)

        self.update_queue = queue.Queue()
        self.paths_data = {}

        self.update_ui_signal.connect(self.update_ui_slot)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        central_widget.setLayout(main_layout)

        self.setup_input_area(main_layout)
        self.setup_tree_and_leaf_widgets(main_layout)
        self.setup_info_view(main_layout)

    def setup_input_area(self, main_layout):
        input_frame = QFrame()
        input_frame.setObjectName("inputFrame")
        input_layout = QVBoxLayout(input_frame)

        prompt_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your initial prompt here")
        prompt_layout.addWidget(self.prompt_input)

        params_layout = QFormLayout()
        self.temperature_input = QLineEdit("1.0")
        self.temperature_input.setValidator(QDoubleValidator(0.0, 100.0, 2))
        self.min_prob_input = QLineEdit("1e-6")
        self.min_prob_input.setValidator(QDoubleValidator(0.0, 1.0, 10))
        self.entropy_factor_input = QLineEdit("3.0")
        self.entropy_factor_input.setValidator(QDoubleValidator(0.0, 100.0, 2))

        params_layout.addRow("Temperature:", self.temperature_input)
        params_layout.addRow("Min Probability:", self.min_prob_input)
        params_layout.addRow("Entropy Factor:", self.entropy_factor_input)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start MCTS")
        self.start_button.clicked.connect(self.start_mcts)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_mcts)
        self.pause_button.setEnabled(False)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_mcts)
        self.reset_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.reset_button)

        input_layout.addLayout(prompt_layout)
        input_layout.addLayout(params_layout)
        input_layout.addLayout(button_layout)
        main_layout.addWidget(input_frame)

    def setup_tree_and_leaf_widgets(self, main_layout):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Tree widget (left side)
        self.tree_widget = MCTSTreeWidget()
        self.tree_widget.setHeaderLabels(["Path", "Probability (%)", "Entropy", "Depth", "Child Count", "Sum of Raw Probs (%)"])
        self.tree_widget.itemSelectionChanged.connect(self.on_tree_item_selected)
        splitter.addWidget(self.tree_widget)
        
        # Leaf table widget (right side)
        self.leaf_table = LeafTableWidget()
        self.leaf_table.setColumnCount(2)
        self.leaf_table.setHorizontalHeaderLabels(["Token", "Conditional Probability (%)"])
        splitter.addWidget(self.leaf_table)
        
        splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
        
        main_layout.addWidget(splitter)

    def setup_info_view(self, main_layout):
        self.info_layout = QHBoxLayout()
        self.total_nodes_label = QLabel("Total nodes: 0")
        self.iteration_label = QLabel("Iteration: 0")
        self.memory_usage_label = QLabel("Memory usage: 0%")
        self.info_layout.addWidget(self.total_nodes_label)
        self.info_layout.addWidget(self.iteration_label)
        self.info_layout.addWidget(self.memory_usage_label)
        main_layout.addLayout(self.info_layout)

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor)
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.reset_button.setEnabled(True)

    def pause_mcts(self):
        self.pause_mcts_signal.emit()
        self.start_button.setText("Resume MCTS")
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setText("Start MCTS")
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.leaf_table.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, paths, total_nodes):
        print(f"MainWindow update_ui called: {len(paths)} paths, {total_nodes} total nodes")
        self.update_ui_signal.emit(paths, total_nodes)

    @pyqtSlot(list, int)
    def update_ui_slot(self, paths, total_nodes):
        try:
            self.update_tree_widget(paths)
            self.total_nodes_label.setText(f"Total nodes: {total_nodes}")
        except Exception as e:
            print(f"Error in update_ui_slot: {str(e)}")
            self.show_error_message(f"Error updating UI: {str(e)}")

    def update_tree_widget(self, paths):
        current_items = set()
        for path_data in paths:
            if len(path_data) >= 7:
                path, prob, entropy, depth, raw_sum, child_count, child_data = path_data
                
                decoded_path = self.decode_path(path)
                item = self.find_or_create_item(decoded_path)
                item.setData(0, Qt.ItemDataRole.UserRole, path)
                item.setText(1, f"{prob * 100:.4f}%")
                item.setText(2, f"{entropy:.4f}")
                item.setText(3, str(depth))
                item.setText(4, str(child_count))
                item.setText(5, f"{raw_sum * 100:.4f}%")  # Display raw_sum as percentage

                self.paths_data[decoded_path] = path_data
                current_items.add(decoded_path)
            else:
                print(f"Unexpected path_data format: {path_data}")

        self.remove_stale_items(current_items)
        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)

    def decode_path(self, path):
        if isinstance(path, (list, np.ndarray)):
            return self.language_model.tokenizer.decode(path)
        elif isinstance(path, tuple) and len(path) > 0:
            if isinstance(path[0], (list, np.ndarray)):
                return self.language_model.tokenizer.decode(path[0])
        return str(path)

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return SortableTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_items):
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_items:
                self.tree_widget.takeTopLevelItem(i)

    def set_language_model(self, language_model):
        self.language_model = language_model

    def on_tree_item_selected(self):
        selected_items = self.tree_widget.selectedItems()
        if selected_items:
            item = selected_items[0]
            path = item.text(0)
            if path in self.paths_data:
                path_data = self.paths_data[path]
                if len(path_data) == 7:
                    _, _, _, _, _, _, child_data = path_data
                else:
                    child_data = []
                self.update_leaf_table(child_data)
            else:
                self.leaf_table.clear()
        else:
            self.leaf_table.clear()

    def update_leaf_table(self, child_data):
        self.leaf_table.clear()
        self.leaf_table.setRowCount(len(child_data))
        for row, child in enumerate(child_data):
            if isinstance(child, tuple) and len(child) >= 2:
                token_id, prob = child[:2]
                try:
                    token = self.language_model.tokenizer.decode([int(token_id)])
                except (OverflowError, ValueError):
                    token = f"<Invalid Token ID: {token_id}>"
                self.leaf_table.setItem(row, 0, QTableWidgetItem(token))
                self.leaf_table.setItem(row, 1, QTableWidgetItem(f"{prob * 100:.4f}%"))
        self.leaf_table.sortItems(1, Qt.SortOrder.DescendingOrder)

    def show_error_message(self, message):
        print(f"Error: {message}")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
        except queue.Empty:
            return

        for path_data in paths:
            path, prob, entropy, depth, raw_sum, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(5, f"{raw_sum:.4f}")

class SortableTreeWidgetItem(QTreeWidgetItem):
    def __lt__(self, other):
        column = self.treeWidget().sortColumn()
        try:
            return float(self.text(column).rstrip('%')) < float(other.text(column).rstrip('%'))
        except ValueError:
            return self.text(column) < other.text(column)