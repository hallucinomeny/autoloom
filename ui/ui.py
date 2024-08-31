import queue
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                             QLabel, QLineEdit, QPushButton, QFrame, QTreeWidget, 
                             QTreeWidgetItem, QFormLayout, QToolTip)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QEvent, QThreadPool, QRunnable, pyqtSlot
from PyQt6.QtGui import QDoubleValidator
from ui.styles import get_styles
import psutil
import os
import time


class SortableTreeWidgetItem(QTreeWidgetItem):
    def __init__(self, parent, values):
        super().__init__(parent, values)
        self.values = values

    def __lt__(self, other):
        column = self.treeWidget().sortColumn()
        try:
            return float(self.values[column]) < float(other.values[column])
        except ValueError:
            return self.values[column] < other.values[column]

class MCTSUI(QMainWindow):
    start_mcts_signal = pyqtSignal(str, float, float, float, float)
    reset_mcts_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCTS Path Probabilities")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(get_styles())

        # Add this line to initialize the thread pool
        self.thread_pool = QThreadPool()

        self.last_update_time = 0
        self.update_interval = 0.1  # Update every 0.1 seconds instead of 0.5

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        central_widget.setLayout(main_layout)

        # Input area
        input_frame = QFrame()
        input_frame.setObjectName("inputFrame")
        input_layout = QVBoxLayout(input_frame)

        # Prompt input
        prompt_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your initial prompt here")
        prompt_layout.addWidget(self.prompt_input)

        # Parameters input
        params_layout = QFormLayout()
        self.temperature_input = QLineEdit("1.0")
        self.temperature_input.setValidator(QDoubleValidator(0.0, 100.0, 2))
        self.min_prob_input = QLineEdit("1e-6")
        self.min_prob_input.setValidator(QDoubleValidator(0.0, 1.0, 10))
        self.entropy_factor_input = QLineEdit("1.5")
        self.entropy_factor_input.setValidator(QDoubleValidator(0.0, 100.0, 2))
        self.eps_input = QLineEdit("0.1")
        self.eps_input.setValidator(QDoubleValidator(0.0, 1.0, 4))

        params_layout.addRow("Temperature:", self.temperature_input)
        params_layout.addRow("Min Probability:", self.min_prob_input)
        params_layout.addRow("Entropy Factor:", self.entropy_factor_input)
        params_layout.addRow("Epsilon:", self.eps_input)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start MCTS")
        self.start_button.clicked.connect(self.start_mcts)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_mcts)
        self.reset_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.reset_button)

        input_layout.addLayout(prompt_layout)
        input_layout.addLayout(params_layout)
        input_layout.addLayout(button_layout)
        main_layout.addWidget(input_frame)

        # Tree view
        self.tree_widget = QTreeWidget()
        self.tree_widget.setColumnCount(6)
        self.tree_widget.setHeaderLabels(["Path", "Probability", "Entropy", "Depth", "Child Count", "Sum of Raw Probs"])
        self.tree_widget.setSortingEnabled(True)
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.viewport().installEventFilter(self)
        main_layout.addWidget(self.tree_widget)

        # Info view
        self.info_layout = QHBoxLayout()
        self.total_nodes_label = QLabel("Total nodes: 0")
        self.iteration_label = QLabel("Iteration: 0")
        self.memory_usage_label = QLabel("Memory usage: 0%")
        self.info_layout.addWidget(self.total_nodes_label)
        self.info_layout.addWidget(self.iteration_label)
        self.info_layout.addWidget(self.memory_usage_label)
        main_layout.addLayout(self.info_layout)

        # Add a timer for periodic updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.process_update_queue)
        self.update_timer.start(100)  # Check for updates every 100ms

        self.update_queue = queue.Queue()
        self.paths_data = {}

    def eventFilter(self, source, event):
        if (source is self.tree_widget.viewport() and
            event.type() == QEvent.Type.MouseMove):
            item = self.tree_widget.itemAt(event.pos())
            if item:
                column = self.tree_widget.columnAt(event.pos().x())
                if column == 5:  # Child Count column
                    # Use globalPosition() instead of globalPos()
                    self.show_child_distribution(item, event.globalPosition().toPoint())
                else:
                    QToolTip.hideText()
        return super().eventFilter(source, event)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
        except queue.Empty:
            return

        self.tree_widget.clear()
        self.paths_data.clear()

        for path_data in paths:
            path, prob, entropy, depth, raw_sum, child_count, child_data = path_data
            item = SortableTreeWidgetItem(self.tree_widget, [
                path,
                f"{prob:.6e}",
                f"{entropy:.4f}",
                str(depth),
                str(child_count),
                f"{raw_sum:.4f}"
            ])
            self.paths_data[path] = path_data

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")

class UpdateWorker(QRunnable):
    def __init__(self, ui, paths, total_nodes):
        super().__init__()
        self.ui = ui
        self.paths = paths
        self.total_nodes = total_nodes

    @pyqtSlot()
    def run(self):
        self.ui.update_queue.put((self.paths, self.total_nodes))
    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        print(f"MCTSUI update_ui called: {len(paths)} paths, {total_nodes} total nodes")
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            print("Skipping update due to interval")
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
            print(f"Processing update: {len(paths)} paths, {total_nodes} total nodes")
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")
        print("UI update completed")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        print(f"MCTSUI update_ui called: {len(paths)} paths, {total_nodes} total nodes")
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            print("Skipping update due to interval")
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
            print(f"Processing update: {len(paths)} paths, {total_nodes} total nodes")
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")
        print("UI update completed")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        print(f"MCTSUI update_ui called: {len(paths)} paths, {total_nodes} total nodes")
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            print("Skipping update due to interval")
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
            print(f"Processing update: {len(paths)} paths, {total_nodes} total nodes")
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")
        print("UI update completed")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        print(f"MCTSUI update_ui called: {len(paths)} paths, {total_nodes} total nodes")
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            print("Skipping update due to interval")
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
            print(f"Processing update: {len(paths)} paths, {total_nodes} total nodes")
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")
        print("UI update completed")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        print(f"MCTSUI update_ui called: {len(paths)} paths, {total_nodes} total nodes")
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            print("Skipping update due to interval")
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
            print(f"Processing update: {len(paths)} paths, {total_nodes} total nodes")
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")
        print("UI update completed")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        print(f"MCTSUI update_ui called: {len(paths)} paths, {total_nodes} total nodes")
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            print("Skipping update due to interval")
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
            print(f"Processing update: {len(paths)} paths, {total_nodes} total nodes")
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")
        print("UI update completed")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        print(f"MCTSUI update_ui called: {len(paths)} paths, {total_nodes} total nodes")
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            print("Skipping update due to interval")
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
            print(f"Processing update: {len(paths)} paths, {total_nodes} total nodes")
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")
        print("UI update completed")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        print(f"MCTSUI update_ui called: {len(paths)} paths, {total_nodes} total nodes")
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            print("Skipping update due to interval")
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
            print(f"Processing update: {len(paths)} paths, {total_nodes} total nodes")
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")
        print("UI update completed")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        print(f"MCTSUI update_ui called: {len(paths)} paths, {total_nodes} total nodes")
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            print("Skipping update due to interval")
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
            print(f"Processing update: {len(paths)} paths, {total_nodes} total nodes")
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")
        print("UI update completed")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        print(f"MCTSUI update_ui called: {len(paths)} paths, {total_nodes} total nodes")
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            print("Skipping update due to interval")
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
            print(f"Processing update: {len(paths)} paths, {total_nodes} total nodes")
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")
        print("UI update completed")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
            item = self.find_or_create_item(path)
            item.setText(1, f"{prob:.6e}")
            item.setText(2, f"{entropy:.4f}")
            item.setText(3, str(depth))
            item.setText(4, str(optimal_topk_length))
            item.setText(5, str(child_count))
            self.paths_data[path] = path_data

        # Remove items that are no longer present
        self.remove_stale_items(paths)

        self.tree_widget.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.total_nodes_label.setText(f"Total nodes: {total_nodes}")

    def find_or_create_item(self, path):
        items = self.tree_widget.findItems(path, Qt.MatchFlag.MatchExactly, 0)
        if items:
            return items[0]
        else:
            return QTreeWidgetItem(self.tree_widget, [path])

    def remove_stale_items(self, current_paths):
        current_path_set = set(path_data[0] for path_data in current_paths)
        for i in range(self.tree_widget.topLevelItemCount() - 1, -1, -1):
            item = self.tree_widget.topLevelItem(i)
            if item.text(0) not in current_path_set:
                self.tree_widget.takeTopLevelItem(i)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if path in self.paths_data:
            _, _, _, _, _, _, child_data = self.paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self.tree_widget)
        else:
            QToolTip.hideText()

    def start_mcts(self):
        prompt = self.prompt_input.text()
        temperature = float(self.temperature_input.text())
        min_prob = float(self.min_prob_input.text())
        entropy_factor = float(self.entropy_factor_input.text())
        eps = float(self.eps_input.text())
        self.start_mcts_signal.emit(prompt, temperature, min_prob, entropy_factor, eps)
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def reset_mcts(self):
        self.reset_mcts_signal.emit()
        self.reset_ui()

    def reset_ui(self):
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.tree_widget.clear()
        self.total_nodes_label.setText("Total nodes: 0")
        self.iteration_label.setText("Iteration: 0")
        self.memory_usage_label.setText("Memory usage: 0%")
        self.paths_data.clear()

    def update_ui(self, root_node, paths, total_nodes):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time
        worker = UpdateWorker(self, paths, total_nodes)
        self.thread_pool.start(worker)
        
        current_iteration = int(self.iteration_label.text().split(": ")[1]) + 1
        self.iteration_label.setText(f"Iteration: {current_iteration}")
        
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        self.memory_usage_label.setText(f"Memory usage: {memory_percent:.2f}%")

    def process_update_queue(self):
        try:
            paths, total_nodes = self.update_queue.get_nowait()
        except queue.Empty:
            return

        # Update existing items and add new ones
        for path_data in paths:
            path, prob, entropy, depth, optimal_topk_length, child_count, child_data = path_data
class UpdateWorker(QRunnable):
    def __init__(self, ui, paths, total_nodes):
        super().__init__()
        self.ui = ui
        self.paths = paths
        self.total_nodes = total_nodes

    @pyqtSlot()
    def run(self):
        self.ui.update_queue.put((self.paths, self.total_nodes))

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MCTSUI()
    window.show()
    sys.exit(app.exec())