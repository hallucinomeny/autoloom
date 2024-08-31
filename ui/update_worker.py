from PyQt6.QtCore import QRunnable, pyqtSlot

class UpdateWorker(QRunnable):
    def __init__(self, ui, paths, total_nodes):
        super().__init__()
        self.ui = ui
        self.paths = paths
        self.total_nodes = total_nodes

    @pyqtSlot()
    def run(self):
        self.ui.update_ui(self.paths, self.total_nodes)