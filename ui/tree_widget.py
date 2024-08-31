from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem, QToolTip, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import Qt, QEvent
import math

class SortableTreeWidgetItem(QTreeWidgetItem):
    def __init__(self, parent, values):
        super().__init__(parent, values)
        self.values = values

    def __lt__(self, other):
        column = self.treeWidget().sortColumn()
        try:
            if column in [1, 6]:  # Log Probability and Top-k Ratio columns
                return float(self.text(column)) > float(other.text(column))  # Sort in descending order
            else:
                return float(self.text(column)) < float(other.text(column))
        except ValueError:
            return self.text(column) < other.text(column)

class MCTSTreeWidget(QTreeWidget):
    def __init__(self):
        super().__init__()
        self.setColumnCount(7)
        self.setHeaderLabels(["Path", "Log Probability", "Entropy", "Depth", "Optimal Top-k", "Child Count", "Optimal Top-k"])
        self.setSortingEnabled(True)
        self.setAlternatingRowColors(True)
        self.viewport().installEventFilter(self)

    def eventFilter(self, source, event):
        if (source is self.viewport() and
            event.type() == QEvent.Type.MouseMove):
            item = self.itemAt(event.pos())
            if item:
                column = self.columnAt(event.pos().x())
                if column == 5:  # Child Count column
                    self.show_child_distribution(item, event.globalPosition().toPoint())
                else:
                    QToolTip.hideText()
        return super().eventFilter(source, event)

    def show_child_distribution(self, item, global_pos):
        path = item.text(0)
        if hasattr(self.parent(), 'paths_data') and path in self.parent().paths_data:
            _, _, _, _, _, _, child_data = self.parent().paths_data[path]
            tooltip_text = "\n".join([f"{token}: {prob:.6f}" for token, prob, _ in child_data])
            QToolTip.showText(global_pos, tooltip_text, self)
        else:
            QToolTip.hideText()

class LeafTableWidget(QTableWidget):
    def __init__(self):
        super().__init__()
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Next Token", "Conditional Probability"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def update_data(self, child_data, tokenizer):
        self.setRowCount(len(child_data))
        for row, (token_id, prob, _) in enumerate(child_data):
            token = tokenizer.decode([token_id])
            self.setItem(row, 0, QTableWidgetItem(token))
            self.setItem(row, 1, QTableWidgetItem(f"{prob:.6f}"))
        self.sortItems(1, Qt.SortOrder.DescendingOrder)

    def clear(self):
        self.setRowCount(0)