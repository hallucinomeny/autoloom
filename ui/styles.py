def get_styles():
    return """
        QMainWindow, QWidget {
            background-color: #2b2b2b;
            color: #e0e0e0;
        }
        QFrame#inputFrame {
            background-color: #333333;
            border: 1px solid #555555;
            border-radius: 5px;
        }
        QTreeWidget {
            background-color: #2b2b2b;
            alternate-background-color: #333333;
            color: #e0e0e0;
            border: 1px solid #555555;
        }
        QTreeWidget::item:selected {
            background-color: #4a4a4a;
        }
        QTreeWidget::item:hover {
            background-color: #3a3a3a;
        }
        QPushButton {
            background-color: #0d47a1;
            color: white;
            border: 1px solid #1565c0;
            padding: 5px 10px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #1565c0;
            border: 1px solid #1976d2;
        }
        QPushButton:pressed {
            background-color: #0d47a1;
            border: 1px solid #0d47a1;
        }
        QPushButton:disabled {
            background-color: #555555;
            color: #888888;
            border: 1px solid #666666;
        }
        QLineEdit {
            background-color: #3a3a3a;
            color: #e0e0e0;
            border: 1px solid #555555;
            border-radius: 3px;
            padding: 2px;
        }
        QLabel {
            color: #e0e0e0;
        }
        QHeaderView::section {
            background-color: #333333;
            color: #e0e0e0;
            border: 1px solid #555555;
        }
        QTableWidget {
            background-color: #2b2b2b;
            alternate-background-color: #333333;
            color: #e0e0e0;
            border: 1px solid #555555;
        }
        QTableWidget::item:selected {
            background-color: #4a4a4a;
        }
        QTableWidget::item:hover {
            background-color: #3a3a3a;
        }
        QHeaderView::section {
            background-color: #333333;
            color: #e0e0e0;
            border: 1px solid #555555;
            padding: 5px;
        }
    """