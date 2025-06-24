import sys
import cv2
import numpy as np
import os
import warnings
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QLineEdit, QMessageBox, QTabWidget, QProgressBar,
                           QFrame, QSplitter, QTableWidget, QTableWidgetItem,
                           QInputDialog, QAction, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon
from face_recognition import FaceRecognition
import time

warnings.filterwarnings('ignore', category=FutureWarning)

class StyledButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)

class StyledLineEdit(QLineEdit):
    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)

class StyledLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)

class AdminWindow(QMainWindow):
    def __init__(self, face_recognition, parent=None):
        super().__init__(parent)
        self.face_recognition = face_recognition
        self.setWindowTitle("Администрирование базы данных")
        self.setGeometry(150, 150, 800, 600)
        
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Создаем таблицу для отображения базы данных
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Имя", "Количество эмбеддингов", "Действия"])
        layout.addWidget(self.table)
        self.refresh_data()
        
    def refresh_data(self):
        self.table.setRowCount(0)
        for name, embeddings in self.face_recognition.face_database.items():
            row = self.table.rowCount()
            self.table.insertRow(row)

            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)
            
            count_item = QTableWidgetItem(str(len(embeddings)))
            count_item.setFlags(count_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, count_item)
            
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            
            edit_btn = StyledButton("Изменить")
            edit_btn.clicked.connect(lambda checked, n=name: self.edit_person(n))
            actions_layout.addWidget(edit_btn)
            delete_btn = StyledButton("Удалить")
            delete_btn.clicked.connect(lambda checked, n=name: self.delete_person(n))
            actions_layout.addWidget(delete_btn)
            self.table.setCellWidget(row, 2, actions_widget)
            
        self.table.resizeColumnsToContents()
        for row in range(self.table.rowCount()):
            self.table.setRowHeight(row, 40)
        
    def add_person(self):
        name, ok = QInputDialog.getText(self, "Добавить личность", "Введите имя:")
        if ok and name:
            if name in self.face_recognition.face_database:
                QMessageBox.warning(self, "Ошибка", "Личность с таким именем уже существует")
                return
            self.face_recognition.face_database[name] = []
            self.face_recognition.save_database()
            self.refresh_data()
            
    def edit_person(self, name):
        new_name, ok = QInputDialog.getText(self, "Изменить имя", 
                                          "Введите новое имя:", text=name)
        if ok and new_name and new_name != name:
            if new_name in self.face_recognition.face_database:
                QMessageBox.warning(self, "Ошибка", "Личность с таким именем уже существует")
                return
                
            embeddings = self.face_recognition.face_database.pop(name)
            self.face_recognition.face_database[new_name] = embeddings
            self.face_recognition.save_database()
            self.refresh_data()
            
    def delete_person(self, name):
        reply = QMessageBox.question(self, "Подтверждение", 
                                   f"Вы уверены, что хотите удалить личность {name}?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.face_recognition.face_database[name]
            self.face_recognition.save_database()
            self.refresh_data()

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система идентификации лиц")
        self.setGeometry(100, 100, 1280, 800)
        icon = QIcon("camera_icon.png")
        self.setWindowIcon(icon)
        self.create_menu()
        self.load_styles()
        self.check_and_download_models()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_webcam)
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video)
        self.current_image = None
        self.webcam = None
        self.video_capture = None
        self.collecting_faces = False
        self.collected_frames = 0
        self.target_frames = 30
        self.is_paused = False
        self.current_frame = None

        self.face_recognition = FaceRecognition()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(20)
        title = StyledLabel("Система идентификации лиц")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.layout.addWidget(self.tabs)
        
        self.create_image_tab()
        self.create_webcam_tab()
        self.create_video_tab()
        self.create_add_person_tab()
        
    def create_menu(self):
        menubar = self.menuBar()
        admin_menu = menubar.addMenu("Администрирование")
        manage_db_action = QAction("Управление базой данных", self)
        manage_db_action.triggered.connect(self.show_admin_window)
        admin_menu.addAction(manage_db_action)
        
    def show_admin_window(self):
        admin_window = AdminWindow(self.face_recognition, self)
        admin_window.show()
        
    def load_styles(self):
        try:
            with open('styles.qss', 'r') as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print(f"Ошибка при загрузке стилей: {str(e)}")
        
    def check_and_download_models(self):
        try:
            if not os.path.exists('buffalo_l'):
                pass
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модели: {str(e)}")
            sys.exit(1)
        
    def on_tab_changed(self, index):
        self.stop_all_processes()
        if hasattr(self, 'result_label'):
            self.result_label.clear()
        if hasattr(self, 'result_image_label'):
            self.result_image_label.clear()
        if hasattr(self, 'result_image'):
            del self.result_image
        if hasattr(self, 'collect_label'):
            self.collect_label.clear()
            # Удаляем старый layout если он есть
            if self.collect_label.layout():
                QWidget().setLayout(self.collect_label.layout())
        
    def stop_all_processes(self):
        try:
            if hasattr(self, 'timer') and self.timer.isActive():
                self.timer.stop()
            if hasattr(self, 'webcam') and self.webcam is not None:
                self.webcam.release()
                self.webcam = None
            if hasattr(self, 'start_webcam_btn'):
                self.start_webcam_btn.setText("Запустить веб-камеру")
            if hasattr(self, 'video_capture') and self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
            if hasattr(self, 'video_timer') and self.video_timer.isActive():
                self.video_timer.stop()
            if hasattr(self, 'start_video_btn'):
                self.video_status_label.setText("")
                self.start_video_btn.setText("Запустить видео")
            self.collecting_faces = False
            self.is_paused = False
            self.current_frame = None
            
        except Exception as e:
            print(f"Ошибка при остановке процессов: {str(e)}")
        
    def create_image_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        info_text = StyledLabel("Загрузите изображение для идентификации личности.\n"
                              "Система автоматически обнаружит лица и попытается их идентифицировать.")
        info_text.setAlignment(Qt.AlignCenter)
        info_text.setObjectName("info_text")
        layout.addWidget(info_text)
        
        self.load_image_btn = StyledButton("Загрузить изображение")
        self.load_image_btn.clicked.connect(self.process_image)
        layout.addWidget(self.load_image_btn)
        
        # область отображения
        self.image_label = StyledLabel()
        self.image_label.setObjectName("image_display")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(400)
        layout.addWidget(self.image_label)
        
        self.status_label = StyledLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Идентификация по изображению")
        
    def create_add_person_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # верхний контейнер для информационного текста и элементов управления
        top_container = QWidget()
        top_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        top_layout = QVBoxLayout(top_container)
        top_layout.setSpacing(10)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # информационный текст
        info_text = StyledLabel("Введите имя и выберите способ добавления личности в базу данных:\n"
                              "Система будет выполнять автоматическое обнаружение лиц и пытаться их идентифицировать")
        info_text.setAlignment(Qt.AlignCenter)
        info_text.setObjectName("info_text")
        info_text.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        top_layout.addWidget(info_text)
        input_buttons_layout = QHBoxLayout()
        input_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.name_input = StyledLineEdit("Введите имя")
        self.name_input.setFixedHeight(40)
        input_buttons_layout.addWidget(self.name_input)

        self.image_add_btn = StyledButton("Загрузить изображение")
        self.image_add_btn.setFixedHeight(40)
        self.image_add_btn.clicked.connect(self.add_person_image)
        input_buttons_layout.addWidget(self.image_add_btn)
                
        self.video_add_btn = StyledButton("Загрузить видео")
        self.video_add_btn.setFixedHeight(40)
        self.video_add_btn.clicked.connect(self.add_person_video)
        input_buttons_layout.addWidget(self.video_add_btn)
        
        self.webcam_add_btn = StyledButton("Запустить веб-камеру")
        self.webcam_add_btn.setFixedHeight(40)
        self.webcam_add_btn.clicked.connect(self.add_person_webcam)
        input_buttons_layout.addWidget(self.webcam_add_btn)
        
        top_layout.addLayout(input_buttons_layout)
        
        # Прогресс-бар для сбора кадров
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setVisible(False)
        top_layout.addWidget(self.progress_bar)

        layout.addWidget(top_container)
        display_container = QWidget()
        display_layout = QVBoxLayout(display_container)
        display_layout.setContentsMargins(0, 0, 0, 0)

        self.collect_label = StyledLabel()
        self.collect_label.setObjectName("image_display")
        self.collect_label.setAlignment(Qt.AlignCenter)
        self.collect_label.setMinimumHeight(500)
        display_layout.addWidget(self.collect_label)
        layout.addWidget(display_container)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Добавить личность")
        
    def create_webcam_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        info_text = StyledLabel("Запустите веб-камеру для идентификации личности в реальном времени.\n"
                              "Система будет автоматически обнаруживать и идентифицировать лица.")
        info_text.setAlignment(Qt.AlignCenter)
        info_text.setObjectName("info_text")
        layout.addWidget(info_text)

        self.start_webcam_btn = StyledButton("Запустить веб-камеру")
        self.start_webcam_btn.clicked.connect(self.toggle_webcam)
        layout.addWidget(self.start_webcam_btn)

        self.webcam_label = StyledLabel()
        self.webcam_label.setObjectName("image_display")
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setMinimumHeight(400)
        layout.addWidget(self.webcam_label)

        self.webcam_status_label = StyledLabel("")
        self.webcam_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.webcam_status_label)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Идентификация через веб-камеру")
        
    def create_video_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        info_text = StyledLabel("Загрузите видеофайл для идентификации личности.\n"
                              "Система будет воспроизводить видео и автоматически идентифицировать обнаруженные лица.")
        info_text.setAlignment(Qt.AlignCenter)
        info_text.setObjectName("info_text")
        layout.addWidget(info_text)

        self.load_video_btn = StyledButton("Загрузить видео")
        self.load_video_btn.clicked.connect(self.load_video)
        layout.addWidget(self.load_video_btn)

        controls_layout = QHBoxLayout()
        
        self.play_video_btn = StyledButton("Воспроизвести")
        self.play_video_btn.clicked.connect(self.toggle_video)
        self.play_video_btn.setVisible(False)
        controls_layout.addWidget(self.play_video_btn)
        
        self.close_video_btn = StyledButton("Закрыть")
        self.close_video_btn.clicked.connect(self.close_video)
        self.close_video_btn.setVisible(False)
        controls_layout.addWidget(self.close_video_btn)
        
        layout.addLayout(controls_layout)

        self.video_label = StyledLabel()
        self.video_label.setObjectName("image_display")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(400)
        layout.addWidget(self.video_label)

        self.video_status_label = StyledLabel("")
        self.video_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_status_label)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Идентификация по видео")
        
    def process_image(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", 
                                                     "Image Files (*.png *.jpg *.jpeg)")
            if not file_name:
                return
                
            image = cv2.imread(file_name)
            if image is None:
                QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
                return
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            recognized_faces_data = self.face_recognition.recognize_face(rgb_image)
            
            if not recognized_faces_data:
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(q_image)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.status_label.setText("Лица не обнаружены")
                return
            
            identified_faces = [(name, i) for i, (name, bbox) in enumerate(recognized_faces_data, 1) if name != "Unknown"]

            if not identified_faces:
                self.status_label.setText("Лица не идентифицированы")
            else:
                status_text = "Идентифицированы как: " + ", ".join([f"{name}({num})" for name, num in identified_faces])
                self.status_label.setText(status_text)
            for i, (name, bbox) in enumerate(recognized_faces_data, 1):
                bbox = bbox.astype(int)
                
                offset = 40
                x1 = max(0, bbox[0] - offset)
                y1 = max(0, bbox[1] - offset)
                x2 = min(image.shape[1], bbox[2] + offset)
                y2 = min(image.shape[0], bbox[3] + offset)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                text = str(i)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

                overlay = image.copy()
                cv2.rectangle(overlay, 
                            (x1, y1 - text_height - 10),
                            (x1 + text_width, y1),
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

                cv2.putText(image, text,
                          (x1, y1 - 10),
                          font, font_scale, (255, 255, 255), thickness)

            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
        except Exception as e:
            print(f"Ошибка при обработке изображения: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при обработке изображения: {str(e)}")
        
    def toggle_webcam(self):
        if self.timer.isActive():
            self.timer.stop()
            self.webcam.release()
            self.webcam = None
            self.start_webcam_btn.setText("Запустить веб-камеру")
        else:
            self.webcam = cv2.VideoCapture(0)
            if self.webcam.isOpened():
                self.timer.start(30)
                self.start_webcam_btn.setText("Остановить веб-камеру")
                
    def update_webcam(self):
        if self.webcam is not None:
            ret, frame = self.webcam.read()
            if ret:
                original_frame = frame.copy()
                display_frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                recognized_faces_data = self.face_recognition.recognize_face(rgb_frame)
                
                if recognized_faces_data:
                    all_unknown = True
                    for name, bbox in recognized_faces_data:
                        if name != "Unknown":
                            all_unknown = False
                            break

                    if all_unknown:
                        self.webcam_status_label.setText("Лицо не идентифицировано")
                    else:
                        identified_names = [name for name, bbox in recognized_faces_data if name != "Unknown"]
                        self.webcam_status_label.setText("Идентифицирован как: " + ", ".join(identified_names))

                    for name, bbox in recognized_faces_data:
                        bbox = bbox.astype(int)
                        offset = 40
                        x1 = max(0, bbox[0] - offset)
                        y1 = max(0, bbox[1] - offset)
                        x2 = min(frame.shape[1], bbox[2] + offset)
                        y2 = min(frame.shape[0], bbox[3] + offset)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = f"{name}"
                        cv2.putText(display_frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    self.webcam_status_label.setText("Лица не обнаружены")

                height, width, channel = display_frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(self.webcam_label.size(), 
                                            Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation)
                self.webcam_label.setPixmap(scaled_pixmap)
            else:
                self.webcam.release()
                self.webcam = None
                self.webcam_status_label.setText("Веб-камера не открыта")
                
    def collect_embeddings_from_video(self, video_path=None, name=None, target_frames=40):
        if not name:
            QMessageBox.warning(self, "Ошибка", "Введите имя")
            return False
            
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            QMessageBox.warning(self, "Ошибка", "Не удалось открыть видео")
            return False
            
        collected_frames = 0
        frames_list = []
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(target_frames)
            
            while collected_frames < target_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.face_recognition.face_analyzer.get(frame_rgb)
                
                if len(faces) > 0:
                    face = faces[0]
                    bbox = face.bbox.astype(int)
                    offset = 40
                    x1 = max(0, bbox[0] - offset)
                    y1 = max(0, bbox[1] - offset)
                    x2 = min(frame.shape[1], bbox[2] + offset)
                    y2 = min(frame.shape[0], bbox[3] + offset)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    frames_list.append(frame_rgb)
                    collected_frames += 1
                    self.progress_bar.setValue(collected_frames)
                    QApplication.processEvents()
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(self.collect_label.size(), 
                                            Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation)
                self.collect_label.setPixmap(scaled_pixmap)
                QApplication.processEvents()
            
            if len(frames_list) > 0:
                self.progress_bar.setValue(0)
                self.progress_bar.setMaximum(4)
                
                if len(frames_list) < 10:
                    num_embeddings = 1
                    frames_per_embedding = len(frames_list)
                else:
                    num_embeddings = min(4, len(frames_list) // 10)
                    frames_per_embedding = 10
                
                processed_count = 0
                for i in range(num_embeddings):
                    start_idx = i * frames_per_embedding
                    end_idx = min((i + 1) * frames_per_embedding, len(frames_list))
                    current_frames = frames_list[start_idx:end_idx]
                    if self.face_recognition.add_averaged_face(current_frames, name):
                        processed_count += 1
                    self.progress_bar.setValue(processed_count)
                    QApplication.processEvents()
                
                if processed_count > 0:
                    num_embeddings = len(self.face_recognition.face_database[name])
                    QMessageBox.information(self, "Успех", 
                                          f"Лицо {name} успешно добавлено в базу данных\n"
                                          f"Создано {num_embeddings} усредненных эмбеддингов")
                    return True
                else:
                    QMessageBox.warning(self, "Ошибка", 
                                      "Не удалось обработать собранные кадры")
                    return False
            else:
                QMessageBox.warning(self, "Ошибка", 
                                  "Не удалось собрать ни одного кадра с лицом")
                return False
                
        finally:
            cap.release()
            self.progress_bar.setVisible(False)
            self.collect_label.clear()
        
    def show_face_selection_dialog(self, image, faces, name):
        display_image = image.copy()
        for i, face in enumerate(faces, 1):
            bbox = face.bbox.astype(int)
            offset = 40
            x1 = max(0, bbox[0] - offset)
            y1 = max(0, bbox[1] - offset)
            x2 = min(image.shape[1], bbox[2] + offset)
            y2 = min(image.shape[0], bbox[3] + offset)
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            text = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            overlay = display_image.copy()
            cv2.rectangle(overlay, 
                        (x1, y1 - text_height - 10),
                        (x1 + text_width, y1),
                        (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_image, 0.4, 0, display_image)
            cv2.putText(display_image, text,
                      (x1, y1 - 10),
                      font, font_scale, (255, 255, 255), thickness)
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.collect_label.size(), 
                                    Qt.KeepAspectRatio,
                                    Qt.SmoothTransformation)
        self.collect_label.setPixmap(scaled_pixmap)
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Выбор лица")
        dialog.setText(f"На изображении обнаружено {len(faces)} лиц.\n"
                      f"Выберите номер лица для добавления в базу данных:")
        buttons = []
        for i in range(len(faces)):
            btn = dialog.addButton(f"Лицо {i+1}", QMessageBox.ActionRole)
            buttons.append(btn)
        cancel_btn = dialog.addButton("Отмена", QMessageBox.RejectRole)
        dialog.exec_()
        
        clicked_button = dialog.clickedButton()
        if clicked_button == cancel_btn:
            return None
        
        return buttons.index(clicked_button)

    def show_image_gallery(self, images, faces_list):
        gallery_widget = QWidget()
        gallery_layout = QVBoxLayout(gallery_widget)
        gallery_layout.setContentsMargins(0, 0, 0, 0)
        image_container = QWidget()
        image_container_layout = QHBoxLayout(image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setObjectName("image_display")
        image_label.setMinimumHeight(500)

        prev_btn = StyledButton("←")
        prev_btn.setFixedWidth(50)
        prev_btn.setFixedHeight(40)
        
        next_btn = StyledButton("→")
        next_btn.setFixedWidth(50)
        next_btn.setFixedHeight(40)

        image_container_layout.addWidget(prev_btn)
        image_container_layout.addWidget(image_label)
        image_container_layout.addWidget(next_btn)
        gallery_layout.addWidget(image_container)
        current_index = 0
        
        def update_image():
            nonlocal current_index
            image = images[current_index]
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(image_label.size(), 
                                        Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            
            prev_btn.setEnabled(current_index > 0)
            next_btn.setEnabled(current_index < len(images) - 1)

        def prev_image():
            nonlocal current_index
            if current_index > 0:
                current_index -= 1
                update_image()
        
        def next_image():
            nonlocal current_index
            if current_index < len(images) - 1:
                current_index += 1
                update_image()
        
        prev_btn.clicked.connect(prev_image)
        next_btn.clicked.connect(next_image)
        update_image()
        self.collect_label.clear()
        self.collect_label.setLayout(gallery_layout)
        
    def add_person_image(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Ошибка", "Введите имя")
            return
            
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Добавления личности по изображению")
        dialog.setText("Выберите способ добавления личности в базу данных:\n\n"
                      "• Одно фото - добавление по одному изображению\n"
                      "• Несколько фото - добавление по нескольким изображениям")
        
        single_btn = dialog.addButton("Одно фото", QMessageBox.ActionRole)
        multiple_btn = dialog.addButton("Несколько фото", QMessageBox.ActionRole)
        cancel_btn = dialog.addButton("Отмена", QMessageBox.RejectRole)
        dialog.exec_()
        clicked_button = dialog.clickedButton()
        
        if clicked_button == cancel_btn:
            return
        elif clicked_button == single_btn:
            file_name, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", 
                                                     "Image Files (*.png *.jpg *.jpeg)")
            if not file_name:
                return
                
            image = cv2.imread(file_name)
            if image is None:
                QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
                return
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.collect_label.setText("Обработка изображения")
            QApplication.processEvents()
            
            faces = self.face_recognition.face_analyzer.get(image_rgb)
            
            if len(faces) == 0:
                QMessageBox.warning(self, "Ошибка", "Лица не обнаружены на изображении")
                self.collect_label.clear()
                return
            elif len(faces) > 1:
                selected_face_index = self.show_face_selection_dialog(image, faces, name)
                if selected_face_index is None:
                    return
                face = faces[selected_face_index]
            else:
                face = faces[0]
            bbox = face.bbox.astype(int)
            offset = 40
            x1 = max(0, bbox[0] - offset)
            y1 = max(0, bbox[1] - offset)
            x2 = min(image.shape[1], bbox[2] + offset)
            y2 = min(image.shape[0], bbox[3] + offset)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.collect_label.size(), 
                                        Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation)
            self.collect_label.setPixmap(scaled_pixmap)
            if self.face_recognition.add_face(image_rgb, name):
                QMessageBox.information(self, "Успех", f"Лицо {name} успешно добавлено в базу данных")
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось добавить лицо в базу данных")
        else:
            file_names, _ = QFileDialog.getOpenFileNames(self, "Выберите изображения", "", 
                                                       "Image Files (*.png *.jpg *.jpeg)")
            if not file_names:
                return
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(len(file_names))
            added_count = 0
            processed_images = []
            faces_list = []
            for i, file_name in enumerate(file_names):
                image = cv2.imread(file_name)
                if image is None:
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = self.face_recognition.face_analyzer.get(image_rgb)
                
                if len(faces) == 0:
                    print(f"Лица не обнаружены на изображении {file_name}, пропускаем.")
                    self.progress_bar.setValue(i + 1)
                    QApplication.processEvents()
                    continue
                elif len(faces) > 1:
                    selected_face_index = self.show_face_selection_dialog(image, faces, name)
                    if selected_face_index is None:
                        continue
                    face = faces[selected_face_index]
                else:
                    face = faces[0]
                
                bbox = face.bbox.astype(int)
                offset = 40
                x1 = max(0, bbox[0] - offset)
                y1 = max(0, bbox[1] - offset)
                x2 = min(image.shape[1], bbox[2] + offset)
                y2 = min(image.shape[0], bbox[3] + offset)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                processed_images.append(image)
                faces_list.append([face])
                if self.face_recognition.add_face(image_rgb, name):
                    added_count += 1
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()
            
            self.progress_bar.setVisible(False)
            if processed_images:
                self.show_image_gallery(processed_images, faces_list)
            
            if added_count > 0:
                QMessageBox.information(self, "Успех", 
                                      f"Успешно добавлено {added_count} из {len(file_names)} фотографий для {name}")
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось добавить ни одной фотографии")
            
    def add_person_webcam(self):
        try:
            name = self.name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "Ошибка", "Введите имя")
                return
            if self.webcam is None or not self.webcam.isOpened():
                QMessageBox.warning(self, "Ошибка", "Веб-камера не активна")
                return
            ret, frame = self.webcam.read()
            if not ret:
                QMessageBox.warning(self, "Ошибка", "Не удалось получить кадр с веб-камеры")
                return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = self.face_recognition.get_single_face_for_addition(frame_rgb)

            if face is None:
                QMessageBox.warning(self, "Ошибка", "Лицо не обнаружено или обнаружено несколько лиц")
                return
            
            display_frame = frame.copy()
            bbox = face.bbox.astype(int)
            offset = 40
            x1 = max(0, bbox[0] - offset)
            y1 = max(0, bbox[1] - offset)
            x2 = min(frame.shape[1], bbox[2] + offset)
            y2 = min(frame.shape[0], bbox[3] + offset)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            height, width, channel = display_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.collect_label.size(), 
                                        Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation)
            self.collect_label.setPixmap(scaled_pixmap)
            
            if self.face_recognition.add_face(frame_rgb, name):
                QMessageBox.information(self, "Успех", f"Лицо {name} успешно добавлено в базу данных")
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось добавить лицо в базу данных")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")
        
    def add_person_video(self):
        try:
            name = self.name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "Ошибка", "Введите имя")
                return
            
            file_name, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", 
                                                     "Video Files (*.mp4 *.avi *.mov)")
            if not file_name:
                return
            
            if self.collect_embeddings_from_video(video_path=file_name, name=name, target_frames=40):
                QMessageBox.information(self, "Успех", f"Лицо {name} успешно добавлено в базу данных")
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось добавить лицо в базу данных")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")
        
    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", 
                                                 "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_capture = cv2.VideoCapture(file_name)
            if self.video_capture.isOpened():
                self.play_video_btn.setVisible(True)
                self.close_video_btn.setVisible(True)
                self.load_video_btn.setVisible(False)
                
                self.video_status_label.setText("Видео загружено")
                self.video_status_label.setObjectName("status_success")
            else:
                self.video_status_label.setText("Ошибка загрузки видео")
                self.video_status_label.setObjectName("status_error")
                
    def toggle_video(self):
        if self.video_timer.isActive():
            self.video_timer.stop()
            self.play_video_btn.setText("Воспроизвести")
            self.is_paused = True
        else:
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            interval = int(1000 / fps)
            self.video_timer.start(interval)
            self.play_video_btn.setText("Пауза")
            self.is_paused = False
            
    def pause_video(self):
        if self.video_capture is not None:
            if not self.is_paused:
                self.current_frame = self.video_capture.read()[1]
                self.video_timer.stop()
                self.is_paused = True
                self.pause_video_btn.setText("Продолжить")
                self.play_video_btn.setEnabled(False)
            else:
                if self.current_frame is not None:
                    frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                    self.display_frame(frame_rgb)
                fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                interval = int(1000 / fps)
                self.video_timer.start(interval)
                self.is_paused = False
                self.pause_video_btn.setText("Пауза")
                self.play_video_btn.setEnabled(True)
                
    def display_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
            
    def close_video(self):
        if self.video_capture is not None:
            self.video_timer.stop()
            self.video_capture.release()
            self.video_capture = None
            self.current_frame = None
            self.is_paused = False
            self.video_label.clear()
            self.video_status_label.clear()
            
            self.play_video_btn.setVisible(False)
            self.close_video_btn.setVisible(False)
            self.load_video_btn.setVisible(True)
            
    def update_video(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                original_frame = frame.copy()
                rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                faces = self.face_recognition.face_analyzer.get(rgb_frame)
                
                if len(faces) > 0:
                    recognition_results = []
                    for i, face in enumerate(faces, 1):
                        bbox = face.bbox.astype(int)
                        offset = 40
                        x1 = max(0, bbox[0] - offset)
                        y1 = max(0, bbox[1] - offset)
                        x2 = min(original_frame.shape[1], bbox[2] + offset)
                        y2 = min(original_frame.shape[0], bbox[3] + offset)
                        cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if hasattr(face, 'embedding') and face.embedding is not None:
                            embedding = face.embedding
                            if len(embedding) > 0:
                                embedding = embedding / np.linalg.norm(embedding)
                                min_dist = float('inf')
                                recognized_name = "Unknown"
                                recognition_threshold = 0.8
                                
                                for name, stored_embeddings in self.face_recognition.face_database.items():
                                    for stored_embedding in stored_embeddings:
                                        stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
                                        dist = 1 - np.dot(embedding, stored_embedding)
                                        if dist < min_dist:
                                            min_dist = dist
                                            recognized_name = name
                                
                                if min_dist > recognition_threshold:
                                    recognized_name = "Unknown"
                                recognition_results.append((i, recognized_name))
                                text = str(i)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 1.2
                                thickness = 2
                                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                                overlay = original_frame.copy()
                                cv2.rectangle(overlay, 
                                            (x1, y1 - text_height - 10),
                                            (x1 + text_width, y1),
                                            (0, 0, 0), -1)
                                cv2.addWeighted(overlay, 0.6, original_frame, 0.4, 0, original_frame)
                                cv2.putText(original_frame, text,
                                          (x1, y1 - 10),
                                          font, font_scale, (255, 255, 255), thickness)
                                
                    identified_faces = [(name, num) for num, name in recognition_results if name != "Unknown"]

                    if not identified_faces:
                        self.video_status_label.setText("Лица не идентифицированы")
                        if hasattr(self, 'last_recognition_results'):
                            print("\nЛица не идентифицированы")
                            self.last_recognition_results = None
                    else:
                        status_text = "Идентифицированы как: " + ", ".join([f"{name}({num})" for name, num in identified_faces])
                        self.video_status_label.setText(status_text)
                        
                        if not hasattr(self, 'last_recognition_results') or self.last_recognition_results != recognition_results:
                            print(f"\nОбнаружено лиц: {len(faces)}")
                            for num, name in recognition_results:
                                print(f"{num} - {name}")
                            self.last_recognition_results = recognition_results
                else:
                    self.video_status_label.setText("Лица не обнаружены")
                    if hasattr(self, 'last_recognition_results'):
                        print("\nЛица не найдены")
                        self.last_recognition_results = None
                
                height, width, channel = original_frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(original_frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
                
                scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                    self.video_label.width(),
                    self.video_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                
                self.video_label.setPixmap(scaled_pixmap)
            else:
                self.video_timer.stop()
                self.video_capture.release()
                self.video_capture = None
                self.play_video_btn.setVisible(False)
                self.close_video_btn.setVisible(False)
                self.load_video_btn.setVisible(True)
                self.video_status_label.setText("Видео закончилось")
                if hasattr(self, 'last_recognition_results'):
                    del self.last_recognition_results

    def closeEvent(self, event):
        self.stop_all_processes()
        if hasattr(self, 'face_recognition'):
            self.face_recognition.save_database()
        event.accept()

    def switch_tab(self, tab_name):
        self.stop_all_processes()
        for tab in self.tabs.values():
            tab.pack_forget()
        self.tabs[tab_name].pack(fill=Qt.BOTH, expand=True)
        self.active_tab = tab_name
        self.result_label.config(text="")
        self.result_image_label.config(image='')
        if hasattr(self, 'result_image'):
            del self.result_image

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_()) 