import sys
import os
import time
import cv2
import yaml
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel, QVBoxLayout
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

from server.video_stream.video_stream import VideoStream


EMOTIONS = {
    'a': 'Злой',
    'd': 'Испытывающий отвращение',
    'f': 'Напуганный',
    'h': 'Счастливый',
    's': 'Огорченный',
    'u': 'Удивленный',
    'n': 'Нейтральное состояние'
}

class VideoWidget(QWidget):

    def readConfig(self, path: str):
        path_to_config = self.resource_path(path)
        self.configs: dict = self.load_config(path_to_config)

    def load_config(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def resource_path(self, relative_path) -> str:
        """Возвращает корректный путь для доступа к ресурсам после сборки .exe"""
        #if getattr(sys, 'frozen', False):
        try:
            # PyInstaller создаёт временную папку _MEIPASS для ресурсов
            base_path = sys._MEIPASS
        except Exception:
            # Если приложение запущено из исходного кода, то используется обычный путь
            base_path = os.path.abspath(".")
    
        return os.path.join(base_path, relative_path)
    def createDir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory {path} created.")
        else:
            print(f"Directory {path} already exists.")

    def __init__(self):
        super().__init__()
        self.capture = VideoStream()
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.emotion_count = {emotion: 0 for emotion in EMOTIONS.values()}
        self.initUI()
        self.readConfig(os.path.join('configs', 'emotion_ubuntu.yaml'))
        self.DIR = self.configs['generate_data_root']
            # Create directories for all emotions
        for emotion_key, emotion_name in EMOTIONS.items():
            emotion_path = os.path.join(self.DIR, emotion_name)
            self.createDir(emotion_path)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(20)

    def initUI(self):
        self.setWindowTitle("Сбор данных о эмоциональном состоянии")  # Заголовок окна
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)  # Установка флагов окна для кнопок управления окном
        self.showFullScreen()  # Открытие окна в полноэкранном режиме
        #self.setWindowState(QtCore.Qt.WindowMaximized)
        self.layout = QHBoxLayout()  # Основной горизонтальный layout для размещения видео и текстовой информации

        # Левая часть: видео
        self.image_label = QLabel(self)  # Метка для отображения видео
        self.image_label.setFixedSize(self.width() // 2, self.height())  # Размер метки подстраивается под половину ширины окна

        # Правая часть: текст и счетчик
        self.text_layout = QVBoxLayout()  # Вертикальный layout для текста и счетчика

        # Метка с инструкциями
        self.instruction_label = QLabel(self)
        instructions = ("<h1>Инструкция:</h1><br><p>Изобразите одну из эмоций, затем нажмите соответствующую клавишу</p>" +
                        "<br>".join(f"{k} - {v}" for k, v in EMOTIONS.items()) + "<br>esc, q - Выход")
        self.instruction_label.setText(instructions)
        self.instruction_label.setFixedSize(self.width() // 2, self.height() // 2)  # Размер метки
        self.instruction_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # Выравнивание текста вверх и влево

        # Метка для отображения счетчика сохраненных изображений
        self.count_label = QLabel(self)
        self.count_label.setText(self.format_emotion_count())  # Текст метки генерируется методом format_emotion_count
        self.count_label.setAlignment(Qt.AlignTop)  # Выравнивание текста вверх
        self.count_label.setWordWrap(True)  # Перенос слов в метке, если текст не помещается

        # Добавление меток в вертикальный layout
        self.text_layout.addWidget(self.instruction_label)
        self.text_layout.addWidget(self.count_label)
        self.text_layout.setSpacing(10)  # Расстояние между элементами в layout

        # Добавление виджетов в основной горизонтальный layout
        self.layout.addWidget(self.image_label)
        self.layout.addLayout(self.text_layout)

        self.setLayout(self.layout)  # Установка основного layout виджета

    
    def format_emotion_count(self):
        return "<h2>Сохраненные изображения:</h2>" + \
            "<br>".join(f"{emotion}: {count}" for emotion, count in self.emotion_count.items())

    
    def updateFrame(self):
        img = self.capture.read_frame()
        if img is None:
            return
        self.img_rgb = self.capture.convert_color(img, to_rgb=True)

        results = self.face_detector.process(self.img_rgb)
        if results.detections:
            self.is_face_on_frame = True
            for detection in results.detections:
                mp.solutions.drawing_utils.draw_detection(self.img_rgb, detection)
 
        else:
            self.is_face_on_frame = False
        height, width, channel = self.img_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(self.img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))
    

    def keyPressEvent(self, event):

        key_char = event.text()
        if self.is_face_on_frame == True and key_char in EMOTIONS and self.img_rgb is not None:
            emotion_name = EMOTIONS[key_char]
            timestamp = "{:.1f}".format(time.time())
            file_path = os.path.join(self.DIR, emotion_name, f'{timestamp}.jpg')
            self.emotion_count[emotion_name] += 1  # Увеличиваем счетчик
            self.count_label.setText(self.format_emotion_count())
            os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Создать директорию, если её нет
            cv2.imwrite(file_path, cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR))
            print(f"IMAGE SAVED => {file_path}")
        elif event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q:
            self.close()


    def closeEvent(self, event):
        self.capture.release()

def main():
    app = QApplication(sys.argv)
    ex = VideoWidget()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
