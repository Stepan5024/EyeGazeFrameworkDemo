from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import Spinner
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2

class VideoStreamModule(BoxLayout):
    def __init__(self, **kwargs):
        super(VideoStreamModule, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.video_stream = cv2.VideoCapture(0)  # Используйте 0 для локальной камеры
        self.image_rgb = Image()
        self.add_widget(self.image_rgb)

        # Запускаем обновление кадров
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)  # Обновление с частотой 30 FPS

    def update_frame(self, dt):
        ret, frame = self.video_stream.read()
        if ret:
            # Преобразуем цвета из BGR в RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = cv2.flip(rgb_frame, -1)
            # Создаём текстуру из кадра
            texture = self.create_texture(rgb_frame)
            self.image_rgb.texture = texture

    def create_texture(self, frame):
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        return texture

    def on_stop(self):
        self.video_stream.release()

class MainApp(App):
    def build(self):
        self.root = BoxLayout(orientation='vertical')
        self.spinner = Spinner(
            text='Выберите модуль',
            values=('Модуль 1 - Модуль чтения видеопотока',
                    'Модуль 2 - поиск портрета человека и выделения информативных признаков',
                    'Модуль 3 - сегментации радужки глаз, склеры, зрачка',
                    'Модуль 4 - определения наклона головы в пространстве', 
                    'Вектор взгляда на модели pytorch CNN'),
            size_hint=(None, None),
            size=(800, 44),
            pos_hint={'center_x': 0.5}
        )
        self.root.add_widget(self.spinner)
        self.start_button = Button(
            text='Запустить',
            size_hint=(None, None),
            size=(200, 44),
            pos_hint={'center_x': 0.5}
        )
        self.start_button.bind(on_press=self.open_module)
        self.root.add_widget(self.start_button)
        
        return self.root
    
    def open_module(self, instance):
        if 'Модуль 1' in self.spinner.text:
            module = VideoStreamModule()
            title = 'Видеопоток'
        else:
            return

        popup_content = BoxLayout(orientation='vertical')
        close_button = Button(text='X', size_hint=(None, None), size=(50, 50), pos_hint={'right': 1})
        popup_content.add_widget(close_button)
        popup_content.add_widget(module)

        popup = Popup(title=title, content=popup_content, size_hint=(None, None), size=(800, 600), auto_dismiss=False)
        close_button.bind(on_press=lambda x: self.close_popup(popup, module))
        popup.open()

    def close_popup(self, popup, module):
        module.on_stop()  # Ensure video stream is released
        popup.dismiss()

if __name__ == '__main__':
    MainApp().run()
