import unittest
import json
import base64
import cv2
import numpy as np
import os

class ServerTest(unittest.TestCase):
    def setUp(self):
        # Путь к тестовому изображению
        image_path = r'.\EyeGazeFrameworkDemo\resources\test\image01.jpg'
        with open(image_path, 'rb') as image_file:
            img_data = image_file.read()
        
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # Подготовка тестового JSON с данными
        self.data = json.dumps({
            "status": "2",
            "image": f"data:image/jpeg;base64,{img_base64}",
            "coordinates": [100, 200],
            "datetime": "28.04.2024 15:45:30.123",
            "emotions": ["3", "0"]
        })

    def test_decode_data(self):
        # Декодирование JSON
        results = json.loads(self.data)

        # Проверка статуса
        self.assertEqual(results['status'], '2', "Status should be '2'")

        # Проверка декодирования изображения
        image_data = results['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Проверка, что изображение было успешно преобразовано
        self.assertIsNotNone(img, "Decoded image should not be None")
        self.assertTrue(img.size > 0, "Decoded image should have size greater than zero")

        # Проверка координат
        x_value = int(results['coordinates'][0])
        y_value = int(results['coordinates'][1])
        self.assertEqual(x_value, 100, "X coordinate should be 100")
        self.assertEqual(y_value, 200, "Y coordinate should be 200")

        # Проверка эмоций
        emotions = results['emotions']
        self.assertIn('3', emotions, "Emotions should contain key 3 = 'happy'")
        self.assertIn('0', emotions, "Emotions should contain key 0 = 'Angry'")

if __name__ == '__main__':
    unittest.main()
