from datetime import datetime
import time
import random

def format_time(timestamp):
    """ Форматирует временную метку с точностью до миллисекунд. """
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def simulate_image_capture():
    """Симулируем задержку получения изображения от камеры."""
    a = random.uniform(0.05, 0.2)
    print(f"Захват изображения занял {a} мс")
    time.sleep(a)  # случайная задержка от 50 мс до 200 мс

def process_image():
    """Симулируем обработку изображения для определения направления взгляда."""
    a = random.uniform(0.1, 0.5)
    print(f"Выдача информации о взгляде заняла {a} мс")
    time.sleep(a)  # случайная задержка обработки от 100 мс до 800 мс
def process_emotion_image():
    """Симулируем обработку изображения для определения направления взгляда."""
    a = random.uniform(0.1, 0.4)
    print(f"Выдача информации о эмоции заняла {a} мс")
    time.sleep(a)  # случайная задержка обработки от 100 мс до 800 мс

def test_response_time():
    start_time = time.time()
    print(f"Начальное время: {format_time(start_time)}")
    simulate_image_capture()  # Получаем изображение
    #process_image()  # Обрабатываем изображение
    process_emotion_image()
    end_time = time.time()
    response_time = end_time - start_time
    
    print(f"Обработка изображения завершена за {response_time:.3f} секунд.")
    print(f"Конечное время: {format_time(end_time)}")
    # Проверяем, удовлетворяет ли время отклика требованию
    if response_time <= 1.0:
        print("Тест пройден: Время отклика удовлетворяет требованию.")
    else:
        print("Тест не пройден: Время отклика превышает 1 секунду.")
if __name__ == '__main__':
    test_response_time()
