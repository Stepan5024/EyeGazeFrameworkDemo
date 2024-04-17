# Используем официальный образ Python с предустановленным Debian
FROM python:3.10

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv

# Установка библиотек Python
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt

# Копирование файлов проекта
COPY . /app

# Открытие порта 9556 для внешних подключений
EXPOSE 9556

# Команда для запуска сервера
CMD ["python", "server.py"]
