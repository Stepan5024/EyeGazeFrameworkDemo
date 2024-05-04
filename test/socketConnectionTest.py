import unittest
import socket

class TestSocketConnection(unittest.TestCase):
    def setUp(self):
        # Параметры сервера
        self.host = '127.0.0.1'
        self.port = 9556
        self.server_address = (self.host, self.port)
        
        # Настройка сокета сервера
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(self.server_address)
        self.server_socket.listen(1)

        # Запуск отдельного потока для имитации сервера
        import threading
        threading.Thread(target=self.server_listener, daemon=True).start()

    def tearDown(self):
        self.server_socket.close()

    def server_listener(self):
        connection, client_address = self.server_socket.accept()
        with connection:
            data = connection.recv(1024)
            connection.sendall(data)  # Эхо ответа

    def test_socket_connection(self):
        # Подключение к серверу
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect(self.server_address)
            # Отправка данных
            message = 'Hello, Server!'
            client_socket.sendall(message.encode())
            # Получение ответа
            response = client_socket.recv(1024).decode()
            # Проверка ответа
            self.assertEqual(message, response, "The response should be an echo of the sent message.")

if __name__ == '__main__':
    unittest.main()
