import socket
import json

def main():
    host = 'localhost'  # The server's hostname or IP address
    port = 9556        # The port used by the server (adjust accordingly)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        while True:
            # Receive the length of the message first
            message_length = s.recv(4)
            if not message_length:
                print("Server has closed the connection.")
                break
            
            # Convert the length to an integer
            message_length = int.from_bytes(message_length, 'big')
            
            # Receive the full message based on the length
            message = b''
            while len(message) < message_length:
                part = s.recv(message_length - len(message))
                if not part:
                    raise Exception("Connection closed prematurely")
                message += part
            
            # Decode the message
            data = json.loads(message.decode('utf-8'))
            
            # Extract and print the coordinates
            x, y = data['coordinates']
            print(f'Coordinates: x={x}, y={y}')

if __name__ == '__main__':
    main()
