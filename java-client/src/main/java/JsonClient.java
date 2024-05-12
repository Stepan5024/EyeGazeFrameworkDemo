import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Base64;

class ImageFrame extends JFrame {
    private JLabel imageLabel;

    public ImageFrame() {
        super("Image Viewer");
        this.imageLabel = new JLabel();
        this.add(imageLabel);
        this.setSize(640, 480);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setVisible(true);
    }

    public void updateImage(BufferedImage image) {
        ImageIcon imageIcon = new ImageIcon(image);
        imageLabel.setIcon(imageIcon);
        this.validate();
        this.repaint();
    }
}

public class JsonClient {
    private static final String SERVER_IP = "127.0.0.1"; // IP-адрес сервера
    private static final int SERVER_PORT = 9556; // Порт сервера
    private static Socket socket;

    public static void main(String[] args) {
        JsonClient client = new JsonClient();
        client.runClient();


    }
    private void closeSocket() {
        try {
            if (socket != null && !socket.isClosed()) {
                socket.close();
                System.out.println("Сокет закрыт успешно");
            }
        } catch (IOException ex) {
            System.out.println("Ошибка при закрытии сокета: " + ex.getMessage());
            ex.printStackTrace();
        }
    }
    public void runClient() {
        ImageFrame frame = new ImageFrame();

        frame.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                closeSocket();
            }
        });

        while (true) { // Бесконечный цикл для попыток подключения
            try {
                if (socket == null || socket.isClosed()) {
                    socket = new Socket(SERVER_IP, SERVER_PORT);
                }

                InputStream inputStream = socket.getInputStream();
                BufferedInputStream bufferedInputStream = new BufferedInputStream(inputStream);

                byte[] lengthBytes = new byte[4];
                while (bufferedInputStream. read(lengthBytes, 0, 4) != -1) {
                    int length = byteArrayToInt(lengthBytes);
                    if (length > 0) {
                        byte[] messageBytes = new byte[length];
                        bufferedInputStream.read(messageBytes, 0, length);
                        String messageStr = new String(messageBytes);

                        JsonElement jsonElement = JsonParser.parseString(messageStr);
                        if (jsonElement.isJsonObject()) {
                            JsonObject jsonObject = jsonElement.getAsJsonObject();
                            String imageData = jsonObject.get("image").getAsString().split(",")[1];
                            String coordinates = String.valueOf(jsonObject.get("coordinates"));
                            System.out.println(coordinates);
                            BufferedImage image = decodeToImage(imageData);
                            frame.updateImage(image);
                        }
                    }
                }
            } catch (IOException e) {
                System.out.println("Ошибка при подключении к серверу: " + e.getMessage());
                e.printStackTrace();

                closeSocket();

                // Переподключение после 5 секунд
                try {
                    Thread.sleep(5000); // Ожидание 5 секунд перед следующей попыткой
                } catch (InterruptedException ie) {
                    System.out.println("Прерывание во время ожидания: " + ie.getMessage());
                }
            }
        }
    }

    private static BufferedImage decodeToImage(String imageString) {
        try {
            byte[] imageByte = Base64.getDecoder().decode(imageString);
            ByteArrayInputStream bis = new ByteArrayInputStream(imageByte);
            BufferedImage image = ImageIO.read(bis);
            bis.close();
            return image;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static int byteArrayToInt(byte[] b) {
        return b[0] << 24 | (b[1] & 0xFF) << 16 | (b[2] & 0xFF) << 8 | (b[3] & 0xFF);
    }

    private static void printJson(JsonObject jsonObject) {
        // Печать всех ключей и значений
        jsonObject.entrySet().forEach(entry -> System.out.println(entry.getKey() + ": " + entry.getValue()));
    }
}
