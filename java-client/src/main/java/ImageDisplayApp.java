import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Base64;

import java.util.logging.Level;
import java.util.logging.Logger;

// Jackson imports


public class ImageDisplayApp extends JFrame {
    private JLabel imageLabel;
    private JLabel coordinatesLabel;
    private JButton reconnectButton;
    private static final int PORT = 9556;
    private static final Logger LOGGER = Logger.getLogger(ImageDisplayApp.class.getName());
    private ServerSocket serverSocket;
    private Socket clientSocket;

    public ImageDisplayApp() {
        super("Image Display");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(600, 400);
        setLayout(new BorderLayout());
        imageLabel = new JLabel();
        coordinatesLabel = new JLabel("Coordinates: ", JLabel.CENTER);
        reconnectButton = new JButton("Reconnect");
        reconnectButton.addActionListener(e -> reconnect());
        add(imageLabel, BorderLayout.CENTER);
        add(coordinatesLabel, BorderLayout.SOUTH);
        add(reconnectButton, BorderLayout.NORTH);
        setVisible(true);
        initServer();
    }

    private void displayImage(BufferedImage image, int x, int y) {
        ImageIcon icon = new ImageIcon(image);
        imageLabel.setIcon(icon);
        coordinatesLabel.setText("Coordinates: X=" + x + ", Y=" + y);
        pack();
    }

    private void initServer() {
        try {
            serverSocket = new ServerSocket(PORT);
            acceptClient();
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Failed to open server socket on port " + PORT, e);
        }
    }

    private void acceptClient() {
        try {
            if (serverSocket != null && !serverSocket.isClosed()) {
                clientSocket = serverSocket.accept();
                listenForMessages();
            }
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Failed to accept client connection", e);
        }
    }

    private void listenForMessages() {
        try (DataInputStream dis = new DataInputStream(clientSocket.getInputStream())) {
            ObjectMapper mapper = new ObjectMapper();
            while (!clientSocket.isClosed()) {
                int length = dis.readInt();
                if (length > 0) {
                    byte[] message = new byte[length];
                    dis.readFully(message, 0, message.length);
                    JsonNode rootNode = mapper.readTree(message);
                    String base64Image = rootNode.path("image").asText().split(",")[1];
                    int x = rootNode.path("coordinates").get(0).asInt();
                    int y = rootNode.path("coordinates").get(1).asInt();

                    BufferedImage image = ImageIO.read(new ByteArrayInputStream(Base64.getDecoder().decode(base64Image)));
                    SwingUtilities.invokeLater(() -> displayImage(image, x, y));
                }
            }
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Error while reading from client", e);
        }
    }

    private void reconnect() {
        try {
            if (clientSocket != null) {
                clientSocket.close();
            }
            acceptClient();
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Failed to reconnect", e);
        }
    }

    public static void main(String[] args) {
        new ImageDisplayApp();
    }
}
