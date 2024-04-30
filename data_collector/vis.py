import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Примерные данные, подставьте реальные данные здесь
face_3d = np.array([
    [682, 506, -0.056],
    [580, 404, 0.007],
    [618, 564, 0.009],
    [677, 638, 0.001],
    [780, 409, 0.015],
    [735, 564, 0.014]
])

center_of_eyes = np.array([0, -37.90856, 43.02182])
gaze_target_3D = np.array([121.32786899, 190.76422928, -403.19840671])

# Настройка фигуры для 3D визуализации
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Визуализация маски лица
ax.scatter(face_3d[:, 0], face_3d[:, 1], face_3d[:, 2], color='blue', label='Face Landmarks')

# Визуализация центра глаз
ax.scatter(center_of_eyes[0], center_of_eyes[1], center_of_eyes[2], color='red', label='Center of Eyes')

# Визуализация вектора взгляда
ax.quiver(center_of_eyes[0], center_of_eyes[1], center_of_eyes[2],
          gaze_target_3D[0], gaze_target_3D[1], gaze_target_3D[2], length=100, color='green', label='Gaze Direction')

# Визуализация плоскости экрана
z = 0  # Предполагается, что экран находится на уровне z=0
x = np.linspace(-1000, 1000, 2)
y = np.linspace(-1000, 1000, 2)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
ax.plot_surface(X, Y, Z, color='orange', alpha=0.5, label='Screen Plane')

# Настройка осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
