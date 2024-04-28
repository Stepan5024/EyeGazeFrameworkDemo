import torch

# Замените 'path_to_checkpoint' на путь к вашему файлу .pth
checkpoint_path = "C:\\Users\\bokar\\Documents\\EyeGazeFrameworkDemo\\server\\models_cnn\\checkpoint_0020.pth"
#checkpoint_path = "C:\\Users\\bokar\\Documents\\EyeGazeFrameworkDemo\\server\\models_cnn\\p00.ckpt"
# Загрузка чекпоинта
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# Вывод всех ключей в чекпоинте
print("Ключи в чекпоинте:")
for key in checkpoint.keys():
    print(key)

# Проверка наличия 'model' в чекпоинте и вывод содержимого
if 'model' in checkpoint:
    print("Содержимое 'model' в чекпоинте:")
    count = 0
    for key, value in checkpoint['model'].items():
        print(f"{key}: {value}")
        count += 1
        if count == 20:  # Ограничение вывода первыми 20 строками
            break
else:
    print("'model' не найден в чекпоинте.")
# Проверка, содержит ли чекпоинт ключ 'state_dict'
if 'state_dict' in checkpoint:
    print("Чекпоинт содержит state_dict.")
else:
    print("Чекпоинт не содержит state_dict.")
