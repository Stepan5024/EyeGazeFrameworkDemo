import h5py
import numpy as np

def merge_h5_files(file_path1, file_path2, output_file_path):
    
    with h5py.File(file_path1, 'a') as h5file1, h5py.File(file_path2, 'r') as h5file2:
        # Перебираем все датасеты во втором файле
        for dataset_name in h5file2.keys():
            # Если датасет уже существует в первом файле, объединяем данные
            if dataset_name in h5file1:
                # Читаем данные из обоих файлов
                data1 = h5file1[dataset_name][:]
                data2 = h5file2[dataset_name][:]

                # Объединяем данные
                combined_data = np.concatenate((data1, data2), axis=0)

                # Удаляем старый датасет в первом файле
                del h5file1[dataset_name]

                # Создаем новый датасет с объединенными данными
                h5file1.create_dataset(dataset_name, data=combined_data)
            else:
                # Если датасета нет в первом файле, копируем его целиком
                h5file2.copy(dataset_name, h5file1)

    print(f"Data from {file_path2} has been successfully merged into {file_path1}")

# Пути к вашим файлам
file_path1 = r'F:\EyeGazeDataset\gaze_user\data.h5'
file_path2 = r'F:\EyeGazeDataset\MPIIFaceGaze_post_proccessed_stepa_pperle\data_only_pperle.h5'
output_file_path = r'F:\EyeGazeDataset\gaze_user\output_data_merge.h5'  # В данном случае первый файл будет являться выходным файлом

# Вызываем функцию объединения
merge_h5_files(file_path1, file_path2, output_file_path)
