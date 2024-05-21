#from pytorch_lightning import seed_everything# seed_everything(42)
#from pytorch_lightning import Trainer
#from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.callbacks import Callback
import pickle
#import pytorch_lightning
import torch.nn as nn
import torch.optim as optim
from server.models import GazeModel 
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os
from torchviz import make_dot

from train.mpii_face_gaze_dataset import get_dataloaders
from train.utils import calc_angle_error, PitchYaw, plot_prediction_vs_ground_truth, log_figure, get_random_idx, get_each_of_one_grid_idx

"""class VersioningCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Добавляем версию PyTorch Lightning, если отсутствует
        if 'pytorch-lightning_version' not in checkpoint:
            checkpoint['pytorch-lightning_version'] = pl.__version__"""



class Model(GazeModel):
    def __init__(self, learning_rate: float = 0.1, weight_decay: float = 0., 
                 k=None, adjust_slope: bool = False, grid_calibration_samples: bool = False, 
                 *args, **kwargs):
        '''
        weight_decay: Величина, контролирующая регуляризацию весов модели для предотвращения 
        переобучения путем добавления штрафа на большие веса. Значение по умолчанию — 0. 
        (без регуляризации).
        
        '''
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.k = [9, 128] if k is None else k
        self.adjust_slope = adjust_slope
        self.grid_calibration_samples = grid_calibration_samples

        self.save_hyperparameters()  # log hyperparameters
    
    def configure_optimizers(self):
        '''
        предназначен для конфигурации и возврата оптимизатора, 
        который будет использоваться для обучения модели. 
        В данном случае для оптимизации используется алгоритм Adam с 
        заданными параметрами скорости обучения (lr=self.learning_rate) 
        и весом распада (weight_decay=self.weight_decay).
        '''
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, 
                                weight_decay=self.weight_decay)

    def __step(self, batch: dict) -> tuple:
        """
        Этот код определяет метод __step, который предназначен для выполнения одного 
        шага обучения или валидации модели во время итерации по данным. Метод принимает 
        на вход пакет данных (batch), извлекает необходимые элементы из этого пакета, производит 
        предсказание с помощью модели и вычисляет потери (loss) с использованием среднеквадратичной 
        ошибки (MSE) между предсказанными и истинными значениями углов взгляда (gaze_pitch и gaze_yaw).
        Возвращаемые значения — это потери, истинные метки и предсказанные значения.
        """

        """
        Operates on a single batch of data.

        :param batch: The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
        :return: calculated loss, given values and predicted outputs
        """
        person_idx = batch['person_idx'].long()
        left_eye_image = batch['left_eye_image'].float()
        right_eye_image = batch['right_eye_image'].float()
        full_face_image = batch['full_face_image'].float()

        gaze_pitch = batch['gaze_pitch'].float()
        gaze_yaw = batch['gaze_yaw'].float()
        labels = torch.stack([gaze_pitch, gaze_yaw]).T

        outputs = self(person_idx, full_face_image, right_eye_image, left_eye_image)  # prediction on the base model
        loss = F.mse_loss(outputs, labels)

        return loss, labels, outputs

    def training_step(self, train_batch: dict, batch_idx: int):
        """
        определяют логику одного шага обучения и валидации соответственно.
        training_step: Метод принимает батч данных для обучения (train_batch) и 
        индекс батча (batch_idx). С помощью внутреннего метода __step он вычисляет потери (loss), 
        истинные метки (labels) и предсказания модели (outputs) на основе входного батча. 
        Затем, с помощью метода self.log, он записывает значение потерь и ошибку угла (angular error) 
        в логи обучения. Возвращает значение потерь.
        """
        loss, labels, outputs = self.__step(train_batch)


        self.log('train/loss', loss)
        self.log('train/angular_error', calc_angle_error(labels, outputs))

        return loss

    def validation_step(self, valid_batch: dict, batch_idx: int):
        """
        validation_step: По аналогии с методом обучения, этот метод обрабатывает 
        батч валидационных данных (valid_batch). Вычисляет потери, истинные метки и 
        предсказания модели. Результаты также логируются. Возвращает словарь с результатами 
        валидации, включая потери, метки, предсказания, информацию о местоположении взгляда и 
        размерах экрана из валидационного батча.
        """
        loss, labels, outputs = self.__step(valid_batch)
                                            
        self.log('valid/offset(k=0)/loss', loss)
        self.log('valid/offset(k=0)/angular_error', calc_angle_error(labels, outputs))
        
        # We append the loss as a scalar (0-dim tensor) if it's not already
        self.validation_outputs.append({'loss': loss.squeeze()})
        
        """if not hasattr(self, "validation_outputs"):
            self.validation_outputs = []
            self.validation_outputs.append(outputs)"""
        return {'loss': loss, 'labels': labels, 'outputs': outputs, 'gaze_locations': valid_batch['gaze_location'], 'screen_sizes': valid_batch['screen_size']}


    def test_step(self, test_batch: dict, batch_idx: int):
        loss, labels, outputs = self.calculate_loss_and_labels(test_batch)
        self.log('test/loss', loss)
        self.log('test/angular_error', calc_angle_error(labels, outputs))
        return {'loss': loss, 'labels': labels, 'outputs': outputs}
       
    def __log_and_plot_details(self, outputs, tag: str):

        """
        Этот код выполняет ряд операций для логирования и анализа результатов 
        тестирования или валидации модели машинного обучения, в частности, для задач, 
        связанных с определением направления взгляда. 
        Давайте подробно разберем, что делает каждая часть кода:
Сбор данных из выходов модели: Код агрегирует данные (labels, outputs, gaze_locations, 
screen_sizes) из всех переданных выходных данных (outputs), собранных за время эпохи 
тестирования или валидации.

Визуализация и логирование: С помощью функции plot_prediction_vs_ground_truth 
строятся графики, сравнивающие предсказанные значения с реальными (labels и outputs) 
для углов наклона и поворота головы. Эти графики логируются в системе логирования 
(logger) с соответствующими тегами.
Калибровка предсказаний: Для последних last_x предсказаний выполняется калибровка 
с использованием двух подходов — либо корректировка среднего смещения (mean offset), 
либо корректировка наклона и пересечения (adjust_slope с использованием линейной 
регрессии np.polyfit). Это делается для улучшения точности предсказаний модели 
на основе набора калибровочных данных.

Логирование результатов калибровки: Вычисляется и логируется средняя и 
стандартное отклонение угловой ошибки (angular_error) после калибровки для 
различных значений k (количество точек для калибровки). Это позволяет оценить 
эффективность калибровки и выбрать оптимальное количество точек.

Лучший случай калибровки: Также вычисляется угловая ошибка для "лучшего случая" 
калибровки, когда используются все доступные данные, кроме последних last_x, и 
результаты также логируются и визуализируются.
        """

        test_labels = torch.cat([output['labels'] for output in outputs])
        test_outputs = torch.cat([output['outputs'] for output in outputs])
        test_gaze_locations = torch.cat([output['gaze_locations'] for output in outputs])
        test_screen_sizes = torch.cat([output['screen_sizes'] for output in outputs])

        figure = plot_prediction_vs_ground_truth(test_labels, test_outputs, PitchYaw.PITCH)
        log_figure(self.logger, f'{tag}/offset(k=0)/pitch', figure, self.global_step)

        figure = plot_prediction_vs_ground_truth(test_labels, test_outputs, PitchYaw.YAW)
        log_figure(self.logger, f'{tag}/offset(k=0)/yaw', figure, self.global_step)

        # find calibration params
        last_x = 500
        calibration_train = test_outputs[:-last_x].cpu().detach().numpy()
        calibration_test = test_outputs[-last_x:].cpu().detach().numpy()

        calibration_train_labels = test_labels[:-last_x].cpu().detach().numpy()
        calibration_test_labels = test_labels[-last_x:].cpu().detach().numpy()

        gaze_locations_train = test_gaze_locations[:-last_x].cpu().detach().numpy()
        screen_sizes_train = test_screen_sizes[:-last_x].cpu().detach().numpy()

        if len(calibration_train) > 0:
            for k in self.k:
                if k <= 0:
                    continue
                calibrated_solutions = []

                num_calibration_runs = 500 if self.grid_calibration_samples else 10_000  # original results are both evaluated with 10,000 runs
                for calibration_run_idx in range(num_calibration_runs):  # get_each_of_one_grid_idx is slower than get_random_idx
                    np.random.seed(42 + calibration_run_idx)
                    calibration_sample_idxs = get_each_of_one_grid_idx(k, gaze_locations_train, screen_sizes_train) if self.grid_calibration_samples else get_random_idx(k, len(calibration_train))
                    calibration_points_x = np.asarray([calibration_train[idx] for idx in calibration_sample_idxs])
                    calibration_points_y = np.asarray([calibration_train_labels[idx] for idx in calibration_sample_idxs])

                    if self.adjust_slope:
                        m, b = np.polyfit(calibration_points_y[:, :1].reshape(-1), calibration_points_x[:, :1].reshape(-1), deg=1)
                        pitch_fixed = (calibration_test[:, :1] - b) * (1 / m)
                        m, b = np.polyfit(calibration_points_y[:, 1:].reshape(-1), calibration_points_x[:, 1:].reshape(-1), deg=1)
                        yaw_fixed = (calibration_test[:, 1:] - b) * (1 / m)
                    else:
                        mean_diff_pitch = (calibration_points_y[:, :1] - calibration_points_x[:, :1]).mean()  # mean offset
                        pitch_fixed = calibration_test[:, :1] + mean_diff_pitch
                        mean_diff_yaw = (calibration_points_y[:, 1:] - calibration_points_x[:, 1:]).mean()  # mean offset
                        yaw_fixed = calibration_test[:, 1:] + mean_diff_yaw

                    pitch_fixed, yaw_fixed = torch.Tensor(pitch_fixed), torch.Tensor(yaw_fixed)
                    outputs_fixed = torch.stack([pitch_fixed, yaw_fixed], dim=1).squeeze(-1)
                    calibrated_solutions.append(calc_angle_error(torch.Tensor(calibration_test_labels), outputs_fixed).item())

                self.log(f'{tag}/offset(k={k})/mean_angular_error', np.asarray(calibrated_solutions).mean())
                self.log(f'{tag}/offset(k={k})/std_angular_error', np.asarray(calibrated_solutions).std())

        # best case, with all calibration samples, all values except the last `last_x` values
        if self.adjust_slope:
            m, b = np.polyfit(calibration_train_labels[:, :1].reshape(-1), calibration_train[:, :1].reshape(-1), deg=1)
            pitch_fixed = torch.Tensor((calibration_test[:, :1] - b) * (1 / m))
            m, b = np.polyfit(calibration_train_labels[:, 1:].reshape(-1), calibration_train[:, 1:].reshape(-1), deg=1)
            yaw_fixed = torch.Tensor((calibration_test[:, 1:] - b) * (1 / m))
        else:
            mean_diff_pitch = (calibration_train_labels[:, :1] - calibration_train[:, :1]).mean()  # mean offset
            pitch_fixed = calibration_test[:, :1] + mean_diff_pitch
            mean_diff_yaw = (calibration_train_labels[:, 1:] - calibration_train[:, 1:]).mean()  # mean offset
            yaw_fixed = calibration_test[:, 1:] + mean_diff_yaw

        pitch_fixed, yaw_fixed = torch.Tensor(pitch_fixed), torch.Tensor(yaw_fixed)
        outputs_fixed = torch.stack([pitch_fixed, yaw_fixed], dim=1).squeeze(-1)
        calibration_test_labels = torch.Tensor(calibration_test_labels)
        self.log(f'{tag}/offset(k=all)/angular_error', calc_angle_error(calibration_test_labels, outputs_fixed))

        figure = plot_prediction_vs_ground_truth(calibration_test_labels, outputs_fixed, PitchYaw.PITCH)
        log_figure(self.logger, f'{tag}/offset(k=all)/pitch', figure, self.global_step)

        figure = plot_prediction_vs_ground_truth(calibration_test_labels, outputs_fixed, PitchYaw.YAW)
        log_figure(self.logger, f'{tag}/offset(k=all)/yaw', figure, self.global_step)


def main(path_to_data: str, validate_on_person: int, test_on_person: int, 
         learning_rate: float, weight_decay: float, batch_size: int, k: int,
           adjust_slope: bool, grid_calibration_samples: bool):
    #seed_everything(42)

    model = Model(learning_rate, weight_decay, 
                  k, adjust_slope, grid_calibration_samples)
    # Получаем версию PyTorch Lightning
    #pl_version = pytorch_lightning.__version__
    # Создаем экземпляр нашего кастомного колбэка
    
    # Пути к директориям
    dirpath = "F:\\EyeGazeDataset\\gaze_user\\checkpoints\\"
    default_root_dir = 'F:\\EyeGazeDataset\\gaze_user\\saved_models\\'
    # Проверка на существование директории dirpath 
    # и её создание при необходимости
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print(f"Директория '{dirpath}' была создана")

    # Проверка на существование директории 
    #default_root_dir и её создание при необходимости
    if not os.path.exists(default_root_dir):
        os.makedirs(default_root_dir)
        print(f"Директория '{default_root_dir}' была создана")


    # Создаем экземпляры наших кастомных колбэков

    
    (train_dataloader, valid_dataloader, test_dataloader) = get_dataloaders(path_to_data, 
                                                                            validate_on_person, 
                                                                            test_on_person, 
                                                                            batch_size)
    epochs = 2
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)

    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        valid_loss = validate(model, valid_dataloader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

    y = model(model)
    #make_dot(y.mean(), params=dict(model.named_parameters()))
    """torch.save(model.state_dict(), 'model_weights.pth')
    # Сохраняем модель после тренировки
    model_path = "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)"""
    print(f"Complete!")


    
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, targets = batch['input'].to(device), batch['target'].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            inputs, targets = batch['input'].to(device), batch['target'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(valid_loader)

if __name__ == '__main__':

    # path_to_data = "C:\\Users\\bokar\\Documents\\train_gaze_stepa\\mpiifacegaze_preprocessed"
    path_to_data = r"F:\EyeGazeDataset\MPIIFaceGaze_post_proccessed_stepa_pperle"
    validate_on_person = 14
    test_on_person = 0
    learning_rate = 0.01 # скорость обучения
    weight_decay = 0.0 # уменьшение веса
    batch_size = 64 # размер блока полезных данных
    k = [9, 128]
    adjust_slope = False # регулировка_склона
    grid_calibration_samples = False # сетка_калибрации_сэмплов

    main(path_to_data, validate_on_person, test_on_person, 
         learning_rate, weight_decay, batch_size, 
         k, adjust_slope, grid_calibration_samples)
