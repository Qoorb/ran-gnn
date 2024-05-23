import torch
import os
from torch.utils.data import DataLoader
from clearml import Task
from classes import PointCloudDataset
from functions import get_all_csv_files, report_clearml_3d, save_to_csv, save_model, predict
from functions import calculate_mape, calculate_smape, calculate_logcosh_loss, calculate_smape_loss
from pointnet import PointNet
from model import DGCNN

# Инициализация задачи ClearML
task = Task.init(project_name='PointNetML', task_name="DGCNN Regression --Model2")

# Параметры
parameters = {
    'root_dir': 'merged_output_taylor',
    'batch_size': 2,
    'epochs': 2,
    'learning_rate': 0.01,
    'k': 2,
    'emb_dims': 512,
    'dropout': 0.2,
    'num_points': 35746
}
task.connect(parameters)

# Определим класс Args для хранения аргументов
class Args:
    def __init__(self, k, emb_dims, dropout):
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout

args = Args(parameters['k'], parameters['emb_dims'], parameters['dropout'])

# Устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Получение списка CSV файлов
csv_files_train, csv_files_test = get_all_csv_files(parameters['root_dir'])

# Создание датасетов и DataLoader
dataset_train = PointCloudDataset(csv_files_train)
dataset_test = PointCloudDataset(csv_files_test)
train_loader = DataLoader(dataset_train, parameters['batch_size'], shuffle=True)
test_loader = DataLoader(dataset_test, parameters['batch_size'], shuffle=False)

# Инициализируем модель, оптимизатор и функцию потерь
#model = PointNet(3,256,3)
model = DGCNN(args, parameters['num_points']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'])
criterion = calculate_smape_loss

# Обучение модели
def train(model, loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_smape = 0
        total_mape = 0  # Добавляем переменную для суммы MAPE
        for points, next_points in loader:
            points, next_points = points.to(device), next_points.to(device)
            optimizer.zero_grad()
            points = points.view(points.size(0), 3, -1)
            next_points = next_points.view(next_points.size(0), 3, -1)
            out = model(points)
            loss = criterion(out, next_points)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_smape += calculate_smape(out, next_points)
            total_mape += calculate_mape(out, next_points)  # Вычисляем MAPE
            
            # Логируем метрики
            task.current_task().get_logger().report_scalar(
                "MSE Loss", "train", loss.item(), epoch)
            task.current_task().get_logger().report_scalar(
                "SMAPE", "train", calculate_smape(out, next_points), epoch)
            task.current_task().get_logger().report_scalar(
                "MAPE", "train", calculate_mape(out, next_points), epoch)

        avg_loss = total_loss / len(loader)
        avg_smape = total_smape / len(loader)
        avg_mape = total_mape / len(loader) 
        
        # Логируем средние значения по эпохе
        task.current_task().get_logger().report_scalar("MSE Loss (Epoch)", "train", avg_loss, epoch)
        task.current_task().get_logger().report_scalar("SMAPE (Epoch)", "train", avg_smape, epoch)
        task.current_task().get_logger().report_scalar("MAPE (Epoch)", "train", avg_mape, epoch)
        
        print(f'Epoch {epoch+1}, MSE Loss: {avg_loss}, SMAPE: {avg_smape}, MAPE: {avg_mape}')

# Оценка модели
def evaluate(model, loader, criterion, epoch):
    model.eval()
    total_loss = 0
    total_smape = 0
    total_mape = 0  # Добавляем переменную для суммы MAPE
    with torch.no_grad():
        for points, next_points in loader:
            points, next_points = points.to(device), next_points.to(device)
            points = points.view(points.size(0), 3, -1)
            next_points = next_points.view(next_points.size(0), 3, -1)
            out = model(points)
            loss = criterion(out, next_points)
            total_loss += loss.item()
            total_smape += calculate_smape(out, next_points)
            total_mape += calculate_mape(out, next_points)  # Вычисляем MAPE
            
            # Логируем метрики
            task.current_task().get_logger().report_scalar("MSE Loss", "evaluate", loss.item(), epoch)
            task.current_task().get_logger().report_scalar("SMAPE", "evaluate", calculate_smape(out, next_points), epoch)
            task.current_task().get_logger().report_scalar("MAPE", "evaluate", calculate_mape(out, next_points), epoch)

    avg_loss = total_loss / len(loader)
    avg_smape = total_smape / len(loader)
    avg_mape = total_mape / len(loader)  # Среднее значение MAPE
    
    # Логируем средние значения
    task.current_task().get_logger().report_scalar("MSE Loss", "evaluate", avg_loss, epoch)
    task.current_task().get_logger().report_scalar("SMAPE", "evaluate", avg_smape, epoch)
    task.current_task().get_logger().report_scalar("MAPE", "evaluate", avg_mape, epoch)
    
    print(f'Evaluation MSE Loss: {avg_loss}, SMAPE: {avg_smape}, MAPE: {avg_mape}')


# Запуск обучения и оценки
train(model, train_loader, optimizer, criterion, parameters['epochs'])
evaluate(model, test_loader, criterion, parameters['epochs'])

# Проверка загрузки данных
for points, next_points in train_loader:
    points, next_points = points.to(device), next_points.to(device)
    report_clearml_3d("Point object", points[0].cpu())
    report_clearml_3d("Target object", next_points[0].cpu())
    prediction = predict(model, points[0].cpu())
    report_clearml_3d("Predict object", prediction.cpu())
    break

# Сохранение модели после обучения
model_save_path = os.path.join('save_model', f'trained_model_ep{parameters["epochs"]}_bch{parameters["batch_size"]}_smape.pth')
save_model(model, model_save_path)
