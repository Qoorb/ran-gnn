import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from classes import PointCloudDataset
from functions import get_all_csv_files, report_clearml_3d
from clearml import Task
from model import DGCNN

# Инициализация задачи ClearML
task = Task.init(project_name='PointNetML', task_name="DGCNN Regression --Model")

# Параметры
parameters = {
    'root_dir': 'merged_output',
    'batch_size': 4,
    'epochs': 100,
    'learning_rate': 0.01,
    'k': 2,
}
task.connect(parameters)

# Получение списка CSV файлов
csv_files_train, csv_files_test = get_all_csv_files(parameters['root_dir'])

# Создание датасетов и DataLoader
dataset_train = PointCloudDataset(csv_files_train)
dataset_test = PointCloudDataset(csv_files_test)
train_loader = DataLoader(dataset_train, parameters['batch_size'], shuffle=True)
test_loader = DataLoader(dataset_test, parameters['batch_size'], shuffle=False)

# Проверка загрузки данных
for points, next_points in train_loader:
    print("Форма входных точек:", points.shape)
    print("Входные точки:")
    print(points[0])
    report_clearml_3d("Now point object", points[0])
    print("Форма целевых точек:", next_points.shape) 
    print("Целевые точки:")
    print(next_points[0])
    report_clearml_3d("Next point object", next_points[0])
    break

# Инициализация модели
model = DGCNN(in_channels=3, out_channels=3, k=parameters['k'])
optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'])
loss_fn = nn.MSELoss()

# Обучение модели
model.train()
for epoch in range(parameters['epochs']):
    epoch_loss = 0
    for points, next_points in train_loader:
        optimizer.zero_grad()
        points = points.float()
        next_points = next_points.float()
        
        # Подготовка данных для DGCNN
        data = Data(x=points.view(-1, 3))  # Преобразование данных
        data.batch = torch.arange(points.size(0)).repeat_interleave(points.size(1)).to(points.device)
        
        output = model(data)
        loss = loss_fn(output.view_as(next_points), next_points)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}')

# Тестирование модели
model.eval()
test_loss = 0
with torch.no_grad():
    for points, next_points in test_loader:
        points = points.float()
        next_points = next_points.float()
        
        data = Data(x=points.view(-1, 3))  # Преобразование данных
        data.batch = torch.arange(points.size(0)).repeat_interleave(points.size(1)).to(points.device)
        
        output = model(data)
        loss = loss_fn(output.view_as(next_points), next_points)
        test_loss += loss.item()
    
    print(f'Test Loss: {test_loss / len(test_loader)}')
