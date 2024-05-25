import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
import torch
import os
from torch_geometric.data import Data, DataLoader
from clearml import Task
from classes import PointCloudDataset
from functions import get_all_csv_files, report_clearml_3d, save_to_csv, save_model, predict
from functions import calculate_mape, calculate_smape, calculate_logcosh_loss, calculate_smape_loss
from pointnet import PointNet
from pointnet import Net,DGCNN

# Инициализация задачи ClearML
task = Task.init(project_name='PointNetML', task_name="DGCNN Regression --Model3")

parameters = {
    'root_dir': 'merged_output_taylor',
    'batch_size': 10,
    'epoch': 2,
    'learning_rate': 0.01,
    'k': 2,
    'emb_dims': 512,
    'dropout': 0.2,
    'num_points': 20
}
task.connect(parameters)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

csv_files_train, csv_files_test = get_all_csv_files(parameters['root_dir'])
dataset_train = PointCloudDataset(csv_files_train,parameters['num_points'])
dataset_test = PointCloudDataset(csv_files_test,parameters['num_points'])

loader = DataLoader(dataset_train, parameters['batch_size'], shuffle=True)
test_loader = DataLoader(dataset_test, parameters['batch_size'], shuffle=False)

#model = Net(3, 64,parameters['num_points'], 3).to(device)
model = DGCNN(3,parameters['num_points'],4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    for epoch in range(epoch):
        total_loss = 0
        total_smape = 0
        total_mape = 0  # Добавляем переменную для суммы MAPE
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            target = data.y.view(data.y.size(0), -1, 3)
            print(target.size(0),target.size(1),target.size(2))
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_smape += calculate_smape(out, target)
            total_mape += calculate_mape(out, target)  # Вычисляем MAPE
            
            # Логируем метрики для каждой итерации
            task.current_task().get_logger().report_scalar("MSE Loss", "train", loss.item(), epoch)
            task.current_task().get_logger().report_scalar("SMAPE", "train", calculate_smape(out, target), epoch)
            task.current_task().get_logger().report_scalar("MAPE", "train", calculate_mape(out, target), epoch)

        avg_loss = total_loss / len(loader)
        avg_smape = total_smape / len(loader)
        avg_mape = total_mape / len(loader) 
        
        # Логируем средние значения по эпохе
        task.current_task().get_logger().report_scalar("MSE Loss (Epoch)", "train", avg_loss, epoch)
        task.current_task().get_logger().report_scalar("SMAPE (Epoch)", "train", avg_smape, epoch)
        task.current_task().get_logger().report_scalar("MAPE (Epoch)", "train", avg_mape, epoch)
        
        print(f'Epoch {epoch + 1}, MSE Loss: {avg_loss}, SMAPE: {avg_smape}, MAPE: {avg_mape}')

def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_smape = 0
    total_mape = 0  
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            target = data.y.view(data.y.size(0), -1, 3)
            loss = criterion(out, target)
            total_loss += loss.item()
            total_smape += calculate_smape(out, target)
            total_mape += calculate_mape(out, target) 
            
            # Логируем метрики для каждой итерации
            task.current_task().get_logger().report_scalar("MSE Loss", "evaluate", loss.item(), epoch)
            task.current_task().get_logger().report_scalar("SMAPE", "evaluate", calculate_smape(out, target), epoch)
            task.current_task().get_logger().report_scalar("MAPE", "evaluate", calculate_mape(out, target), epoch)

        avg_loss = total_loss / len(loader)
        avg_smape = total_smape / len(loader)
        avg_mape = total_mape / len(loader)  
        
        # Логируем средние значения
        task.current_task().get_logger().report_scalar("MSE Loss", "evaluate", avg_loss, epoch)
        task.current_task().get_logger().report_scalar("SMAPE", "evaluate", avg_smape, epoch)
        task.current_task().get_logger().report_scalar("MAPE", "evaluate", avg_mape, epoch)
        
        print(f'Evaluation MSE Loss: {avg_loss}, SMAPE: {avg_smape}, MAPE: {avg_mape}')

# Запуск обучения и оценки
train(model, loader, optimizer, criterion,device, parameters['epoch'])
evaluate(model, test_loader, criterion,device, parameters['epoch'])