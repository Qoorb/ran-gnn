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
from functions import get_all_csv_files, report_clearml_3d, save_to_csv, save_model
from functions import calculate_mape, calculate_smape, calculate_logcosh_loss, calculate_smape_loss
from pointnet import PointNet
from pointnet import Net,DGCNN
from tqdm import tqdm

# Инициализация задачи ClearML
task = Task.init(project_name='PointNetML', task_name="DGCNN --CompleteWithoutPoint_CPU")

parameters = {
    'root_dir': 'merged_output_taylor',
    'batch_size': 2,
    'epoch': 2,
    'learning_rate': 0.01,
    'k': 30,
    'emb_dims': 512,
    'dropout': 0.2,
    'num_points': 2000
}
task.connect(parameters)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

csv_files_train, csv_files_test = get_all_csv_files(parameters['root_dir'])
dataset_train = PointCloudDataset(csv_files_train,parameters['num_points'],parameters["k"])
dataset_test = PointCloudDataset(csv_files_test,parameters['num_points'],parameters["k"])

loader = DataLoader(dataset_train, parameters['batch_size'], shuffle=True)
test_loader = DataLoader(dataset_test, parameters['batch_size'], shuffle=False)


#model = Net(3, 64,parameters['num_points'], 3).to(device)
model = DGCNN(3,parameters['num_points'],parameters['k']).to(device)
optimizer = torch.optim.Adam(model.parameters(), parameters['learning_rate'])
criterion = torch.nn.MSELoss()



def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    for epoch in range(epoch):
        total_loss = 0
        total_smape = 0
        total_mape = 0
        avg_loss = 0
        avg_smape = 0
        avg_mape = 0
        progress_bar = tqdm(loader, desc=f'Epoch {epoch + 1}')
        for batch_idx, data in enumerate(progress_bar):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            target = data.y.view(data.y.size(0), -1, 3)
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

            # Обновляем описание `tqdm` с текущими значениями
            progress_bar.set_description(f'Epoch {epoch + 1}, Loss: {total_loss/(batch_idx+1):.4f}, SMAPE: {total_smape/(batch_idx+1):.4f}')

        avg_loss = total_loss / len(loader)
        avg_smape = total_smape / len(loader)
        avg_mape = total_mape / len(loader)

        # Логируем средние значения по эпохе
        task.current_task().get_logger().report_scalar("MSE Loss (Epoch)", "train", avg_loss, epoch)
        task.current_task().get_logger().report_scalar("SMAPE (Epoch)", "train", avg_smape, epoch)
        task.current_task().get_logger().report_scalar("MAPE (Epoch)", "train", avg_mape, epoch)

        print(f'Epoch {epoch + 1}, MSE Loss: {avg_loss:.4f}, SMAPE: {avg_smape:.4f}, MAPE: {avg_mape:.4f}')

def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_smape = 0
    total_mape = 0
    avg_loss = 0
    avg_smape = 0
    avg_mape = 0
    print("Evaluate")
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f'Epoch {epoch + 1}')
        for batch_idx, data in enumerate(progress_bar):
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

            # Обновляем описание `tqdm` с текущими значениями
            progress_bar.set_description(f'Epoch {epoch + 1}, Loss: {total_loss/(batch_idx+1):.4f}, SMAPE: {total_smape/(batch_idx+1):.4f}')

        avg_loss = total_loss / len(loader)
        avg_smape = total_smape / len(loader)
        avg_mape = total_mape / len(loader)

        # Логируем средние значения
        task.current_task().get_logger().report_scalar("MSE Loss", "evaluate", avg_loss, epoch)
        task.current_task().get_logger().report_scalar("SMAPE", "evaluate", avg_smape, epoch)
        task.current_task().get_logger().report_scalar("MAPE", "evaluate", avg_mape, epoch)

        print(f'Evaluation MSE Loss: {avg_loss:.4f}, SMAPE: {avg_smape:.4f}, MAPE: {avg_mape:.4f}')


# Запуск обучения и оценки
train(model, loader, optimizer, criterion,device, parameters['epoch'])
evaluate(model, test_loader, criterion,device, parameters['epoch'])

def predict(model, data, edge_index, batch):
    model.eval()
    with torch.no_grad():
        output = model(data, edge_index, batch)
    return output

for batch_idx, data in enumerate(loader):
    points, next_points = data.x, data.y
    points, next_points = points.to(device), next_points.to(device)
    edge_index, batch = data.edge_index.to(device), data.batch.to(device)  # Добавлено
    print(f'Batch {batch_idx + 1}')
    print('Points:', points)
    print('Next Points:', next_points[0])
    report_clearml_3d("Point object", points)
    report_clearml_3d("Target object", next_points[0])
    prediction = predict(model, points, edge_index, batch)  # Передаем edge_index и batch
    print("asdasd",prediction[0])
    report_clearml_3d("Predict object", prediction[0])
    
    # Прерываем после первого батча для примера
    break


# Сохранение модели после обучения
model_save_path = os.path.join('save_model', f'DGCNN_ep{parameters["epoch"]}_bch{parameters["batch_size"]}_lr{parameters["learning_rate"]}_k{parameters["k"]}.pth')
save_model(model, model_save_path)