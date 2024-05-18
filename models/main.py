from torch.utils.data import DataLoader
from classes import PointCloudDataset
from functions import get_all_csv_files,report_clearml_3d
from clearml import Task


task = Task.init(project_name = 'PointNetML',task_name="DGCNN Regression --LoadData")

parametrs = {
    'root_dir': 'merged_output', # Путь к главной директории с экспериментами
    'batch_size': 1,
}
task.connect(parametrs)

# Получаем список всех CSV файлов
csv_files = get_all_csv_files(parametrs['root_dir'])

# Создаем экземпляр датасета
dataset = PointCloudDataset(csv_files)

# Создаем DataLoader для загрузки данных
data_loader = DataLoader(dataset, parametrs['batch_size'], shuffle=True)

# Проверка загрузки данных
for points, next_points in data_loader:
    print("Input points shape:", points.shape)
    print("Input points:")
    print(points[0])
    report_clearml_3d("Now point object", points[0])
    print("Target points shape:", next_points.shape) 
    print("Target points:")
    print(next_points[0])
    report_clearml_3d("Next point object", next_points[0])
    break





