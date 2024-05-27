from clearml import Logger
import pandas as pd
import os
import random
import torch

def get_all_csv_files(root_dir):
    '''This function recursively takes your CSV files with points and returns
    two lists: one for training (70%) and one for testing (30%)'''

    # Собираем все CSV файлы
    csv_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    print(f'Found {len(csv_files)} CSV files')
    random.shuffle(csv_files)
    train_size = int(0.7 * len(csv_files))
    train_files = csv_files[:train_size]
    test_files = csv_files[train_size:]
    return train_files, test_files

# Обновленная функция для сохранения данных в CSV
def save_to_csv(points, next_points, filename):
    points = points.view(-1, 3).cpu().numpy()
    next_points = next_points.view(-1, 3).cpu().numpy()
    
    df = pd.DataFrame(points, columns=['X', 'Y', 'Z'])
    next_df = pd.DataFrame(next_points, columns=['X', 'Y', 'Z'])
    
    # Объединение текущих точек и предсказанных
    merged_df = pd.concat([df, next_df], axis=1)
    merged_df.columns = ['X', 'Y', 'Z', 'X1', 'Y1', 'Z1']
    
    # Сохранение в CSV файл
    merged_df.to_csv(filename, index=False)




def report_clearml_3d(title, scatter):

    '''This functions needed for report 
    3D graphs into ClearML Server'''

    Logger.current_logger().report_scatter3d(
        title=title,
        series="series_xyz",
        scatter= scatter,
        xaxis="x",
        yaxis="y",
        zaxis="z",
    )



def report_clearml_scatter(title,scatter2d):

    '''This functions needed for report 
    2D scatter plot graphs into ClearML Server'''

    Logger.current_logger().report_scatter2d(
    title,
    "series_xy",
    scatter=scatter2d,
    xaxis="title x",
    yaxis="title y",
    )   

def load_model(model_class, filename):
    model = model_class(3, 64, 3)  # Инициализация модели
    state_dict = torch.load(filename)  # Загрузка состояния модели из файла
    model.load_state_dict(state_dict)  # Загрузка состояния в модель
    model.eval()  # Переключаем модель в режим оценки
    print(f"Model loaded from {filename}")
    return model
# Функция для загрузки данных из всех CSV файлов
def load_data(directory,name_target):
    data_frames = []
    for i in range(1, 8):
            filename = os.path.join(directory, f"merged_{i}_{i+1}.csv")
            if os.path.exists(filename):  # Проверка, существует ли файл
                df = pd.read_csv(filename)
                df['file'] = f"{name_target}_{i}_{i+1}"
                data_frames.append(df)
            else:
                print(f"File not found: {filename}")
    return pd.concat(data_frames, ignore_index=True)

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def predict(model, data, edge_index, batch):
    model.eval()
    with torch.no_grad():
        output = model(data, edge_index, batch)
    return output

# Функция для расчета MAPE
def calculate_mape(output, target):
    absolute_percentage_error = torch.abs(output - target) / target
    mape = torch.mean(absolute_percentage_error) * 100
    return mape.item()

def calculate_smape(output, target):
    numerator = torch.abs(output - target)
    denominator = (torch.abs(output) + torch.abs(target)) / 2
    smape = torch.mean(numerator / denominator) * 100
    return smape.item()

def calculate_smape_loss(output, target):
    numerator = torch.abs(output - target)
    denominator = (torch.abs(output) + torch.abs(target)) / 2
    smape = torch.mean(numerator / denominator) * 100
    return smape

def calculate_logcosh(output, target):
    logcosh = torch.log(torch.cosh(output - target))
    return torch.mean(logcosh).item()

def calculate_logcosh_loss(output, target):
    logcosh = torch.log(torch.cosh(output - target))
    return torch.mean(logcosh)

def calculate_exponential_loss(output, target):
    diff = output - target
    exp_loss = torch.mean(torch.exp(diff) - 1)
    return exp_loss
