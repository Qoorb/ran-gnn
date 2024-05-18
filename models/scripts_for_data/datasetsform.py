import os
import pandas as pd

def merge_csv_in_folders(folder_path, output_base_folder):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for subfolder in subfolders:
        output_folder = os.path.join(output_base_folder, os.path.basename(subfolder))
        os.makedirs(output_folder, exist_ok=True)
        csv_files = [f for f in os.listdir(subfolder) if f.endswith('.csv')]
        csv_files.sort()  # Сортировка файлов для правильного объединения
        for i in range(len(csv_files) - 1):  # Итерация по всем файлам, кроме последнего
            file1 = os.path.join(subfolder, csv_files[i])
            file2 = os.path.join(subfolder, csv_files[i+1])
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
            df2.columns = [f"{col}1" for col in df2.columns]
            merged_df = pd.concat([df1, df2.reset_index(drop=True)], axis=1)
            output_file = os.path.join(output_folder, f"merged_{csv_files[i][6:9]}_{csv_files[i+1][6:9]}.csv")
            merged_df.to_csv(output_file, index=False)

# Задаем путь к основной папке и путь к папке, куда будут сохранены объединенные файлы
folder_path = 'data_csv'
output_base_folder = 'merged_output'

# Вызываем функцию для объединения CSV файлов внутри папок и сохранения их в новых папках
merge_csv_in_folders(folder_path, output_base_folder)
