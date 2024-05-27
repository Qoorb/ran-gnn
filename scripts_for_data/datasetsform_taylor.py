import os
import pandas as pd

def merge_csv_in_folders(folder_path, output_base_folder):
    def average_every_three_rows(df):
        # Вычисляем среднее значение каждых трех строк
        avg_df = df.groupby(df.index // 3).mean().reset_index(drop=True)
        return avg_df

    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for subfolder in subfolders:
        output_folder = os.path.join(output_base_folder, os.path.basename(subfolder))
        os.makedirs(output_folder, exist_ok=True)
        
        # Переименование файлов nodes10 в nodes99
        for f in os.listdir(subfolder):
            if f.endswith('.csv') and 'nodes' in f:
                new_name = f.replace('10', '99')
                os.rename(os.path.join(subfolder, f), os.path.join(subfolder, new_name))
        
        # После переименования создаем список файлов и сортируем его
        csv_files = [f for f in os.listdir(subfolder) if f.endswith('.csv') and 'nodes' in f]
        csv_files.sort()

        print(csv_files)
        
        for i in range(len(csv_files) - 1):  # Итерация по всем файлам, кроме последнего
            file1 = os.path.join(subfolder, csv_files[i])
            file2 = os.path.join(subfolder, csv_files[i+1])
            
            df1 = pd.read_csv(file1)
            df1 = df1[str(df1.columns[0])].str.split(';', expand=True)
            df1.columns = ['X', 'Y', 'Z']
            df1 = df1.applymap(lambda x: x.replace('.0', '0.0').replace('-.0', '-0.0'))
            df1 = df1.astype(float)
            avg_df1 = average_every_three_rows(df1)

            df2 = pd.read_csv(file2)
            df2 = df2[str(df2.columns[0])].str.split(';', expand=True)
            df2.columns = ['X1', 'Y1', 'Z1']
            df2 = df2.applymap(lambda x: x.replace('.0', '0.0').replace('-.0', '-0.0'))
            df2 = df2.astype(float)
            avg_df2 = average_every_three_rows(df2)

            merged_df = pd.concat([avg_df1, avg_df2.reset_index(drop=True)], axis=1)

            print(merged_df.columns)
            output_file = os.path.join(output_folder, f"merged_{csv_files[i][5:6]}_{csv_files[i+1][5:6]}.csv")
            merged_df.to_csv(output_file, index=False)

# Задаем путь к основной папке и путь к папке, куда будут сохранены объединенные файлы
folder_path = 'taylor'
output_base_folder = 'merged_output_taylor'

# Вызываем функцию для объединения CSV файлов внутри папок и сохранения их в новых папках
merge_csv_in_folders(folder_path, output_base_folder)
