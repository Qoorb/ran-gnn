import os
import pandas as pd

def merge_csv_in_folders(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    merged_df = pd.DataFrame()
    subfolders = ['src/taylor\\420'] #delete if you want use all dataset experiments
    for subfolder in subfolders:
        csv_files = [f for f in os.listdir(subfolder) if f.endswith('.csv') and 'node' in f]
        subfolder_df = pd.DataFrame()
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(subfolder, csv_file))
            df[['X', 'Y', 'Z']] = df[str(df.columns[0])].str.split(';', expand=True)
            df = df.applymap(lambda x: x.replace('.0', '0.0').replace('-.0', '-0.0'))
            df = df.drop(columns=[str(df.columns[0])])
            df = df.rename(columns={'X': f'X{csv_file[5:6]}_E{subfolder[11:]}', 'Y': f'Y{csv_file[5:6]}_E{subfolder[11:]}', 'Z': f'Z{csv_file[5:6]}_E{subfolder[11:]}'})
            df.reset_index(drop=True, inplace=True)
            df =df.astype(float)*100000
            print(df.describe())
            subfolder_df = pd.concat([subfolder_df, df], axis=1)

        # Объединяем DataFrame текущей папки с основным DataFrame
        merged_df = pd.concat([merged_df, subfolder_df], axis=0)

    return merged_df

# Задаем путь к основной папке
folder_path = 'src/taylor'

# Вызываем функцию для объединения CSV файлов внутри папок
merged_data = merge_csv_in_folders(folder_path)

# Выводим объединенные данные
print(merged_data)
