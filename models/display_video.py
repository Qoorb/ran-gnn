import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
import torch
from functions import load_data,load_model,save_to_csv,predict
from classes import PointCloudDataset
from torch.utils.data import DataLoader
from pointnet import PointNet

number_to_video = '3'
dataset_test_video = PointCloudDataset([f'merged_output_taylor\\{number_to_video}\\merged_1_2.csv'])
test_loader_video = DataLoader(dataset_test_video, 1, shuffle=False)
output_directory = r'output_data' 
os.makedirs(output_directory, exist_ok=True)
model = load_model(PointNet,'save_model/trained_model_ep3_bch2_smape.pth' )

for points, next_points in test_loader_video:
    prediction = points[0]
    for i in range(1, 10):
        # Сохранение текущих и предсказанных точек в CSV
        filename = os.path.join(output_directory, f'merged_{i}_{i+1}.csv')
        prediction2 = predict(model, prediction)
        save_to_csv(prediction, prediction2, filename)
        prediction = prediction2

def make_video(data_directory,output_file,name_target):
    data = load_data(data_directory,name_target)
    files = sorted(data['file'].unique())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter([], [], [], s=50)

    def set_equal_aspect(ax):
        """ Устанавливает одинаковый масштаб на всех осях """
        extents = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        centers = np.mean(extents, axis=1)
        max_range = np.max(extents[:, 1] - extents[:, 0])
        bounds = np.array([centers - max_range / 2, centers + max_range / 2]).T
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_zlim(bounds[2])

    def init():
        ax.set_xlim(data['X'].min(), data['X'].max())
        ax.set_ylim(data['Y'].min(), data['Y'].max())
        ax.set_zlim(data['Z'].min(), data['Z'].max())
        set_equal_aspect(ax)  # Установка равных масштабов осей
        return scatter,

    # Обновление графика для каждого кадра
    def update(frame):
        current_file = files[frame]
        df = data[data['file'] == current_file]
        scatter._offsets3d = (df['X'].values, df['Y'].values, df['Z'].values)
        ax.set_title(f"Frame: {current_file}")
        set_equal_aspect(ax)  # Установка равных масштабов осей
        return scatter

    anim = FuncAnimation(fig, update, frames=len(files), init_func=init, blit=False)
    anim.save(output_file, writer='ffmpeg')
    #plt.show()

make_video(f'merged_output_taylor\\{number_to_video}',f'video/object_movement_target{number_to_video}.gif', 'Target')
make_video('output_data',f'video/object_movement_predict{number_to_video}.gif','Predict')