from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from clearml import Task, Dataset
from torchvision import transforms
from classes import read_off, report_clearml_3d
from classes import PointSampler, Normalize, RandRotation_z, RandomNoise, PointCloudData, ToTensor
from model import PointNet,pointnetloss

import numpy as np
import random
import os
import torch
import pandas as pd

number_ex = 360

data = pd.read_csv(f'src/taylor/{number_ex}/nodes1.csv')
data10 = pd.read_csv(f'src/taylor/{number_ex}/nodes10.csv')
task = Task.init(project_name = 'PointNetML',task_name=f'Taylor graph E{number_ex} S1 and S10')

data[['X','Y','Z']] = data[str(data.columns[0])].str.split(';', expand=True)
data = data.applymap(lambda x: x.replace('.0', '0.0').replace('-.0', '-0.0'))
data = data.drop(columns=[str(data.columns[0])])

data10[['X','Y','Z']] = data10[str(data10.columns[0])].str.split(';', expand=True)
data10 = data10.applymap(lambda x: x.replace('.0', '0.0').replace('-.0', '-0.0'))
data10 = data10.drop(columns=[str(data.columns[0])])

pointcloud = data.values.astype(float)
pointcloud10 = data.values.astype(float)
report_clearml_3d("Scatter_3d_1", pointcloud)
report_clearml_3d("Scatter_3d_10", pointcloud10)

norm_pointcloud = Normalize()(pointcloud)
norm_pointcloud10 = Normalize()(pointcloud10)
report_clearml_3d("Normalize_Scatter_3d_1", norm_pointcloud)
report_clearml_3d("Normalize_Scatter_3d_10", norm_pointcloud10)

noisy_pointcloud = RandomNoise()(norm_pointcloud)
noisy_pointcloud10 = RandomNoise()(norm_pointcloud10)
report_clearml_3d("Noisy_Scatter_3d_1", noisy_pointcloud)
report_clearml_3d("Noisy_Scatter_3d_10", noisy_pointcloud10)