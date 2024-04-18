from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from clearml import Task, Dataset
from torchvision import transforms

import plotly.graph_objects as go
import numpy as np
import random
import os
import torch
from classes import read_off, report_clearml_3d
from classes import PointSampler, Normalize, RandRotation_z, RandomNoise, PointCloudData, ToTensor
from model import PointNet,pointnetloss


parametrs = {
    "optimazer": "Adam",
    "Learning_rate": 0.001,
    "epochs": 3,
    "train_batch_size": 32,
    "valid_batch_size": 64,
}

task = Task.init(project_name = 'PointNetML',task_name="Pointnet for classification")

data_path = Dataset.get(dataset_name = 'ModelNet10').get_local_copy()
path = Path(data_path)

folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)};

task.connect(parametrs)
task.upload_artifact(name ="Classes",artifact_object = classes)

with open(path/"bed/train/bed_0001.off", 'r') as f:
  verts, faces = read_off(f)

pointcloud = PointSampler(3000)((verts, faces))
report_clearml_3d("Scatter_3d", pointcloud)

norm_pointcloud = Normalize()(pointcloud)
report_clearml_3d("Normalize_Scatter_3d", norm_pointcloud)

rot_pointcloud = RandRotation_z()(norm_pointcloud)
noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)
report_clearml_3d("Noisy_Scatter_3d", noisy_rot_pointcloud)

train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])


train_ds = PointCloudData(path, transform=train_transforms)
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)
inv_classes = {i: cat for cat, i in train_ds.classes.items()};

task.upload_artifact(name ='Train dataset size',artifact_object = len(train_ds))
task.upload_artifact(name ="Valid dataset size",artifact_object = len(valid_ds))
task.upload_artifact(name ='Number of classes',artifact_object = len(train_ds.classes))
task.upload_artifact(name ="Sample pointcloud shape",artifact_object = train_ds[0]['pointcloud'].size())

train_loader = DataLoader(dataset=train_ds, batch_size=parametrs["train_batch_size"], shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=parametrs["valid_batch_size"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
task.upload_artifact(name ="Cuda_device",artifact_object = device)

pointnet = PointNet()
pointnet.to(device);

optimizer = torch.optim.Adam(pointnet.parameters(), lr=parametrs["Learning_rate"])
def train(model, optimizer,train_loader, val_loader=None,  epochs=15, device = "cpu", save=True):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(inputs.transpose(1,2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches

                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                    running_loss = 0.0

        model.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

        # save the model
        if save:
            torch.save(model.state_dict(), "save_"+str(epoch)+".pth")

train(
  model = pointnet,
  optimizer = optimizer, 
  train_loader = train_loader, 
  val_loader = valid_loader,
  epochs=parametrs['epochs'],  
  device = device,
  save=True)