import os
import os.path as op
from collections import OrderedDict

import pandas as pd

import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy


def vtu_to_csv(dir_path: str, save_path: str) -> None:
    if not op.exists(save_path): os.mkdir(save_path)

    reader = vtk.vtkXMLUnstructuredGridReader()

    files = []
    for f in os.listdir(dir_path):
        if f.endswith(".vtu"):
            files.append(op.join(dir_path, f))

    for file in files:
        reader.SetFileName(file)
        reader.Update()

        output = reader.GetOutput()

        data = OrderedDict()
        for idx in range(output.GetPointData().GetNumberOfArrays()):
            arr = output.GetPointData().GetArray(idx)
            data[arr.GetName()] = vtk_to_numpy(arr)

        coordinates = vtk_to_numpy(output.GetPoints().GetData())
        data['X'] = coordinates[:, 0]
        data['Y'] = coordinates[:, 1]
        data['Z'] = coordinates[:, 2]
        
        pd.DataFrame(data).to_csv(op.join(save_path,
                                          op.splitext(
                                              op.basename(file))[0] + '.csv'), index=False)


if __name__ == '__main__':
    vtu_to_csv('./data', './data_csv')
