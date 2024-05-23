import os
import os.path as op
from collections import OrderedDict
import pandas as pd
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

def vtu_to_csv(dir_path: str, save_path: str) -> None:
    if not op.exists(save_path):
        os.makedirs(save_path)

    reader = vtk.vtkXMLUnstructuredGridReader()

    # Рекурсивно обходим директорию
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".vtu"):
                file_path = op.join(root, file)
                relative_path = op.relpath(file_path, dir_path)
                output_dir = op.join(save_path, op.dirname(relative_path))
                if not op.exists(output_dir):
                    os.makedirs(output_dir)

                reader.SetFileName(file_path)
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

                df = pd.DataFrame(data)
                df = df.iloc[:, -3:]  # Оставляем три последних столбца

                csv_filename = op.splitext(op.basename(file))[0] + '.csv'
                csv_filepath = op.join(output_dir, csv_filename)
                df.to_csv(csv_filepath, index=False)

if __name__ == '__main__':
    vtu_to_csv('./data', './data_csv')
