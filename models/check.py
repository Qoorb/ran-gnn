import pandas as pd
from clearml import Task
from functions import get_all_csv_files,report_clearml_3d

task = Task.init(project_name = 'PointNetML',task_name="DGCNN Regression --LoadData")
data1 = pd.read_csv("merged_output/210/merged_001_002.csv")
data2 = pd.read_csv("merged_output/210/merged_004_999.csv")

report_clearml_3d("Now point object",data1.iloc[:,:3].values.tolist())
report_clearml_3d("Next point object",data2.iloc[:,3:6].values.tolist())


