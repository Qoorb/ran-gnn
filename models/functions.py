from clearml import Logger
import pandas as pd
import os



def get_all_csv_files(root_dir):

    '''This function recursiving takes your csv 
    files with points and return their place list'''

    csv_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    print(f'Found {len(csv_files)} CSV files')
    return csv_files



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

