import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


files = ["../data/aspen.csv", "../data/breckenridge.csv", "../data/steamboat.csv", "../data/vail.csv"]


#TODO: Function - for each ski resort read in the file and return the data in the form of a pandas a df
def read_ski_resort_data(files_list):
    dfs = []
    for file in files_list:
        df = pd.read_csv(file, encoding='utf-8',delimiter="," ,header=0,names=["date","maxTemp","minTemp", "precipitation","snowFall"])
        dfs.append(df)
    return dfs

dfs = read_ski_resort_data(files)
print(dfs[0].head(5))