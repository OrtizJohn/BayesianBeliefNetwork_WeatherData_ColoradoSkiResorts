import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


files = ["../data/aspen.csv", "../data/steamboat.csv", "../data/vail.csv"]

# brechenridge does not work at all, missing too much data
# "../data/breckenridge.csv",


#TODO: Function - for each ski resort read in the file and return the data in the form of a pandas a df
def read_ski_resort_data(files_list):
    dfs = []
    for file in files_list:
        df = pd.read_csv(file, encoding='utf-8',delimiter="," ,header=0,names=["date","maxTemp","minTemp", "precipitation","snowFall"])

        # my own sanity, remove this later
        # print(file)
        # print(df['maxTemp'].unique())
        # print(df['minTemp'].unique())
        # print(df['precipitation'].unique())
        # print(df['snowFall'].unique())

        # clean rows
        indexCleaned = df[ (df['date'] == 'M') | (df['maxTemp'] == 'M') | (df['minTemp'] == 'M') | (df['precipitation'] == 'M') | (df['snowFall'] == 'M')].index
        df.drop(indexCleaned , inplace=True)

        indexCleaned = df[ (df['date'] == 'T') | (df['maxTemp'] == 'T') | (df['minTemp'] == 'T') | (df['precipitation'] == 'T') | (df['snowFall'] == 'T')].index
        df.drop(indexCleaned , inplace=True)

        dfs.append(df)
    return dfs

def make_boolean(dfs):
    for df in dfs:
        date = df["date"]
        high = df["maxTemp"].astype(float)
        low = df["minTemp"].astype(float)
        precip = df["precipitation"].astype(float)
        snow = df["snowFall"].astype(float)

        # TODO: check the values we are comparing against here (they're mildly a guess so this will need refining)
        df['date'] = pd.to_datetime(date)
        df['high'] = (high < 50).astype(int)
        df['low'] = (low < 32).astype(int)
        df['tempDiff'] = ((high - low) > 40).astype(int)
        df['precip'] = (precip > 0).astype(int)
        df['snow'] = (snow > 0).astype(int)

        df.drop(columns=["maxTemp","minTemp", "precipitation","snowFall"], inplace=True)

        # TODO: add boolean for if it rained/snowed yesterday (might not work since not ever date is included)
        # precip_y = ??
        # snow_y = ??
    
    return dfs



dfs = read_ski_resort_data(files)
print(dfs[0].head(5))

dfs = make_boolean(dfs)
print(dfs[0].head(5))
print(dfs[0].dtypes)
