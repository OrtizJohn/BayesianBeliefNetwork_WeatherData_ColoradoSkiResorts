import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


# files = ["../data/aspen.csv", "../data/steamboat.csv", "../data/vail.csv"]

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
        df = df.reset_index()

        dfs.append(df)
    return dfs

def split_dataframes(dfs: List[pd.DataFrame], R: float) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Splits each dataframe in an array of dataframes into train and test dataframes based on a given percentage R.
    Returns two lists of dataframes, one for the training data and another for the testing data.
    """
    if not isinstance(dfs, list) or not all(isinstance(df, pd.DataFrame) for df in dfs):
        raise TypeError("Input must be an array of pandas dataframes.")
    if not isinstance(R, float) or R <= 0 or R >= 1:
        raise ValueError("R must be a float between 0 and 1 exclusive.")
        
    train_dfs = []
    test_dfs = []
    for df in dfs:
        num_rows = df.shape[0]
        num_train = round(num_rows * R)
        num_test = num_rows - num_train
        
        train_df = df.head(num_train)
        test_df = df.tail(num_test)
        
        train_dfs.append(train_df)
        test_dfs.append(test_df)
        
    return train_dfs, test_dfs


def make_binary(dfs):
    #dfs_copy = dfs.copy()
    for df in dfs:
        date = df["date"]
        high = df["maxTemp"].astype(float)
        low = df["minTemp"].astype(float)
        precip = df["precipitation"].astype(float)
        snow = df["snowFall"].astype(float)

        # TODO: check the values we are comparing against here (they're mildly a guess so this will need refining)
        df['D'] = pd.to_datetime(date)
        df['H'] = (high > np.mean(high)).astype(int)
        df['L'] = (low < np.mean(low)).astype(int)
        # do we want this? complicates things
        # df['tempDiff'] = ((high - low) > 40).astype(int)
        df['P'] = (precip > 0).astype(int)
        df['S'] = (snow > 0).astype(int)

        df.drop(columns=["date", "maxTemp","minTemp", "precipitation","snowFall"], inplace=True)

        # TODO: add boolean for if it rained/snowed yesterday (might not work since not ever date is included)
        # precip_y = ??
        # snow_y = ??
    
    return dfs



# dfs = read_ski_resort_data(files)
# print(dfs[0].head(5))

# dfs = make_binary(dfs)
# print(dfs[0].head(5))
# print(dfs[0].dtypes)
