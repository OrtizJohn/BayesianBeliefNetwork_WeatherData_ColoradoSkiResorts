import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from readFiles import *
from probability import *
from nets import *

def main():

    files = ["../data/aspen.csv", "../data/steamboat.csv", "../data/vail.csv"]
    # brechenridge does not work at all, missing too much data
    # "../data/breckenridge.csv",

    dfs = read_ski_resort_data(files)
    dfs = make_binary(dfs)


    #split train/test
    train_dfs,test_dfs = split_dataframes(dfs,.8)
    
    # for simplicity while figuring this out
    df = train_dfs[0]

    marg = find_marginals(df)
    joint = find_joint(df)
    bayes(joint, marg, df)




main()

