import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from readFiles import *
from probability import *
from nets import *
from bnn import BNN

def main():

    files = ["../data/aspen.csv", "../data/steamboat.csv", "../data/vail.csv"]
    # brechenridge does not work at all, missing too much data
    # "../data/breckenridge.csv",

    dfs = read_ski_resort_data(files)

    # plt.hist(dfs[0]['precipitation'])
    # plt.show()

    # plt.hist(dfs[0]['snowFall'])
    # plt.show()

    #dfs = make_binary(dfs)
    dfs = alterDfs(dfs)

    # #split train/test
    train_dfs,test_dfs = split_dataframes(dfs,.8)
    
    # # for simplicity while figuring this out
    xTrain = train_dfs[0][['H', 'L', 'P']].to_numpy()
    yTrain = train_dfs[0]['S'].to_numpy()
    xTest = test_dfs[0][['H', 'L', 'P']].to_numpy()
    yTest = test_dfs[0]['S'].to_numpy()


    model = BNN()
    model.train(xTrain,yTrain)
    predict, mae = model.predict(xTest, yTest)

    # print(predict)

    model.plotPredictions(predict, yTest)
    # marg = find_marginals(df)
    # joint = find_joint(df)
    # # bayes(joint, marg, df)

    # pred = make_model(train, test)

    # print(np.sum(pred == (test['S'] > 0))/len(pred))




main()

