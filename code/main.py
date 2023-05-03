import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode

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
    preds = []
    maes = []

    for i in range(len(files)):
    
        xTrain = train_dfs[i][['H', 'L', 'P']].to_numpy()
        yTrain = train_dfs[i]['S'].to_numpy()
        xTest = test_dfs[i][['H', 'L', 'P']].to_numpy()
        yTest = test_dfs[i]['S'].to_numpy()


        model = BNN()
        model.train(xTrain,yTrain)
        predict, mae = model.predict(xTest, yTest)
        print(min(predict), min(yTest))
        print(mode(predict))

        # print(predict)

        model.plotPredictions(predict, yTest)
        preds.append(predict)
        maes.append(mae)
    # marg = find_marginals(df)
    # joint = find_joint(df)
    # # bayes(joint, marg, df)

    # pred = make_model(train, test)

    # print(np.sum(pred == (test['S'] > 0))/len(pred))




main()

