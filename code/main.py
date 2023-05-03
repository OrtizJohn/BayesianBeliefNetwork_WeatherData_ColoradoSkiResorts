import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode

from readFiles import *
from probability import *
from nets import *
from bnn import BNN

def main():

    # files used
    files = ["../data/Aspen_1SW_5.csv"]#, "../data/Aspen_full.csv"]
    #, "../data/aspen.csv", "../data/steamboat.csv", "../data/vail.csv"]

    # brechenridge does not work at all, missing too much data
    # "../data/breckenridge.csv",

    # read data
    dfs = read_ski_resort_data(files)


    # Some checks, remove?
    # plt.hist(dfs[0]['precipitation'])
    # plt.show()

    # plt.hist(dfs[0]['snowFall'])
    # plt.show()

    # change with attempting binary data
    #dfs = make_binary(dfs)
    dfs = alterDfs(dfs)

    # #split train/test
    train_dfs,test_dfs = split_dataframes(dfs,.8)
    preds = []
    MAEs = []
    RMSEs = []

    for i in range(len(files)):
    # for i in range(1):
    
        xTrain = train_dfs[i][['H', 'L', 'P']].to_numpy()
        yTrain = train_dfs[i]['S'].to_numpy()
        train_dates = train_dfs[i]['D']
        xTest = test_dfs[i][['H', 'L', 'P']].to_numpy()
        yTest = test_dfs[i]['S'].to_numpy()
        test_dates = test_dfs[i]['D']


        model = BNN()
        model.train(xTrain,yTrain)
        predict, MAE, RMSE = model.predict(xTest, yTest)
        # print(min(predict), min(yTest))
        # print(mode(predict))


        # model.plotPredictions(predict, yTest, test_dates, files[i], MAE, RMSE, type=0, save=False)
        model.plotPredictions(predict, yTest, test_dates, files[i], MAE, RMSE, type=2, save=True)

        preds.append(predict)
        MAEs.append(MAE)
        RMSEs.append(RMSE)


        maeFile = open(f"../data/errs/{files[i][8:-4]}_mae.txt", "a")
        maeFile.write(f"{MAE}\n")
        maeFile.close()

        rmseFile = open(f"../data/errs/{files[i][8:-4]}_rmse.txt", "a")
        rmseFile.write(f"{RMSE}\n")
        rmseFile.close()


    # TODO: probably remove the thinge below here
    # marg = find_marginals(df)
    # joint = find_joint(df)
    # # bayes(joint, marg, df)

    # pred = make_model(train, test)

    # print(np.sum(pred == (test['S'] > 0))/len(pred))

    # print(train_dfs[1]['D'])
    # print(test_dfs[1]['D'])



main()

