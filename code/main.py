import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
# from sklearn.preprocessing import RobustScaler

from readFiles import *
from probability import *
from nets import *
from bnn import BNN

def main():

    # files used
    files = ["../data/Aspen_1SW_5.csv", "../data/Steamboat_5.csv", "../data/Vail_5.csv"]

    # brechenridge does not work at all, missing too much data
    # "../data/breckenridge.csv",
    # Short sets: "../data/aspen.csv", "../data/steamboat.csv", "../data/vail.csv"
    # very long set, different years: "../data/Aspen_full.csv"

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
    model = BNN()

    for i in range(len(files)):
    # for i in range(1):

        xTrain = train_dfs[i][['H', 'L', 'P']].to_numpy()
        xTest = test_dfs[i][['H', 'L', 'P']].to_numpy()

        yTrain = train_dfs[i]['S'].to_numpy()
        yTest = test_dfs[i]['S'].to_numpy()
        test_dates = test_dfs[i]['D']

        model.train(xTrain,yTrain)

    preds = []
    MAEs = []
    RMSEs = []

    t_preds = []
    t_MAEs = []
    t_RMSEs = []
    for i in range(len(files)):
        xTrain = train_dfs[i][['H', 'L', 'P']].to_numpy()
        xTest = test_dfs[i][['H', 'L', 'P']].to_numpy()

        yTrain = train_dfs[i]['S'].to_numpy()
        yTest = test_dfs[i]['S'].to_numpy()

        train_dates = train_dfs[i]['D']
        test_dates = test_dfs[i]['D']

        predict, MAE, RMSE = model.predict(xTest, yTest)
        # print(min(predict), min(yTest))
        # print(mode(predict))


        model.plotPredictions(predict, yTest, test_dates, files[i], MAE, RMSE, type=2, save=False)
        # model.plotPredictions(predict, yTest, test_dates, files[i], MAE, RMSE, type=2, save=True)

        preds.append(predict)
        MAEs.append(MAE)
        RMSEs.append(RMSE)


        maeFile = open(f"../data/errs/{files[i][8:-4]}_mae.txt", "a")
        maeFile.write(f"{MAE}\n")
        maeFile.close()

        rmseFile = open(f"../data/errs/{files[i][8:-4]}_rmse.txt", "a")
        rmseFile.write(f"{RMSE}\n")
        rmseFile.close()

        # test on training set to compare
        t_predict, t_MAE, t_RMSE = model.predict(xTrain, yTrain)

        model.plotPredictions(t_predict, yTrain, train_dates, files[i], t_MAE, t_RMSE, type=2, save=False)
        # model.plotPredictions(t_predict, yTrain, train_dates, files[i], t_MAE, t_RMSE, type=2, save=True)

        t_preds.append(t_predict)
        t_MAEs.append(t_MAE)
        t_RMSEs.append(t_RMSE)


    for i in range(len(files)):
        print(files[i])
        print("MAE:", MAEs[i], t_MAEs[i])
        print("RMSE:", RMSEs[i], t_RMSEs[i])

    # TODO: probably remove the thinge below here
    # marg = find_marginals(df)
    # joint = find_joint(df)
    # # bayes(joint, marg, df)

    # pred = make_model(train, test)

    # print(np.sum(pred == (test['S'] > 0))/len(pred))

    # print(train_dfs[1]['D'])
    # print(test_dfs[1]['D'])



main()

