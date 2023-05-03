import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import math
from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork

def bayes(joint, marginal, df):
    nodes = ['H', 'L', 'P', 'S']
    # edges = [['H', 'P'], ['L', 'P'], ['H', 'S'], ['L', 'S'], ['P', 'S']]
    X = df[nodes].to_numpy()
    print(X[0])
    # X = [1:,:]
    # conditionals
    # p_h = np.zeros([2,2])
    # p_l = np.zeros([2,2])
    # s_h = np.zeros([2,2])
    # s_l = np.zeros([2,2])
    # s_p = np.zeros([2,2])

    p_hl = np.zeros([2,2,2])
    s_hlp = np.zeros([2,2,2,2])

    for i in range(2):
        for j in range(2):
            # p_h[i][j] = len(df[df['H'] == j & df['P'] == i]) / len(df[df['H'] == j])
            # p_l[i][j] = len(df[df['L'] == j & df['P'] == i]) / len(df[df['L'] == j])
            # s_h[i][j] = len(df[df['H'] == j & df['S'] == i]) / len(df[df['H'] == j])
            # s_l[i][j] = len(df[df['L'] == j & df['S'] == i]) / len(df[df['L'] == j])
            # s_p[i][j] = len(df[df['P'] == j & df['S'] == i]) / len(df[df['P'] == j])
            
            for k in range(2):
                numerator = len(df[(df['H'] == i) & (df['L'] == j) & (df['P'] == k)])
                denominator = len(df[(df['H'] == i) & (df['L'] == j)])
                if denominator > 0:
                    p_hl[i][j][k] = numerator/denominator
                else:
                    p_hl[i][j][k] = 0
                
                for l in range(2):
                    numerator = len(df[(df['H'] == i) & (df['L'] == j) & (df['P'] == k) & (df['S'] == l)])
                    denominator = len(df[(df['H'] == i) & (df['L'] == j) & (df['P'] == k)])
                    if denominator > 0:
                        s_hlp[i][j][k][l] = numerator/denominator
                    else:
                        s_hlp[i][j][k][l] = 0
                    # s_hlp[i][j][k][l] = len(df[(df['H'] == j) & (df['L'] == k) & (df['P'] == l) & (df['S'] == i)]) / len(df[(df['H'] == j) & (df['L'] == k) & (df['P'] == l)])

    print(p_hl)
    print(s_hlp)
    H = Categorical([[marginal['H'][0], marginal['H'][1]]])
    L = Categorical([[marginal['L'][0], marginal['L'][1]]])
    P = ConditionalCategorical([p_hl])
    S = ConditionalCategorical([s_hlp])
    
    model = BayesianNetwork([H, L, P, S], [(H, P), (L, P), (H, S), (L, S), (P, S)])
    print(X.shape)
    model.fit(X)
    # model.from_summaries()
    print(model.distributions[0].probs, model.distributions[1].probs)

def markov():
    pass


from hmmlearn import hmm
def make_model(train, test):
    columns = ['H', 'L', 'P']#, 'S'
    X_train = train[columns].to_numpy()
    X_test = test[columns].to_numpy()

    # model = hmm.MultinomialHMM()
    model = hmm.GaussianHMM(n_components = 2, covariance_type = "full", n_iter = 500, random_state = 42, init_params="mcs")
    model.fit(X_train)
    Z = model.predict(X_test)
    print(Z)
    print(len(Z), X_test.shape)
    # print(test['D'])
    return Z

import tensorflow as tf
from tensorflow import keras
from keras import layers
# import tensorflow_datasets as tfds
import tensorflow_probability as tfp

def try_again():
    pass