import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# TODO: compute conditional independence from joint/marginal etc (lol maybe not)
# def find_conditional_independence(df, x, y, z):
#     # print(citest(df[x], df[y], df[z], test_args={'statistic': 'ksg_mi', 'n_jobs': 2}))
#     pass

# TODO: find joint probability from data
def find_joint(df):
    length = df.shape[0]
    columns = ['H', 'L', 'P', 'S']

    jointpdf = np.zeros(shape=[2,2,2,2])

    for i in range(length):
        jointpdf[df['H'][i]][df['L'][i]][df['P'][i]][df['S'][i]] += 1
    
    jointpdf = jointpdf/length
    return jointpdf

    



# TODO: find marginal probabilities from data
def find_marginals(df):
    length = df.shape[0]
    columns = ['H', 'L', 'P', 'S']
    probs = {i : {0: 0, 1: 1} for i in columns}

    for i in columns:
        ones = np.sum(df[i])
        probs[i][1] = ones/length
        probs[i][0] = 1 - probs[i][1]

    return probs

