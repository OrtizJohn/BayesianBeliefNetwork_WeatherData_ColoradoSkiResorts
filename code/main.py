import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from readFiles import *
from probability import *

def main():

    files = ["../data/aspen.csv", "../data/steamboat.csv", "../data/vail.csv"]
    # brechenridge does not work at all, missing too much data
    # "../data/breckenridge.csv",

    dfs = read_ski_resort_data(files)
    dfs = make_binary(dfs)

    # for simplicity while figuring this out
    df = dfs[0]

    print(find_marginals(df))
    print(find_joint(df))



main()

