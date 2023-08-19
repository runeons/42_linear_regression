import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils_constants import IMAGES_PATH, FIG_SIZE, C2
from utils_colors import Colors
from utils_plot import scatter_plot, histogram_plot

def init():
    os.makedirs(IMAGES_PATH, exist_ok=True)
    np.random.seed(42) # pour generer toujours le meme set de test

def load_data(csv_path):
    return pd.read_csv(csv_path)

def explore_data(data):
    print(data.head())
    print(data.info())
    scatter_plot(data, "init_scatter_plot")
    histogram_plot(data, "init_histogram_plot")

def split_train_test(data, ratio):
    shuf_i = np.random.permutation(len(data))
    test_size = int(len(data) * ratio)
    test_i = shuf_i[:test_size]
    train_i = shuf_i[test_size:]
    return data.iloc[train_i], data.iloc[test_i]

def main():
    try:
        init()
        full_data = load_data(os.path.join("dataset", "data.csv"))
        # explore_data(data)
        train_set, test_set = split_train_test(full_data, 0.2)
        train_set = train_set.reset_index()
        print(train_set.head())


    except ValueError as e:
        print(e)

if (__name__ == "__main__"):
    main()