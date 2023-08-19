import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils_constants import IMAGES_PATH
from utils_colors import Colors
from utils_plot import scatter_plot, histogram_plot
import os
from utils_constants import IMAGES_PATH, FIG_SIZE, C2
import matplotlib.pyplot as plt
from utils_colors import Colors

class Data:
    def __init__(self, data_path, image_path="./images/", fig_size=(10, 7)):
        self.data_path = data_path
        self.full_data = self.load()
        self.train_set, self.test_set = self.split_sets(0.2)
        # self.reset_index()
        self.fig_size = fig_size
        self.image_path = image_path
        os.makedirs(image_path, exist_ok=True)
        self.normed_train_set = self.normalise_train_set()

    def load(self):
        return pd.read_csv(self.data_path)

    def split_sets(self, test_ratio):
        np.random.seed(42)
        rand_i = np.random.permutation(len(self.full_data))
        test_size = int(len(self.full_data) * test_ratio)
        test_i = rand_i[:test_size]
        train_i = rand_i[test_size:]
        return self.full_data.iloc[train_i], self.full_data.iloc[test_i]

    def summary(self, d_set):
        print(f"{Colors.DATASET}[DATASET SUMMARY] head:\n{Colors.RES}{d_set.head()}")
        print(f"{Colors.DATASET}[DATASET SUMMARY] count:\n{Colors.RES}{d_set.count()}")
        print(f"{Colors.DATASET}[DATASET SUMMARY] min:\n{Colors.RES}{d_set.min()}")
        print(f"{Colors.DATASET}[DATASET SUMMARY] max:\n{Colors.RES}{d_set.max()}")

    def reset_index(self):
        self.train_set = self.train_set.reset_index()
        self.test_set = self.test_set.reset_index()

    def save_plot(self, name):
        path = os.path.join(self.image_path, name + ".png")
        plt.tight_layout()
        plt.savefig(path, format="png", dpi=300)
        print(f"{Colors.GREEN}Plot {Colors.RES}{name} {Colors.GREEN}saved{Colors.RES}")

    def scatter(self, data, x, y, name, save=True):
        data.plot(kind='scatter', x=x, y=y, alpha=0.5, color='green', figsize=self.fig_size)
        if save == True:
            self.save_plot(name + "_scatter")
        plt.show()

    def histogram(self, data, bins, name, save=True):
        data.hist(bins=bins, figsize=self.fig_size, color=C2) # 6 car 24 paires de données, donc pratique
        if save == True:
            self.save_plot(name + "_histogram")
        plt.show()

    def normalise_train_set(self):
        return self.train_set / self.train_set.max()
        # return (self.train_set - self.train_set.min()) / (self.train_set.max() - self.train_set.min()) 

# def to_file(t0, t1):
#     s = f"{t0} {t1}"
#     f = open("thetas", "w+")
#     f.write(s)
#     f.close()

# def cost_plot(summary):
#     plt.figure(figsize=(10, 6))
#     plt.plot(summary['iteration'], summary['squared_residuals'], marker='o')
#     plt.grid(True)
#     plt.show()

# class LinReg:
#     def __init__(self):
#         self.y_min = self._coordinates_to_int(0, 0)
#         self.y_max = self._coordinates_to_int(65535, 65535)
#         self.denormalise_max = 4294967295

def main():
    d = Data("dataset/data.csv")
    d.summary(d.train_set)
    # d.scatter(d.train_set, "km", "price", "init train_set")
    # d.histogram(d.train_set, 6, "init train_set")
    d.normalise_train_set()
    d_train = d.normed_train_set.reset_index()
    print(d_train.head())
    # print(type(d_train))                    # DataFrame
    # print(type(d_train["km"]))              # Series
    # print(type(d_train["km"].to_numpy()))   # ndarray
    kms = d_train["km"].to_numpy()
    prices = d_train["price"].to_numpy()
    # t0, t1 = train(kms, prices, 0.01, 4000)
    # to_file(t0, t1)
    # to_file(5, 8)

if (__name__ == "__main__"):
    main()

## price = t0 + t1 * km
    # 1 random intercept
        # residuals = sum of the squared residuals (type de loss function)
        # residual = observed - predicted point
        # sum of each residual^2
        # bonus ==> graph squared residuals
        # quand la courbe est au plus bas, on pick
            # c'est la qu'on utilise gradient descent, pour trouver le point EXACTEMENT le plus bas
            # optimise car grandes steps quand on en est loin, et petites quand on s'approche du but