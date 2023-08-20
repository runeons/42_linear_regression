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
        self.normed_full_data = self.normalise_full_data()
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

    def norm_min_max(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    def norm_max(self, x):
        return x / x.max()
    
    def normalise_train_set(self):
        return self.norm_min_max(self.train_set)

    def normalise_full_data(self):
        return self.norm_min_max(self.full_data)

def to_file(t0, t1):
    s = f"{t0} {t1}"
    f = open("thetas", "w+")
    f.write(s)
    f.close()

# def cost_plot(summary):
#     plt.figure(figsize=(10, 6))
#     plt.plot(summary['iteration'], summary['squared_residuals'], marker='o')
#     plt.grid(True)
#     plt.show()

class LinearRegression:
    def __init__(self, data):
        self.data = data
        self.alpha = 0.1
        self.thetas = [1, 1]
        self.kms = self.data["km"].to_numpy()
        self.prices = self.data["price"].to_numpy()
        self.m = len(self.kms)
        self.convergence = 0.000001
        self.cost_plot = []
    
    def plot_cost(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_plot)
        plt.grid(True)
        plt.show()        

    def predict(self, x, thetas):
        return thetas[0] + x * thetas[1]

    def get_new_thetas(self):
        tmp_thetas = [1, 1]
        cost = self.predict(self.kms, self.thetas) - self.prices
        self.cost_plot.append(np.sum(cost ** 2) / self.m)
        tmp_thetas[0] = self.thetas[0] - (self.alpha * np.sum(cost) / self.m)
        tmp_thetas[1] = self.thetas[1] - (self.alpha * np.sum(cost * self.kms) / self.m)
        return tmp_thetas

    def launch(self):
        for i in range(2000):
            tmp_thetas = self.get_new_thetas()
            if (i % 100 == 0):
                print(i, self.thetas, tmp_thetas)
            if (np.abs(np.array(tmp_thetas) - np.array(self.thetas)) < self.convergence).all(): # attention arrondis
                break
            self.thetas = tmp_thetas
        return self.thetas

def main():
    d = Data("dataset/data.csv")
    d.summary(d.train_set)
    # d.scatter(d.train_set, "km", "price", "init train_set")
    # d.histogram(d.train_set, 6, "init train_set")
    # d_full = d.normed_full_data
    d_train = d.normed_train_set.reset_index()

    lr = LinearRegression(d_train)
    thetas = lr.launch()
    x = 42000
    y = thetas[0] + thetas[1] * x
    # y = y * (d.train_set["price"].max() - d.train_set["price"].min()) + d.train_set["price"].min()  # Dénormalise le prix
    print(thetas)
    print(f"{Colors.RES}The estimated price for a car that has a mileage of {Colors.GREEN}{x}{Colors.RES} is {Colors.GREEN}{y}{Colors.RES}.\n")
    to_file(thetas[0], thetas[1])
    lr.plot_cost()
    # np.save("thetas2", thetas)
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