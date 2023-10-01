import os
import pandas as pd
import matplotlib.pyplot as plt
from utils_colors import Colors

class Data:
    def __init__(self, data_path, x, y, image_path="./images/", fig_size=(10, 7)):
        self.data_path = data_path
        self.full_data = self.load()
        self.x = self.full_data[x]
        self.y = self.full_data[y]
        self.fig_size = fig_size
        self.image_path = image_path
        os.makedirs(image_path, exist_ok=True)

    def load(self):
        return pd.read_csv(self.data_path)

    def summary(self, d_set):
        print(f"{Colors.DATASET}[INFO] Dataset summary - head:\n{Colors.RES}{d_set.head()}")
        print(f"{Colors.DATASET}[INFO] Dataset summary - count:\n{Colors.RES}{d_set.count()}")
        print(f"{Colors.DATASET}[INFO] Dataset summary - min:\n{Colors.RES}{d_set.min()}")
        print(f"{Colors.DATASET}[INFO] Dataset summary - max:\n{Colors.RES}{d_set.max()}")

    def save_plot(self, name):
        path = os.path.join(self.image_path, name + ".png")
        plt.savefig(path, format="png", dpi=300)
        print(f"{Colors.GREEN}[INFO] Plot {name} saved{Colors.RES}")

    def scatter(self, x, y, name, save=True):
        plt.figure()
        plt.plot(x, y, 'o', alpha=0.5, color='green')
        plt.title("Initial data")
        plt.xlabel(self.x.name)
        plt.ylabel(self.y.name)
        if save == True:
            self.save_plot(name + "_scatter")
        else:
            plt.show()

    def histogram(self, data, bins, name, save=True):
        plt.figure()
        data.hist(bins=bins, figsize=self.fig_size, color="#B15B7C")
        if save == True:
            self.save_plot(name + "_histogram")
        else:
            plt.show()
