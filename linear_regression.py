import os
import numpy as np
import matplotlib.pyplot as plt
from utils_colors import Colors

class LinearRegression:
    def __init__(self, x, y):
        self.init_x = x
        self.init_y = y
        self.max_x = x.max()
        self.max_y = y.max()
        self.x = x
        self.y = y
        self.t = [0.0, 0.0]
        self.alpha = 0.1
        self.max_epochs = 1000000
        self.convergence = 0.0000000001
        self.m = len(self.x)
        self.cost = []
        self.conv = []

    def norm_x_y(self):
        self.x = self.x / self.x.max()
        self.y = self.y / self.y.max()

    def predict(self, x, thetas):
        return thetas[0] + x * thetas[1]

    def find_thetas(self):
        for i in range(self.max_epochs):
            predicted_y = self.predict(self.x, self.t)
            d_1 = np.sum(((predicted_y - self.y) * self.x)) / self.m
            d_0 = np.sum(((predicted_y - self.y))) / self.m
            self.t = [self.t[0] - self.alpha * d_0, self.t[1] - self.alpha * d_1]
            cost = np.sum(((predicted_y - self.y) ** 2) * (2 / self.m))
            self.cost.append(cost)
            if i != 0:
                conv = self.cost[-1] - self.cost[-2]
                self.conv.append(conv)
                if abs(conv) < self.convergence:
                    print(f"{Colors.BLUE}[INFO] The gradient descent finished in {i} steps{Colors.RES}")
                    break
                if i % 100 == 0:
                    print(f"{Colors.GREY}[INFO] iteration {i}: cost = {cost}, conv = {conv}, thetas = {self.t}{Colors.RES}")
                
    def renorm_thetas(self):
        self.t[0] = self.t[0] * self.max_y
        self.t[1] = self.t[1] * (self.max_y / self.max_x)

    def launch(self):
        self.norm_x_y()
        self.find_thetas()
        self.renorm_thetas()

    def save(self):
        np.save("thetas", self.t)

    def evaluate(self):
        predicted_y = self.predict(self.init_x, self.t)
        ss_reg = np.sum(((predicted_y - np.mean(self.init_y)) ** 2))
        ss_tot = np.sum(((self.init_y - np.mean(self.init_y)) ** 2))
        rsquared = ss_reg / ss_tot
        print(f"{Colors.BLUE}[INFO] The coefficient of determination is {rsquared * 100:.2f}%{Colors.RES}")

    def save_plot(self, name):
        path = os.path.join("./images/", name + ".png")
        plt.savefig(path, format="png", dpi=300)
        print(f"{Colors.GREEN}[INFO] Plot {name} saved{Colors.RES}")

    def plot_line(self, save=True):
        plt.figure()
        plt.plot(self.init_x, self.init_y, 'o', alpha=0.5, color='green')
        plt.plot(self.init_x, self.t[1] * self.init_x + self.t[0])
        plt.title("Linear function")
        plt.xlabel("kms")
        plt.ylabel("prices")
        if save == True:
            self.save_plot("linear_regression")
        else:
            plt.show()
    
    def simple_plot(self, val_array, title, xlabel, ylabel, save=True):
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(val_array)
        plt.grid(True)
        if save == True:
            self.save_plot(title)
        else:
            plt.show()        
    