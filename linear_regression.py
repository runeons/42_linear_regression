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
        self.alpha = 0.1
        self.epochs = 1000000
        self.t = [0.0, 0.0]
        self.m = len(self.x)
        self.convergence = 0.000001
        self.cost = []
        self.conv = []
        self.theta0 = []
        self.theta1 = []

    def norm_max(self, x):
        return x / x.max()

    def norm_min_max(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def norm_x_y(self):
        self.x = self.norm_max(self.x)
        self.y = self.norm_max(self.y)

    def predict(self, x, thetas):
        return thetas[0] + x * thetas[1]

    def find_thetas(self):
        for i in range(self.epochs):
            predicted_y = self.predict(self.x, self.t)
            d_1 = np.sum(((predicted_y - self.y) * self.x)) / self.m
            d_0 = np.sum(((predicted_y - self.y))) / self.m
            self.t = [self.t[0] - self.alpha * d_0, self.t[1] - self.alpha * d_1]
            self.theta0.append(self.t[0])
            self.theta1.append(self.t[1])
            cost = np.sum(((predicted_y - self.y) ** 2) * (2 / self.m))
            self.cost.append(cost)
            if i != 0:
                conv = self.cost[-1] - self.cost[-2]
                self.conv.append(conv)
                if abs(conv) < self.convergence:
                    print(f"{Colors.BLUE}[INFO] gradient descent finished in {i} steps{Colors.RES}")
                    break
    
    def renorm_thetas(self):
        self.t[0] = self.t[0] * self.max_y
        self.t[1] = self.t[1] * (self.max_y / self.max_x)

    def launch(self):
        self.norm_x_y()
        self.find_thetas()
        self.renorm_thetas()

    def save(self):
        np.save("thetas", self.t)

    def plot_line(self):
        plt.plot(self.init_x, self.init_y, 'o', alpha=0.5, color='green')
        plt.plot(self.init_x, self.t[1] * self.init_x + self.t[0])
        plt.title("Linear function")
        plt.xlabel("kms")
        plt.ylabel("prices")
        plt.show()
    
    def plot_cost(self):
        plt.figure(figsize=(10, 6))
        plt.title("Cost function")
        plt.xlabel("iterations")
        plt.ylabel("mean squared error")
        plt.plot(self.cost)
        plt.grid(True)
        plt.show()        
    
    def plot_conv(self):
        plt.figure(figsize=(10, 6))
        plt.title("Convergence")
        plt.xlabel("iterations")
        plt.ylabel("convergence")
        plt.plot(self.conv)
        plt.grid(True)
        plt.show()        
    
    def plot_theta0(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.theta0)
        plt.grid(True)
        plt.show()        
    
    def plot_theta1(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.theta1)
        plt.grid(True)
        plt.show()        

    def evaluate(self):
        pass
        # sse = 
        # ssr = np.sum((self.predict(self.x, self.t) - self.y) ** 2)
        # r = (sst - ssr) / sst
        # print(r)