from utils_colors import Colors
from data import Data
from linear_regression import LinearRegression

def main():
    # Data exploration
    dt = Data("dataset/data.csv", "km", "price")
    dt.summary(dt.full_data)
    dt.scatter(dt.full_data["km"], dt.full_data["price"], "initial")
    dt.histogram(dt.full_data, 6, "initial")

    # Linear regression
    lr = LinearRegression(dt.x, dt.y)
    lr.launch()
    lr.plot_line()

    # Analysis
    lr.simple_plot(lr.cost, "Cost function", "iterations", "mean squared error")
    lr.simple_plot(lr.conv, "Convergence", "iterations", "convergence")
    lr.evaluate()

    # Save to predict
    lr.save()

    test = 42000
    print(f"{Colors.RES}The estimated price for a car that has a mileage of {Colors.GREEN}{test}{Colors.RES} is {Colors.GREEN}{lr.predict(test, lr.t)}{Colors.RES}.\n")


if (__name__ == "__main__"):
    main()
