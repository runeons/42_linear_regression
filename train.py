from data import Data
from linear_regression import LinearRegression

def main():
    try:
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

        # Save coefficients
        lr.save()
    except Exception as e:
        print(e)

if (__name__ == "__main__"):
    main()
