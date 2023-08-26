import os
from utils_colors import Colors
import numpy as np

def predict(x, t0, t1):
    return (t0 + t1 * x)

def main():
    print(f"{Colors.INFO}[INFO]    Welcome to this car price prediction tool.{Colors.RES}")
    thetas = [0, 0]
    if os.path.isfile("thetas.npy"):
        thetas = np.load("thetas.npy")
    else:
        print(f"{Colors.WARNING}[WARNING] It has not been trained yet.{Colors.RES}")
    print(f"{Colors.INFO}[INFO]    Thetas are {thetas[0]} and {thetas[1]}{Colors.RES}\n")
    while True:
        try:
            prompt = input(">> Please, enter a mileage: ")
            if prompt == "exit":
                exit()
            x = int(prompt)
            if x < 0:
                print(f"{Colors.WARNING}[WARNING] This does not make any sense.{Colors.RES}")
            y = predict(x, thetas[0], thetas[1])
            print(f"{Colors.RES}The estimated price for a car that has a mileage of {Colors.GREEN}{x}{Colors.RES} is {Colors.GREEN}{y}{Colors.RES}.\n")
        except ValueError:
            print(f"{Colors.RED}Error: {Colors.RES}The mileage should be a natural number. If you are done, type \"exit\" to quit the program.\n")

if (__name__ == "__main__"):
    main()