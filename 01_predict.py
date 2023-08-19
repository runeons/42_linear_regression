import os
from utils_colors import Colors

def predict(x, t0, t1):
    return (t0 + t1 * x)

def parse_file():
    t0 = t1 = 0
    if os.path.isfile("thetas"):
        f = open("thetas", "r")
        thetas = f.read().split()
        if len(thetas) != 2:
            print(f"{Colors.RED}Error: {Colors.RES}Unexpected file content. Please train again.")
            exit()
        t0, t1 = map(float, thetas)
    return t0, t1

def main():
    print(f"{Colors.PREDICT}WELCOME to this car price prediction tool.{Colors.PREDICT}{Colors.RES}")
    t0, t1 = parse_file()
    print(f"{Colors.PREDICT}PS: The secrets thetas are {t0} and {t1}.{Colors.RES}\n")
    while True:
        try:
            prompt = input(">> Please, enter a mileage: ")
            if prompt == "exit":
                exit()
            x = int(prompt)
            if x < 0: # ca n'a pas de sens pratique (meme si le calcul est possible)
                raise ValueError
            y = predict(x, t0, t1)
            print(f"{Colors.RES}The estimated price for a car that has a mileage of {Colors.GREEN}{x}{Colors.RES} is {Colors.GREEN}{y}{Colors.RES}.\n")
        except ValueError:
            print(f"{Colors.RED}Error: {Colors.RES}The mileage must be a natural number. If you are done, type \"exit\" to quit the program.\n")

if (__name__ == "__main__"):
    main()