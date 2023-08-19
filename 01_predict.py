def estimated_price(x, t0, t1):
    return (t0 + t1 * x)

def main():
    t0 = t1 = 0
    print("Welcome to this car price prediction tool.\nWhen you are done, type \"exit\" to quit the program.\n")
    while True:
        try:
            prompt = input("Please, enter a mileage: ")
            if prompt == "exit":
                exit()
            x = int(prompt)
            if x < 0:
                raise ValueError
            y = estimated_price(x, t0, t1)
            print(f"The estimated price for a car that has a mileage of {x} is {y}.\n")
        except ValueError:
            print("The mileage must be a natural number.")

if (__name__ == "__main__"):
    main()