import time
from utils import load_data
from forward_selection import forward_selection
from backward_elimination import backward_elimination
from optimized_forward_selection import optimized_forward_selection
from optimized_backward_elimination import optimized_backward_elimination

def main():
    print("Welcome to Bertie Woosters Feature Selection Algorithm.")
    print("Please select one : 1 - Synthetic Dataset, 2 - Real-World Dataset")
    a=int(input())
    if a==1:
        file = input("Type in the name of the file to test: ")
    else:
        print("Please select one : 1 - Red wine dataset, 2 - White wine dataset")
        p=int(input())
        if p==1:
            file="./preprocessed_winequality-red.csv"
        else:
            file="./preprocessed_winequality-white.csv"
    data = load_data(file)
    print(f"This dataset has {data.shape[1] - 1} features (not including the class attribute), with {data.shape[0]} instances.")
    print("Please select the search algorithm: 1 - Forward Selection, 2 - Backward Elimination, 3 - Optimized Forward Selection, 4 - Optimized Backward Elimination")
    choice = int(input())
    start_time = time.time()
    if choice == 1:
        print("Starting Forward Selection...")
        best_features, best_accuracy, top_three_features, top_three_accuracy = forward_selection(data)
    elif choice == 2:
        print("Starting Backward Elimination...")
        best_features, best_accuracy, top_three_features, top_three_accuracy = backward_elimination(data)
    elif choice == 3:
        print("Starting Optimized Forward Selection...")
        best_features, best_accuracy = optimized_forward_selection(data)
    elif choice == 4:
        print("Starting Optimized Backward Elimination...")
        best_features, best_accuracy = optimized_backward_elimination(data)
    error_rate = 1 - best_accuracy
    #print(f"The best feature subset is {best_features}, which has an error rate of {error_rate:.3f}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time:0.3f} seconds")

if __name__ == "__main__":
    main()
