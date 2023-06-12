import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

def load_data(file):
    return pd.read_csv(file, sep='\s+', header=None)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def nearest_neighbor(train, test_instance):
    distances = train.apply(lambda x: euclidean_distance(x[1:], test_instance[1:]), axis=1)
    return train.loc[distances.idxmin()][0]

def leave_one_out_cross_validation(data, target):
    loo = LeaveOneOut()
    predictions = []
    for train_index, test_index in loo.split(data):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        prediction = nearest_neighbor(train, test.iloc[0])
        predictions.append(prediction)
    return accuracy_score(data[target], predictions)

def forward_selection(data, target):
    features = list(data.columns)
    features.remove(target)
    selected_features = []
    print(f"Running nearest neighbor with all {len(features)} features, using \"leaving-one-out\" evaluation, I get accuracy of {leave_one_out_cross_validation(data, target)*100:.1f}%")
    print("Beginning search.")
    while len(features) > 0:
        best_accuracy = 0
        best_feature = None  #best_feature
        for feature in features:
            temp_features = selected_features + [feature]
            accuracy = leave_one_out_cross_validation(data[temp_features + [target]], target)
            print(f"Using feature(s) {temp_features} accuracy is {accuracy*100:.1f}%")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature
        if best_feature:  #append and remove if best_feature is not None
            selected_features.append(best_feature)
            features.remove(best_feature)
            print(f"Feature set {selected_features} was best, accuracy is {best_accuracy*100:.1f}%")
        else:
            break  #if no imprv in accuracy - break
    return selected_features

def main():
    print("Welcome to Bertie Woosters Feature Selection Algorithm.")
    file = input("Type in the name of the file to test : ")
    algorithm = int(input("Type the number of the algorithm you want to run.\n1) Forward Selection\n2) Backward Elimination\n"))

    data = load_data(file)
    target = 0
    if algorithm == 1:
        print(f"This dataset has {data.shape[1] - 1} features (not including the class attribute), with {data.shape[0]} instances.")
        print("Starting Forward Selection...")
        selected_features = forward_selection(data, target)
        print(f"Finished search!! The best feature subset is {selected_features}, which has an accuracy of {leave_one_out_cross_validation(data[selected_features + [target]], target)*100:.1f}%")
    else:
        print("Backward Elimination is not implemented in this script.")

if __name__ == "__main__":
    main()
