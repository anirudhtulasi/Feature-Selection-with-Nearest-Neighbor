import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

def load_data(file):
    return pd.read_csv(file, sep='\s+', header=None)

def leave_one_out_cross_validation(data, target):
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    for train_index, test_index in loo.split(data):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train.drop(target, axis=1))
        distances, indices = nbrs.kneighbors(test.drop(target, axis=1))
        y_true.append(test[target].values[0])
        y_pred.append(train.iloc[indices[0]][target].values[0])
    return accuracy_score(y_true, y_pred)

def forward_selection(data, target):
    features = list(data.columns)
    features.remove(target)
    selected_features = []
    print(f"Running nearest neighbor with all {len(features)} features, using \"leaving-one-out\" evaluation, I get accuracy of {leave_one_out_cross_validation(data, target)*100:.1f}%")
    print("Beginning search.")
    while len(features) > 0:
        best_accuracy = 0
        best_feature = None
        for feature in features:
            temp_features = selected_features + [feature]
            accuracy = leave_one_out_cross_validation(data[temp_features + [target]], target)
            print(f"Using feature(s) {temp_features} accuracy is {accuracy*100:.1f}%")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature
        if best_feature:
            selected_features.append(best_feature)
            features.remove(best_feature)
            print(f"Feature set {selected_features} was best, accuracy is {best_accuracy*100:.1f}%")
        else:
            break
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
