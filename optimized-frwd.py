import numpy as np
import time

def load_data(filename):
    return np.loadtxt(filename)

def calculate_distances(feature_set, data):
    distances_new = np.zeros((data.shape[0], data.shape[0]))
    for feature in feature_set:
        distances_new += (data[:, feature].reshape(-1, 1) - data[:, feature]) ** 2
    return np.sqrt(distances_new)

def leave_one_out_cross_validation(distances, labels):
    np.fill_diagonal(distances, np.inf)
    nearest_neighbors = np.argmin(distances, axis=1)
    predictions = labels[nearest_neighbors]
    accuracy = np.mean(predictions == labels)
    return accuracy

def forward_selection(data):
    features = list(range(1, data.shape[1]))
    selected_features = []
    best_accuracy_so_far = 0
    best_features_so_far = []
    labels = data[:, 0]

    print(f"Running nearest neighbor with all {len(features)} features, using \"leaving-one-out\" evaluation, I get accuracy of {leave_one_out_cross_validation(calculate_distances(features, data), labels)*100:.1f}%\n")

    while len(features) > 0:
        best_accuracy_this_level = 0
        best_feature_this_level = None

        for feature in features:
            current_features = selected_features + [feature]
            current_distances = calculate_distances(current_features, data)
            accuracy = leave_one_out_cross_validation(current_distances, labels)

            print(f"Using feature(s) {current_features} accuracy is {accuracy*100:.1f}%")

            if accuracy > best_accuracy_this_level:
                best_accuracy_this_level = accuracy
                best_feature_this_level = feature
                best_distances_this_level = current_distances

        if best_feature_this_level is not None:
            selected_features.append(best_feature_this_level)
            features.remove(best_feature_this_level)

        print(f"Progress: {len(selected_features)}/{len(selected_features) + len(features)} features selected.\n")

        if best_accuracy_this_level > best_accuracy_so_far:
            print(f"\nFeature set {selected_features} was best, accuracy is {best_accuracy_this_level*100:.1f}%\n")
            best_accuracy_so_far = best_accuracy_this_level
            best_features_so_far = selected_features.copy()
        else:
            print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n")

    print(f"Finished search!! The best feature subset is {best_features_so_far}, which has an accuracy of {best_accuracy_so_far*100:.1f}%")

    return best_features_so_far, best_accuracy_so_far

def backward_elimination(data):
    features = list(range(1, data.shape[1]))
    labels = data[:, 0]
    best_accuracy_so_far = leave_one_out_cross_validation(calculate_distances(features, data), labels)

    print(f"Running nearest neighbor with all {len(features)} features, using \"leaving-one-out\" evaluation, I get accuracy of {best_accuracy_so_far*100:.1f}%\n")

    while len(features) > 1:
        best_accuracy_this_level = 0
        worst_feature_this_level = None

        for feature in features:
            current_features = features.copy()
            current_features.remove(feature)
            current_distances = calculate_distances(current_features, data)
            accuracy = leave_one_out_cross_validation(current_distances, labels)

            print(f"Using feature(s) {current_features} accuracy is {accuracy*100:.1f}%")

            if accuracy > best_accuracy_this_level:
                best_accuracy_this_level = accuracy
                worst_feature_this_level = feature

        if worst_feature_this_level is not None:
            features.remove(worst_feature_this_level)

        print(f"Progress: {len(features)} features left.\n")

        if best_accuracy_this_level > best_accuracy_so_far:
            print(f"\nFeature set {features} was best, accuracy is {best_accuracy_this_level*100:.1f}%\n")
            best_accuracy_so_far = best_accuracy_this_level
        else:
            print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n")

    print(f"Finished search!! The best feature subset is {features}, which has an accuracy of {best_accuracy_so_far*100:.1f}%")

    return features, best_accuracy_so_far

def main():
    print("Welcome to Bertie Woosters Feature Selection Algorithm.")
    file = input("Type in the name of the file to test : ")
    data = load_data(file)
    start_time = time.time()
    print(f"This dataset has {data.shape[1] - 1} features (not including the class attribute), with {data.shape[0]} instances.")
    print("Please select the search algorithm: 1 - Forward Selection, 2 - Backward Elimination")
    choice = int(input())
    if choice == 1:
        print("Starting Forward Selection...")
        best_features, best_accuracy = forward_selection(data)
    elif choice == 2:
        print("Starting Backward Elimination...")
        best_features, best_accuracy = backward_elimination(data)
    error_rate = 1 - best_accuracy
    print(f"\nThe best feature subset is {best_features}, which has an error rate of {error_rate:.3f}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()
