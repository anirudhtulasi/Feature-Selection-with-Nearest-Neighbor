import numpy as np
from utils import calculate_distances, leave_one_out_cross_validation_optimized

def optimized_backward_elimination(data):
    features = list(range(1, data.shape[1]))
    selected_features = features.copy()
    best_accuracy_so_far = 0
    best_features_so_far = []
    labels = data[:, 0]
    distances = sum(calculate_distances(feature, data) for feature in features)

    best_three_features_accuracy = 0
    best_three_features = []

    print(f"Running nearest neighbor with all {len(features)} features, using \"leaving-one-out\" evaluation, I get accuracy of {leave_one_out_cross_validation_optimized(distances, labels)*100:.1f}%\n")

    while len(selected_features) > 1:
        best_accuracy_this_level = 0
        worst_feature_this_level = None

        for feature in selected_features:
            current_features = selected_features.copy()
            current_features.remove(feature)
            current_distances = distances - calculate_distances(feature, data)
            accuracy = leave_one_out_cross_validation_optimized(current_distances, labels)

            print(f"Using feature(s) {current_features} accuracy is {accuracy*100:.1f}%")

            if accuracy > best_accuracy_this_level:
                best_accuracy_this_level = accuracy
                worst_feature_this_level = feature
                best_distances_this_level = current_distances

        if worst_feature_this_level is not None:
            selected_features.remove(worst_feature_this_level)
            distances = best_distances_this_level

            print(f"Progress: feature size is : {len(selected_features)}\n")

            if len(selected_features) == 3:
                best_three_features_accuracy = best_accuracy_this_level
                best_three_features = selected_features.copy()

            if best_accuracy_this_level > best_accuracy_so_far:
                print(f"Feature set {selected_features} was best, accuracy is {best_accuracy_this_level*100:.1f}%")
                best_accuracy_so_far = best_accuracy_this_level
                best_features_so_far = selected_features.copy()

    print(f"The best feature subset is {best_features_so_far}, which has an accuracy of {best_accuracy_so_far*100:.1f}%")
    #print(f"The best three-feature subset is {best_three_features}, which has an accuracy of {best_three_features_accuracy*100:.1f}%")

    return best_features_so_far, best_accuracy_so_far
