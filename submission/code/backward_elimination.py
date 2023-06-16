import numpy as np
from utils import calculate_distances, leave_one_out_cross_validation

def backward_elimination(data):
    features = list(range(1, data.shape[1]))
    selected_features = features.copy()
    best_accuracy_so_far = 0
    best_features_so_far = []

    best_three_features_accuracy = 0
    best_three_features = []

    print(f"Running nearest neighbor with all {len(features)} features, using \"leaving-one-out\" evaluation, I get accuracy of {leave_one_out_cross_validation(data, features)*100:.1f}%\n")

    while len(selected_features) > 1:
        best_accuracy_this_level = 0
        worst_feature_this_level = None

        for feature in selected_features:
            current_features = selected_features.copy()
            current_features.remove(feature)
            accuracy = leave_one_out_cross_validation(data, current_features)

            print(f"Using feature(s) {current_features} accuracy is {accuracy*100:.1f}%")

            if accuracy > best_accuracy_this_level:
                best_accuracy_this_level = accuracy
                worst_feature_this_level = feature

        if worst_feature_this_level is not None:
            selected_features.remove(worst_feature_this_level)

        print(f"Progress: feature size is : {len(selected_features)}.\n")

        if len(selected_features) == 3:
            best_three_features_accuracy = best_accuracy_this_level
            best_three_features = selected_features.copy()

        if best_accuracy_this_level >= best_accuracy_so_far:
            if best_accuracy_this_level > best_accuracy_so_far:
                print(f"\nFeature set {selected_features} was best, accuracy is {best_accuracy_this_level*100:.1f}%\n")
            best_accuracy_so_far = best_accuracy_this_level
            best_features_so_far = selected_features.copy()
        else:
            print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n")

    print(f"Finished search!! The best feature subset is {best_features_so_far}, which has an accuracy of {best_accuracy_so_far*100:.1f}%\n")
    error_rate = 1 - best_accuracy_so_far
    print(f"The error rate is {error_rate:.3f} when using only features {best_features_so_far}")

    three_features_error_rate = 1 - best_three_features_accuracy
    #print(f"The error rate is {three_features_error_rate:.3f} when using the three features {best_three_features}")

    return best_features_so_far, best_accuracy_so_far, best_three_features, best_three_features_accuracy
