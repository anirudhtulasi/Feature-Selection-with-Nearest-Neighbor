import numpy as np
from utils import calculate_distances, leave_one_out_cross_validation

def forward_selection(data):
    features = list(range(1, data.shape[1]))
    selected_features = []
    best_accuracy_so_far = 0
    best_features_so_far = []

    best_three_features_accuracy = 0
    best_three_features = []

    print(f"Running nearest neighbor with all {len(features)} features, using \"leaving-one-out\" evaluation, I get accuracy of {leave_one_out_cross_validation(data, features)*100:.1f}%\n")

    while len(features) > 0:
        best_accuracy_this_level = 0
        best_feature_this_level = None

        for feature in features:
            current_features = selected_features + [feature]
            accuracy = leave_one_out_cross_validation(data, current_features)

            print(f"Using feature(s) {current_features} accuracy is {accuracy*100:.1f}%")

            if accuracy > best_accuracy_this_level:
                best_accuracy_this_level = accuracy
                best_feature_this_level = feature

        if best_feature_this_level is not None:
            selected_features.append(best_feature_this_level)
            features.remove(best_feature_this_level)

        print(f"Progress: {len(selected_features)}/{len(selected_features) + len(features)} features selected.\n")

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
    print(f"The error rate is {error_rate:.3f} when using only features {best_features_so_far}\n")

    three_features_error_rate = 1 - best_three_features_accuracy
    print(f"The error rate is {three_features_error_rate:.3f} when using the three features {best_three_features}\n")

    return best_features_so_far, best_accuracy_so_far, best_three_features, best_three_features_accuracy
