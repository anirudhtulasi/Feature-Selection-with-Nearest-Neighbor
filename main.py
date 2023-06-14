import numpy as np
import time
def load_data(filename):
    return np.loadtxt(filename)

def leave_one_out_cross_validation(data, feature_set):
    data_feature_set = data[:, feature_set]
    num_instances = data.shape[0]
    labels = data[:, 0]
    correct_predictions = 0
    
    for i in range(num_instances):
        #eucdistances
        distances = np.sqrt(np.sum((data_feature_set - data_feature_set[i])**2, axis=1))
        distances[i] = np.inf  #avoid same instance
        nearest_neighbor_index = np.argmin(distances)
        if labels[nearest_neighbor_index] == labels[i]:
            correct_predictions += 1

    return correct_predictions / num_instances



def forward_selection(data):
    features = list(range(1, data.shape[1]))
    selected_features = []
    best_accuracy_so_far = 0
    best_features_so_far = []

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
        if best_accuracy_this_level >= best_accuracy_so_far:
            if best_accuracy_this_level > best_accuracy_so_far:
                print(f"\nFeature set {selected_features} was best, accuracy is {best_accuracy_this_level*100:.1f}%\n")
            best_accuracy_so_far = best_accuracy_this_level
            best_features_so_far = selected_features.copy()
        else:
            print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n")

    print(f"Finished search!! The best feature subset is {best_features_so_far}, which has an accuracy of {best_accuracy_so_far*100:.1f}%")
    return best_features_so_far, best_accuracy_so_far

def main():
    print("Welcome to Bertie Woosters Feature Selection Algorithm.")
    file = input("Type in the name of the file to test : ")
    data = load_data(file)
    start_time = time.time()
    print(f"This dataset has {data.shape[1] - 1} features (not including the class attribute), with {data.shape[0]} instances.")
    print("Starting Forward Selection...")
    #best_features = forward_selection(data)
    best_features, best_accuracy = forward_selection(data)
    #print(f"Finished search!! The best feature subset is {best_features}, which has an accuracy of {leave_one_out_cross_validation(data, best_features):.3f}")
    error_rate = 1 - best_accuracy
    print(f"\nOn '{file}', the error rate is {error_rate:.3f} when using only features {best_features}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()
