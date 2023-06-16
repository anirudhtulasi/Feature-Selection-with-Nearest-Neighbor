import numpy as np

def load_data(filename):
    return np.loadtxt(filename)

def calculate_distances(feature, data):
    return np.abs(data[:, feature].reshape(-1, 1) - data[:, feature]) ** 2

def leave_one_out_cross_validation(data, feature_set):
    data_feature_set = data[:, feature_set]
    num_instances = data.shape[0]
    labels = data[:, 0]
    correct_predictions = 0
    
    for i in range(num_instances): #eucdistances
        distances = np.sqrt(np.sum((data_feature_set - data_feature_set[i])**2, axis=1))
        distances[i] = np.inf #avoid same instance
        nearest_neighbor_index = np.argmin(distances)
        if labels[nearest_neighbor_index] == labels[i]:
            correct_predictions += 1

    return correct_predictions / num_instances
def leave_one_out_cross_validation_optimized(distances,labels):
    np.fill_diagonal(distances, np.inf)
    nearest_neighbors = np.argmin(distances, axis=1)
    predictions = labels[nearest_neighbors]
    accuracy = np.mean(predictions == labels)
    return accuracy
