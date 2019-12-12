import os
from PIL import Image
import numpy as np
from random import randint
from matplotlib import pyplot as plt
import math

num_of_classes = 10
epochs = 500
add_ones = False
number_of_iterations = 30

def read_images(folder: str):
    np_images = [[] for filename in os.listdir(folder) if filename.endswith(".jpg")]

    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            index = int(filename.split(".")[0]) - 1
            image = Image.open(os.path.join(folder, filename))
            np_image = np.intc(image).flatten()

            if add_ones:
                np_image = np.append(np_image, 1)

            del np_images[index]
            np_images.insert(index, np_image)

    with open(os.path.join(folder, "Labels.txt"), "r") as f:
        labels = [int(label.strip()) for label in f.readlines()]

    return np_images, labels

def gray_scale_to_binary(np_images):
    for np_image in np_images:
        for i in range(np_image.shape[0]):
                np_image[i] = 0 if np_image[i] < 140 else 255
    return np_images

def get_initial_means(dataset, k):
    initial_means = [[] for i in range(k)]
    previous_cluster_mean = dataset[randint(0, len(dataset))]
    initial_means[0] = previous_cluster_mean

    for j in range(1, k):
        current_cluster_mean = get_farthest_point(previous_cluster_mean, dataset, initial_means)
        initial_means[j] = current_cluster_mean
        previous_cluster_mean = current_cluster_mean

    return np.array(initial_means)

def get_farthest_point(origin_point, dataset, means):
    distance_to_farthest_point = 0
    farthest_point_index = 0 # This value does not really matter

    for i, data_point in enumerate(dataset):
        distance_from_origin = np.linalg.norm(origin_point - data_point)
        if (distance_from_origin > distance_to_farthest_point):
            if all(not np.array_equal(data_point, mean) for mean in means):
                distance_to_farthest_point = distance_from_origin
                farthest_point_index = i

    return dataset[farthest_point_index]

# def get_distance(origin_point, destination_point):
#     # Not sure if this is the right way to calculate distances
#
#     distance_vector = origin_point - destination_point
#
#     sum = 0
#     for dimension in distance_vector:
#         sum += dimension ** 2
#
#     return sum

def k_means_clustering(dataset, labels, k):
    means = get_initial_means(dataset, k)
    points_belonging_to_each_cluster = [[] for i in range(k)]

    while True:
        for j, data_point in enumerate(dataset):
            distance_to_nearest_mean = np.linalg.norm(data_point - means[0])
            index_of_nearest_mean = 0

            for i, mean in enumerate(means):
                distance_to_mean = np.linalg.norm(data_point - mean)

                if distance_to_mean < distance_to_nearest_mean:
                    distance_to_nearest_mean = distance_to_mean
                    index_of_nearest_mean = i

            points_belonging_to_each_cluster[index_of_nearest_mean].append((j, data_point))

        new_means = update_means(means, points_belonging_to_each_cluster)
        if all_zeros(np.array(new_means) - np.array(means)):
            break

        means = new_means

    return compute_mean_for_each_clusters(points_belonging_to_each_cluster)
    # return points_belonging_to_each_cluster

def update_means(old_means, points_belonging_to_each_cluster):
    for i, cluster in enumerate(points_belonging_to_each_cluster):
        sum = np.zeros(shape=784)
        for _, data_point in cluster:
            sum += data_point
        if any(dimension != 0 for dimension in sum):
            # If there were some points in that cluster, update its mean
            old_means[i] = sum / len(points_belonging_to_each_cluster[i])
    return old_means

def all_zeros(np_array):
    for row in np_array:
        for value in row:
            if value != 0:
                return False
    return True

def compute_mean_for_each_clusters(clusters):
    clusters_with_means = [() for cluster in clusters]

    for i, cluster in enumerate(clusters):
        mean_for_current_cluster = np.zeros(shape=784)
        for _, data_point in cluster:
            mean_for_current_cluster += data_point

        mean_for_current_cluster /= len(cluster)
        clusters_with_means[i] = (mean_for_current_cluster, cluster)

    return clusters_with_means

def distance_between_points_and_their_means(clusters_with_means):
    # distances = [0 for cluster in clusters_with_means]
    sum_of_distances_between_each_cluster_and_its_mean = 0
    for i, (mean, cluster) in enumerate(clusters_with_means):

        for _, data_point in cluster:
            # distances[i] +=np.linalg.norm(mean - data_point)
            sum_of_distances_between_each_cluster_and_its_mean + np.linalg.norm(mean - data_point)

    return sum_of_distances_between_each_cluster_and_its_mean

# def find_index_of_cluster_for_point(clusters_with_means, dataset):
#     partition_size = 240
#     offset = 0
#     max_count_
#     for i, element in enumerate(dataset):
#         if np.array_equal(element, data_point):
#             return i
#
#     raise AssertionError("Could not find the data point in the dataset")

def find_max_count_for_each_cluster(clusters_with_means):
    partition_size = 240
    offset = 0

    final_counts = np.zeros(shape=10)
    counts_of_points_for_each_digit = np.zeros(shape=10)

    for digit in range(10):
        for i, (_, cluster) in enumerate(clusters_with_means):
            for index_of_point, point in cluster:
                if offset < index_of_point < offset + partition_size:
                    counts_of_points_for_each_digit[i] += 1
        final_counts[offset // partition_size] = np.max(counts_of_points_for_each_digit)
        counts_of_points_for_each_digit = np.zeros(shape=10)
        offset += partition_size

    return final_counts
def main():
    dataset, data_labels = read_images("./images")
    dataset = gray_scale_to_binary(dataset)

    min_sum_of_distances_between_each_cluster_and_its_mean = math.inf
    for i in range(number_of_iterations):
        clusters_with_means = k_means_clustering(dataset, data_labels, 10)
        sum_of_distances_between_each_cluster_and_its_mean = distance_between_points_and_their_means(clusters_with_means)
        if min_sum_of_distances_between_each_cluster_and_its_mean > sum_of_distances_between_each_cluster_and_its_mean:
            clusters_with_min_sum_of_distances = clusters_with_means
            min_sum_of_distances_between_each_cluster_and_its_mean = sum_of_distances_between_each_cluster_and_its_mean

    counts = find_max_count_for_each_cluster(clusters_with_min_sum_of_distances)
    x_values = [i for i in range(10)]

    plt.barh(x_values, counts, align='center', alpha=0.5)
    plt.show()
    print("finished")
if __name__ == '__main__':
    main()
