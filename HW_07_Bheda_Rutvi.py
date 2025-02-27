"""
Rutvi Bheda
BDA Homework 7
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans


def cross_correlation(data):
    """
    function to calculate the cross correlation for the given dataset
    """
    without_class = data.drop("ID", axis=1)
    correlation = without_class.corr().round(2)
    return correlation


def read_file():
    """
    function to read the data to a csv file
    """
    directory = (r'C:\Users\rutvi\OneDrive\Desktop\Sem 3\720. Big Data Analytics\homework '
                 r'7\HW_CLUSTERING_SHOPPING_CART_v2235b.csv')
    data = pd.read_csv(directory)
    without_id_data = data.iloc[:, 1:]
    new_without_id_data = without_id_data.values
    return new_without_id_data, data


def cluster_details(data):
    """
    initializing the important cluster details needed for agglomeration
    """
    cluster = {}
    centers = {}
    cluster_size = {}

    for i in range(len(data)):
        cluster[i] = [i]

    for i in range(len(data)):
        centers[i] = data[i]

    for i in range(len(data)):
        cluster_size[i] = 1

    return cluster, centers, cluster_size


def calculate_manhattan_distance(a, b):
    """
    calculation of manhattan distance between the data points
    """
    dist = np.sum(np.abs(a - b))
    return dist


def track_distance(data):
    distance_matrix = {}
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distance_matrix[(i, j)] = calculate_manhattan_distance(data[i], data[j])

    # can print distances here to check
    # for p, d in distance_matrix.items():
    #     print(f"distance between point {p[0]} and {p[1]} is : {d}")
    return distance_matrix


def agglomerative_clustering(data, cluster, distance_matrix, cluster_center, cluster_size):
    """
    function to perform agglomerative clustering
    """
    smallest_cluster_size = []
    new_cluster_id = len(data)
    linkage = []
    while len(cluster) > 1:
        (c1, c2), d = min(distance_matrix.items(), key=lambda x: x[1])

        if c1 > c2:
            c1, c2 = c2, c1

        small_size_temp = min(len(cluster[c1]), len(cluster[c2]))
        smallest_cluster_size.append(small_size_temp)
        if len(smallest_cluster_size) > 20:
            smallest_cluster_size.pop(0)

        print(f"merged clusters {c1} and {c2} with distance {d} and size of the smallest is {small_size_temp}")

        new_cluster = cluster[c1] + cluster[c2]
        new_center = np.mean([data[index] for index in new_cluster], axis=0)
        cluster[new_cluster_id] = new_cluster
        cluster_center[new_cluster_id] = new_center
        cluster_size[new_cluster_id] = len(new_cluster)

        linkage.append([c1,c2, d, len(new_cluster)])

        del cluster[c1], cluster[c2]
        del cluster_center[c1], cluster_center[c2]
        del cluster_size[c1], cluster_size[c2]

        remaining = list(cluster.keys())

        for d in remaining:
            if d != new_cluster_id:
                distance_matrix[min(new_cluster_id, d), max(d, new_cluster_id)] = calculate_manhattan_distance(cluster_center[new_cluster_id],
                                                                                       cluster_center[d])

        distance_matrix = {(x, y): dist for (x, y), dist in distance_matrix.items() if c2 not in [x, y] and c1 not in [x, y]}

        new_cluster_id = new_cluster_id + 1

    return smallest_cluster_size, linkage


def create_dendrogram(linkage):
    """
    function to create a dendrogram for the last 20 clusters
    """
    linkage_data = np.array(linkage)
    plt.figure(figsize = (10,7))
    dendrogram(linkage_data, truncate_mode='lastp', p = 20 , leaf_rotation=90., leaf_font_size=10., show_contracted=True)
    plt.title('Dendrogram for Agglomerative Clustering')
    plt.xlabel('Clusters')
    plt.ylabel('Distance')
    plt.show()


def perform_KMeans(data):
    """
    function to perform K Means Clustering
    """
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_

    for x in range(k):
        print(f"K-Means cluster {x + 1} is of size {np.sum(labels == x)}")


def main():
    data, file_data = read_file()
    # print(data)
    correlation = cross_correlation(file_data)
    print(correlation)

    cluster, centers, cluster_size = cluster_details(data)

    distance_matrix = track_distance(data)
    smallest_cluster_size, linkage = agglomerative_clustering(data, cluster, distance_matrix, centers, cluster_size)
    print("size of the last 10 smallest clusters merged", smallest_cluster_size[-10:])
    create_dendrogram(linkage)

    perform_KMeans(data)


if __name__ == '__main__':
    main()
