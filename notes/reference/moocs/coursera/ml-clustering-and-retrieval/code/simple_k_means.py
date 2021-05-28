import pprint

import math

datapoints = dict(
    dp1=(-1.88, 2.05),
    dp2=(-0.71, 0.42),
    dp3=(2.41, -0.67),
    dp4=(1.85, -3.80),
    dp5=(-3.69, -1.33)
)

clusters = dict(
    cluster_1=(2., 2.),
    cluster_2=(-2., -2.)
)


def calc_distance(datapoint, cluster):
    return math.sqrt(sum([(c - d)**2 for c, d in zip(cluster, datapoint)]))


def revise_cluster_centers(assigned_cluster, clusters):
    output = {}
    for cluster_name, dp_names in assigned_cluster.items():
        cluster_result = []
        for dp_index in range(len(clusters[cluster_name])):
            dp_index_total = 0.
            for dp_name in dp_names:
                dp_index_total += datapoints[dp_name][dp_index]
            dp_index_total = dp_index_total / len(dp_names)
            cluster_result.append(dp_index_total)

        output[cluster_name] = tuple(cluster_result)

    return output


for iter_count in range(0, 5):
    assigned_clusters = dict(cluster_1=[], cluster_2=[])

    for datapoint, values in datapoints.items():
        lowest_dist = float('inf')
        assigned_cluster = None
        for cluster, cluster_values in clusters.items():
            distance = calc_distance(values, cluster_values)
            if distance < lowest_dist:
                lowest_dist = distance
                assigned_cluster = cluster

        assigned_clusters[assigned_cluster].append(datapoint)

    clusters = revise_cluster_centers(assigned_clusters, clusters)
    print "Assigned clusters", assigned_clusters
