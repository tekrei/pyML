#!/usr/bin/python3
# coding=utf-8
"""
@author: tekrei
"""
import matplotlib.pyplot as plot
import numpy
from sklearn.cluster import KMeans as SKLKMeans

from utility import euclidean


class KMeans():

    def cluster(self, data, cluster_count, max_iteration=1000):
        data_count = data.shape[0]
        # select the clusters randomly from data
        clusters = data[numpy.random.randint(
            data_count, size=cluster_count), :]
        # assign random memberships to the data
        memberships = numpy.zeros(data_count, dtype=numpy.int8)
        iteration = 0
        _inertia = 1e308
        # k-means loop starting
        while True:
            changed = False
            # reset new cluster variables
            _clusters = numpy.zeros((cluster_count, data.shape[1]))
            _cluster_size = numpy.zeros(cluster_count)

            # CLUSTER ASSIGNMENT STEP
            # assign each data to the nearest cluster
            for i, datum in enumerate(data):
                dmin = float('Inf')
                # find the smallest distance cluster center
                for j, cluster in enumerate(clusters):
                    distance = euclidean(datum, cluster)
                    if distance < dmin:
                        dmin = distance
                        n = j
                # assign closest cluster to the datum
                if memberships[i] != n:
                    memberships[i] = n
                    changed = True
                # store the sum of the all data belonging to the same cluster
                _clusters[memberships[i]] = _clusters[memberships[i]] + datum
                # store the data count of cluster
                _cluster_size[memberships[i]] += 1

            # UPDATE STEP
            # calculate new cluster centers using data cluster information
            clusters = numpy.divide(_clusters, _cluster_size[:, numpy.newaxis])

            # COST CALCULATION
            inertia = numpy.sum((data - clusters[memberships])**2)
            print("iteration: %d cost: %f" % (iteration, inertia))
            if _inertia == inertia:
                break
            else:
                _inertia = inertia
            iteration += 1
            # check for stop criteria
            if iteration > max_iteration or changed is False:
                break
        # print final inertia
        print("inertia: %f" % numpy.sum((data - clusters[memberships])**2))
        # data cluster memberships and cluster centers are returned
        return clusters, memberships


if __name__ == "__main__":
    # testing
    ae = KMeans()
    data = numpy.random.rand(25, 2)
    clusters, memberships = ae.cluster(data, 2)
    sklKMeans = SKLKMeans(n_clusters=2)
    sklKMeans.fit(data)
    print("scikit-learn inertia: %f" % sklKMeans.inertia_)
    # plot results
    f, (ax1, ax2) = plot.subplots(1, 2, sharey=True)
    ax1.scatter(data[:, 0], data[:, 1], c=memberships)
    ax1.plot(clusters[:, 0], clusters[:, 1], 'g^')
    ax1.set_title("code")
    ax2.scatter(data[:, 0], data[:, 1], c=sklKMeans.labels_)
    ax2.plot(sklKMeans.cluster_centers_[:, 0],
             sklKMeans.cluster_centers_[:, 1], 'g^')
    ax2.set_title("scikit-learn")
    plot.show()
