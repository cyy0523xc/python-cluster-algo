# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年11月30日 星期六 10时42分17秒
import numpy as np


class Point:
    def __init__(self, row):
        self.row = np.array(row)
        self.cd = None  # core distance
        self.rd = None  # reachability distance
        self.processed = False  # has this point been processed?

    def distance(self, point):
        # calculate the distance between any two points on earth
        # convert coordinates to radians
        return np.sum(np.square(self.row - point.row))

    def __repr__(self):
        return ", ".join([str(i) for i in self.row])


class Cluster:
    def __init__(self, points):
        self.points = points


class Optics:
    def __init__(self, points, max_radius, min_cluster_size):
        self.points = points
        self.max_radius = max_radius  # maximum radius to consider
        self.min_cluster_size = min_cluster_size  # minimum points in cluster

    def _setup(self):
        # get ready for a clustering run
        for p in self.points:
            p.rd = None
            p.processed = False
        self.unprocessed = [p for p in self.points]
        self.ordered = []

    def _core_distance(self, point, neighbors):
        # distance from a point to its nth neighbor (n = min_cluser_size)
        if point.cd is not None:
            return point.cd
        if len(neighbors) >= self.min_cluster_size - 1:
            sorted_neighbors = sorted([n.distance(point) for n in neighbors])
            point.cd = sorted_neighbors[self.min_cluster_size - 2]
            return point.cd

    def _neighbors(self, point):
        # neighbors for a point within max_radius
        return [p for p in self.points if p is not point and
                p.distance(point) <= self.max_radius]

    def _processed(self, point):
        """ mark a point as processed """
        point.processed = True
        self.unprocessed.remove(point)
        self.ordered.append(point)

    def _update(self, neighbors, point, seeds):
        # update seeds if a smaller reachability distance is found
        # for each of point's unprocessed neighbors n...
        for n in [n for n in neighbors if not n.processed]:
            # find new reachability distance new_rd
            # if rd is null, keep new_rd and add n to the seed list
            # otherwise if new_rd < old rd, update rd
            new_rd = max(point.cd, point.distance(n))
            if n.rd is None:
                n.rd = new_rd
                seeds.append(n)
            elif new_rd < n.rd:
                n.rd = new_rd

    def run(self):
        # run the OPTICS algorithm
        self._setup()
        # for each unprocessed point (p)...
        while self.unprocessed:
            point = self.unprocessed[0]
            # mark p as processed
            # find p's neighbors
            self._processed(point)
            point_neighbors = self._neighbors(point)
            # if p has a core_distance, i.e has min_cluster_size - 1 neighbors
            if self._core_distance(point, point_neighbors) is not None:
                # update reachability_distance for each unprocessed neighbor
                seeds = []
                self._update(point_neighbors, point, seeds)
                # as long as we have unprocessed neighbors...
                while (seeds):
                    # find the neighbor n with smallest reachability distance
                    seeds.sort(key=lambda n: n.rd)
                    n = seeds.pop(0)
                    # mark n as processed
                    # find n's neighbors
                    self._processed(n)
                    n_neighbors = self._neighbors(n)
                    # if p has a core_distance...
                    if self._core_distance(n, n_neighbors) is not None:
                        # update reachability_distance for each of n's neighbors
                        self._update(n_neighbors, n, seeds)

        # when all points have been processed
        # return the ordered list
        return self.ordered

    def cluster(self, cluster_threshold):
        clusters = []
        separators = []
        for i in range(len(self.ordered)):
            this_i = i
            this_p = self.ordered[i]
            this_rd = this_p.rd if this_p.rd else float('infinity')
            # use an upper limit to separate the clusters
            if this_rd > cluster_threshold:
                separators.append(this_i)

        separators.append(len(self.ordered))
        for i in range(len(separators) - 1):
            start = separators[i]
            end = separators[i + 1]
            if end - start >= self.min_cluster_size:
                clusters.append(Cluster(self.ordered[start:end]))

        return clusters


if __name__ == "__main__":
    # LOAD SOME POINTS
    points = [
        Point((1, 1)),  # cluster #1
        Point((1, 3)),  # cluster #1
        Point((2, 2)),  # cluster #1
        Point((4, 6)),  # cluster #2
        Point((5, 7)),  # cluster #2
    ]

    optics = Optics(points, 4, 2)
    optics.run()
    clusters = optics.cluster(2)

    for cluster in clusters:
        print(cluster.points)
