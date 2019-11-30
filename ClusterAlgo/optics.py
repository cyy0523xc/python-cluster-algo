# -*- coding: utf-8 -*-
#
# optics聚类算法
# Author: alex
# Created Time: 2019年11月30日 星期六 10时42分17秒
import numpy as np


def euclidean(p1, p2):
    """欧氏距离"""
    return np.sum(np.square(p1.row - p2.row))


class Point:
    def __init__(self, row):
        self.row = row
        self.cd = None  # core distance
        self.rd = None  # reachability distance
        self.processed = False  # has this point been processed?

    def __repr__(self):
        return ", ".join([str(i) for i in self.row])


class Optics:
    inf = float('infinity')

    def __init__(self, max_radius, min_cluster_size, distance=euclidean):
        """
        :param max_radius: int|float, 最大半径
        :param min_cluster_size: int, 最小聚类的数据点的数量
        :param distance: function, 距离函数，可以在外部自定义，默认为欧氏距离
        """
        self.max_radius = max_radius
        self.min_cluster_size = min_cluster_size
        self.distance = distance

    def _core_distance(self, point, neighbors):
        # distance from a point to its nth neighbor (n = min_cluser_size)
        if point.cd is not None:
            return point.cd
        if len(neighbors) < self.min_cluster_size - 1:
            return None
        sorted_neighbors = sorted([self.distance(point, n) for n in neighbors])
        point.cd = sorted_neighbors[self.min_cluster_size - 2]
        return point.cd

    def _neighbors(self, point):
        # neighbors for a point within max_radius
        return [p for p in self.points if p is not point and
                self.distance(point, p) <= self.max_radius]

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
            new_rd = max(point.cd, self.distance(n, point))
            if n.rd is None:
                n.rd = new_rd
                seeds.append(n)
            elif new_rd < n.rd:
                n.rd = new_rd

    def fit(self, points):
        self.points = [Point(row) for row in points]
        self.unprocessed = [p for p in self.points]
        self.ordered = []

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
            this_rd = this_p.rd if this_p.rd else self.inf
            # use an upper limit to separate the clusters
            if this_rd > cluster_threshold:
                separators.append(this_i)

        separators.append(len(self.ordered))
        for i in range(len(separators) - 1):
            start = separators[i]
            end = separators[i + 1]
            if end - start >= self.min_cluster_size:
                clusters.append(self.ordered[start:end])

        return clusters


if __name__ == "__main__":
    # LOAD SOME POINTS
    points = [
        np.array((1, 1)),
        np.array((1, 3)),
        np.array((2, 2)),
        np.array((4, 6)),
        np.array((5, 7)),
    ]

    optics = Optics(4, 2)
    optics.fit(points)
    clusters = optics.cluster(2)

    for cluster in clusters:
        print(len(cluster))
        print(cluster)