# -*- coding: utf-8 -*-
#
# optics聚类算法
# 参考：https://www.biaodianfu.com/optics.html
# Author: alex
# Created Time: 2019年11月30日 星期六 10时42分17秒
import numpy as np


def euclidean(p1, p2):
    """欧氏距离"""
    return np.linalg.norm(p1 - p2)


class Point:
    def __init__(self, data):
        self.data = data
        self.cd = None  # core distance
        self.rd = None  # reachability distance
        self.processed = False  # has this point been processed?

    def __repr__(self):
        return str(self.data)


class Optics:
    inf = float('infinity')

    def __init__(self, max_radius, min_cluster_size, distance=euclidean):
        """
        :param max_radius: int|float, 邻域半径
        :param min_cluster_size: int, 最小聚类的数据点的数量
        :param distance: function, 距离函数，可以在外部自定义，默认为欧氏距离
        说明：
        距离函数定义：def function_name(point1, point2)
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
        sorted_neighbors = sorted([self.distance(point.data, n.data)
                                   for n in neighbors])
        point.cd = sorted_neighbors[self.min_cluster_size - 2]
        return point.cd

    def _neighbors(self, point):
        """找到其所有直接密度可达样本点"""
        return [p for p in self.points if p is not point and
                self.distance(point.data, p.data) <= self.max_radius]

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
            new_rd = max(point.cd, self.distance(n.data, point.data))
            if n.rd is None:
                n.rd = new_rd
                seeds.append(n)
            elif new_rd < n.rd:
                n.rd = new_rd

    def fit(self, points):
        self.points = [Point(row) for row in points]

        # 待处理队列
        self.unprocessed = [p for p in self.points]
        # 结果队列用来存储样本点的输出次序
        self.ordered = []
        seeds = []   # 核心点周围的未处理的邻点

        # 选择一个未处理且为核心对象的样本点，找到其所有直接密度可达样本点
        while self.unprocessed or seeds:
            # 优先从seeds选择一个点
            if seeds:
                seeds.sort(key=lambda n: n.rd)
                point = seeds.pop(0)
            else:
                point = self.unprocessed[0]

            # mark p as processed
            self._processed(point)
            # find p's neighbors
            point_neighbors = self._neighbors(point)
            if self._core_distance(point, point_neighbors) is None:
                # point不满足核心点的条件
                continue

            # update reachability_distance for each unprocessed neighbor
            self._update(point_neighbors, point, seeds)

        # when all points have been processed
        # return the ordered list
        return self.ordered

    def cluster(self, cluster_threshold):
        separators = []
        for i in range(len(self.ordered)):
            this_i = i
            this_p = self.ordered[i]
            this_rd = this_p.rd if this_p.rd else self.inf
            # use an upper limit to separate the clusters
            if this_rd > cluster_threshold:
                separators.append(this_i)

        clusters = []
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

    """
    输出：
    [[1 1], [2 2], [1 3]]
    [[4 6], [5 7]]
    """
    for cluster in clusters:
        print(cluster)
