#!/usr/bin/python
# coding=utf-8

from numpy import *
import numpy as np
import copy
import operator

# tree node
class Node:
    """docstring for ClassName"""
    def __init__(self, left_child = None, right_child = None, node_feature = None, node_label = None, axis = None):
        self.left_child = left_child
        self.right_child = right_child
        self.node_feature = node_feature
        self.node_label = node_label
        self.axis = axis

# kdtree 过程
def kdtree(data, labels, depth=0):
    assert(data.shape[0]==labels.shape[0])
    k = data.shape[1]
    axis = depth % k

    if data.shape[0] < 1:
        return None
    elif data.shape[0] == 1:
        return Node(left_child=None, right_child=None,
                    node_feature=data[0, :], node_label=labels[0],
                    axis=axis)
    sorted_idx = data[:, axis].argsort()
    sorted_data = data[sorted_idx]
    sorted_labels = labels[sorted_idx]
    median = data.shape[0]//2
    return Node(left_child=kdtree(sorted_data[:median], sorted_labels[:median], depth+1),
                right_child=kdtree(sorted_data[median+1:], sorted_labels[median+1:], depth+1),
                node_feature=sorted_data[median, :], node_label=sorted_labels[median], axis=axis)

# 创建kd tree
def construct_kdtree(data, labels):
    data = np.atleast_2d(data)
    assert(data.shape[0] == labels.shape[0])
    return kdtree(data, labels)

# 有界优先队列,保留最优的k个元素
class BPQ:
    def __init__(self, length=5, hold_max=False):
        self.data = []
        self.length = length
        self.hold_max = hold_max

    def append(self, point, distance, label):
        self.data.append((point, distance, label))
        self.data.sort(key=operator.itemgetter(1), reverse=self.hold_max)
        self.data = self.data[:self.length]

    def get_data(self):
        return [item[0] for item in self.data]

    def get_label(self):
        labels = [item[2] for item in self.data]
        uniques, counts = np.unique(labels, return_counts=True)
        return uniques[np.argmax(counts)]

    def get_threshold(self):
        return np.inf if len(self.data) == 0 else self.data[-1][1]

    def full(self):
        return len(self.data) >= self.length

# knn 搜索过程
def knn_search(test_point, node, queue):
    if node is not None:
        cur_dist = get_distance(test_point, node.node_feature)
        if cur_dist < queue.get_threshold():
            queue.append(node.node_feature, cur_dist, node.node_label)

        axis = node.axis
        search_left = False
        if test_point[axis] < node.node_feature[axis]:
            search_left = True
            queue = knn_search(test_point, node.left_child, queue)
        else:
            queue = knn_search(test_point, node.right_child, queue)

        if not queue.full() or np.abs(node.node_feature[axis] - test_point[axis]) < queue.get_threshold():
            if search_left:
                queue = knn_search(test_point, node.right_child, queue)
            else:
                queue = knn_search(test_point, node.left_child, queue)

    return queue


def knn(test_point, tree, k):
    queue = BPQ(k)
    queue = knn_search(test_point, tree, queue)
    return queue.get_data(),queue.get_label()

# 根据欧式距离计算
def get_distance(a, b):
    sum = 0.0  
    for i in range(len(a)):  
        sum = sum + (a[i] - b[i]) * (a[i] - b[i])  
    #return math.sqrt(sum)
    return sum

# 得到最近的分类标签
def findSimilarLable(dataSet, labels, testData, k):
    group = np.array(dataSet)   # 将list 转成array
    labels = np.array(labels)
    tree = construct_kdtree(group, labels)
    testData = np.array(testData)
    # 调用分类函数对未知数据分类
    nnlist, outputLabel = knn(testData, tree, k)
    return outputLabel

if __name__ == '__main__':
    # 生成数据集和类别标签
    dataSet, labels = createDataSet()
    tree = construct_kdtree(dataSet, labels)

    # 定义一个未知类别的数据
    testX = array([56.607985, 101.326679, 79.548094, 43.831216, 93.196007, 43.250454, 58.059891, 69.965517, 86.226860, 103.940109, 1, 1, 2, 3])
    k = 6
    # 调用分类函数对未知数据分类
    nnlist, outputLabel = knn(testX, tree, k)
    print("Your input is: " + str(testX) + str(k) + "个近邻点为: ")
    print(str(np.array(nnlist)))
    print("be classified to class: " + str(outputLabel))
