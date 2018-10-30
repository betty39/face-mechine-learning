import data_conversion1
import knn_kdtree1
import os
import operator
from numpy import *
import pylab as pl

def pcaAndFaceFindAnalysis(train_data, train_labels, test_data, test_labels, lowDimen, knn):
    data_train_new,data_mean,V = data_conversion1.pca(train_data, lowDimen)
    num_train = data_train_new.shape[0]
    num_test = test_data.shape[0]
    temp_face = test_data - tile(data_mean,(num_test,1))
    data_test_new = temp_face*V #得到测试脸在特征向量下的数据
    data_test_new = array(data_test_new) # mat change to array
    data_train_new = array(data_train_new)
    true_num = 0
    for i in range(num_test):
        testFace = data_test_new[i,:]
        outputLabel = knn_kdtree1.findSimilarLable(data_train_new, train_labels, testFace, knn)
        if outputLabel == test_labels[i]:
            true_num += 1
    accuracy = float(true_num)/num_test
    return accuracy

path = 'E:/研一上/机器学习/作业/Face recognition/face-mechine-learning/att_faces'
# 获取样本图片原始数据
print('analysis att_faces')
train_data, train_labels, test_data, test_labels = data_conversion1.loadDataSetAnalysis(path, 8)
# 画图
x = []  # 横轴的数据
y = []
dim = 10
while (dim < 100):
    y.append(pcaAndFaceFindAnalysis(train_data, train_labels, test_data, test_labels, dim, 4))
    x.append(dim)
    dim += 10
pl.plot(x, y)  # 调用pylab的plot函数绘制曲线
pl.show()  # 显示绘制出的图

path = 'E:/研一上/机器学习/作业/Face recognition/face-mechine-learning/jaffe'
# 获取样本图片原始数据
print('analysis jaffe')
train_data, train_labels, test_data, test_labels = data_conversion1.loadDataJaffeAnalysis(path, 13)

# 画图
x = []  # 横轴的数据
y = []
dim = 10
while (dim < 100):
	y.append(pcaAndFaceFindAnalysis(train_data, train_labels, test_data, test_labels, dim, 4))
	x.append(dim)
	dim += 10
pl.plot(x, y)  # 调用pylab的plot函数绘制曲线
pl.show()  # 显示绘制出的图

path = '/Users/heyijia/master/机器学习/人脸识别/jaffe_face'
# 获取样本图片原始数据
print('analysis jaffe face')
train_data, train_labels, test_data, test_labels = data_conversion1.loadDataJaffeAnalysis(path, 13, 100, 100)
# 画图
x = []  # 横轴的数据
y = []
dim = 10
while (dim < 100):
    y.append(pcaAndFaceFindAnalysis(train_data, train_labels, test_data, test_labels, dim, 4))
    x.append(dim)
    dim += 10
pl.plot(x, y)  # 调用pylab的plot函数绘制曲线
pl.show()  # 显示绘制出的图