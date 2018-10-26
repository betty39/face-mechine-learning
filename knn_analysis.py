import data_conversion1
import knn_kdtree1
import os
import operator
from numpy import *

path = '/Users/heyijia/master/机器学习/人脸识别/att_faces'
# 获取样本图片原始数据
train_data, train_labels, test_data, test_labels = data_conversion1.loadDataSetAnalysis(path, 3)
data_train_new,data_mean,V = data_conversion1.pca(train_data, 30)
num_train = data_train_new.shape[0]
num_test = test_data.shape[0]
temp_face = test_data - tile(data_mean,(num_test,1))
data_test_new = temp_face*V #得到测试脸在特征向量下的数据
data_test_new = array(data_test_new) # mat change to array
data_train_new = array(data_train_new)
true_num = 0
for i in range(num_test):
    testFace = data_test_new[i,:]
    outputLabel = knn_kdtree1.findSimilarLable(data_train_new, train_labels, testFace, 6)
    if outputLabel == test_labels[i]:
        true_num += 1

accuracy = float(true_num)/num_test
print('The classify accuracy is: %.2f%%'%(accuracy * 100))