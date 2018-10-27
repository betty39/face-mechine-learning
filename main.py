import data_conversion1
import knn_kdtree1
import os
import operator
from numpy import *

path = '/Users/heyijia/master/机器学习/人脸识别/jaffe'
# 获取样本图片原始数据
train_data, train_labels = data_conversion1.loadDataSet(path)
data_train_new,data_mean,V = data_conversion1.pca(train_data, 30)
test_path = '/Users/heyijia/master/机器学习/人脸识别/test_face' + '/KA.AN1.39.tiff'
test_face = data_conversion1.img2vector(test_path)
num_test = test_face.shape[0]
temp_face = test_face - tile(data_mean,(num_test,1))
data_test_new = temp_face*V # 得到测试脸在特征向量下的数据
data_test_new = array(data_test_new)
outputLabel = knn_kdtree1.findSimilarLable(data_train_new, train_labels, data_test_new[0,:], 6)
print('预测结果:' + str(outputLabel))