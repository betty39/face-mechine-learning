import data_conversion1
import knn_kdtree1
from numpy import *
import matplotlib.pyplot as pl
import constant
from time import *

BEST_K = 4
BEST_DIM = 20

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

def precisionWithBestKAndDim():
    path = constant.JAFFE['last_path']
    # 获取样本图片原始数据
    start_time =time()
    train_data, train_labels, test_data, test_labels = data_conversion1.loadTwoLayerDataAnalysis(path, 14, constant.JAFFE['last_height'], constant.JAFFE['last_weight'])
    precision = pcaAndFaceFindAnalysis(train_data, train_labels, test_data, test_labels, BEST_DIM, BEST_K)
    return precision, time() - start_time

def analysis():
    path = constant.JAFFE['last_path']
    # 获取样本图片原始数据
    print('analysis jaffe face')
    train_data, train_labels, test_data, test_labels = data_conversion1.loadTwoLayerDataAnalysis(path, 14, constant.JAFFE['last_height'], constant.JAFFE['last_weight'])
    # 画图
    for k in range(10):
        # x = []  # 横轴的数据
        x = linspace(5, 100, 19)
        y = []
        for i in x:
            dim = int(i)
            y.append(pcaAndFaceFindAnalysis(train_data, train_labels, test_data, test_labels, dim, k + 1))
        pl.plot(x, y, label = k + 1)  # 调用pylab的plot函数绘制曲线
    pl.xlabel("Dimension")
    pl.ylabel("Precision")
    pl.title('Different k')
    pl.legend()
    pl.show()  # 显示绘制出的图