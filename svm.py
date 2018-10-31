#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/10/30 20:20
# @Author : yaxinrong
# @File   : svm.py
import data_conversion1
import data_preprocess
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import constant
from time import *


def pcaAndSvmFaceFindAnalysis():
    start_time = time()
    path = constant.ATT_FACE['path']
    xTrain_, yTrain, xTest_, yTest = data_conversion1.loadDataSetAnalysis(path, 8)
    num_train, num_test = xTrain_.shape[0], xTest_.shape[0]
    xTrain, data_mean, V = data_conversion1.pca(xTrain_, 16)
    xTest = np.array((xTest_ - np.tile(data_mean, (num_test, 1))) * V)  # 得到测试脸在特征向量下的数据
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5]}  # 自动穷举出最优的C参数
    clf = GridSearchCV(SVC(kernel='linear', class_weight='balanced', gamma=0.01),
                           param_grid)
    clf = clf.fit(xTrain, yTrain)
    test_pred = clf.predict(xTest)
    # print classification_report(y[test], test_pred)
    # 计算平均准确率
    precision = 0
    precision_average = 0
    for i in range(0, len(yTest)):
        if (yTest[i] == test_pred[i]):
            precision = precision + 1
    precision_average = float(precision) / len(yTest)
    return precision_average, time() - start_time

def pcaAndSvmFaceFind(path, file_path):
    star_time = time()
    train_data, train_labels = data_conversion1.loadDataSet(path, constant.ATT_FACE['height'], constant.ATT_FACE['weight'])
    train_data_new,data_mean,V = data_conversion1.pca(train_data,16)
    test_path = data_preprocess.doPreprocess(file_path, constant.ATT_FACE['height'],
                                             constant.ATT_FACE['weight'])  # 图片预处理
    test_face = data_conversion1.img2vector(test_path, (constant.ATT_FACE['height'], constant.ATT_FACE['weight']))
    num_test = test_face.shape[0]

    temp_face = test_face - np.tile(data_mean, (num_test, 1))
    data_test_new = temp_face * V  # 得到测试脸在特征向量下的数据
    data_test_new = np.array(data_test_new)
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5]}  # 自动穷举出最优的C参数
    clf = GridSearchCV(SVC(kernel='linear', class_weight='balanced', gamma=0.01),
                       param_grid)
    clf = clf.fit(train_data_new, train_labels)
    test_pred = clf.predict(data_test_new)
    return test_pred[0], star_time-time()

def main():
    acc = pcaAndSvmFaceFindAnalysis()
    print(acc)


if __name__=='__main__':
    main()


