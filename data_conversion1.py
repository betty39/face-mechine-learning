# -*- coding: utf-8 -*-

import os
import operator
from numpy import *
import cv2

# define PCA
'''
: data: 原始矩阵 k: 降低维度值
'''
def pca(data,k):
    data = float32(mat(data)) 
    rows,cols = data.shape#取大小
    data_mean = mean(data,0)#对列求均值
    data_mean_all = tile(data_mean,(rows,1))
    Z = data - data_mean_all
    T1 = Z*Z.T #使用矩阵计算，所以前面mat
    D,V = linalg.eig(T1) #特征值与特征向量
    V1 = V[:,0:k]#取前k个特征向量
    V1 = Z.T*V1
    for i in range(k): #特征向量归一化
        L = linalg.norm(V1[:,i])
        V1[:,i] = V1[:,i]/L

    data_new = Z*V1 # 降维后的数据
    return data_new,data_mean,V1

#covert image to vector
def img2vector(filename):
    img = cv2.imread(filename,0) #read as 'gray'
    rows,cols = img.shape
    imgVector = zeros((1,rows*cols)) #create a none vectore:to raise speed
    imgVector = reshape(img,(1,rows*cols)) #change img from 2D to 1D      
    return imgVector

#load dataSet
'''
:param dataSetDir:读取文件夹下原始图片矩阵及标签(只解析子文件图片)
:return train_face: 样本图片原始矩阵 train_face_lables: 样本图片标签
'''
def loadDataSet(dataSetDir):
    ##step 1:Getting data set
    fileNum = 0
    for lists in os.listdir(dataSetDir):
        sub_path = os.path.join(dataSetDir, lists)
        if os.path.isfile(sub_path):
            fileNum = fileNum+1                     # 统计图片数量

    train_face = zeros((fileNum,256*256))
    train_face_labels = []
    for parent,dirnames,filenames in os.walk(dataSetDir):
         index = 0
         for filename in filenames:
            img = img2vector(parent+'/'+filename)
            train_face[index,:] = img
            train_face_labels.append(parent+'/'+filename)
            index += 1
    return train_face,train_face_labels

'''
: param dataSetDir: 文件夹路径(att_faces 的数据) k: 取的训练集个数
'''
def loadDataSetAnalysis(dataSetDir, k):
    choose = random.permutation(10)+1 #随机排序1-10 (0-9）+1
    train_face = zeros((40*k,112*92))
    train_face_number = zeros(40*k)
    test_face = zeros((40*(10-k),112*92))
    test_face_number = zeros(40*(10-k))
    for i in range(40): #40 sample people
        people_num = i+1
        for j in range(10): #everyone has 10 different face
            if j < k:
                filename = dataSetDir+'/s'+str(people_num)+'/'+str(j+1)+'.pgm'
                img = img2vector(filename)     
                train_face[i*k+j,:] = img
                train_face_number[i*k+j] = people_num
            else:
                filename = dataSetDir+'/s'+str(people_num)+'/'+str(j+1)+'.pgm'
                img = img2vector(filename)     
                test_face[i*(10-k)+(j-k),:] = img
                test_face_number[i*(10-k)+(j-k)] = people_num
    return train_face,train_face_number,test_face,test_face_number

'''
: param dataSetDir: 文件夹路径(jaffe 的数据) k: 作为训练集个数
'''
def loadDataJaffeAnalysis(dataSetDir, k):
    fileNum = 0
    for lists in os.listdir(dataSetDir):
        sub_path = os.path.join(dataSetDir, lists)
        if os.path.isfile(sub_path):
            fileNum = fileNum+1                  # 统计图片数量
    train_face = zeros((10*k, 256*256))
    train_face_labels = []
    test_face = zeros((fileNum - 10*k,256*256))
    test_face_labels = []
    for parent,dirnames,filenames in os.walk(dataSetDir):
        per = 0
        index = 0
        lastLabel = ''
        test_num = 0
        for filename in filenames:
            img = img2vector(parent+'/'+filename)
            person = filename.split('.')
            if index == 0 or person[0] != lastLabel:
                per = 0
                index += 1
            if per < k :
                train_face[(index - 1) *k + per, :] = img
                train_face_labels.append(person[0])
            else :
                test_face[test_num, :] = img
                test_face_labels.append(person[0])
                test_num += 1
            lastLabel = person[0]
            per += 1
    return train_face,train_face_labels,test_face,test_face_labels