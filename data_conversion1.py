# -*- coding: utf-8 -*-

import os
import operator
import constant
from numpy import *
import cv2


# define PCA
'''
: data: 原始矩阵 k: 降低维度值
'''
def pca(data,k):
    data = float32(mat(data)) 
    rows,cols = data.shape  # 取大小
    data_mean = mean(data,0) # 对列求均值
    data_mean_all = tile(data_mean,(rows,1))
    Z = data - data_mean_all
    T1 = Z*Z.T  # 使用矩阵计算，所以前面mat
    D,V = linalg.eig(T1)  # 特征值与特征向量
    V1 = V[:,0:k]  # 取前k个特征向量
    V1 = Z.T*V1
    for i in range(k):  # 特征向量归一化
        L = linalg.norm(V1[:,i])
        V1[:,i] = V1[:,i]/L

    data_new = Z*V1 # 降维后的数据
    return data_new,data_mean,V1

#covert image to vector
#covert image to vector
def img2vector(filename, dimsize = (50, 50)):
    #file_path_gbk = filename.encode('gbk')
    img = cv2.imread(filename, 0)
    retImg = cv2.resize(img, dimsize) # 缩放成一定尺寸
    retImg=LBP(retImg)
    rows,cols = retImg.shape
    imgVector = zeros((1,rows*cols)) #create a none vectore:to raise speed
    imgVector = reshape(retImg,(1,rows*cols)) #change img from 2D to 1D
    return imgVector

#提取lbp
# 算法主过程

def LBP(I, radius=2, count=8):       #得到图像的LBP特征
    dh = np.round([radius*math.sin(i*2*math.pi/count) for i in range(count)])
    dw = np.round([radius*math.cos(i*2*math.pi/count) for i in range(count)])

    height ,width = I.shape
    lbp = np.zeros(I.shape, dtype = np.int)
    I1 = np.pad(I, radius, 'edge')
    for k in range(count):
        h,w = radius+dh[k], radius+dw[k]
        lbp += ((I>I1[h:h+height, w:w+width])<<k)
    return lbp

# load dataSet
'''
:param dataSetDir:读取文件夹下原始图片矩阵及标签(适合两层的文件结构)
:return train_face: 样本图片原始矩阵 train_face_lables: 样本图片标签
'''
def loadDataSet(dataSetDir, height = 100, weight = 100):
    ##step 1:Getting data set
    dimsize = height * weight
    fileNum = 0
    for parent,dirnames,filenames in os.walk(dataSetDir):
        for dirname in dirnames:
            for sub, sub_dir, sub_files in os.walk(parent + constant.SLASH + dirname):
                for sub_file in sub_files:
                    fileNum += 1

    train_face = zeros((fileNum, dimsize))
    train_face_labels = []
    for parent,dirnames,filenames in os.walk(dataSetDir):
        index = 0
        for dirname in dirnames:
            for sub, sub_dir, sub_files in os.walk(parent + constant.SLASH + dirname):
                for sub_file in sub_files:
                    img = img2vector(parent+constant.SLASH+dirname + constant.SLASH + sub_file, (height, weight))
                    train_face[index,:] = img
                    train_face_labels.append(dirname)
                    index += 1
    return train_face, train_face_labels

'''
: param dataSetDir: 文件夹路径(att_faces 的数据) k: 取的训练集个数
: 作为 Holdout Method(保留)的交叉验证方式去分析
'''
def loadDataSetAnalysis(dataSetDir, k):
    height = 112
    weight = 92
    dimsize = height * weight # 原图为112 * 92

    choose = random.permutation(10)+1 #随机排序1-10 (0-9）+1
    train_face = zeros((40*k, dimsize))
    train_face_number = zeros(40*k)
    test_face = zeros((40*(10-k),dimsize))
    test_face_number = zeros(40*(10-k))
    for i in range(40): #40 sample people
        people_num = i+1
        for j in range(10): #everyone has 10 different face
            if j < k:
                filename = dataSetDir+ constant.SLASH +'s'+str(people_num)+constant.SLASH+str(choose[j])+'.pgm'
                img = img2vector(filename, (height, weight))
                train_face[i*k+j,:] = img
                train_face_number[i*k+j] = people_num
            else:
                filename = dataSetDir+ constant.SLASH +'s'+str(people_num)+constant.SLASH+str(choose[j])+'.pgm'
                img = img2vector(filename, (height, weight))
                test_face[i*(10-k)+(j-k),:] = img
                test_face_number[i*(10-k)+(j-k)] = people_num
    return train_face,train_face_number,test_face,test_face_number

'''
: param dataSetDir: 文件夹路径(jaffe 的数据) k: 作为训练集个数
: 作为 Holdout Method(保留)的交叉验证方式去分析
'''
def loadDataJaffeAnalysis(dataSetDir, k, height = 256, weight = 256):
    dimsize = height * weight # 原图为256 * 256
    fileNum = 0
    for lists in os.listdir(dataSetDir):
        sub_path = os.path.join(dataSetDir, lists)
        if os.path.isfile(sub_path):
            fileNum = fileNum+1                  # 统计图片数量
    train_face = zeros((10*k, dimsize))
    train_face_labels = []
    test_face = zeros((fileNum - 10*k, dimsize))
    test_face_labels = []
    for parent,dirnames,filenames in os.walk(dataSetDir):
        per = 0
        index = 0
        lastLabel = ''
        test_num = 0
        for filename in filenames:
            img = img2vector(parent+constant.SLASH+filename, (height, weight))
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

'''
: param: dataSetDir: 数据文件夹路径, k: 每个类中划分作为训练集的个数, height: 图片统一高度, weight: 图片统一宽度
'''
def loadTwoLayerDataAnalysis(dataSetDir, k, height = 100, weight = 100):
    dimsize = height * weight
    personNum = 0
    fileNum = 0
    for parent,dirnames,filenames in os.walk(dataSetDir):
        for dirname in dirnames:
            personNum += 1
            for sub, sub_dir, sub_files in os.walk(parent + constant.SLASH + dirname):
                for sub_file in sub_files:
                    fileNum += 1
    train_face = zeros((k * personNum, height * weight))
    train_face_labels = []
    test_face = zeros((fileNum - k * personNum, height * weight))
    test_face_labels = []
    for parent,dirnames,filenames in os.walk(dataSetDir):
        index = 0
        test_num = 0
        for dirname in dirnames:
            each_file_num = 0
            for sub, sub_dir, sub_files in os.walk(parent + constant.SLASH + dirname):
                for file in sub_files:
                    each_file_num += 1
            choose = random.permutation(each_file_num)+1 #随机排序1-10 (0-9）+1
            j = 0
            for sub, sub_dir, sub_files in os.walk(parent + constant.SLASH + dirname):
                for sub_file in sub_files:
                    filename = sub_file.split('.')
                    filename = parent+constant.SLASH+dirname + constant.SLASH + str(choose[j])+ '.' + filename[-1]
                    img = img2vector(filename, (height, weight))
                    if j < k:
                        train_face[index*k+j,:] = img
                        train_face_labels.append(dirname)
                    else:
                        test_face[test_num,:] = img
                        test_face_labels.append(dirname)
                        test_num += 1
                    j += 1
                index += 1
    return train_face, train_face_labels, test_face, test_face_labels
