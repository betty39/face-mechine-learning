#-*-coding:utf8-*-#

import os
import cv2
import constant
from datetime import datetime
from PIL import Image,ImageDraw
import time

#detectFaces()返回图像中所有人脸的矩形坐标（矩形左上、右下顶点）
#使用haar特征的级联分类器haarcascade_frontalface_default.xml，在haarcascades目录下还有其他的训练好的xml文件可供选择。
#注：haarcascades目录下训练好的分类器必须以灰度图作为输入。
def detectFaces(image_name):
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result


#保存人脸图 且框出人脸图进行保存
def saveFaces(image_name, save_dir):
    draw_path = ''
    faces = detectFaces(image_name)
    if faces:
        #将人脸保存在save_dir目录下。
        #Image模块：Image.open获取图像句柄，crop剪切图像(剪切的区域就是detectFaces返回的坐标)，save保存。
        file_name = image_name.split(constant.SLASH)[-1]
        #os.mkdir(save_dir)
        count = 0
        img = Image.open(image_name)
        draw_instance = ImageDraw.Draw(img)
        for (x1,y1,x2,y2) in faces:
            file_name1 = os.path.join(save_dir,file_name)
            Image.open(image_name).crop((x1,y1,x2,y2)).save(file_name1)
            draw_instance.rectangle((x1,y1,x2,y2), outline=255)
            count+=1
        draw_path = save_dir + constant.SLASH +  'draw-' + file_name
        img.save(draw_path)
        return draw_path
    return draw_path


#在原图像上画矩形，框出所有人脸。
#调用Image模块的draw方法，Image.open获取图像句柄，ImageDraw.Draw获取该图像的draw实例，然后调用该draw实例的rectangle方法画矩形(矩形的坐标即
#detectFaces返回的坐标)，outline是矩形线条颜色(B,G,R)。
#注：原始图像如果是灰度图，则去掉outline，因为灰度图没有RGB可言。drawEyes、detectSmiles也一样。
def drawFaces(image_name):
    faces = detectFaces(image_name)
    if faces:
        img = Image.open(image_name)
        draw_instance = ImageDraw.Draw(img)
        for (x1,y1,x2,y2) in faces:
            draw_instance.rectangle((x1,y1,x2,y2), outline= 255)
        img.save('drawfaces_'+image_name)

if __name__ == '__main__':
    needHandlePath = constant.JAFFE['first_path']
    handledPath = constant.JAFFE['last_path']
    for parent,dirnames,filenames in os.walk(needHandlePath):
        index = 0
        for filename in filenames:
            saveFaces(parent+constant.SLASH+filename, handledPath)
    # saveFaces('/Users/heyijia/master/机器学习/人脸识别/jaffe/KA.SU3.38.tiff', 'upload')
    '''
    result=detectFaces('KA.AN1.39.tiff')
    if len(result)>0:
        print("有人存在！！---》人数为："+str(len(result)))
    else:
        print('视频图像中无人！！')
    '''
