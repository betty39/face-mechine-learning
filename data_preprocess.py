import face_recognition
from PIL import Image,ImageDraw
import cv2
import os
import sys
import constant
from PIL import Image,ImageDraw

# 二值判断,如果确认是噪声,用改点的上面一个点的灰度进行替换
# 该函数也可以改成RGB判断的,具体看需求如何
def getPixel(image,x,y,G,N):
    L = image.getpixel((x,y))
    if L > G:
        L = True
    else:
        L = False

    nearDots = 0
    if L == (image.getpixel((x - 1,y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1,y)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1,y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x,y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x,y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1,y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1,y)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1,y + 1)) > G):
        nearDots += 1

    if nearDots < N:
        return image.getpixel((x,y-1))
    else:
        return None

# 降噪
# 根据一个点A的RGB值，与周围的8个点的RBG值比较，设定一个值N（0 <N <8），当A的RGB值与周围8个点的RGB相等数小于N时，此点为噪点
# G: Integer 图像二值化阀值
# N: Integer 降噪率 0 <N <8
# Z: Integer 降噪次数
# 输出
#  0：降噪成功
#  1：降噪失败
def clearNoise(img_file,G,N,Z, save_img):
    #打开图片
    image = Image.open(img_file)
    draw = ImageDraw.Draw(image)
    for i in range(0,Z):
        for x in range(1,image.size[0] - 1):
            for y in range(1,image.size[1] - 1):
                color = getPixel(image,x,y,G,N)
                if color != None:
                    draw.point((x,y),color)
    #保存图片
    image.save(save_img)

def doPreprocess(img_path, height = 100, weight = 100):
    gray_img = cv2.imread(img_path, 0) # 灰度图
    retImg = cv2.resize(gray_img, (height, weight)) # 缩放成一定尺寸
    retImg = cv2.equalizeHist(retImg) # 直方图均衡化
    cv2.imwrite(img_path, retImg)
    clearNoise(img_path, 20, 2, 4, img_path)
    return img_path

if __name__ == '__main__':
    path = constant.JAFFE['first_path']
    handle = constant.JAFFE['last_path']
    for parent,dirnames,filenames in os.walk(path):
        index = 0
        for filename in filenames:
            gray_img = cv2.imread(parent + constant.SLASH + filename, 0) # 灰度图
            retImg = cv2.resize(gray_img, (100, 100)) # 缩放成一定尺寸
            rows,cols = retImg.shape
            retImg = cv2.equalizeHist(retImg) # 直方图均衡化
            cv2.imwrite(handle + constant.SLASH + filename, retImg)
    for parent,dirnames,filenames in os.walk(handle):
        index = 1
        for filename in filenames:
            person = filename.split('.')
            #判断文件夹是否存在，如果不存在则创建
            after_path = handle + constant.SLASH + person[0]
            if not os.path.exists(after_path):
                index = 1
                os.makedirs(after_path)
            else:
                pass
            clearNoise(parent + constant.SLASH + filename, 20, 2, 4, after_path + constant.SLASH + str(index) + '.tiff')
            index += 1
