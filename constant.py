import platform

if platform.system() == "Windows":
    SLASH = '\\'
else:
    platform.system()=="Linux"
    SLASH = '/'
JAFFE = {
    'first_path': '/Users/qinlaodewenzi/Desktop/机器学习/jaffe',
    'last_path': '/Users/qinlaodewenzi/Desktop/机器学习/jaffe_face1', # 经过处理之后的jaffe 数据集目录
    'first_height': 256,
    'first_weight': 256,
    'last_height': 100,
    'last_weight': 100,
}

ATT_FACE = {
    'path': '/Users/qinlaodewenzi/Desktop/机器学习/att_faces',
    'height': 112,
    'weight': 92,
}