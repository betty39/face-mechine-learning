import platform

if platform.system() == "Windows":
    SLASH = '\\'
else:
    platform.system()=="Linux"
    SLASH = '/'
JAFFE = {
    'first_path': '/Users/heyijia/master/机器学习/人脸识别/jaffe',
    'last_path': '/Users/heyijia/master/机器学习/人脸识别/jaffe_face1', # 经过处理之后的jaffe 数据集目录
    'first_height': 256,
    'first_weight': 256,
    'last_height': 100,
    'last_weight': 100,
}

ATT_FACE = {
    'path': '/Users/heyijia/master/机器学习/人脸识别/att_faces',
    'height': 112,
    'weight': 92,
}