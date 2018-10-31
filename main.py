import data_conversion1
import knn_kdtree1
import os
import operator
from numpy import *
import uuid
import platform
from flask import Flask, abort, request, jsonify
import face_recognition

if platform.system() == "Windows":
    slash = '\\'
else:
    platform.system()=="Linux"
    slash = '/'

UPLOAD_FOLDER = 'upload'
ALLOW_EXTENSIONS = set(['jpg', 'png', 'pgm', 'tiff'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#判断文件夹是否存在，如果不存在则创建
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
else:
    pass
# 判断文件后缀是否在列表中
def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1] in ALLOW_EXTENSIONS

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/knn-find-face', methods=['POST'])
def knn_findFace():
	'''
	post 传输接受参数例子
	if not request.json or 'id' not in request.json or 'info' not in request.json:
        abort(400)
    task = {
        'id': request.json['id'],
        'info': request.json['info']
    }
	'''
	if not request.files or 'file' not in request.files:
		return buildResponse({}, 400, 'no file upload')
	file_path = ''
	#获取post过来的文件名称，从name=file参数中获取
	file = request.files['file']
	if file and allowed_file(file.filename):
		filename = file.filename
		file_name = str(uuid.uuid4()) + '.' + filename.rsplit('.', 1)[1]
		#file.save(os.path.join(app.config['UPLOAD_FOLDER']), file_name)
		file.save(app.config['UPLOAD_FOLDER'] + slash + file_name)
		file_path = UPLOAD_FOLDER + slash + file_name
		print(file_path)
	if file_path == '':
		return buildResponse({}, 400, 'no file upload')

	# 对上传图片进行人脸检测
	ifHasFace = face_recognition.detectFaces(file_path)
	if len(ifHasFace) <= 0:
		return buildResponse({}, 400, 'no face in upload file')

	path = '/Users/qinlaodewenzi/Desktop/机器学习/att_faces'
	# 获取样本图片原始数据
	train_data, train_labels = data_conversion1.loadDataSet(path)
	data_train_new,data_mean,V = data_conversion1.pca(train_data, 30)
	test_path = file_path
	test_face = data_conversion1.img2vector(test_path)
	num_test = test_face.shape[0]
	temp_face = test_face - tile(data_mean,(num_test,1))
	data_test_new = temp_face*V # 得到测试脸在特征向量下的数据
	data_test_new = array(data_test_new)
	outputLabel = knn_kdtree1.findSimilarLable(data_train_new, train_labels, data_test_new[0,:], 6)
	return buildResponse({'facePath': outputLabel})

def buildResponse(data, code = 0, message = 'success'):
	return jsonify({'code': code, 'data': data,'message': message})

if __name__ == '__main__':
    app.run(debug=True)

