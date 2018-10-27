# Start

## 文件说明
```
data_conversion1.py: 图片转换处理
    loadDataSet: 读取 jaffe 文件夹的图片
    loadDataSetAnalysis: 读取 att_faces 文件夹的图片
main.py 接口最外层文件
knn_analysis.py: 算法分析文件

```

### 使用说明
```
pip install matplotlib
pip install flask
python main.py 运行之后为接口项目启动, 可在postman中根据路由访问接口
python knn_analysis.py 为knn算法分析文件, 目前可生成一张根据降低维度不同准确率不同的图片
```
