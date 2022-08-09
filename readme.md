python preprocess_datasets.py 进行数据预处理
python train_test_datasets.py 进行数据分割#这里放了三种数据分割方式，进行了一些尝试

cd model/ 进入模型文件夹，里面有四种模型
运行cnn
cd model/cnn_model/    
python main_cnn_model.py

运行lstm
cd model/lstm_model/    
python main_lstm_model.py

运行resnet
cd model/resnet_model/    
python main_resnet_model.py

运行xgb
cd model/xgb_model/    
python main_xgb_model.py

版本：深度学习方法使用pytorch平台，python 3.8.10，pytorch 1.10+cu111

一、几种文件和文件夹的作用、结构
preprocess_datasets.py 数据预处理作用
train_test_datasets.py 划分训练集和测试集，其中30%用作测试集

model 文件夹下存放4种模型 cnn_model,lstm_model,resnet_model,xgb_model
model文件夹中cnn_model文件夹下包含main_cnn.py  model.py 其中model.py写了cnn模型的网络结构，main_cnn.py包含数据加载、运行模型、评价指标
model文件夹中lstm_model文件夹下包含main_lstm.py  model.py 其中model.py写了lstm模型的网络结构，main_lstm.py包含数据加载、运行模型、评价指标
model文件夹中resnet_model文件夹下包含main_resnet.py  model.py 其中model.py写了resnet模型的网络结构，main_resnet.py包含数据加载、运行模型、评价指标
model文件夹中xgb_model文件夹下包含xgb.py  xgb.py包含数据加载、运行模型、评价指标

raw_datasets 存放原数据
pre_datasets 存放预处理后数据
train_test_datasets 存放训练集和测试集
picture 存放loss曲线图及混淆矩阵图片

二、四种模型介绍
每个模型都把模型部分和实验部分分开，具体详见相关代码
