"""
查看soda数据集中不同场景下的图片数量，因为服务器上的内存容量有限，现在只是对不同场景下的图片进行一个统计，并没有将他们进行分类
"""

"""
本脚本有两个功能：
1.将voc数据集标注信息(.xml)转为yolo标注格式(.txt)，并将图像文件复制到相应文件夹
2.根据json标签文件，生成对应names标签(my_data_label.names)
"""
import os
from tkinter import X
from tqdm import tqdm

import json
import shutil


# voc数据集根目录以及版本
soda_root = "/home/ifpp/ye/datasets/SODA10M/SSLAD-2D/labeled"
soda_train="/home/ifpp/ye/datasets/SODA10M/SSLAD-2D/labeled/train"
soda_val="/home/ifpp/ye/datasets/SODA10M/SSLAD-2D/labeled/val"  # 这里是图片的位置
annotations_json_path="/home/ifpp/ye/datasets/SODA10M/SSLAD-2D/labeled/annotations/" # 这里是标注信息的位置

tong_ji_de_zi_dian={}
tu_pian_shu=0

train_data_json_path=annotations_json_path+"instance_train"+".json"  # 这个是训练的数据的标注文件
json_f=open(train_data_json_path,'r')
json_dict=json.load(json_f) 
image_infos=json_dict["images"]  # list 里面是每张图片的信息 
for yi_zhang_tu_de_zi_dian in image_infos:
    tu_pian_shu+=1
    if yi_zhang_tu_de_zi_dian["location"] in tong_ji_de_zi_dian.keys():
    # if tong_ji_de_zi_dian.has_key(yi_zhang_tu_de_zi_dian["location"]):
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["location"]]+=1
    else:
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["location"]]=1

    if yi_zhang_tu_de_zi_dian["period"] in tong_ji_de_zi_dian.keys():
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["period"]]+=1
    else:
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["period"]]=1

    if yi_zhang_tu_de_zi_dian["weather"] in tong_ji_de_zi_dian.keys():
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["weather"]]+=1
    else:
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["weather"]]=1
print(tong_ji_de_zi_dian, tu_pian_shu)

val_data_json_path=annotations_json_path+"instance_val"+".json"  # 这个是验证集的数据的标注文件
json_f=open(val_data_json_path,'r')
json_dict=json.load(json_f)
image_infos=json_dict["images"]  # list 里面是每张图片的信息 
for yi_zhang_tu_de_zi_dian in image_infos:
    tu_pian_shu+=1
    if yi_zhang_tu_de_zi_dian["location"] in tong_ji_de_zi_dian.keys():
    # if tong_ji_de_zi_dian.has_key(yi_zhang_tu_de_zi_dian["location"]):
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["location"]]+=1
    else:
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["location"]]=1

    if yi_zhang_tu_de_zi_dian["period"] in tong_ji_de_zi_dian.keys():
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["period"]]+=1
    else:
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["period"]]=1

    if yi_zhang_tu_de_zi_dian["weather"] in tong_ji_de_zi_dian.keys():
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["weather"]]+=1
    else:
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["weather"]]=1
print(tong_ji_de_zi_dian, tu_pian_shu)

test_data_json_path=annotations_json_path+"instance_test"+".json"  # 这个是测试集的数据的标注文件
json_f=open(test_data_json_path,'r')
json_dict=json.load(json_f)
image_infos=json_dict["images"]  # list 里面是每张图片的信息 
for yi_zhang_tu_de_zi_dian in image_infos:
    tu_pian_shu+=1
    if yi_zhang_tu_de_zi_dian["location"] in tong_ji_de_zi_dian.keys():
    # if tong_ji_de_zi_dian.has_key(yi_zhang_tu_de_zi_dian["location"]):
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["location"]]+=1
    else:
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["location"]]=1

    if yi_zhang_tu_de_zi_dian["period"] in tong_ji_de_zi_dian.keys():
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["period"]]+=1
    else:
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["period"]]=1

    if yi_zhang_tu_de_zi_dian["weather"] in tong_ji_de_zi_dian.keys():
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["weather"]]+=1
    else:
        tong_ji_de_zi_dian[yi_zhang_tu_de_zi_dian["weather"]]=1
print(tong_ji_de_zi_dian, tu_pian_shu)

