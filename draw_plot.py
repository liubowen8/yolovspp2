"""
画折线图
"""
import matplotlib.pyplot as plt
import numpy as np

easy_file_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/easy_cases.txt"
open_easy_file = open(easy_file_path)
easy_data = open_easy_file.readlines()
easy_data_=[]
for yihang in easy_data:
    easy_data_.append([float(i) for i in yihang.strip('\n').split()])

easy_data_ = np.array(easy_data_)
easy_data_ = easy_data_[:,:-1]
print(easy_data_)
adjusted_kls= easy_data_[:,0]
pes = easy_data_[:,1]
ious = easy_data_[:,2]
sigmas = easy_data_[:,3]
print(pes)
plt.scatter(pes[:150], adjusted_kls[:150], marker='o')
plt.show()
print(11111111)


