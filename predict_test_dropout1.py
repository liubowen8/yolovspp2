from cgi import print_directory
from cmath import pi
import os
import json
import re
import this
import time

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from build_utils import img_utils, torch_utils, utils
from models import Darknet
from draw_box_utils import draw_objs, draw_objs_dropout

def pemi_pre_class(x):

    mean_class_score=np.mean(x, axis=0)
    encropy=0

    for i in list(mean_class_score):
        encropy=encropy+i*(-np.log(i))

    import scipy.stats
    kls=[]
    for one_predict in x:
        kl=scipy.stats.entropy(mean_class_score, one_predict)
        kls.append(kl)
    kls=np.array(kls)
    return encropy, np.var(kls)

def pemi_pre_object(x):
    """
    x的数据形式 [[],[],[],...]
    是一个二维数组，表示框住这个对象的多个预测。这里的x中的元素是一个预测，是16维度的
    """
    x=np.array(x)
    classes_score=x[:,6:-4]
    mean_class_score=np.mean(classes_score, axis=0)
    encropy=0

    for i in list(mean_class_score):
        encropy=encropy+i*(-np.log(i))

    import scipy.stats
    kls=[]
    for one_predict in classes_score:
        kl=scipy.stats.entropy(mean_class_score, one_predict)
        kls.append(kl)
    kls=np.array(kls)
    return encropy, np.var(kls)

def pemi(x):
    """
    这个函数的作用：
    对于每一个类别 计算pe和互信息 然后放在列表中
    这里应该是对每一个object的多个检查计算 预测熵和互信息，而不是每一个类别
    现在所使用的检测图片上只有一个对象，所以可以暂时使用
    """
    sss=[[],[],[],[],[],[]] # 三维list ，每一个类别是一个二维list， 每一个类别有很多框，每一个框就是一个检查，就是一个list
    for i in x:
        this_class=int(i[0])
        if(len(sss)<this_class):
            # sss.append([])
            sss[this_class-1].append(i[1:])
        else:
            sss[this_class-1].append(i[1:])
    pes=[]
    kls=[]

    for i in sss: # 每一个类别
        if len(i)==0:
            #没有这个类别
            pes.append(0)
            kls.append(0)
        else:
            pe, kl=pemi_pre_class(i)
            pes.append(pe)
            kls.append(kl)
    return pes, kls
    

def bbox_iou(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4


    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) * \
            (min(b1_y2, b2_y2) - max(b1_y1, b2_y1))
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    return iou

def process(pred):
    
    xx=pred.detach().cpu().numpy()
    xx=list(xx)
    xx=sorted(xx, key=lambda x:x[4], reverse=True)
    # xx=np.array(xx)
    # pred=torch.from_numpy(xx)
    pred=xx
  
    the_number_of_objects=0
    object_list=[]
    for each_pred in pred:
        all_ious=[]
        if len(object_list)!=0:
            for choiced_one in object_list: 
                all_ious.append(bbox_iou(each_pred, choiced_one[0]))
        if len(all_ious)==0:
            object_list.append([each_pred])
            the_number_of_objects+=1
        else:
            if max(all_ious)<0.75:
                object_list.append([each_pred])
                the_number_of_objects+=1
            else:
                max_index = all_ious.index(max(all_ious))
                object_list[max_index].append(each_pred)
    pemis=[]
    for each_object in object_list:
        pemis.append(pemi_pre_object(each_object))
    
    result=[]
    for i in range(len(pemis)):
        one_pred=object_list[i][0]
        one_pred = list(one_pred)
        pe, mi=pemis[i]
        one_pred.append(pe)
        one_pred.append(mi)
        one_pred = np.array(one_pred)
        result.append(one_pred)
    
    result=np.array(result)
    result=torch.from_numpy(result)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return result.to(device=device)

# dropout保持开启
def apply_dropout(m):
    if type(m)==torch.nn.Dropout:
        m.train()

def main():
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/my_yolov3.cfg"  # 改成生成的.cfg文件
    weights = "weights/dropout_weights/yolov3spp-430.pt"  # 改成自己训练好的权重文件
    # weights = "weights/yolov3spp-366.pt"  # 改成自己训练好的权重文件
 
    # weights = "weights/yolov3spp-365.pt"  # 改成自己训练好的权重文件
    json_path = "./data/soda_classes.json"  # json标签文件
    img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_000005_SH_001.jpg"
    # img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_000240_SZ_010.jpg"
    img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_004607_SZ_010.jpg"

    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}


    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location='cpu')["model"])
    model.to(device)

    model.eval()
    model.apply(apply_dropout)
    list_for_detection=[]
    times=10
    
    with torch.no_grad():
        
        # init
        img = torch.zeros((1, 3, img_size, img_size), device=device)
        model(img)

        img_o = cv2.imread(img_path)  # BGR
        assert img_o is not None, "Image Not Found " + img_path

        img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
    
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # scale (0, 255) to (0, 1)
        img = img.unsqueeze(0)  # add batch dimension

        t1 = torch_utils.time_synchronized()
        #fms=[]
        for i in range(times):  
            pred = model(img)[0]  # only get inference result
            t2 = torch_utils.time_synchronized()
            # print(t2 - t1)
            pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
            # 这里的pred就是在 512*320上的xy，xy 还没有减去padding
            t3 = time.time()
            # print(t3 - t2)
            if pred is not None:  # 有些时候检查不到目标
                list_for_detection.append(pred)
        pred=torch.cat(list_for_detection,0)
        if pred is None:
            print("No target detected.")
            exit(0)

        # process detections
        pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
        # 对 sigma进行放大
        gain=max(img.shape[2:])/ max(img_o.shape)
        pred[:,-4:]/=gain
        # 对 sigma进行放大
        sigmas=pred[:,-4:].detach().cpu().numpy()

        # 在考虑类别时，预测的输出是16维度的，前6维度和原来一样 坐标四维度，conf，cls， 后十个前六维度表示类别分数，后面四个维度表示sigmas
        scores = pred[:, 4].detach().cpu().numpy()
        idxs = np.greater(scores, 0.2)
        pred=pred[idxs]  # 只对那些置信度大于0.2的检测结果进行处理。

        pred=process(pred)   # !!!!!!

        bboxes = pred[:, :4].detach().cpu().numpy()
        sigmas=pred[:,-4:].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1
        pemis=pred[:,-2:].detach().cpu().numpy()
        
      

        # 计算sigmas
        bboxes_tiaoxuan=pred
        lefts=torch.var(bboxes_tiaoxuan[:,0])
        tops=torch.var(bboxes_tiaoxuan[:,1])
        rights=torch.var(bboxes_tiaoxuan[:,2])
        lows=torch.var(bboxes_tiaoxuan[:,3])
        print(lefts,tops, rights, lows, 66666666)

        pil_img = Image.fromarray(img_o[:, :, ::-1])
        plot_img = draw_objs_dropout(pil_img,
                             bboxes,
                             classes,
                             scores,
                             pemis=pemis,
                             sigmas=sigmas,
                             category_index=category_index,
                             box_thresh=0.2,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        
        # plt.imshow("xx", plot_img)
        # plt.show()
        # 保存预测的图片结果
        plot_img.save("ratest_result.jpg")
        


if __name__ == "__main__":
    main()
