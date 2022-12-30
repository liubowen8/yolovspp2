"""
从annotation的角度出发，去逐个验证训练数据和验证数据
"""
# from __future__ import annotations
# from cProfile import label

import json
import os
import time

import cv2
import numpy as np
import scipy.stats
import torch
from matplotlib import pyplot as plt
from PIL import Image

from build_utils import img_utils, torch_utils, utils
from draw_box_utils import draw_objs
from models import Darknet
from val_utils import bbox_iou, xywh2xyxy

annotations_json_path="/home/ifpp/ye/datasets/SODA10M/SSLAD-2D/labeled/annotations/"
train_json_path = annotations_json_path+"instance_train.json"
json_f=open(train_json_path,'r')
t_json_dict=json.load(json_f)
t_annotations=t_json_dict["annotations"]  # list 里面是每一个object的标注信息，一个对象使用一个字典标注
# 这里的annotation是从soda的annotation中得到的
# 所以类别编号是从1开始的，另外bbox是左上角，w，h ！！！
t_image_infos=t_json_dict["images"]  # list 里面是每张图片的信息
val_json_path = annotations_json_path+"instance_val.json"
json_f=open(val_json_path,'r')
v_json_dict=json.load(json_f)
v_annotations=v_json_dict["annotations"]  # list 里面是每一个object的标注信息，一个对象使用一个字典标注
v_image_infos=v_json_dict["images"]  # list 里面是每张图片的信息

images_root = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/"
total=0
easy_cases_txtfile_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/easy_cases.txt"
hard_cases_txtfile_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/hard_cases.txt"

def main():
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/my_yolov3.cfg"  # 改成生成的.cfg文件xclock
    weights = "weights/yolov3spp-399.pt"  # 改成自己训练好的权重文件
    json_path = "./data/soda_classes.json"  # json标签文件
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location='cpu')["model"])
    model.to(device)

    model.eval()
    with torch.no_grad():
        # init
        img = torch.zeros((1, 3, img_size, img_size), device=device)
        model(img)

        # # 训练数据的测试
        # for imageinfo_dict in t_image_infos:
        #     image_file_name = imageinfo_dict["file_name"]
        #     image_id = imageinfo_dict["id"]
        #     image_period = imageinfo_dict["period"]
        #     image_weather = imageinfo_dict["weather"]
        #     img_path = os.path.join(images_root,"train/","images/",image_file_name)
        #     one_img_labels =[]
       
        #     for i in range(len(t_annotations)):
        #         if(t_annotations[i]["image_id"]==image_id):
        #             class_and_bbox= t_annotations[i]["bbox"]
        #             class_and_bbox.insert(0, t_annotations[i]["category_id"])  # 类别，leftx，top_y, w, h
        #             one_img_labels.append(class_and_bbox)
                    
        #         else :
        #             del t_annotations[0:i]
        #             break
        #     if len(one_img_labels)==0:
        #         continue
        #     img_o = cv2.imread(img_path)  # BGR
        #     assert img_o is not None, "Image Not Found " + img_path
        #     img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
        
        #     # Convert
        #     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        #     img = np.ascontiguousarray(img)
        #     img = torch.from_numpy(img).to(device).float()
        #     img /= 255.0  # scale (0, 255) to (0, 1)
        #     img = img.unsqueeze(0)  # add batch dimension

        #     t1 = torch_utils.time_synchronized()
        #     #fms=[]
        #     pred = model(img)[0]  # only get inference result
            
        #     # 这里每一个预测的深度是15， 其中最后面的四个是sigmas， 所以还有前面的11维度， 前面四个维度是表示边界框的，还有七个维度，一个表示obj存在的概率，剩下的6个，表示六个类别的概率
        
        #     t2 = torch_utils.time_synchronized()
        #     print(t2 - t1)
            
        
        #     pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
           
        #     #torch.Size([5, 16]) 66666
        #     # 这里的pred就是在 512*320上的xy，xy 还没有减去padding
        #     t3 = time.time()
        #     print(t3 - t2)

        #     if pred is None:
        #         print("No target detected.")
        #         # exit(0)
        #         continue

        #     # process detections
        #     pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
            
        #     # 对 sigma进行放大
        #     gain=max(img.shape[2:])/ max(img_o.shape)
        #     pred[:,-4:]/=gain
        #     # 对 sigma进行放大
            
        #     # 过滤掉低概率的目标
        #     scores = pred[:, 4].detach().cpu().numpy()
        #     idxs = np.greater(scores, 0.2)
        #     pred=pred[idxs]
        #     # 过滤掉低概率的目标

        #     bboxes = pred[:, :4].detach().cpu().numpy()
        #     scores = pred[:, 4].detach().cpu().numpy()
        #     classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1
        #     class_scores=pred[:, 6:12].detach().cpu().numpy()
        #     sigmas=pred[:,-4:].detach().cpu().numpy()
        #     # 这就是16个维度表示的内容
        #     pes=[]
        #     for xx in list(class_scores):
        #         pe=0
        #         for i in xx:
        #             pe=pe+i*(-np.log(i))
        #         round(pe, 3)
        #         pes.append(pe)
        #     pes=np.array(pes)
            
        #     one_img_labels = np.array(one_img_labels)
        #     gt_bbox = one_img_labels[:,1:]
        #     gt_bbox = xywh2xyxy(gt_bbox)  # n*4维度的数据
         
        #     for i in range(len(pred)): # 对于每一个预测框而言的
        #         ious= utils.bbox_iou(pred[i][:4],torch.from_numpy(gt_bbox).to(pred.device))

        #         #得到这个预测框 和 一张图片中的所有真值框计算IOU，找到最大的IOU
        #         max_iou, maxiou_index = torch.max(ious,dim=0)
        #         if(max_iou>=0.25):
                   
        #             index = int(maxiou_index.cpu().item())
        #             gt_class_box_of_this_predict_box= one_img_labels[index]
        #             # gt中的类别下标是从0开始的
        #             print("预测:", classes[i], "真值:", gt_class_box_of_this_predict_box[0])
        #             one_hot_list = [0 for _ in range(6)]
        #             one_hot_list[gt_class_box_of_this_predict_box[0]-1]=1
        #             kl=scipy.stats.entropy(one_hot_list, class_scores[i])
        #             pe= pes[i]
        #             is_classification_correct = 1
        #             iou_ = max_iou.item()
        #             box_width = bboxes[i][2]-bboxes[i][0]
        #             box_height = bboxes[i][3] - bboxes[i][1]
        #             sigma_ = (sigmas[i][0]/box_width+sigmas[i][2]/box_width+sigmas[i][1]/box_height+sigmas[i][3]/box_height)/4
        #             if(classes[i] == gt_class_box_of_this_predict_box[0]):
        #                 print("类别判断正确...")
        #                 # 对类别真值减1，因为类别真值是从1开始的
        #                 is_classification_correct = 1
        #             else :
        #                 print("类别判断错误... O.O'...")
        #                 is_classification_correct = 0
                    
        #             yihang = [kl, pe, iou_, sigma_, is_classification_correct]
        #             yihang = [str(i) for i in yihang]
        #             if image_period=="Daytime" and image_weather=="Clear" :
        #                 f=open(easy_cases_txtfile_path,"a")
        #                 if os.path.getsize(easy_cases_txtfile_path):
        #                     f.write("\n"+" ".join(yihang))
        #                 else:
        #                     f.write(" ".join(yihang))
        #             else :
        #                 f=open(hard_cases_txtfile_path,"a")
        #                 if os.path.getsize(easy_cases_txtfile_path):
        #                     f.write("\n"+" ".join(yihang))
        #                 else:
        #                     f.write(" ".join(yihang))



        # 验证数据的测试
        for imageinfo_dict in v_image_infos:
            print(imageinfo_dict)
            image_file_name = imageinfo_dict["file_name"]
            image_id = imageinfo_dict["id"]
            image_period = imageinfo_dict["period"]
            image_weather = imageinfo_dict["weather"]
            img_path = os.path.join(images_root,"val/","images/",image_file_name)
            one_img_labels =[]
       
            for i in range(len(v_annotations)): 
                if(v_annotations[i]["image_id"]==image_id):
                    class_and_bbox= v_annotations[i]["bbox"]
                    class_and_bbox.insert(0, v_annotations[i]["category_id"])  # 类别，leftx，top_y, w, h
                    one_img_labels.append(class_and_bbox)
                    
                else :
                    del v_annotations[0:i]
                    break
            if len(one_img_labels)==0:
                continue
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
            pred = model(img)[0]  # only get inference result
            
            # 这里每一个预测的深度是15， 其中最后面的四个是sigmas， 所以还有前面的11维度， 前面四个维度是表示边界框的，还有七个维度，一个表示obj存在的概率，剩下的6个，表示六个类别的概率
        
            t2 = torch_utils.time_synchronized()
            print(t2 - t1)
            
        
            pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
           
            #torch.Size([5, 16]) 66666
            # 这里的pred就是在 512*320上的xy，xy 还没有减去padding
            t3 = time.time()
            print(t3 - t2)

            if pred is None:
                print("No target detected.")
                # exit(0)
                continue

            # process detections
            pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
            
            # 对 sigma进行放大
            gain=max(img.shape[2:])/ max(img_o.shape)
            pred[:,-4:]/=gain
            pred[:,-4:]*=3 # 表示3σ
            # 对 sigma进行放大
            
            # 过滤掉低概率的目标
            scores = pred[:, 4].detach().cpu().numpy()
            idxs = np.greater(scores, 0.2)
            pred=pred[idxs]
            # 过滤掉低概率的目标

            bboxes = pred[:, :4].detach().cpu().numpy()
            scores = pred[:, 4].detach().cpu().numpy()
            classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1
            class_scores=pred[:, 6:12].detach().cpu().numpy()
            sigmas=pred[:,-4:].detach().cpu().numpy()
            # 这就是16个维度表示的内容
            pes=[]
            for xx in list(class_scores):
                pe=0
                for i in xx:
                    pe=pe+i*(-np.log(i))
                round(pe, 3)
                pes.append(pe)
            pes=np.array(pes)
            
            one_img_labels = np.array(one_img_labels)
            gt_bbox = one_img_labels[:,1:]
            gt_bbox = xywh2xyxy(gt_bbox)  # n*4维度的数据
         
            for i in range(len(pred)): # 对于每一个预测框而言的
                ious= utils.bbox_iou(pred[i][:4],torch.from_numpy(gt_bbox).to(pred.device))

                #得到这个预测框 和 一张图片中的所有真值框计算IOU，找到最大的IOU
                max_iou, maxiou_index = torch.max(ious,dim=0)
                if(max_iou>=0.25):
                   
                    index = int(maxiou_index.cpu().item())
                    gt_class_box_of_this_predict_box= one_img_labels[index]
                    # gt中的类别下标是从0开始的
                    print("预测:", classes[i], "真值:", gt_class_box_of_this_predict_box[0])
                    one_hot_list = [0 for _ in range(6)]
                    one_hot_list[gt_class_box_of_this_predict_box[0]-1]=1
                    kl=scipy.stats.entropy(one_hot_list, class_scores[i])
                    # mean_list = [1/6 for _ in range(6)]
                    # mcp = scipy.stats.entropy(mean_list, class_scores[i])
                    # kl = kl + mcp
                    pe= pes[i]
                    is_classification_correct = 1
                    iou_ = max_iou.item()
                    box_width = bboxes[i][2]-bboxes[i][0]
                    box_height = bboxes[i][3] - bboxes[i][1]
                    sigma_ = (sigmas[i][0]/box_width+sigmas[i][2]/box_width+sigmas[i][1]/box_height+sigmas[i][3]/box_height)/4
                    if(classes[i] == gt_class_box_of_this_predict_box[0]):
                        print("类别判断正确...")
                        # 对类别真值减1，因为类别真值是从1开始的
                        is_classification_correct = 1
                    else :
                        print("类别判断错误... O.O'...")
                        is_classification_correct = 0
                    
                    yihang = [kl, pe, iou_, sigma_, is_classification_correct]
                    yihang = [str(i) for i in yihang]
                    if image_period=="Daytime" and image_weather=="Clear" :
                        f=open(easy_cases_txtfile_path,"a")
                        if os.path.getsize(easy_cases_txtfile_path):
                            f.write("\n"+" ".join(yihang))
                        else:
                            f.write(" ".join(yihang))
                    else:
                        f=open(hard_cases_txtfile_path,"a")
                        if os.path.getsize(easy_cases_txtfile_path):
                            f.write("\n"+" ".join(yihang))
                        else:
                            f.write(" ".join(yihang))
            


if __name__ == "__main__":
    main()
