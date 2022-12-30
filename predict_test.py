import json
import os
import time
from operator import gt

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from build_utils import img_utils, torch_utils, utils
from draw_box_utils import draw_objs
from models import Darknet

# from torchinfo import summary
"""
1122 今天的修改时想在原来的图上画上gt bbox ，想要的效果是预测box和gt box相差比较大时，reg的不确定性也比较大
"""

# dropout保持开启
def apply_dropout(m):
    if type(m)==torch.nn.Dropout:
        m.train()

def main():
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/my_yolov3.cfg"  # 改成生成的.cfg文件xclock
    # weights = "weights/yolov3spp-365.pt"  # 改成自己训练好的权重文件
    weights = "weights/yolov3spp-399.pt"  # 改成自己训练好的权重文件

    # weights = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/weights/dropout_weights/yolov3spp-400.pt"
    json_path = "./data/soda_classes.json"  # json标签文件
    # img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_000005_SH_001.jpg"
    # img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_000012_SH_001.jpg"
    # img_path = "/home/ifpp/ye/datasets/kitti/tracking/data/training/image_02/0020/000000.png"
    # img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_002158_SH_011.jpg"
    img_path = "/home/ifpp/ye/datasets/SODA10M/SSLAD-2D/labeled/val/HT_VAL_000016_SZ_230.jpg"
    img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_004607_SZ_010.jpg"
    # img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/snowy/57753a1ab95cdb9ed0e1fe590cccd991.jpeg"
    img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_000033_SZ_010.jpg"
    # 上图中有一个ood数据
    img_path= "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/train/images/HT_TRAIN_000004_SH_000.jpg"
    # 上为十字路口的图

    # img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_000176_SH_012.jpg"
    # rainy1
    # img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_000254_SH_012.jpg"
    # rainy 2
    # img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_000772_SH_012.jpg"
    # 下雨 又 路况复杂
    # img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_002591_SH_012.jpg"
    # 下雨

    # img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_000364_SH_012.jpg"
    # # 远光灯

    # img_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/HT_VAL_000411_SH_230.jpg"
    # 光线差
    img_path = "/home/ifpp/ye/datasets/SODA10M/SSLAD-2D/labeled/val/HT_VAL_000033_SZ_010.jpg"
    # gt_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/labels/HT_VAL_000915_GZ_000.txt"
    # # 上述两个应该一起出现，一个是图片，一个是它对应的label
    # gt_box = np.loadtxt(gt_path)
    # gt_box = gt_box[:,1:]
    # gt_box[:, 0::2]*=1920
    # gt_box[:,1::2 ]*=1080
    # print(gt_box)
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
    #model.apply(apply_dropout)
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
        pred = model(img)[0]  # only get inference result
        print(pred.shape,555555555)
        # 这里每一个预测的深度是15， 其中最后面的四个是sigmas， 所以还有前面的11维度， 前面四个维度是表示边界框的，还有七个维度，一个表示obj存在的概率，剩下的6个，表示六个类别的概率
       
        t2 = torch_utils.time_synchronized()
        print(t2 - t1)
        
      
        pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
        print(pred.shape, 66666)
        #torch.Size([5, 16]) 66666
        # 这里的pred就是在 512*320上的xy，xy 还没有减去padding
        locations=pred.clone()
        t3 = time.time()
        print(t3 - t2)

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
        
        print(img.shape[2:], pred[:, :4], img_o.shape)
        print(pred.shape,123)

        bboxes = pred[:, :4].detach().cpu().numpy()
        sigmas=pred[:,-4:].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1
        class_scores=pred[:, 6:12].detach().cpu().numpy()
        pes=[]
        for xx in list(class_scores):
            pe=0
            for i in xx:
                pe=pe+i*(-np.log(i))
            round(pe, 3)
            pes.append(pe)
        pes=np.array(pes)
        # print(pes, 77777)
        if  pred.shape[1]==7:
            uns=pred[:, 6].detach().cpu().numpy()
         # 过滤掉低概率的目标
        idxs = np.greater(scores, 0.2)
        locations=locations[idxs]



        pil_img = Image.fromarray(img_o[:, :, ::-1])
        plot_img = draw_objs(pil_img,
                             bboxes,
                             classes,
                             scores,
                             sigmas=sigmas,
                             pes=pes,
                             gt_box=None,
                             category_index=category_index,
                             box_thresh=0.20,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20,
                             )
        # plot_img = draw_objs(pil_img,
        #                      bboxes,
        #                      classes,
        #                      scores,
        #                      uns=uns,
        #                      category_index=category_index,
        #                      box_thresh=0.2,
        #                      line_thickness=3,
        #                      font='arial.ttf',
        #                      font_size=20)
        
        # plt.imshow("xx", plot_img)
        # plt.show()
        # 保存预测的图片结果
        plot_img.save("ratest_result1.jpg")
        
        # all_location_fms=[]

        # # # 整个特征图
        # # this_location_fms=[]
        # # for fm in fms:   #  对应于 每一个框框
        # #     fm = fm.reshape((1,-1))
        # #     pinjie=[]
        # #     for i in range(10):
        # #         suiji=torch.randn(1,fm.shape[1]).to(device)
        # #         pinjie.append(fm+suiji)
        # #     pinjie=torch.cat(pinjie,0)
        # #     #fm=fm.repeat(10,1)
        # #     this_location_fms.append(pinjie)
        # # all_location_fms.append(this_location_fms)
        # # # 整个特征图

        # for location in locations:  # 每一个 位置
        #     sigmas_=location[-4:]
        #     input_image_h, input_image_w=img.shape[-2:]
        #     x_left, y_top, x_right, y_down= location[:4]
        #     x_left_n, x_right_n= x_left/input_image_w, x_right/input_image_w
        #     y_top_n, y_down_n= y_top/input_image_h, y_down/input_image_h
        #     this_location_fms=[]
            
        #     for fm in fms:   #  对应于 每一个框框
        #         fm_h, fm_w= fm.shape[-2:]
        #         fm_xl, fm_xr=int(x_left_n.item()*fm_w), int(x_right_n.item()*fm_w)+1
        #         fm_yt, fm_yd=int(y_top_n.item()*fm_h), int(y_down_n.item()*fm_h)+1
        #         fm=fm[:,:,fm_yt:fm_yd,fm_xl:fm_xr]
        #         fm = fm.reshape((1,-1))
        #         pinjie=[]
        #         for i in range(10):
        #             suiji=torch.randn(1,fm.shape[1]).to(device)
        #             pinjie.append(fm+suiji*5.0)
        #         pinjie=torch.cat(pinjie,0)
        #         xx=pinjie.clone()
        #         pinjie=pca(xx.to("cpu"), 50)
        #         print(pinjie.shape)
        #         #fm=fm.repeat(10,1)
        #         this_location_fms.append(pinjie)
        #     all_location_fms.append(this_location_fms)
        # all_location_fms=np.array(all_location_fms)
        # np.save( 'all_location_fms.npy', all_location_fms)


        # print(locations[:,:4])
        # print(locations.shape)
        # print(img.shape)


if __name__ == "__main__":
    main()
