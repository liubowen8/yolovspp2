"""
本脚本有两个功能：
1.将voc数据集标注信息(.xml)转为yolo标注格式(.txt)，并将图像文件复制到相应文件夹
2.根据json标签文件，生成对应names标签(my_data_label.names)
"""
import json
import os
import shutil
from tkinter import X

from tqdm import tqdm

# voc数据集根目录以及版本
soda_root = "/home/ifpp/ye/datasets/SODA10M/SSLAD-2D/labeled"
soda_train="/home/ifpp/ye/datasets/SODA10M/SSLAD-2D/labeled/train"
soda_val="/home/ifpp/ye/datasets/SODA10M/SSLAD-2D/labeled/val"
annotations_json_path="/home/ifpp/ye/datasets/SODA10M/SSLAD-2D/labeled/annotations/"

# 转换的训练集以及验证集对应txt文件
train_txt = "train.txt"
val_txt = "val.txt"

# 转换后的文件保存目录
save_file_root = "./my_yolo_dataset"

# label标签对应json文件
label_json_path = '/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/data/soda_classes.json'

# 检查文件/文件夹都是否存在
assert os.path.exists(label_json_path), "label_json_path does not exist..."
if os.path.exists(save_file_root) is False:
    os.makedirs(save_file_root)



def translate_info(file_names: list, save_root: str, class_dict: dict, annotations_json_path, train_val='train'):
    """
    将对应xml文件信息转为yolo中使用的txt文件信息
    :param file_names:
    :param save_root:
    :param class_dict:
    :param train_val:
    :return:
    """
    save_txt_path = os.path.join(save_root, train_val, "labels")
    if os.path.exists(save_txt_path) is False:
        os.makedirs(save_txt_path)
    save_images_path = os.path.join(save_root, train_val, "images")
    if os.path.exists(save_images_path) is False:
        os.makedirs(save_images_path)
    
    json_path=annotations_json_path+"instance_"+train_val+".json"
    json_f=open(json_path,'r')
    json_dict=json.load(json_f)
    annotations=json_dict["annotations"]  # list 里面是每一个object的标注信息，一个对象使用一个字典标注
    image_infos=json_dict["images"]  # list 里面是每张图片的信息 
    # 生成label文件夹和 其下的txt文件
    for object_info in tqdm(annotations):
        #object_info是一个字典
        image_id=object_info["image_id"]
        #"category_id": 3, "bbox": [65, 667, 174, 126], "area": 21924, "id": 1, "iscrowd": 0
        category_id=object_info["category_id"]
        bbox=object_info["bbox"]
        left, top, w, h= bbox 
        xcenter=left+w/2
        ycenter=top+h/2
        #"height": 1080, "width": 1920,
        image_info=image_infos[image_id-1]
        image_height=image_info["height"]
        image_width=image_info["width"]
        file=image_info["file_name"][:-4]
        with open(os.path.join(save_txt_path, file + ".txt"), "a") as f:
            class_index=category_id-1
            # 绝对坐标转相对坐标，保存6位小数
            xcenter = round(xcenter / image_width, 6)
            ycenter = round(ycenter / image_height, 6)
            w = round(w / image_width, 6)
            h = round(h / image_height, 6)
            if 0<=xcenter<=1 and 0<=ycenter<=1 and w<=1 and h<=1 :
                info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]
                if os.path.getsize(os.path.join(save_txt_path, file + ".txt")):
                    f.write("\n" + " ".join(info))
                else:
                    f.write(" ".join(info))
                

    for file in tqdm(file_names, desc="translate {} file...".format(train_val)):
        # 检查下图像文件是否存在
        img_path = os.path.join(soda_root, train_val, file + ".jpg")
        assert os.path.exists(img_path), "file:{} not exist...".format(img_path)

        # copy image into save_images_path
        path_copy_to = os.path.join(save_images_path, img_path.split(os.sep)[-1])
        if os.path.exists(path_copy_to) is False:
            shutil.copyfile(img_path, path_copy_to)


def create_class_names(class_dict: dict):
    keys = class_dict.keys()
    with open("./data/my_data_label.names", "w") as w:
        for index, k in enumerate(keys):
            if index + 1 == len(keys):
                w.write(k)
            else:
                w.write(k + "\n")


def main():
    # read class_indict
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)


    #train
    train_images=[i[:-4] for i in os.listdir(soda_train)]
    train_images.sort(key=lambda x:int(x[9:15]))
    translate_info(train_images, save_file_root, class_dict, annotations_json_path , "train")

    # val
    val_images=[i[:-4] for i in os.listdir(soda_val)]
    val_images.sort(key=lambda x:int(x[7:13]))
    translate_info(val_images, save_file_root, class_dict, annotations_json_path , "val")


    # 创建my_data_label.names文件
    create_class_names(class_dict)


if __name__ == "__main__":
    main()
