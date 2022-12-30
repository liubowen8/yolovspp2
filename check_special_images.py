"""
这个python，用于查看 特定特征的图片
soda数据集中图片的特征信息："city": "Shanghai", "location": "Highway", "period": "Daytime", "weather": "Rainy"
城市、位置、时间和天气
"""
import json

import cv2

annotations_tjson_path="/home/ifpp/ye/datasets/SODA10M/SSLAD-2D/labeled/annotations/instance_train.json"
annotations_vjson_path="/home/ifpp/ye/datasets/SODA10M/SSLAD-2D/labeled/annotations/instance_val.json"
def draw(city=None, location=None, period=None, weather=None):
    json_f=open(annotations_tjson_path,'r')
    json_dict=json.load(json_f)
    image_infos=json_dict["images"]  # list 里面是每张图片的信息 
    json_f_=open(annotations_vjson_path,'r')
    json_dict_=json.load(json_f_)
    image_infos_=json_dict_["images"]  # list 里面是每张图片的信息 
    for image_info in image_infos:
        #"file_name": "HT_VAL_000003_GZ_001.jpg", "id": 3, "height": 1080, "width": 1920, 
        #"city": "Guangzhou", "location": "Citystreet", "period": "Daytime", "weather": "Overcast"
        image_name= image_info["file_name"]
        image_city= image_info["city"]
        image_location = image_info["location"]
        image_period = image_info["period"]
        image_weather = image_info["weather"]
        if((city==None or city==image_city) and (location==None or location==image_location) and (period==None or period==image_period) and (weather==None or weather == image_weather)):
            image_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/train/images/"+image_name
            imagex = cv2.imread(image_path)
            cv2.imshow(image_path, imagex)
            cv2.waitKey(300)
            cv2.destroyAllWindows()

    for image_info in image_infos_:
        #"file_name": "HT_VAL_000003_GZ_001.jpg", "id": 3, "height": 1080, "width": 1920, 
        #"city": "Guangzhou", "location": "Citystreet", "period": "Daytime", "weather": "Overcast"
        image_name= image_info["file_name"]
        image_city= image_info["city"]
        image_location = image_info["location"]
        image_period = image_info["period"]
        image_weather = image_info["weather"]
        if((city==None or city==image_city) and (location==None or location==image_location) and (period==None or period==image_period) and (weather==None or weather == image_weather)):
            image_path = "/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/my_yolo_dataset/val/images/"+image_name
            imagex = cv2.imread(image_path)
            cv2.imshow(image_path, imagex)
            cv2.waitKey(300)
            cv2.destroyAllWindows()

    

if __name__=="__main__":
    draw(period="Night")
