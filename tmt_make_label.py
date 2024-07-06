import os
import numpy as np
import random


import json
from easydict import EasyDict
from pathlib import Path
from copy import deepcopy
from PIL import Image
from lib.contour_after_process import contour_area, contour_center_dis,contour_center_X_or_Y,contour_big2small_n_order
from lib.warp_and_reverse_warp import warp_polar, reverse_warp
from lib_save.read_params import *
from lib.compare_img_module import FeatureVisualizationModule
from utils import crop_circle, rotate_image, crop_circle_by_warp, fill_balck_circle, is_image

### simple_tiny
from lib.custom_circle_detection import fit_circle_2d, get_x_y_from_contour

def find_class_in_classes(class_name, classes):

    for i, class_title in enumerate(classes):
        if class_title in class_name:
            return i, class_title

def contour_to_yolo(contour, extend_percent = 0):

    '''

    :param contour:
    :param extend_percent: number in range 0-100
    :return:
    '''
    x, y, w, h = cv2.boundingRect(contour)
    x_center = (x + w / 2) / img.shape[1]
    y_center = (y + h / 2) / img.shape[0]
    width = w*(1+ extend_percent/100)  / img.shape[1]
    height = h*(1+ extend_percent/100) / img.shape[0]

    return x_center, y_center, width, height



if __name__ == '__main__':
    path = "F:\\pud_ploy\\tomato_project\\All"
    path = "F:\\pud_ploy\\All"
    path = "F:\\pud_ploy\\tomato_project\\All_with_class"

    read = read_save()


    names = os.listdir(path)
    try:
        result_path = os.path.join(path, "result")
        os.mkdir(result_path)
    except:
        print("error mkdir")

    try:
        label_path = os.path.join(path, "labels_index_extended")
        os.mkdir(label_path)
    except:
        print("error mkdir")

    params = {"crop": [100, 100], "HSV": [0, 178, 95, 23, 255, 255], "erode": [21, 2], "dilate": [10, 2], "contour_area": [0, 3000, 18, 1, 1, 1000]}

    classes = ["original", "cold", "wet", "other"]

    for name in names:
        index_nameclass = find_class_in_classes(name, classes)
        print(f"{index_nameclass} {name}")
    for name in names:
        is_image_format = is_image(name)
        if is_image_format is not None:

            img = cv2.imread(os.path.join(path, name))

            result, _, _ = read.read_params(params,img)

            img_proc = result["final"]

            try:
                _, contours, _ = cv2.findContours(img_proc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except:
                _, contours = cv2.findContours(img_proc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Define the YOLO label names
            print(len(contours))
            # Loop through all contours and create a YOLO label for each
            label_file = os.path.join(label_path, name.split(".")[0] + ".txt")

            index = True

            print(name)

            with open(label_file, 'w') as f:
                for i, contour in enumerate(contours):
                    print(f"box {i}")
                    index_nameclass = find_class_in_classes(name, classes)
                    x_center, y_center, width, height = contour_to_yolo(contour, extend_percent= 15)
                    # Save the label as a YOLO text file
                    # with open(label_file, 'w') as f:
                    # f.write(f"{classes[0]} {x_center} {y_center} {width} {height}\n")

                    print(index_nameclass)
                    if index_nameclass is None:
                        index_nameclass = 0, classes[0]
                    print(index_nameclass)

                    if index:
                        f.write(f"{index_nameclass[0]} {x_center} {y_center} {width} {height}\n")
                    else:
                        f.write(f"{index_nameclass[1]} {x_center} {y_center} {width} {height}\n")

            cv2.imshow("test", img_proc)

            cv2.imwrite(os.path.join(result_path, name), img_proc)

            cv2.waitKey(1)


