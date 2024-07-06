import os
import numpy as np
import random
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

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

def plot_histrogram(cv2_img, color_space = "rgb"):
    if color_space == "rgb":

        axis_name = ["Blue", "Green", "Red"]
        # cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        # cv2_img = np.clip(cv2_img[:, :, :], 1, 255)
    elif color_space == "hsv":
        axis_name = ["H", "S", "V"]
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)

    blue_histogram = cv2.calcHist([cv2_img], [0], None, [256], [1, 256])
    red_histogram = cv2.calcHist([cv2_img], [1], None, [256], [1, 256])
    green_histogram = cv2.calcHist([cv2_img], [2], None, [256], [1, 256])

    plt.subplot(3, 1, 1)
    plt.title(f"histogram of {axis_name[0]}")
    plt.hist(blue_histogram, color="darkblue")

    plt.subplot(3, 1, 2)
    plt.title(f"histogram of {axis_name[1]}")
    plt.hist(green_histogram, color="green")

    plt.subplot(3, 1, 3)
    plt.title(f"histogram of {axis_name[0]}")
    plt.hist(red_histogram, color="red")

    plt.tight_layout()
    return plt

def plot_line(cv2_img, color_space="rgb"):

    if color_space == "rgb":
        axis_name = ["Blue", "Green", "Red"]
        cv2_img = np.clip(cv2_img[:, :, :], 1, 255)
    elif color_space == "hsv":
        axis_name = ["H", "S", "V"]
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)

    blue_histogram = cv2.calcHist([cv2_img], [0], None, [256], [1, 256])
    red_histogram = cv2.calcHist([cv2_img], [1], None, [256], [1, 256])
    green_histogram = cv2.calcHist([cv2_img], [2], None, [256], [1, 256])

    plt.subplot(3, 1, 1)
    plt.title(f"histogram of {axis_name[0]}")
    plt.plot(blue_histogram, color="darkblue")

    plt.subplot(3, 1, 2)
    plt.title(f"histogram of {axis_name[1]}")
    plt.plot(green_histogram, color="green")

    plt.subplot(3, 1, 3)
    plt.title(f"histogram of {axis_name[2]}")
    plt.plot(red_histogram, color="red")

    plt.tight_layout()
    return plt

def plot_multiple_line(img_list, color_space =  "rgb"):

    for img in img_list:
        plot_line(img, color_space)

    return plt


def plot3D(cv2_img, color_space = "rgb"):
    if color_space == "rgb":
        mask_and_crop_plt = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    elif color_space == "hsv":
        mask_and_crop_plt = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)


    mask_and_crop_plt = mask_and_crop_plt.astype('float32') / 255.0
    r, g, b = cv2.split(mask_and_crop_plt)
    fig = plt.figure()

    default = ["rgb", "hsv"]
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = mask_and_crop_plt.reshape((np.shape(mask_and_crop_plt)[0] * np.shape(mask_and_crop_plt)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()


    if color_space == "hsv":
        axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("H")
        axis.set_ylabel("S")
        axis.set_zlabel("V")
    elif color_space == "rgb":
        axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Red")
        axis.set_ylabel("Green")
        axis.set_zlabel("Blue")

    return plt

if __name__ == '__main__':
    path = "F:\\pud_ploy\\tomato_project\\test_class_improc"

    read = read_save()


    names = os.listdir(path)
    try:
        result_path = os.path.join(path, "test_class_result")
        os.mkdir(result_path)
    except:
        print("error mkdir")

    try:
        result_path_crop = os.path.join(path, "test_class_result_crop")
        os.mkdir(result_path_crop)
    except:
        print("error mkdir")

    try:
        result_path_crop_plot3d = os.path.join(path, "test_class_result_crop_plot3D")
        os.mkdir(result_path_crop_plot3d)
    except:
        print("error mkdir")

    params = {"crop": [100, 100], "HSV": [0, 178, 95, 23, 255, 255], "erode": [21, 2], "dilate": [10, 2], "contour_area": [0, 3000, 18, 1, 1, 1000]}

    classes = ["original", "cold", "wet", "other"]

    # for name in names:
    #     index_nameclass = find_class_in_classes(name, classes)
    #     print(f"{index_nameclass} {name}")
    for name in names:
        is_image_format = is_image(name)
        if is_image_format is not None:

            img = cv2.imread(os.path.join(path, name))
            try:
                result, _, _ = read.read_params(params,img)
            except Exception as e:
                print(e)
                continue

            img_proc = result["final"]

            try:
                _, contours, _ = cv2.findContours(img_proc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except:
                _, contours = cv2.findContours(img_proc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Define the YOLO label names
            print(len(contours))
            # Loop through all contours and create a YOLO label for each
            # label_file = os.path.join(label_path, name.split(".")[0] + ".txt")

            index = True

            print(name)

            # with open(label_file, 'w') as f:

            tomato_images = []
            for i, contour in enumerate(contours):
                print(f"box {i}")
                x, y, w, h = cv2.boundingRect(contour)
                # Create a black image with dimensions 500x500 and 3 channels (BGR)
                black_image = np.zeros(img_proc.shape[0:2], dtype=np.uint8)
                print(black_image.shape)
                cv2.drawContours(black_image, [contour], -1, (255, 255, 255), -1)
                type_kernel = cv2.MORPH_ELLIPSE  # ok
                kernel = cv2.getStructuringElement(type_kernel, (20, 20))

                black_image = cv2.dilate(black_image, kernel, iterations=1)
                # cv2.imshow("black_img", black_image)

                mask_and = cv2.bitwise_and(img,img, mask=black_image)
                mask_and_crop = mask_and[y:y+h, x: x+w]

                tomato_images.append(mask_and_crop)

                # Convert the image data type from uint8 to float32
                # cv2.imshow("mask_and", mask_and)

                #### save image
                cv2.imwrite(os.path.join(result_path, f"{name}_{i}.jpg"),mask_and)
                cv2.imwrite(os.path.join(result_path_crop, f"{name}_{i}.jpg"), mask_and_crop)




                # plt.show()
                #### save 3D
                # plot3D(mask_and_crop, "hsv")
                # plt.savefig(os.path.join(result_path_crop_plot3d,f"{name}_{i}_hsv.png"))
                # plot3D(mask_and_crop, "rgb")
                # plt.savefig(os.path.join(result_path_crop_plot3d,f"{name}_{i}_rgb.png"))

                #### save plot
                plot_histrogram(mask_and_crop, "rgb")
                plt.savefig(os.path.join(result_path_crop_plot3d,f"{name}_{i}_HIS_rgb.png"))
                plt.clf()
                plot_histrogram(mask_and_crop, "hsv")
                plt.savefig(os.path.join(result_path_crop_plot3d,f"{name}_{i}_HIS_hsv.png"))
                plt.clf()

                plot_line(mask_and_crop, "rgb")
                plt.savefig(os.path.join(result_path_crop_plot3d, f"{name}_{i}_LINE_rgb.png"))
                plt.clf()
                plot_line(mask_and_crop, "hsv")
                plt.savefig(os.path.join(result_path_crop_plot3d, f"{name}_{i}_LINE_hsv.png"))
                plt.clf()



                # plt.savefig('output_image.png')

            plot_multiple_line(tomato_images,"hsv")
            plt.savefig(os.path.join(result_path_crop_plot3d, f"{name}_LINE_hsv.png"))
            plt.clf()
            cv2.waitKey(1)

            plot_multiple_line(tomato_images, "rgb")
            plt.savefig(os.path.join(result_path_crop_plot3d, f"{name}_LINE_rgb.png"))
            plt.clf()
            cv2.waitKey(1)

