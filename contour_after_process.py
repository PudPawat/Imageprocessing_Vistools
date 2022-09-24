import cv2
from lib_save import Imageprocessing,read_save
from copy import deepcopy
import os

def contour_area(bi_image, draw_img,area_min = 0,area_max =1,write_area = True):
    try:
        _,contours,_ = cv2.findContours(bi_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        _,contours = cv2.findContours(bi_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    right_contours = []
    if type(draw_img) != type(None):
        draw = True
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        if area >= area_min and area <= area_max:
            right_contours.append(contour)
            if draw:
                cv2.drawContours(draw_img,[contour],-1,(255,0,0),5)
                pass
            if write_area:
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(draw_img,str(area),(cX,cY),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),5)
    return right_contours, draw_img

def main_contour_proc(image,proc_param):
    min_area = 400
    improc = read_save()
    frame = image
    draw_frame = deepcopy(frame)
    img_params = improc.read_params(proc_param, frame)
    print(img_params[0])
    img = img_params[0]["final"]
    cv2.namedWindow("after proc",cv2.WINDOW_NORMAL)
    cv2.imshow("after proc",img)
    right_contours,draw_img = contour_area(img, draw_frame, 400 ,5000)
    # for contour in right_contours:
    #     cv2.drawContours(draw_frame)
    cv2.namedWindow("Final_contour",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Draw",cv2.WINDOW_NORMAL)
    cv2.imshow("Final_contour", img)  # ori_write
    cv2.imshow("Draw", draw_img)  # ori_write
    print("asdasd")
    if not os.path.exists("./output/casting/"+str(min_area)+"/"):
        os.mkdir("./output/casting/"+str(min_area)+"/")
    cv2.imwrite("./output/casting/"+str(min_area)+"/" + name, draw_img)
    key = cv2.waitKey(0)
    return draw_img

if __name__ == '__main__':
    source = "F:\Pawat\Projects\Imageprocessing_Vistools\data\casting"
    # params = {"gaussianblur": [5, 29], "sobel": [5, 100, 1], "blur": 14, "HSV": [0, 0, 76, 180, 255, 255], "erode": [5, 0], "dilate": [4, 0]}
    # params = {'gaussianblur': (13, 13), 'sobel': (3, 100, 6), 'blur': 16, 'HSV': [0, 0, 120, 180, 255, 255],"dilate": [2, 0], "erode": [5, 0]}
    params = {'gaussianblur': (21, 33), 'sobel': (5, 56, 1), 'blur': 1, 'HSV': [0, 0, 110, 180, 255, 220],"erode": [10, 0], "dilate": [14, 0]}
    im_name = os.listdir(source)
    # resize_img_baseon_FOV(im_name,source,params,output_path)
    for name in im_name:
        frame = cv2.imread(source + "/" + name)
        main_contour_proc(frame,params)
