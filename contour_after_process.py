import cv2
from lib_save import Imageprocessing
from copy import deepcopy
import os

def contour_area(bi_image, draw_img,area_min = 0,area_max =1,write_area = True):
    contours, _ = cv2.findContours(bi_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    right_contours = []
    if draw_img != None:
        draw = True
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= area_min and area <= area_max:
            right_contours.append(contour)
            if draw:
                cv2.drawContours(draw_img,contour)
            if write_area:
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(draw_img,str(area),(cX,cY),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
    return right_contours, draw_img

def main_contour_proc(image,proc_param):
    frame = image
    draw_frame = frame.deepcopy()
    img_params = Imageprocessing.read_params(proc_param, frame)
    print(img_params[0])
    img = img_params[0]["final"]
    right_contours,draw_img = contour_area(img, draw_frame, 10 ,1000)
    # for contour in right_contours:
    #     cv2.drawContours(draw_frame)

    cv2.imshow("Final", img)  # ori_write
    # cv2.imwrite(output_path + name, crop_img)
    key = cv2.waitKey(0)

if __name__ == '__main__':
    source = ""
    params = ""
    im_name = os.listdir(source)
    # resize_img_baseon_FOV(im_name,source,params,output_path)
    for name in im_name:
        frame = cv2.imread(source + "/" + name)
