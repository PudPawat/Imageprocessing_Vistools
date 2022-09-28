import cv2
import os
from lib_save.read_params import *
from copy import deepcopy
import numpy as np


from contour_after_process import contour_area, contour_center_distance_by_img,contour_area_by_img,contour_center_dis
from other_project.container.warp_and_reverse_warp import warp_polar,warp_reverser_warp


def resize_scale(img, scale = 0.3):
    resize = cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)))
    return resize

class OrientationDetection():
    def __init__(self, params):
        self.params = params
        self.read = read_save()
        pass

    def remove_background_by_diff_imgs(self, img1, img2):
        result1, _, _ = read.read_params(self.params, img1)
        img1 = result1["final"]
        self.img1_process = img1
        result2, _, _ = read.read_params(self.params, img2)
        img2 = result2["final"]
        self.img2_process = img2

        diff_img = img2 - img1
        # diff_img = img2

        # diff_img = cv2.blur(diff_img,(30,30))
        _, diff_img = cv2.threshold(diff_img, 100, 255, cv2.THRESH_BINARY)

        self.diff_img = diff_img

        return diff_img

    def remove_background(self,img1, img2):
        result1, _, _ = self.read.read_params(self.params, img1)
        img1 = result1["final"]
        self.img1_process = img1
        result2, _, _ = self.read.read_params(self.params, img2)
        img2 = result2["final"]
        self.img2_process = img2

        diff_img = img2 - img1
        # diff_img = img2

        # diff_img = cv2.blur(diff_img,(30,30))
        _, diff_img = cv2.threshold(diff_img, 100, 255, cv2.THRESH_BINARY)

        self.diff_img = diff_img
        # cv2.imshow("diff", diff_img)
        # cv2.waitKey(0)
        # diff_img = cv2.erode(diff_img,(k,k))
        # diff_img = cv2.dilate(diff_img,(k,k))
        # _, diff_img = cv2.threshold(diff_img,150,255,cv2.THRESH_BINARY)

        # diff_result,_ ,_ = read.read_params(params_after, diff_img)
        # diff_img = diff_result["final"]
        # print(diff_img.shape)
        try:
            _, contours = cv2.findContours(diff_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            _, contours, _ = cv2.findContours(diff_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for contour in contours:
        #     area = cv2.contourArea(contour)
        # right_contours, draw_img = contour_center_distance(diff_img, img2_copy, (crop_circle[0],crop_circle[1]),dis_criterion= 100 )
        right_contours = contour_area(contours, area_min=1000, area_max=100000000, write_area=False)
        right_contours = contour_center_dis(right_contours, (self.img2_process.shape[1] / 2, self.img2_process.shape[0] / 2), 30)
        self.right_contours = right_contours
        # right_contours, draw_img = contour_center_distance_by_img(diff_img, img2_copy,(img2_copy.shape[1]/2,img2_copy.shape[0]/2),30)
        print(right_contours)
        right_mask = np.zeros(diff_img.shape[:2], dtype="uint8")
        # for contour in right_contours:
        #     right_mask = cv2.drawContours(right_mask,[contour],-1,(255,255,255),10)
        right_mask = cv2.drawContours(right_mask, right_contours, -1, (255, 255, 255), -1)
        self.right_mask = right_mask
        return diff_img, right_mask, right_contours

    def detect_orientaion(self):
        if self.right_contours != []:
            M = cv2.moments(self.right_contours[0])

            if M["m00"] == 0.0:
                M["m00"] = 0.01

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            boundRect = cv2.boundingRect(self.right_contours[0])
            print(boundRect)
            self.circle = (cX,cY,boundRect[2]/2)
            # cv2.rectangle(drawing, (int(boundRect[0]), int(boundRect[1])), \
            #              (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
            _, self.warp = warp_polar(self.right_mask, self.circle)

            cv2.imshow("selfwarp", self.warp)
            # cv2.waitKey(0)



if __name__ == '__main__':
    read = read_save()
    folder = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\light2"
    folder = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\\0927_200000_focus"
    folder = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\image\\60000_focus"
    folder = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\image\\120000_focus"
    crop_circle = (546.1550518881522, 421.04824877486305, 375)
    names = os.listdir(folder)
    i = 0
    j = 5

    params = {'sobel': (3, 1, 1), 'gaussianblur': (2, 2), 'canny': (350, 350), "dilate": [40, 0],"erode": [40, 0]}
    params = {'HSV': [0, 0, 90, 180, 255, 255], 'erode': (20, 0), 'dilate': (20, 0)}
    # params = {'HSV': [52, 0, 80, 108, 118, 255], 'dilate': (20, 0), 'erode': (20, 0), 'dilate': (50, 0),'erode': (50, 0),}
    params_after = { "dilate": [30, 0],"erode": [30, 0]}
    img1 = cv2.imread(os.path.join(folder, names[i]),1)
    if max(img1.shape) > 1000:
        img1 = resize_scale(img1)
    img1_copy = deepcopy(img1)


    orientation_detection = OrientationDetection(params)


    for j in range(len(names)):

        img2 = cv2.imread(os.path.join(folder, names[j]),1)
        flag_rotate = [cv2.ROTATE_90_COUNTERCLOCKWISE,cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_180]
        img2 = cv2.rotate(img2, flag_rotate[j%3])
        img2_copy = deepcopy(img2)
        if max(img2.shape) > 1000:
            img2 = resize_scale(img2)
        cv2.imshow(names[i], img1)
        cv2.imshow(names[j], img2)



        # right_contours, draw_img = contour_area(diff_img, img2_copy, area_min=0, area_max=1000)
        # cv2.imshow(names[i],img1)
        # cv2.imshow(names[j],img2)

        diff_img, right_mask, right_contours = orientation_detection.remove_background(img1,img2)
        orientation_detection.detect_orientaion()

        # if j == 0:
        #     continue
        # print(right_mask.shape, type(right_mask))
        # print(diff_img.shape, type(diff_img))
        _, right_mask = cv2.threshold(right_mask,10,255,cv2.THRESH_BINARY)
        and_img = cv2.bitwise_and(img2_copy,img2_copy, mask= right_mask)

        try:
            and_img = cv2.bitwise_and(img2_copy,img2_copy, mask= right_mask)
            print("right mask")
        except:
            and_img = cv2.bitwise_and(img2_copy,img2_copy, mask= diff_img)

        _, warp = warp_polar(right_mask,(img1.shape[0]/2,img1.shape[1]/2,img1.shape[1]/2))
        cv2.imshow("warp",warp)

        cv2.imshow("and_img", and_img)

        # cv2.imshow("right_mask", right_mask)
        cv2.imshow("img2_copy", img2_copy)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

        img1 = deepcopy(img1_copy)
        cv2.destroyAllWindows()