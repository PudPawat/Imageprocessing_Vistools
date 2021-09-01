import cv2
import numpy as np
from read_all_save import *
from lib_save import read_save
import imutils

class ShapeDetector:
    def __init__(self,img):
        self.img = img
        pass
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        # cv2.arcLength(c, True)
        # for factor in range(0,100,1):
        #     Fac = factor*0.001
        #     print(Fac)
        #     approx = cv2.approxPolyDP(c, Fac* peri, True)
        #     print(approx,len(approx))
        #     cv2.drawContours(self.img,approx,-1,(0, 255, 0), 2)
        #     cv2.imshow("app",self.img)
        approx = cv2.approxPolyDP(c, 0.006 * peri, True)  #0.008
        print(approx,len(approx))
        approx_y = sorted(approx, key = lambda x: x[0][1])
        print(approx_y,"approx_1\n",approx_y[0:3])
        cv2.drawContours(self.img,[approx],-1,(0, 255, 0), 2)
        cv2.imshow("app",self.img)

# read .json
def read_json(path2json):
    path2json = "config/params_propeller2.json"
    with Path(path2json).open("r") as f:
        opt = json.load(f)
        print(opt)


img_path = "data/Image__2021-08-30__20-16-51.jpg"
img = cv2.imread(img_path)
img = img[0:-1,int(img.shape[1]*0.3):int(img.shape[1]*0.7)]
cv2.imshow("test",img)
cv2.waitKey(0)

reading = read_save()
proc_img, _, _ = reading.read_params(opt,img)


cv2.imshow("test1",proc_img["final"])
cv2.waitKey(0)

# ShapeDetector().detect(proc_img)

cnts = cv2.findContours(proc_img["final"].copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
sd = ShapeDetector(img)

ratio = 1

for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)
    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    cv2.putText(img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 2)
    # show the output image
    cv2.imshow("Image", img)
    cv2.waitKey(0)