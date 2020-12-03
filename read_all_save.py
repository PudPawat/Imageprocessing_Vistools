import cv2
import os
import json
import numpy as np
import math
import argparse
import random as rng
from pypylon import pylon
from pathlib import Path
from lib_save import Imageprocessing, read_save

def listDir(dir):
    fileNames = os.listdir(dir)
    Name = []
    for fileName in fileNames:
        Name.append(fileName)
        # print(fileName)

    return Name

def find_closest(boxes, reference): #array of boxes and center point of ref
    lengths = []
    index = None
    if boxes != []:
        for i,box in enumerate(boxes):
            x0,y0 = box[0]
            x1,y1 = box[1]
            center = [(x0+x1)/2,(y0+y1)/2]
            lengthX = (center[0] - reference[0]) ** 2
            lengthY = (center[1] - reference[1]) ** 2
            length = math.sqrt(lengthY + lengthX)
            lengths.append(length)
        # print(lengths)
        if lengths != []:
            lengths = np.array(lengths)
            min_length = np.min(lengths)
            if min_length <= 20 :
                for i, length in enumerate(lengths):
                    if length == min_length:
                        index = i
    return index


def process(img_bi,img_ori_result):
    '''
        process every selected region
    the process is detection system for detecting lift pin by using contour
    :param img:
    :return: img after process
    '''
    img_bi_write = cv2.cvtColor(img_bi, cv2.COLOR_GRAY2BGR)
    img = img_bi
    for i in range(len(rect_params['main'])):
        main = rect_params['main'][str(i)]
        # imCrop = im[int(main[1]):int(main[1] + main[3]), int(main[0]):int(main[0] + main[2])]
        y0 = int(main[1])
        y1 = int(main[1] + main[3])
        x0 = int(main[0])
        x1 = int(main[0] + main[2])
        cv2.rectangle(img_ori_result, (x0, y0), (x1, y1), (255, 0, 0), 3)
        cv2.rectangle(img_bi_write, (x0, y0), (x1, y1), (255, 0, 0), 3)

        crop = img_bi[y0:y1, x0:x1]

        plot = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        # cv2.imshow("qweqwe",plot)
        crop_contours, __ = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        filter_crop_contours = []
        boxes = []
        for contour in crop_contours:
            area = cv2.contourArea(contour)
            print(area)
            if area > 5:
                filter_crop_contours.append(contour)

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                topleft = box.min(axis=0)
                botright = box.max(axis=0)
                # cv2.drawContours(plot, [box], 0, (0, 0, 255), 1)
                # cv2.drawContours(plot, contour, 0, (255, 0, 0), 1)
                boxes.append([topleft, botright])
                print(area)

                # cv2.imshow("qwe", plot)
                # cv2.waitKey(0)

        for j in range(len(rect_params['sub'][str(i)])):
            sub = rect_params['sub'][str(i)][str(j)]
            sub_y0 = int(sub[1])
            sub_y1 = int(sub[1] + sub[3])
            sub_x0 = int(sub[0])
            sub_x1 = int(sub[0] + sub[2])
            center_ref = ((sub_x0 + sub_x1) / 2, (sub_y0 + sub_y1) / 2)
            index = find_closest(boxes, center_ref)
            if index != None:
                # print(index)
                area = cv2.contourArea(filter_crop_contours[index])
                print("area", area)
                if area > 24:
                    result = "NG"
                    cv2.rectangle(img_ori_result, (x0 + sub_x0, y0 + sub_y0), (x0 + sub_x1, y0 + sub_y1), (0, 0, 255),
                                  3)  # draw in big picture
                    cv2.rectangle(img_bi_write, (x0 + sub_x0, y0 + sub_y0), (x0 + sub_x1, y0 + sub_y1), (0, 0, 255),
                                  3)
                    cv2.drawContours(plot,[filter_crop_contours[index]], 0, (0, 255, 255), 3)
                    # cv2.drawContours(img_ori_result, filter_crop_contours[index], 1, (0, 255, 255), 1)

                    cv2.imshow("qwe", plot)
                    cv2.waitKey(0)

                else:
                    result = "OK"
            else:
                result = "OK"
            print(result)
        # cv2.imshow("asd", crop)
        # cv2.waitKey(0)
    return img_bi_write, img_ori_result


def main(params, rect_params):
    source = opt.source
    print(source)
    webcam = source.isnumeric()
    # print()
    imgproc = Imageprocessing()
    reading = read_save()
    # params ={}


    if webcam:
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    elif source == "pylon":
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        converter = pylon.ImageFormatConverter()
        # camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        # ========== Grabing Continusely (video) with minimal delay ==========
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # ========== converting to opencv bgr format ==========
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        rng.seed(12345)
    else:
        im_name = listDir(source)
        for name in im_name:
            key = ord("a")
            while key != ord("q"):
                frame = cv2.imread(source + "/" + name)
                frame0 = frame.copy()
                img_params = reading.read_params(params,frame)
                img_params_write, ori_write = process(img_params, frame0)
                cv2.imshow("Final", ori_write) # ori_write
                key = cv2.waitKey(0)

    frame = None
    while True:
        # frame = cv.imread(read_dir + "/" + name)

        if webcam:
            _, frame = cap.read()
            # print(frame.shape)

        elif source == "pylon":
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                frame = image.GetArray()
        else:
            break

        if frame is None:
            break
        frame_ori = frame.copy()
        frame_rect = reading.read_rectangle(rect_params, frame)
        cv2.imshow("show",frame_rect)
        key = cv2.waitKey(1)
        if key == ord("c"):
            img_params = reading.read_params(params, frame_ori)
            # print(frame.shape)
            img_params_write, ori_write = process(img_params, frame_ori)
            cv2.imshow("show", img_params_write)
            cv2.waitKey(0)

if __name__ == "__main__":
    # read params
    with Path("config/params.json").open("r") as f:
        params = json.load(f)

    # read rectangle
    with Path("config/rectangles_2.json").open("r") as f:
        rect_params = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='pylon',
                        help='source pylon number for webcam')  # file/folder, 0 for webcam

    opt = parser.parse_args()

    main(params,rect_params)