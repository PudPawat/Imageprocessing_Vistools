from lib_save import read_save
import json
import cv2

file_path = "lib_save/sample_params.json"
image_path = "cosmetic/Image__2020-11-22__22-51-14.jpg"
with open(file_path, 'r') as file:
    param = json.load(file)

img = cv2.imread(image_path)

read_save().read_params(param, img, show= True)