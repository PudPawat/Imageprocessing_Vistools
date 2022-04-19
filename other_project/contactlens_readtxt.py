import pytesseract  
import os
import cv2

path = "../data/CONTACTLENS_READTXT/"
names = os.listdir(path)
print(names)
for name in names:
    read_im_path = os.path.join(path,name)
    # print(im)
    image = cv2.imread(read_im_path)

    boxes = pytesseract.image_to_boxes(image)
    print(boxes, type(boxes))


    str_read = pytesseract.image_to_string(image)
    print("--"*10,"\n"*2,str_read)
    cv2.imshow("asd", image)
    cv2.waitKey(0)


