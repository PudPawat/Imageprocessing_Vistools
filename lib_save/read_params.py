from .improc_save import Imageprocessing
import cv2

class read_save(object):
    def __init__(self):
        self.imgproc = Imageprocessing()


    def read_params(self, params, frame):
        for key in params.keys():
            print(key)
            if key == "HSV":
                # frame_HSV, params['HSV'] = imgproc.HSV_range(frame, params[key])
                frame, params['HSV'] = self.imgproc.HSV_range(frame, params[key])

            elif key == "erode":
                # frame_erode, params['erode'] = imgproc.erode(frame, params[key])
                frame, params['erode'] = self.imgproc.erode(frame, params[key])

            elif key == "dilate":
                # frame_dialte, params['dilate'] = imgproc.dilate(frame, params[key])
                frame, params['dilate'] = self.imgproc.dilate(frame, params[key])

            elif key == "thresh":
                # frame_binary, params['thresh'] = imgproc.threshold(frame, params[key])
                frame, params['thresh'] = self.imgproc.threshold(frame, params[key])

            elif key == "sharp":
                # frame_sharp, params['sharp'] = imgproc.sharpen(frame, params[key])
                frame, params['sharp'] = self.imgproc.sharpen(frame, params[key])

            elif key == "blur":
                # frame_blur, params['blur'] = imgproc.blur(frame, params[key])
                frame, params['blur'] = self.imgproc.blur(frame, params[key])

            elif key == "line":
                # frame_line, lines, params['line'] = imgproc.line_detection(frame, frame0, params[key])
                if len(frame.shape) == 2:
                    frame0 = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame, lines, params['line'] = self.imgproc.line_detection(frame, frame0, params[key])

            elif key == "canny":
                # frame_canny, params['canny'] = imgproc.canny(frame, params[key], show=True)
                frame, params['canny'] = self.imgproc.canny(frame, params[key], show=True)

            elif key == "circle":
                # frame_circle, circle, params['circle'] = imgproc.circle_detection(frame, frame0, params[key], show=False)
                if len(frame.shape) == 2:
                    frame0 = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame, circle, params['circle'] = self.imgproc.circle_detection(frame, frame0, params[key], show=False)

        return frame

    def read_rectangle(self,config,img):
        for i in range(len(config['main'])):
            main = config['main'][str(i)]
            # imCrop = im[int(main[1]):int(main[1] + main[3]), int(main[0]):int(main[0] + main[2])]
            y0 = int(main[1])
            y1 = int(main[1] + main[3])
            x0 = int(main[0])
            x1 = int(main[0] + main[2])
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 3)

            for j in range(len(config['sub'][str(i)])):
                sub = config['sub'][str(i)][str(j)]
                sub_y0 = int(sub[1])
                sub_y1 = int(sub[1] + sub[3])
                sub_x0 = int(sub[0])
                sub_x1 = int(sub[0] + sub[2])
                cv2.rectangle(img, (x0 + sub_x0, y0 + sub_y0), (x0 + sub_x1, y0 + sub_y1), (0, 0, 255), 3)
        return img