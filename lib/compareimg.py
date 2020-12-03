import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt
import os


class FeatureVisualization():
    def __init__(self,  selected_layer=0):
        # self.img_path = img_path
        self.selected_layer = selected_layer
        # Load pretrained model
        self.pretrained_model = models.vgg16(pretrained=True).features
        # print(self.pretrained_model)
        self.pretrained_model2 = models.vgg16(pretrained=True)

    # @staticmethod
    def preprocess_image(self, cv2im, resize_im=True):

        # Resize image
        if resize_im:
            cv2im = cv2.resize(cv2im, (224, 224))
        im_as_arr = np.float32(cv2im)
        im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
        im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
        # Normalize the channels
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
        # Convert to float tensor
        im_as_ten = torch.from_numpy(im_as_arr).float()
        # Add one more channel to the beginning. Tensor shape = 1,3,224,224
        im_as_ten.unsqueeze_(0)
        # Convert to Pytorch variable
        im_as_var = Variable(im_as_ten, requires_grad=False)
        return im_as_var

    def process_image(self, img):
        # print('input image:')
        img = self.preprocess_image(img)
        return img

    def get_feature(self,img):
        # Image  preprocessing
        input = self.process_image(img)
        print("input.shape:{}".format(input.shape))
        x = input
        self.pretrained_model.eval()
        with torch.no_grad():
            for index, layer in enumerate(self.pretrained_model):
                x = layer(x)
                #             print("{}:{}".format(index,x.shape))
                if (index == self.selected_layer):
                    return x

    def get_conv_feature(self,img):
        # print("1")
        # Get the feature map
        features = self.get_feature(img)
        print("output.shape:{}".format(features.shape))
        result_path = './feat_' + str(self.selected_layer)

        if not os.path.exists(result_path):
            os.makedirs(result_path)

    def plot_probablity(self, outputs):

        outputs = outputs.data.numpy()
        outputs = np.ndarray.tolist(outputs)
        x = range(0, 4096)
        plt.bar(x, outputs[0])
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        plt.title("FC feature")
        plt.show()

    def get_fc_feature(self,img):
        input = self.process_image(img)
        self.pretrained_model2.eval()
        self.pretrained_model2.classifier = nn.Sequential(*list(self.pretrained_model2.classifier.children())[0:4])
        with torch.no_grad():
            outputs = self.pretrained_model2(input)
        # self.plot_probablity(outputs)
        return outputs