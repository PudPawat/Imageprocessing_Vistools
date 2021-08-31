import os
# from .usbrelay import *
import time
import numpy as np
import cv2
import json
import pandas as pd
import copy
import math

def listDir(dir):
    """
    Function Name: listDir

    Description: Input directory and return list of all name in that directory

    Argument:
        dir [string] -> [directory]

    Return:
        [list] -> [name of all files in the directory]

    Edited by: 12-4-2020 [Pawat]
    """
    fileNames = os.listdir(dir)
    Name = []
    for fileName in fileNames:
        Name.append(fileName)
        # print(fileName)

    return Name



