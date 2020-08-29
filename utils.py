from typing import Tuple
import numpy as np
from cv2 import cv2

def str2img(str_img:str, img_size:Tuple=(96,96)) -> np.ndarray:
    """converts string image to numpy image"""
    return np.array([int(pixel) for pixel in str_img.split(" ")]).astype(np.uint8).reshape(img_size)

def load_img(img_path:str) -> np.ndarray:
    """loads image with given file path"""
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

def visualize(img:np.ndarray, keypoints:np.ndarray) -> int:
    """show image with keypoints

    Args:
        img (np.ndarray): h,w or h,w,3 channeled image
        keypoints (np.ndarray): N,2 keypoints

    Returns:
        int: clicked cv2.waitKey key
    """
    simg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    assert len(keypoints.shape) = 2 and keypoints.shape[-1] == 2,"wrong keypoints format"
    for x,y in keypoints.astype(np.int32):
        simg = cv2.circle(simg, (x,y), (2), (0,0,255))
    cv2.imshow("", simg)
    return cv2.waitKey(0)

def seed_everything(magic_number:int):
    # TODO
    pass