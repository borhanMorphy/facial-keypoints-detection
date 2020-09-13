import transforms
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as vis_F
import random
import math
from typing import Tuple
from cv2 import cv2

class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
            Boolean: True if image is flipped
        """
        if torch.rand(1) < self.p:
            return vis_F.hflip(img),True
        return img,False

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        self.degrees = (-degrees, degrees)
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
            angle: float
        """

        angle = self.get_params(self.degrees)

        return vis_F.rotate(img, angle, self.resample, self.expand, self.center, self.fill),angle

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ')'
        return format_string

def rotate_points_counter_clockwise(points:np.ndarray, degree:float, center:Tuple):
    # points: N,2 as [(x,y),... ]
    new_points = np.empty(points.shape, dtype=np.float32)
    tmp_points = points.copy()
    tmp_points[:, 0] -= center[0]
    tmp_points[:, 1] -= center[1]

    radian = math.radians(degree)
    new_points[:,0] = tmp_points[:,0]*math.cos( radian ) + tmp_points[:,1]*math.sin( radian )
    new_points[:,1] = -tmp_points[:,0]*math.sin( radian ) + tmp_points[:,1]*math.cos( radian )

    new_points[:, 0] += center[0]
    new_points[:, 1] += center[1]

    return new_points

class TrainTransforms():
    def __init__(self, img_size:int, original_size:int, mean:float=0, std:float=1,
            brightness:float=0.3, contrast:float=0.5, saturation:float=0.5, hue:float=0.3,
            rotation_degree:int=10, hflip:float=0.5, debug:bool=False):

        self.original_size = original_size
        self.target_size = img_size
        self.to_pil = transforms.ToPILImage()
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.resize = transforms.Resize(img_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean,std)
        self.r_horizontal_flip = RandomHorizontalFlip(p=hflip)
        self.r_rotation = RandomRotation(rotation_degree)
        self.debug = debug

    def __call__(self, oimg:np.ndarray, targets:np.ndarray):
        targets = targets / self.original_size
        img = self.to_pil(oimg)
        img,angle = self.r_rotation(img)
        if self.debug: print(f"image rotated {angle} degree")
        targets = rotate_points_counter_clockwise(targets.reshape(-1,2), angle, (0.5,0.5))
        img,flipped = self.r_horizontal_flip(img)
        if flipped:
            if self.debug: print("image horizontally flipped")
            targets[:,0] = 1 - targets[:,0]
            tmp_targets = targets.copy()
            targets[1],targets[2] = tmp_targets[2],tmp_targets[1]
            targets[4],targets[6] = tmp_targets[6],tmp_targets[4]
            targets[5],targets[7] = tmp_targets[7],tmp_targets[5]
            targets[8],targets[10] = tmp_targets[10],tmp_targets[8]
            targets[9],targets[11] = tmp_targets[11],tmp_targets[9]
            targets[12],targets[13] = tmp_targets[13],tmp_targets[12]

            # 1 <=> 2
            # 4 <=> 6
            # 5 <=> 7
            # 8 <=> 10
            # 9 <=> 11
            # 12 <=> 13
            """
                nose_tip_x, nose_tip_y
                left_eye_center_x, left_eye_center_y
                right_eye_center_x, right_eye_center_y
                mouth_center_bottom_lip_x, mouth_center_bottom_lip_y
                left_eye_inner_corner_x, left_eye_inner_corner_y
                left_eye_outer_corner_x, left_eye_outer_corner_y
                right_eye_inner_corner_x, right_eye_inner_corner_y
                right_eye_outer_corner_x, right_eye_outer_corner_y
                left_eyebrow_inner_end_x, left_eyebrow_inner_end_y
                left_eyebrow_outer_end_x, left_eyebrow_outer_end_y
                right_eyebrow_inner_end_x, right_eyebrow_inner_end_y
                right_eyebrow_outer_end_x, right_eyebrow_outer_end_y
                mouth_left_corner_x, mouth_left_corner_y
                mouth_right_corner_x, mouth_right_corner_y
                mouth_center_top_lip_x, mouth_center_top_lip_y
            """

        targets = torch.from_numpy(targets.reshape(-1))
        img = self.color_jitter(img)
        img = self.resize(img)
        if self.debug:
            targets *= self.target_size
            return np.array(img,dtype=np.uint8), targets.numpy()
        img = self.to_tensor(img)
        img = self.normalize(img)

        return img,targets