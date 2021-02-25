import os
import cv2
import numpy as np
import pandas as pd
import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import math
import random
import numpy as np
from scipy.stats import beta
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform, NoOp, to_tuple
from PIL import Image
import albumentations
from PIL import Image, ImageOps, ImageEnhance
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset


class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.
    
    Author: Qishen Ha
    Email: haqishen@gmail.com
    2020/01/29

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                
                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        
        if (image.shape[2]==3) | (image.shape[2]==1): #channel last 
            h, w = image.shape[:2]         
            mask = mask[:, :, np.newaxis]
            image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
            
        else: # channel first
            h, w = image.shape[1:]
            mask = mask[np.newaxis, :, :]
            image *= mask[:,rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
                            
#         h, w = image.shape[:2]
#         mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
#         mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
#         mask = mask[:, :, np.newaxis] if (image.shape[2]==3) & (image.shape[2]==1) else mask[np.newaxis, :, :]
#         image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        if (img.shape[2]==3) | (img.shape[2]==1): #channel last 
            height, width = img.shape[:2]
                                
        else: # channel first
            height, width = img.shape[1:]
        self.init_masks(height, width)
#         print(f'height : {height}, width : {width}') 
        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')
    
    
################## lens distortion ##########################
class LensDistortion(ImageOnlyTransform):

    def __init__(self, exp_min = 0.8, exp_max = 1.2, scale=1, always_apply=False, p=0.5):
        super(LensDistortion, self).__init__(always_apply, p)
        self.exp_min = exp_min
        self.exp_max = exp_max
        self.scale = scale
        
    def apply(self, image, exp, **params):
        rows, cols = image.shape[:2]
        
        mapy, mapx = np.indices((rows, cols),dtype=np.float32)
        mapx = 2*mapx/(cols-1)-1
        mapy = 2*mapy/(rows-1)-1

        r, theta = cv2.cartToPolar(mapx, mapy)
        r[r< self.scale] = r[r<self.scale] **exp  
        mapx, mapy = cv2.polarToCart(r, theta)

        mapx = ((mapx + 1)*cols-1)/2
        mapy = ((mapy + 1)*rows-1)/2
        distorted = cv2.remap(image ,mapx,mapy,cv2.INTER_LINEAR)

        return distorted

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        
        exp = np.random.choice([self.exp_min+0.05, self.exp_max-0.05], size=1)[0]

        
        return { "exp": exp}
 
    
    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("var_limit",)

    
################## Vinyl shining ##########################    
class VinylShining(ImageOnlyTransform):

    def __init__(self, n_shinnings, always_apply=False, p=0.5):
        super(VinylShining, self).__init__(always_apply, p)
        self.n_shinnings = n_shinnings

    def apply(self, image, vertices_list, **params):
        image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) ## Conversion to HSV
#         mask = np.zeros_like(image, np.uint8)
        mask = np.zeros((image.shape), dtype=np.uint8)
#         print('satuation max value  : ' , image[:,:,1].max())
        for vertices in vertices_list[:]:
            mask = cv2.fillPoly(mask, vertices, (255, 255,255)) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
        
        values = image_HSV[:,:,2][mask[:,:,1]==255] * 1.5 # value, higher, brighter
        values = [255 if x > 255 else x for x in values] # max clipping
        sat = image_HSV[:,:,1][mask[:,:,1]==255] * 0.2  # saturation, lower, whiter

        # Saturation, Value all of these has max value  = 255
        image_HSV[:,:,2][mask[:,:,1]==255] = values ## if red channel is hot, image's "Lightness" channel's brightness is lowered
        image_HSV[:,:,1][mask[:,:,1]==255] = sat 
        image_RGB = cv2.cvtColor(image_HSV,cv2.COLOR_HSV2RGB)

        return image_RGB

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
#         if (image.shape[0]==3) or (image.shape[0]==1): # channel first
#             image = np.transpose(image, (1,2,0))
         
        vertices_list = self.generate_coordinates(image.shape)
        return { "vertices_list": vertices_list}
    
    
    def generate_coordinates(self, imshape):
        vertices_list=[]    
        for index in range(self.n_shinnings):        
            vertex=[]        
            for dimensions in range(np.random.randint(3,7)): ## Dimensionality of the shadow polygon            
                vertex.append(( imshape[1]*np.random.uniform(),imshape[0]*np.random.uniform()))        
                vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices         
                vertices_list.append(vertices)    
        return vertices_list ## List of shadow vertices
    
    
    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("var_limit",)
    
# class VinylShining(ImageOnlyTransform):

#     def __init__(self, n_shinnings, always_apply=False, p=0.5):
#         super(VinylShining, self).__init__(always_apply, p)
#         self.n_shinnings = n_shinnings

#     def apply(self, image, vertices_list, **params):
#         image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) ## Conversion to HSV
# #         mask = np.zeros_like(image, np.uint8)
#         mask = np.zeros((image.shape), dtype=np.uint8)
        
#         for vertices in vertices_list:
#             mask = cv2.fillPoly(mask, vertices, (255, 255,255)) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
#             image_HSV[:,:,2][mask[:,:,1]==255] = image_HSV[:,:,2][mask[:,:,1]==255] =220 ## if red channel is hot, image's "Lightness" channel's brightness is lowered
#             image_HSV[:,:,1][mask[:,:,1]==255] = image_HSV[:,:,1][mask[:,:,1]==255] * 0.9
#             image_RGB = cv2.cvtColor(image_HSV,cv2.COLOR_HSV2RGB)
#         return image_RGB

#     def get_params_dependent_on_targets(self, params):
#         image = params["image"]
# #         if (image.shape[0]==3) or (image.shape[0]==1): # channel first
# #             image = np.transpose(image, (1,2,0))
         
#         vertices_list = self.generate_coordinates(image.shape)
#         return { "vertices_list": vertices_list}
    
    
#     def generate_coordinates(self, imshape):
#         vertices_list=[]    
#         for index in range(self.n_shinnings):        
#             vertex=[]        
#             for dimensions in range(np.random.randint(3,7)): ## Dimensionality of the shadow polygon            
#                 vertex.append(( imshape[1]*np.random.uniform(),imshape[0]*np.random.uniform()))        
#                 vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices         
#                 vertices_list.append(vertices)    
#         return vertices_list ## List of shadow vertices
    
    
#     @property
#     def targets_as_params(self):
#         return ["image"]

#     def get_transform_init_args_names(self):
#         return ("var_limit",)


######################## Fmix################################
def fftfreqnd(h, w=None, z=None):
    """ Get bin values for discrete fourier transform of size (h, w, z)
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)

        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)


def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    """ Samples a fourier image with given size and frequencies decayed by decay power
    :param freqs: Bin values for the discrete fourier transform
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param ch: Number of channels for the resulting mask
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)

    scale = np.expand_dims(scale, -1)[None, :]

    return scale * param


def make_low_freq_image(decay, shape, ch=1):
    """ Sample a low frequency image from fourier space
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param ch: Number of channels for desired mask
    """
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, ch, *shape)#.reshape((1, *shape[:-1], -1))
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))

    if len(shape) == 1:
        mask = mask[:1, :shape[0]]
    if len(shape) == 2:
        mask = mask[:1, :shape[0], :shape[1]]
    if len(shape) == 3:
        mask = mask[:1, :shape[0], :shape[1], :shape[2]]

    mask = mask
    mask = (mask - mask.min())
    mask = mask / mask.max()
    return mask


def sample_lam(alpha, reformulate=False):
    """ Sample a lambda from symmetric beta distribution with given alpha
    :param alpha: Alpha value for beta distribution
    :param reformulate: If True, uses the reformulation of [1].
    """
    if reformulate:
        lam = beta.rvs(alpha+1, alpha)
    else:
        lam = beta.rvs(alpha, alpha)

    return lam


def binarise_mask(mask, lam, in_shape, max_soft=0.0):
    """ Binarises a given low frequency image such that it has mean lambda.
    :param mask: Low frequency image, usually the result of `make_low_freq_image`
    :param lam: Mean value of final mask
    :param in_shape: Shape of inputs
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :return:
    """
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

    eff_soft = max_soft
    if max_soft > lam or max_soft > (1-lam):
        eff_soft = min(lam, 1-lam)

    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft

    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))

    mask = mask.reshape((1, *in_shape))
    return mask


def sample_mask(alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """ Samples a mean lambda from beta distribution parametrised by alpha, creates a low frequency image and binarises
    it based on this lambda
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    """
    if isinstance(shape, int):
        shape = (shape,)

    # Choose lambda
    lam = sample_lam(alpha, reformulate)

    # Make mask, get mean / std
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam, shape, max_soft)

    return lam, mask


def sample_and_apply(x, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """
    :param x: Image batch on which to apply fmix of shape [b, c, shape*]
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    :return: mixed input, permutation indices, lambda value of mix,
    """
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    index = np.random.permutation(x.shape[0])

    x1, x2 = x * mask, x[index] * (1-mask)
    return x1+x2, index, lam


class FMixBase:
    r""" FMix augmentation
        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].
    """

    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        super().__init__()
        self.decay_power = decay_power
        self.reformulate = reformulate
        self.size = size
        self.alpha = alpha
        self.max_soft = max_soft
        self.index = None
        self.lam = None

    def __call__(self, x):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        raise NotImplementedError
        
        
################### mix up ########################        
def mixup(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.7)
    data = lam*data + (1-lam)*shuffled_data
    targets = (target, shuffled_target, lam)

    return data, targets

############# augmix #############################

def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)

def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
#     translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
#     translate_x, translate_y, 
    color, contrast, brightness, sharpness
]


def apply_op(image, op, severity):
    #   image = np.clip(image, 0, 255)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img)

def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
    mixed: Augmented and mixed image.
    """
    ws = np.float32(
      np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

#     mix = np.zeros_like(image).astype(np.float32)
    mix = np.zeros((image.shape)).astype(np.float32)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug
#         mix += ws[i] * normalize(image_aug)

    mixed = (1 - m) * image + m * mix
#     mixed = (1 - m) * normalize(image) + m * mix
    return mixed.astype(np.uint8)

class RandomAugMix(ImageOnlyTransform):

    def __init__(self, severity=3, width=3, depth=-1, alpha=1., always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def apply(self, image, **params):

        image = augment_and_mix(
            image,
            self.severity,
            self.width,
            self.depth,
            self.alpha
        )
#     def get_params_dependent_on_targets(self, params):
#         image_ = params["image"]

        return image