import numpy as np
import pandas as pd
import cv2
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm

def norm(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

def resize_img(x, size):
    return resize(x, (size,size), order = 1, mode = 'constant', preserve_range=True,
            anti_aliasing=True, clip = True)

def resize_mask(x, size):
    return resize(x, (size,size), order=0, mode = 'constant', preserve_range = True, 
                  anti_aliasing=False, clip=True)

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """
    Elastic deformation of images as described in [Simard2003]_ (with modifications).
    [Simard2003] Simard, Steinkraus and Platt, 'Best Practices for Convolutional Neural 
    Networks applied to Visual Document Analysis', in Proc. of the International Conference 
    on Document Analysis and Recognition, 2003.

    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    
    deformed = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    rotated = np.rot90(deformed, np.random.randint(4))
    
    return rotated[...,0], rotated[...,1]

def draw_grid(im, grid_size):
    # Draw grid lines
	for i in range(0, im.shape[1], grid_size):
	    cv2.line(im, (i, 0), (i, im.shape[0]), color=(1,))
	for j in range(0, im.shape[0], grid_size):
	    cv2.line(im, (0, j), (im.shape[1], j), color=(1,))


def generate_dataset(imgs, masks, img_size, samples):
    x_o = imgs.get_fdata()
    y_o = masks.get_fdata()

    x_o = np.array([norm(resize_img(x_o[:,:,i], img_size)) for i in range(imgs.shape[2])])
    y_o = np.array([resize_mask(y_o[:,:,i], img_size) for i in range(imgs.shape[2])])
    x = []
    y = []

    for i in tqdm(range(samples)):
        sample = np.random.randint(imgs.shape[2])
        im = x_o[sample,:,:]
        im_mask = y_o[sample,:,:]
        im_merge = np.concatenate((im[...,None], im_mask[...,None]), axis=2)
        im_t, im_mask_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.07, im_merge.shape[1] * 0.07)
        x.append(im_t)
        y.append(im_mask_t)

    x = np.array(x)
    y = np.array(y)

    # Repeat the channel
    x = np.repeat(x[..., np.newaxis], 3, -1)
    x_o = np.repeat(x_o[..., np.newaxis], 3, -1)

    # We adjust the size to match the images
    y = y.reshape(samples, img_size, img_size, 1)
    y_o = y_o.reshape(imgs.shape[2], img_size, img_size, 1)
    
    return x_o, y_o, x, y

def generate_original_dataset(imgs, masks, img_size):
    x_o = imgs.get_fdata()
    y_o = masks.get_fdata()

    x_o = np.array([norm(resize_img(x_o[:,:,i], img_size)) for i in range(imgs.shape[2])])
    y_o = np.array([resize_mask(y_o[:,:,i], img_size) for i in range(imgs.shape[2])])

    x_o = np.repeat(x_o[..., np.newaxis], 3, -1)
    y_o = y_o.reshape(100, img_size, img_size,1)
    
    return x_o, y_o

def generate_unknown_dataset(unknown_dataset, img_size):
    x_o = unknown_dataset.get_fdata()

    x_o = np.array([norm(resize_img(x_o[:,:,i], img_size)) for i in range(unknown_dataset.shape[2])])

    x_o = np.repeat(x_o[..., np.newaxis], 3, -1)
    
    return x_o



