#!/usr/bin/env python
import math
import numpy as np
from PIL import Image
import tifffile

# *************************************************************
# *                    From Photography Notebook              *
# *************************************************************

# ======================= white_balance =======================
# Input:
#   I: an RGB image -- a numpy array of shape (height, width, 3)
#   black_level: an RGB offset to be subtracted from all the pixels
#   gray: the RGB color of a gray object (includes the black level)
# Output:
#   The corrected image: black level subtracted, then color channels scale to make gray come out gray
def white_balance(I, black_level, gray):
    # A3TODO: Complete this function
    I = I - black_level
    I = I / gray
    np.clip(I,0,255,out = I)
    I = I.astype(np.float32)
    return I # Replace this with your implementation


# ======================= color_transform =======================
# Input:
#   I: an RGB image -- a numpy array of shape (height, width, 3)
#   M: a 3x3 matrix, to be multiplied with each RGB triple in I
# Output:
#   The image with each RGB triple multiplied by M
def color_transform(I, M):
    # A3TODO: Complete this function
    I = np.array([[M @ I[i,j,:] for j in range(I.shape[1])] for i in range(I.shape[0])])
    return I # Replace this with your implementation


# *************************************************************
# *                    From Distortion Notebook               *
# *************************************************************

# ======================= shift_image_to_left =======================
# Input:
#   img: 2D numpy array of a grayscale image
#   k: The number of units/pixels to be shifted to the left (you can assume k < width of image)
# Output:
#   A 2D array of img shifted to the left by k pixels
#  For points that fall out of range on the right side, repeat the rightmost pixel.
def shift_image_to_left(img, k):
    new_img = np.zeros(img.shape, np.uint8)
    # A3TODO: Complete this function
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i,j,:] = img[i,(j+k) % img.shape[1],:]

    return new_img


# ======================= rotate_image =======================
# Input:
#   img: 2D numpy array of a grayscale image
#   k: The angle (in degrees) to be rotated counter-clockwise around the image center
#   interp_mode: 0 for nearest neighbor, 1 for bilinear
# Output:
#   A 2D array of img rotated around the original image's center by k degrees
def rotate_image(img, k, interp_mode=0):
    new_img = np.zeros(img.shape, np.uint8)
    # A3TODO: Complete this function
    center = np.array([img.shape[0]/2,img.shape[1]/2])
    angle = -math.pi * k/180;
    M = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)],
    ])

    if interp_mode == 0:
        # nearest neighbor
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                position = np.array([i,j])
                vector = position - center
                newvector = M @ vector
                newPos = center + newvector
                ix = newPos[0].astype(np.int)
                iy = newPos[1].astype(np.int)

                if (ix < img.shape[0] and iy < img.shape[1] and ix>= 0 and iy >= 0):
                    new_img[i,j,:] = img[ix,iy,:]

    else:
        # bilinear
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):

                position = np.array([i,j])
                vector = position - center
                newvector = M @ vector
                newPos = center + newvector
                
                if newPos[0]>=0 and newPos[0]<img.shape[0]-1:
                    x0 = math.trunc(newPos[0])
                    x1 = x0+1
                    dx = newPos[0]-x0
                elif newPos[0]<0:
                    x0 = 0
                    x1 = 0
                    dx = 0
                elif newPos[0]>img.shape[0]-1:
                    x0 = img.shape[0]-1
                    x1 = img.shape[0]-1
                    dx = 1

                if newPos[1]>=0 and newPos[1]<img.shape[1]-1:
                    y0 = math.trunc(newPos[1])
                    y1 = y0+1
                    dy = newPos[1]-y0
                elif newPos[1]<0:
                    y0 = 0
                    y1 = 0
                    dy = 0
                elif newPos[1]>img.shape[1]-1:
                    y0 = img.shape[1]-1
                    y1 = img.shape[1]-1
                    dy = 1

                RGBx0 = img[x0,y0,:]
                RGBx1 = img[x1,y0,:]
                RGBy0 = img[x0,y1,:]
                RGBy1 = img[x1,y1,:]
                xRGB = (dx * RGBx1) + (1-dx) * RGBx0
                yRGB = (dx * RGBy1) + (1-dx) * RGBy0
                finalRGB =  (dy * yRGB) + (1-dy) * xRGB
                new_img[i,j,:] = finalRGB

    return new_img


# ======================= undistort_image =======================
# Input:
#   img: A distorted image, with coordinates in the distorted space
#   k1, k2: distortion model coefficients (see explanation above)
#   M: affine transformation from pixel coordinates to distortion-model coordinates
#   interp_mode: 0 for nearest neighbor, 1 for bilinear
# Output:
#   An undistorted image, with pixels in the image coordinates
# Write down the formula for calculating the distortion model first (see exercise above)
# Put black in for points that fall out of bounds
def undistort_image(img, k1, k2, M, interp_mode=0):
    Mi = np.linalg.inv(M)
    output = np.zeros_like(img)
    # A3TODO: Complete this function
    h, w = img.shape[:2]

    if interp_mode == 0:
        # nearest neighbor
        for i in range(h):
            for j in range(w):
                position = np.array([j,i,1])
                x = Mi @ position
                vector = np.array([x[0],x[1]])
                r = np.linalg.norm(vector)
                sr = 1 + (k1 * r**2) + (k2 * r**4)
                scale = np.array([
                    [sr, 0,0],
                    [0, sr,0],
                    [0, 0,1]
                    ])
                gx = scale @ x
                homo = gx[2]
                gx /= homo
                GPos = M @ gx
                ix = round(GPos[1]).astype(np.int)
                iy = round(GPos[0]).astype(np.int)
                if (ix < h and iy < w and ix>=0 and iy >= 0):
                    output[i,j,:] = img[ix,iy,:]

    else:
        # bilinear
        for i in range(h):
            for j in range(w):
                position = np.array([j,i,1])
                x = Mi @ position
                vector = np.array([x[0],x[1]])
                r = np.linalg.norm(vector)
                sr = 1 + (k1 * r**2) + (k2 * r**4)
                scale = np.array([
                    [sr, 0,0],
                    [0, sr,0],
                    [0, 0,1]
                    ])
                gx = scale @ x
                homo = gx[2]
                gx /= homo
                GPos = M @ gx

                newPos = GPos
                y0 = math.trunc(newPos[0])
                y1 = y0+1
                x0 = math.trunc(newPos[1])
                x1 = x0+1
                dy = newPos[0]-y0
                dx = newPos[1]-x0

                if (x1 < img.shape[0] and y1 < img.shape[1] and x0>=0 and y0 >= 0):

                    RGBx0 = img[x0,y0,:]
                    RGBx1 = img[x1,y0,:]
                    RGBy0 = img[x0,y1,:]
                    RGBy1 = img[x1,y1,:]

                    xRGB = (dx * RGBx1) + (1-dx) * RGBx0
                    yRGB = (dx * RGBy1) + (1-dx) * RGBy0
                    finalRGB =  (dy * yRGB) + (1-dy) * xRGB
                    np.clip(finalRGB,0,255,out = finalRGB)
                    finalRGB = finalRGB.astype(np.uint8)
                    output[i,j,:] = finalRGB

    return output


# *************************************************************
# *                    From Convolution Notebook              *
# *************************************************************

# ======================= gen_gaussian_filter =======================
# Input:
#   dim: size of the filter in both x and y direction
#   sigma: standard deviation of the gaussian filter
# Output:
#   A 2-dimensional numpy array of size dim*dim
#   (Note that the array should be normalized)
# Hint: Use linspace or mgrid from numpy
def gen_gaussian_filter(dim, sigma):
    # A3 implement
    #pass # Replace this line with your implementation
    f = np.zeros([dim, dim])
    if (dim % 2 ==0):
        center = dim/2
    else:
        center = (dim-1)/2

    for i in range(dim):
        for j in range(dim):
            const = 2 * sigma**2
            x = i - center
            y = j - center
            f[i,j] = (1/const*np.pi)*math.exp( -(x**2+y**2)/const )
    sum = np.sum(f)
    f = f / sum
    return f

# ======================= convolve =======================
# Input:
#   I: A 2D numpy array containing pixels of an image
#   f: A squared/non-squared filter of odd/even-numbered dimensions
# Output:
#   A 2D numpy array resulting from applying the convolution filter f to I
#   All the entries of the array should be of type uint8, and restricted to [0,255]
#   You may use clip and astype in numpy to enforce this
# Note: When convolving, do not operate on the entries outside of the image bound,
#           i.e. clamp the ranges to the width and height of the image
#       Tie-breaking: If f has an even number of dimensions in some direction (assume the dimension is 2r),
#           sweep through [i-r+1, i+r] (i.e. length of left half = length of right half - 1)
#           With odd # of dimensions (2r+1), you would sweep through [i-r, i+r].
def convolve(I, f):
    # A3TODO: Complete this function
    output = np.zeros_like(I)
    kernel_h = f.shape[0]
    kernel_w = f.shape[1]
    h = math.trunc(kernel_h / 2)
    w = math.trunc(kernel_w / 2)
    image_pad = np.pad(I, pad_width=(
    (kernel_h // 2, kernel_h // 2),(kernel_w // 2,
    kernel_w // 2),(0,0)), mode='constant',
    constant_values=0).astype(np.float32)
    print(image_pad.shape)
    for i in range(h,image_pad.shape[0]-h):
        for j in range(w,image_pad.shape[0]-h):
            s = np.zeros(im.shape[2])
            for ii in range(-(kernel_h-h-1),h):
                for jj in range(-(kernel_w-w-1),w):
                    s += f[ii+h,jj+w]*image_pad[i-ii,j-jj,:]
            np.clip(s,0,255,out = s)
            s = s.astype(np.uint8)
            output[i-h,j-w]=s

    return output


# ======================= convolve_sep =======================
# Input:
#   I: A 2D numpy array containing pixels of an image
#   f: A squared/non-squared filter of odd/even-numbered dimensions
# Output:
#   A 2D numpy array resulting from applying the convolution filter f to I
#   All the entries of the array should be of type uint8, and restricted to [0,255]
#   You may use clip and astype in numpy to enforce this
# Note: When convolving, do not operate on the entries outside of the image bound,
#           i.e. clamp the ranges to the width and height of the image in the for loop
#       Tie-breaking: If f has an even number of dimensions in some direction (assume the dimension is 2r),
#           sweep through [i-r+1, i+r] (i.e. length of left half = length of right half - 1)
#           With odd # of dimensions (2r+1), you would sweep through [i-r, i+r].
#       You will convolve with respect to the direction corresponding to I.shape[0] first, then I.shape[1]
def convolve_sep(I, f):
    output = np.zeros_like(I)

    # A3TODO: Complete this function
    kernal_h = f.shape[0]
    kernal_w = f.shape[1]
    h = math.trunc(kernal_h / 2)
    w = math.trunc(kernal_w / 2)
    fh = f[h,:]
    fw = f[:,w]
    fh = fh / np.sum(fh)
    fw = fw / np.sum(fw)
    image_pad = np.pad(I, pad_width=(
    (kernal_h // 2, kernal_h // 2),(kernal_w // 2,
    kernal_w // 2),(0,0)), mode='constant',
    constant_values=0).astype(np.float32)
    temp = np.zeros_like(image_pad)

    for i in range(h,image_pad.shape[0]-h):
        for j in range(w,image_pad.shape[1]-w):
            s = np.zeros(im.shape[2])
            for ii in range(-(kernal_h-h-1),h):
                s+= fh[ii+h]*image_pad[i-ii,j,:]
            temp[i,j]=s
    for i in range(h,image_pad.shape[0]-h):
        for j in range(w,image_pad.shape[1]-w):
            s = np.zeros(im.shape[2])
            for jj in range(-(kernal_w-w-1),w):
                s += fw[jj+w]*temp[i,j-jj,:]

            np.clip(s,0,255,out = s)
            s = s.astype(np.uint8)
            output[i-h,j-w]=s
    return output


# ======================= unsharp_mask =======================
# This function essentially subtracts a (scaled) blurred version of an image from (scaled version of) itself
# Input:
#   I: A 2D numpy array containing pixels of an image
#   sigma: Gassian std.dev. for blurring
#   w: Sharpening weight
# Output:
#   A sharpened version of I
def unsharp_mask(I, sigma, w):
    output = np.zeros_like(I)
    # A3TODO: Complete this function
    f = gen_gaussian_filter(7, sigma)
    blurI = convolve_sep(I, f)
    o = (1+w)*I - w * blurI
    np.clip(o,0,255,out = o)
    o = o.astype(np.uint8)
    output = o
    return output
