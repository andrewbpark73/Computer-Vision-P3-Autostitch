import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    
    
    # Compute the four corners of the image
    height, width = img.shape[:2]
    corners = np.array([[0, 0, 1], 
                        [width-1, 0, 1], 
                        [width-1, height-1, 1], 
                        [0, height-1, 1]])
    
    # Transform the corners to find the bounding box
    transformed_corners = np.dot(M, corners.T).T
    # Normalize the coordinates
    transformed_corners = transformed_corners / transformed_corners[:, 2][:, np.newaxis]
    
    # Calculate the bounding box
    minX = np.min(transformed_corners[:, 0])
    minY = np.min(transformed_corners[:, 1])
    maxX = np.max(transformed_corners[:, 0])
    maxY = np.max(transformed_corners[:, 1])
    
    
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    
    # Optimized version, take 3, issue with black lines on the border
    
    h, w, c = img.shape  # Height, width, and channel count of the image

    # Determine the bounding box for the transformed image
    minX, minY, maxX, maxY = imageBoundingBox(img, M)

    # Generate grid of coordinates (x, y) for the accumulator image
    x, y = np.meshgrid(np.arange(minX, maxX), np.arange(minY, maxY))
    homogenous_coordinates = np.stack((x.ravel(), y.ravel(), np.ones_like(x).ravel()), axis=1).T

    # Apply inverse transformation to get source coordinates
    src_coordinates = np.dot(np.linalg.inv(M), homogenous_coordinates)
    src_coordinates /= src_coordinates[2, :]  # Normalize the coordinates

    # Reshape back to image dimensions
    map_x, map_y = src_coordinates[:2].reshape(2, maxY - minY, maxX - minX)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # Use remap for efficient inverse warping with linear interpolation
    warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Identify non-black pixels in the warped image
    non_black_mask = np.any(warped_img > 0, axis=2)

    # Create a blend mask for feathering effect, avoiding black pixels
    blend_mask = np.zeros((maxY - minY, maxX - minX), dtype=np.float32)
    distance_from_left_edge = np.arange(maxX - minX)
    distance_from_right_edge = np.arange(maxX - minX)[::-1]
    blend_mask = np.minimum(distance_from_left_edge, distance_from_right_edge) / blendWidth
    blend_mask = np.minimum(blend_mask, 1)  # Cap the values at 1 for distances greater than blendWidth
    blend_mask = np.tile(blend_mask, (maxY - minY, 1))

    # Apply the non-black mask to the blend mask
    blend_mask *= non_black_mask

    # Expand dimensions to match the image channels for multiplication
    blend_mask_expanded = np.expand_dims(blend_mask, axis=2)
    weights = np.ones_like(warped_img, dtype=np.float32) * blend_mask_expanded

    # Blend warped image into the accumulator using feathering, ignoring black pixels
    acc[minY:maxY, minX:maxX, :3] *= 1 - weights
    acc[minY:maxY, minX:maxX, :3] += warped_img * weights
    
    # Update the accumulator's weights channel with the maximum weight encountered
    acc[minY:maxY, minX:maxX, 3] = np.maximum(acc[minY:maxY, minX:maxX, 3], blend_mask_expanded[:,:,0])
    
    
    ''' Normal Version

    h, w, c = img.shape  # Height, width, and channel count of the image

    # Determine the bounding box for the transformed image
    minX, minY, maxX, maxY = imageBoundingBox(img, M)

    # Iterate through each pixel within the bounding box
    for i in range(minX, maxX):
        for j in range(minY, maxY):
            # Transform pixel location from accumulator space back to image space
            pixel_in_acc_space = np.array([i, j, 1.0])
            pixel_in_img_space = np.dot(np.linalg.inv(M), pixel_in_acc_space)
            x, y = pixel_in_img_space[0] / pixel_in_img_space[2], pixel_in_img_space[1] / pixel_in_img_space[2]

            # Ensure the transformed coordinates are within the image bounds
            if not (0 <= x < w and 0 <= y < h):
                continue
            
            # Skip processing for fully black pixels
            if np.all(img[int(y), int(x), :3] == 0):
                continue

            # Calculate the blending weight
            weight = 1.0
            # Adjust weight for horizontal blending near edges
            if minX + blendWidth > x > minX:
                weight *= (x - minX) / blendWidth
            elif maxX - blendWidth < x < maxX:
                weight *= (maxX - x) / blendWidth
            
            # Update the accumulator's weight sum (alpha channel)
            acc[j, i, 3] += weight

            # Update the accumulated image with the weighted pixel color
            for k in range(3):  # Loop over color channels
                acc[j, i, k] += img[int(y), int(x), k] * weight'''
    
    ''' Optimized version, take 2
    height, width, channel = img.shape

    #Normalization to remove brightness.
    for chan in range(3):
        c_max = np.max(img[:, :, chan])

        c_min = np.min(img[:, :, chan])
        ranging = c_max - c_min
        img[:, :, chan] = int((float(img[:, :, chan] - c_min) / float(ranging)) * 255)

    addon = np.zeros(acc.shape)

    alpha = np.ones((height ,width))
    for c in range(blendWidth):
        alpha[:, c] = [float(c) / blendWidth] * height
        alpha[:, width - c - 1] = [float(c) / blendWidth] * height


    feathering = np.zeros((height, width, channel + 1))
    for chan in range(3):
        feathering[:, :, chan] = img[:, :, chan] * alpha
    feathering[:, :, -1] = alpha

    for chan in range(4):
        addon[:, :, chan] = cv2.warpPerspective(feathering[:, :, chan], M, dsize=(acc.shape[1], acc.shape[0]), flags= cv2.INTER_LINEAR)

    acc += addon
    '''
        
    ''' Original version - unoptimized
    # Get the dimensions of the images
    height, width = img.shape[:2]
    acc_height, acc_width = acc.shape[:2]
    
    # compute the inverse transform
    M_inv = np.linalg.inv(M)
    
    for y in range(acc_height):
        for x in range(acc_width):
            # Compute the position of the pixel in the original image
            pos = np.dot(M_inv, np.array([x, y, 1]))
            pos = pos / pos[2]
            src_x, src_y = pos[:2]
            
            # Check if the pixel is within the bounds of the original image
            if 0 <= src_x < width-1 and 0 <= src_y < height-1:
                # Linear interpolation
                x0, y0 = int(src_x), int(src_y)
                x1, y1 = x0 + 1, y0 + 1
                
                # Compute the weights
                a = src_x - x0
                b = src_y - y0
                
                # Pixel value interpolation
                f00 = img[y0, x0]
                f01 = img[y0, x1]
                f10 = img[y1, x0]
                f11 = img[y1, x1]
                
                # Interpolate the pixel value
                pixel_val = (1 - a) * (1 - b) * f00 + a * (1 - b) * f01 + (1 - a) * b * f10 + a * b * f11
                
                # Skip black pixels
                if np.all(pixel_val == 0):
                    continue
                
                # Compute feathering weights
                distance_to_center = min(abs(x - acc_width / 2), blendWidth) / blendWidth
                weight = 1 - distance_to_center
                
                # Update the accumulator
                acc[y, x, :3] *= (1 - weight) # Weighted sum of pixel colors
                acc[y, x, :3] += weight * pixel_val # Weighted sum of pixel colors
                acc[y, x, 3] += weight # Sum of the weights
    '''
    
    ''' Optimized version, take 1

    def accumulateBlend(img, acc, M, blendWidth):
        acc_height, acc_width = acc.shape[:2]
        M_inv = np.linalg.inv(M)
        
        # Generate grid of coordinates (x, y) for the accumulator image
        x, y = np.meshgrid(np.arange(acc_width), np.arange(acc_height))
        homogenous_coordinates = np.stack((x.ravel(), y.ravel(), np.ones_like(x).ravel()), axis=1).T
        
        # Apply inverse transformation to get source coordinates
        src_coordinates = M_inv.dot(homogenous_coordinates)
        src_coordinates /= src_coordinates[2, :]
        
        # Reshape back to image dimensions
        map_x, map_y = src_coordinates[:2].reshape(2, acc_height, acc_width)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # Use remap for efficient inverse warping with linear interpolation
        warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Create a blend mask based on distance from the center for feathering effect
        blend_mask = np.minimum(np.abs(x - acc_width / 2) / blendWidth, 1)
        blend_mask = np.dstack([blend_mask] * 3)  # Assuming img is 3-channel; adjust if not
        
        # Compute weights for blending (take into account only non-black pixels from the warped image)
        weights = np.logical_or.reduce(warped_img != 0, axis=2).astype(np.float32)
        weights = np.expand_dims(weights, axis=2) * (1 - blend_mask)
        
        # Blend warped image into the accumulator
        acc[:, :, :3] *= (1 - weights)
        acc[:, :, :3] += weights * warped_img
        acc[:, :, 3] += weights[:,:,0]  # Assuming the accumulator's 4th channel is for weights
        
    '''
       
                
    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    
    
    # Copy the accumulator to avoid modifying the original image
    img = acc.copy()
    
    weights = img[:, :, 3] # Alpha channel contains the weights
    weights[weights == 0] = 1 # Avoid division by zero
    weights = weights[:, :, np.newaxis] # Add an axis to weights to make it (height x width x 1)
    
    # Normalize the RGB channels
    img[:, :, :3] = img[:, :, :3] / weights
    
    # Set the alpha channel to opaque
    img[:, :, 3] = 255
    
    # Clip the pixel values to the range [0, 255] and convert to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)

    
    #TODO-BLOCK-END
    # END TODO
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = np.Inf
    minY = np.Inf
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        
        
        # Get the bounding box of the current image
        minX_img, minY_img, maxX_img, maxY_img = imageBoundingBox(img, M)
        
        # Update the mosaic bounding box
        minX = min(minX, minX_img)
        minY = min(minY, minY_img)
        maxX = max(maxX, maxX_img)
        maxY = max(maxY, maxY_img)
        
        
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    
    
    if is360:
        # Compute the drift correction matrix
        A = computeDrift(x_init, y_init, x_final, y_final, width)
    
    
    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

