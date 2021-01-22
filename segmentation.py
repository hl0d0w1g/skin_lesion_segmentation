#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv
import numpy as np
import skimage
from skimage import io, filters, color, morphology, segmentation, measure, feature
from sklearn.metrics import jaccard_score
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from scipy import ndimage as ndi


# -------------------------- Preprocessing functions --------------------------
@adapt_rgb(each_channel)
def median_filter_rgb(image, selem=morphology.disk(30)):
    '''
    Median filter for RGB images.
    
    Args: 
    - image (numpy ndarray): Image to apply the median filter.
    - selem (numpy ndarray): Kernel of the filter.
    
    Return:
    - The filtered RGB image.
    '''
    return filters.median(image, selem=selem)

def crop_img(img, threshold=100):
    '''
    Crop the image to get the region of interest. Remove the vignette frame.
    Analyze the value of the pixels in the diagonal of the image, from 0,0 to h,w and
    take the points where this value crosses the threshold by the first time and for last.

    Args:
    - img (numpy ndarray): Image to crop.
    - threshold (int): Value to split the diagonal into image and frame.

    Return:
    - The coordinates of the rectangle and the cropped image.
    '''
    # Get the image dimensions
    h, w = img.shape[:2]

    # Get the coordinates of the pixels in the diagonal
    y_coords = ([i for i in range(0, h, 3)], [i for i in range(h - 3, -1, -3)])
    x_coords = ([i for i in range(0, w, 4)], [i for i in range(0, w, 4)])

    # Get the mean value of the pixels in the diagonal, form 0,0 to h,w 
    # and from h,0 to 0,w
    coordinates = {'y1_1': 0, 'x1_1': 0, 'y2_1': h, 'x2_1': w, 'y1_2': h, 'x1_2': 0, 'y2_2': 0, 'x2_2': w}
    for i in range(2):
        d = []
        y1_aux, x1_aux = 0, 0
        y2_aux, x2_aux = h, w 
        for y, x in zip(y_coords[i], x_coords[i]):
            d.append(np.mean(img[y, x, :]))

        # Get the location of the first point where the threshold is crossed
        for idx, value in enumerate(d):
            if value >= threshold:
                coordinates['y1_' + str(i + 1)] = y_coords[i][idx]
                coordinates['x1_' + str(i + 1)] = x_coords[i][idx]
                break

        # Get the location of the last point where the threshold is crossed
        for idx, value in enumerate(reversed(d)):
            if value >= threshold:
                coordinates['y2_' + str(i + 1)] = y_coords[i][-idx if idx != 0 else -1]
                coordinates['x2_' + str(i + 1)] = x_coords[i][-idx if idx != 0 else -1]
                break

    # Set the coordinates to crop the image
    y1 = max(coordinates['y1_1'], coordinates['y2_2'])
    y2 = min(coordinates['y2_1'], coordinates['y1_2'])
    x1 = max(coordinates['x1_1'], coordinates['x1_2'])
    x2 = min(coordinates['x2_1'], coordinates['x2_2'])
    
    return y1, y2, x1, x2, img[y1:y2, x1:x2, :]

def remove_hair(img, canny_sigma=0.1, dilation_selem=morphology.disk(10), min_hair_index=2.0):
    '''
    Removes the hair from a given image with a median filter applied only over the hair.
    
    Args:
    - img (numpy ndarray): The image to filter.
    - canny_sigma (float): Sigma of the canny edge detector.
    - dilation_selem (numpy ndarray): Kernel of the dilation of the detected edges.
    - min_hair_index (float): Minimun hair presence to apply the filter.
    
    Return:
    - The filtered image without (or with less) hair.
    '''
    img_gray = color.rgb2gray(img)

    # Compute the edges of the image with canny and dilation them
    edges = feature.canny(img_gray, sigma=canny_sigma)
    edges = morphology.dilation(edges, selem=dilation_selem)

    # Get the 1s ratio of the edges mask
    hair_index = (np.sum(edges) / (edges.shape[0] * edges.shape[1])) * 100

    if hair_index >= min_hair_index:
        # Compute the median filter over the original filter
        img_median = median_filter_rgb(img)

        # Multiply the median filtered image by the edges mask, to get the value of the 
        # pixels only in the mask
        result = img_median * edges[:, :, np.newaxis].astype(int)

        # Compute the maximum to replace the pixels of the hair by the median value
        final_img = np.maximum(img, result)

        return final_img

    else:
        return img

def preprocessing_(img_data):
    # Crop the image to get the region of interest
    y1, y2, x1, x2, cropped_image = crop_img(img_data['original_image'])
    img_data['roi_coords'] = {'y1': y1, 'y2': y2, 'x1': x1, 'x2': x2}
    img_data['roi'] = cropped_image
    
    # Blur the image with a gaussian filter
    img_data['roi'] = filters.gaussian(img_data['roi'], sigma=3, multichannel=True)

    # Remove (or reduce) the hair
    img_data['roi'] = remove_hair(img_data['roi'])

    return img_data

# -------------------------- Segmentation functions --------------------------
def thresholding(img):
    '''
    Converts the image to HSV and compute an otsu thresholding over the saturation channel
    
    Args:
    - img (numpy ndarray): Image to apply the threshold.
    
    Return:
    - The mask generated by the threshold.
    '''
    # Convert the image to HSV and get only the saturation channel
    image_hsv = color.rgb2hsv(img)[:, :, 1]

    # Compute an otsu thresholding over the saturation channel
    otsu_th = filters.threshold_otsu(image_hsv)
    predicted_mask = (image_hsv > otsu_th).astype(int)

    return predicted_mask, otsu_th

def segmentation_(img_data):
    # Compute the thresholding
    mask_thresholding, otsu_th = thresholding(img_data['roi'])
    
    img_data['otsu_th'] = otsu_th
    img_data['mask_thresholding'] = mask_thresholding

    return img_data

# -------------------------- Postprocessing functions --------------------------
def calc_dist(p1, p2):
    '''
    Calculates the euclidean distance between two points.
    
    Args:
    - p1 (tuple): Point 1 (y1, x1).
    - p2 (tuple): Point 2 (y2, x2).
    
    Return:
    - The euclidean distance.
    '''
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

def recover_original_mask_shape(original_shape, mask, roi_coords):
    '''
    Takes the mask of the region of interest and enlarge it to the original image shape.
    
    Args:
    - original_shape (tuple): The original shape of the mask to achive.
    - mask (numpy ndarray): The mask to enlarge.
    - roi_cords (dict): Dictionnaire with the coordinates of the region of interest.
    
    Return:
    - The mask with the same shape of original_sahpe.
    '''
    original_mask = np.zeros(original_shape)

    for i, y in enumerate(range(roi_coords['y1'], roi_coords['y2'])):
        for j, x in enumerate(range(roi_coords['x1'], roi_coords['x2'])):
            original_mask[y, x] = mask[i, j]

    return original_mask

def postprocessing_(img_data):
    # Fill the holes of the mask regions
    img_data['mask_thresholding'] = ndi.binary_fill_holes(img_data['mask_thresholding'])

    # Remove the small artifacts of the mask
    selem = morphology.disk(12)
    img_data['mask_thresholding'] = morphology.erosion(img_data['mask_thresholding'], selem)
    img_data['mask_thresholding'] = morphology.dilation(img_data['mask_thresholding'], selem)

    # Label the mask to get the properties of the regions 
    label_img = measure.label(img_data['mask_thresholding'])
    regions = measure.regionprops(label_img)

    # Compute the distance between each region centroid and the center of the image and get the closest one
    dist = [(props.label, calc_dist(props.centroid, tuple(ti / 2 for ti in img_data['mask_thresholding'].shape[:2]))) for props in regions]
    dist = sorted(dist, key=lambda tup: tup[1])
    aux = dist[0][0]

    label_img[label_img != aux] = 0
    label_img[label_img == aux] = 1
    img_data['mask_thresholding'] = label_img 

    # Enlarges the mask and makes it more polygonal
    img_data['mask_thresholding'] = morphology.dilation(img_data['mask_thresholding'], selem=morphology.disk(20))
    img_data['mask_thresholding'] = morphology.convex_hull_image(img_data['mask_thresholding'], offset_coordinates=True)

    # Recover the original shape of the mask to match with image
    img_data['mask'] = recover_original_mask_shape(img_data['img_shape'], img_data['mask_thresholding'], img_data['roi_coords'])
    
    return img_data

# ----------------------------- Segmentation function -------------------------
def skin_lesion_segmentation(img_root):
    '''
    Segment the image by creating a mask

    Args:
    - img_root (str): file name of the image showing the skin lesion     
    
    Returns:
    - predicted_mask (numoy ndarray): predicted segmentation mask   
    
    '''
    image = io.imread(img_root)

    # Dictionary to store the image and all the relevant information
    img_data = {}
    img_data['original_image'] = image
    img_data['img_shape'] = image.shape[:2]

    # Process the image
    img_data = preprocessing_(img_data)
    img_data = segmentation_(img_data)
    img_data = postprocessing_(img_data)

    predicted_mask = img_data['mask']

    return predicted_mask

# ----------------------------- Evaluation function ---------------------------        
def evaluate_masks(img_roots, gt_masks_roots):
    ''' 
    It receives two lists, for each image on the list performs the segmentation
    and determines Jaccard Index. Finally, it computes an average Jaccard Index for all the images in 
    the list.

    Args:
    - img_roots (list(str)): A list of the file names of the images to be analysed.
    - gt_masks_roots (list(str)): A list of the file names of the corresponding Ground-Truth (GT). 
            
    Returns:
    - Mean scorefor the full set.
    '''
    score = []
    for i in np.arange(np.size(img_roots)):
        print('I%d' %i)
        predicted_mask = skin_lesion_segmentation(img_roots[i])
        gt_mask = io.imread(gt_masks_roots[i])/255     
        score.append(jaccard_score(np.ndarray.flatten(gt_mask),np.ndarray.flatten(predicted_mask)))
    mean_score = np.mean(score)
    print('Average Jaccard Index: '+str(mean_score))
    return mean_score

def rle_encode(img):
    '''
    This function encodes the images using a format suitable for Kaggle

    Args:
    - img (numpy ndarray): 1 - mask, 0 - background
    
    Returns:
    - Length as string formated.
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# --- This function generates a CSV file for performing a Kaggle submission ---
def test_prediction_csv(dir_images_name, csv_name):
    '''
    This function generates a CSV file for performing a Kaggle submission

    Args:
    - dir_images_name (str): Directory where the test images are stored.
    - csv_name (str): Name of the csv file.

    Returns:
    - None
    '''
    dir_images = np.sort(os.listdir(dir_images_name))

    with open(csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImageId", "EncodedPixels"])
        for i in np.arange(np.size(dir_images)):        
            # Segmentation
            predicted_mask = skin_lesion_segmentation(dir_images_name+'/'+dir_images[i]) 
            
            # RLE coding and generation of CSV file
            encoded_pixels = rle_encode(predicted_mask)
            writer.writerow([dir_images[i][:-4], encoded_pixels])
            print('Mask ' + str(i))

# Reading images
data_dir= os.curdir

train_imgs_files = [ os.path.join(data_dir,'train/images',f) for f in sorted(os.listdir(os.path.join(data_dir,'train/images'))) 
            if (os.path.isfile(os.path.join(data_dir,'train/images',f)) and f.endswith('.jpg')) ]

train_masks_files = [ os.path.join(data_dir,'train/masks',f) for f in sorted(os.listdir(os.path.join(data_dir,'train/masks'))) 
            if (os.path.isfile(os.path.join(data_dir,'train/masks',f)) and f.endswith('.png')) ]

# train_imgs_files.sort()
# train_masks_files.sort()
print("Number of train images", len(train_imgs_files))
print("Number of image masks", len(train_masks_files))

# Segmentation and evaluation
mean_score = evaluate_masks(train_imgs_files, train_masks_files)

# Generation of the CSV file for Kaggle submissions 
# (https://www.kaggle.com/c/bip-segmentation-project-2020/leaderboard)
test_images_dir=os.path.join(data_dir,'test/images')
test_prediction_csv(test_images_dir, 'test_prediction.csv')
