import numpy as np
from evaluator import evaluate_single_image
import cv2
from skimage import measure


def fillHole(im_in):
    im_floodfill = im_in.copy().astype(np.uint8)
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = (im_in | im_floodfill_inv)/255

    return im_out.astype(float)

def fill(img_bin):
    fill_bin = fillHole(np.array(img_bin,dtype='uint8'))
    return fill_bin


def bubbleSort(arr):
    n = len(arr)
    index = np.array(range(n))
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j],arr[j+1] = arr[j+1],arr[j]
                index[j],index[j + 1] = index[j+1],index[j]
    return index[-5:]

def search_largest_region(image):
    labeling = measure.label(image)
    regions_all = measure.regionprops(labeling)
    regions = []
    for i,region in enumerate(regions_all):
        if region.area >100:
            regions.append(regions_all[i])
    if len(regions) > 5:
        regions_list=[]
        for region in regions:
            regions_list.append(region.area)
        five_top_index = bubbleSort(regions_list)
    else:
        five_top_index = np.array(range(len(regions)))
    return regions,five_top_index

def five_top_region(fill_bin):
    regions,five_top_index = search_largest_region(fill_bin)
    bin_image = np.zeros_like(fill_bin)
    for i in range(len(five_top_index)):
        region = regions[five_top_index[i]]
        for coord in region.coords:
            bin_image[coord[0], coord[1]] = 1
    return bin_image

def post_treatment(images):
    images = np.transpose(np.squeeze(images),(2,0,1))
    post_images = np.zeros_like(images).astype('float32')
    for i in range(images.shape[0]):
        fill_bin = five_top_region(images[i])
        post_images[i]= fill_bin

    return np.expand_dims(post_images.transpose((1,2,0)),axis=3)
