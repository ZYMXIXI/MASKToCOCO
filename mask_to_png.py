import cv2
import numpy as np
import os
import sys



def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return ''  # no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return ''  # ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


if __name__ == "__main__":
    path = "1.png"
    image = cv2.imread("1.png", cv2.IMREAD_GRAYSCALE)
    res = rle_encode(image)
    with open("rle.csv", "w+") as f:
        input_line = path + ","+ str(res)
        f.writelines(input_line)
    # print(res)