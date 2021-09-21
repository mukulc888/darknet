import pandas as pd
import numpy as np
from darknet import *
import cv2
import glob

def convert4cropping(image, bbox):
    x, y, w, h = bbox

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)
    return bbox_cropping


darknet_height = 416
darknet_width = 416

path = r"/home/anikets2002/darknet/data/obj/"
list_images = glob.glob( path + '*.jpg')
list_txtfiles = glob.glob(path + '*.txt')
print(list_images[:10], " *********************", list_txtfiles[:10])

for file in list_images:
    df = pd.read_csv('{}.txt'.format(file[:-4]), delimiter=' ', header=None)
    df = df.iloc[:, :5]
    img = cv2.imread(file)

    for inx, row in df.iterrows():
        row = row.to_numpy()
        bbox = row[1:]
        point =  convert4cropping(img , bbox)
        cords.append(point)
        # print(row)
        # rows.append(row)

    print(img.shape)

    for i, cdt in enumerate(cords):
        left, top, right, bottom = cdt
        cone_img = img[top:bottom, left:right]
        lowerY = np.array([20, 80, 80])
        upperY = np.array([30, 255, 255])
        lowerB = np.array([100, 150, 0])
        upperB = np.array([140, 255, 255])
        img_hsv = cv2.cvtColor(cone_img, cv2.COLOR_BGR2HSV)
        img_cvt_Y = cv2.inRange(img_hsv, lowerY, upperY)
        img_cvt_B = cv2.inRange(img_hsv, lowerB, upperB)

        wh_px_Y = np.count_nonzero(img_cvt_Y == 255)
        wh_px_B = np.count_nonzero(img_cvt_B == 255)
        flag_Y = wh_px_Y / (img_cvt_Y.size)
        flag_B = wh_px_B / (img_cvt_B.size)

        if flag_Y >= 0.075:
            cv2.rectangle(img, (left, top), (right, bottom), (30, 255, 255), 2)
            df.loc[i, 0] = 1
        elif flag_B >= 0.075:
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
            df.loc[i, 0] = 0
        else:
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 0), 2)
    cv2.imshow('image', img)

    df.to_csv(r'train\test1.txt', header=None, index=None, sep=' ', mode='a')

    cv2.waitKey(0)

    print(df)
