# coding: utf-8

__author__ = 'cleardusk'

import numpy as np
import cv2
from math import sqrt
import matplotlib.pyplot as plt
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
color_val = (0,0,255)

def filter(img,points,scale=5,masked=False,cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask,np.int32([points]),(255,255,255))
        img = cv2.bitwise_and(img,mask)
    if cropped:
        bounding_box = cv2.boundingRect(points)
        x,y,w,h = bounding_box
        cropped_part = img[y:y+h,x:x+w]
        cropped_part = cv2.resize(cropped_part,(0,0),None,scale,scale)
        return cropped_part
    else:
        return mask

def get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos:]


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def calc_hypotenuse(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)

    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box


def plot_image(img):
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    plt.imshow(img[..., ::-1])
    plt.show()



def draw_landmarks(img, pts, style='fancy', wfp=None, show_flag=False, **kwargs):
    """Draw landmarks using matplotlib"""
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))
    plt.imshow(img[..., ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    dense_flag = kwargs.get('dense_flag')

    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        if dense_flag:
            plt.plot(pts[i][0, ::6], pts[i][1, ::6], 'o', markersize=0.4, color='c', alpha=0.7)
        else:
            alpha = 0.8
            markersize = 4
            lw = 1.5
            color = kwargs.get('color', 'w')
            markeredgecolor = kwargs.get('markeredgecolor', 'black')

            #nums = [48, 60, 68] # lips
            nums = [48, 68]

            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                #plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)
                plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize, color='red',markeredgecolor=markeredgecolor, alpha=alpha)

    if wfp is not None:
        plt.savefig(wfp, dpi=150)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plt.show()


def cv_draw_landmark(img_ori, pts, box=None, color=RED, size=3):
    img = img_ori.copy()
    n = pts.shape[1]
    # mouth height
    distance = sqrt( (pts[0, 63] - pts[0, 65])**2 +
                      (pts[1, 63] - pts[1, 65])**2)
    print ("\n Mouth_distance: ", distance)

    if (distance > 30):
        xcoor, ycoor = [], []
        for r in range (len(pts[0])):
             xcoor.append(pts[0][r])
             ycoor.append(pts[1][r])
        #print (face_landmarks[49:61])
        xcoor = np.array(xcoor)
        ycoor = np.array(ycoor)
        xcoor = xcoor.reshape(-1,1)
        ycoor = ycoor.reshape(-1,1)
        face_landmarks = np.concatenate((xcoor,ycoor),axis = 1)
        img_lips = filter(img,face_landmarks[49:61],3,masked=True,cropped=False)


        img_color_lips = np.zeros_like(img_lips)
        img_color_lips[:] = color_val  # Creating a fully colored image of the color selected .
        img_color_lips = cv2.bitwise_and(img_lips,img_color_lips)  # Getting colored lips.
        img_color_lips = cv2.GaussianBlur(img_color_lips,(7,7),10) # Blurring to get better effect on merging.

        final_image = cv2.addWeighted(img,1,img_color_lips,0.4,0)  # Merging with original image.
        # Can work around with the weight of img_color_lips to get the best desired effect.
        img = final_image
        #img = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

    else:
        # mouth edges
        cv2.circle(img, (int(round(pts[0, 48])), int(round(pts[1, 48]))), size, color, -1)
        cv2.circle(img, (int(round(pts[0, 54])), int(round(pts[1, 54]))), size, color, -1)

    return img
