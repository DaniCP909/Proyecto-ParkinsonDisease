import numpy as np
import math
import cv2

def fit_into_normalized_canvas(img, max_h, max_w):
    final_w = int(round_up_to(max_w, 50) * .5)
    final_h = int(round_up_to(max_h, 16) * .5)
    canvas = np.zeros((final_h, final_w),dtype=np.float64)
    img_h, img_w = img.shape

    h_ratio = final_h / img_h
    w_ratio = final_w / img_w

    if h_ratio > w_ratio:
        resized_img = cv2.resize(img, dsize=None, fx=w_ratio, fy=w_ratio, interpolation=cv2.INTER_LINEAR)
    else:
        resized_img = cv2.resize(img, dsize=None, fx=h_ratio, fy=h_ratio, interpolation=cv2.INTER_LINEAR)

    resized_h, resized_w = resized_img.shape

    start_y = (final_h - resized_h) // 2


    canvas[start_y: start_y + resized_h, :resized_w] = resized_img

    return canvas

def round_up_to(x, base):
    return base * math.ceil(x / base)