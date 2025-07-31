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


def bresenham_line(x1, y1, x2, y2, height, width, thickness=1):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    half = thickness // 2

    pixel_coords = []

    def collect_pixel_block(cx, cy):
        y_start = max(0, cy - half)
        y_end = min(height, cy + half + 1)
        x_start = max(0, cx - half)
        x_end = min(width, cx + half + 1)

        for yy in range(y_start, y_end):
            for xx in range(x_start, x_end):
                if (xx - cx)**2 + (yy - cy)**2 <= half**2:
                    pixel_coords.append((yy, xx))

    if dx > dy:
        err = dx / 2.0
        while x != x2:
            collect_pixel_block(x, y)
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            collect_pixel_block(x, y)
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    collect_pixel_block(x2, y2)
    return pixel_coords

    
def normalize(values: list[int], fallback: float = 0.5) -> list[int]:
        min_v, max_v = min(values), max(values)
        if max_v - min_v == 0:
            return [fallback] * len(values)
        return [(v - min_v) / (max_v - min_v) for v in values]