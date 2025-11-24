import numpy as np
import math
import cv2

def fit_into_normalized_canvas(img, max_h, max_w):
    final_w = int(round_up_to(max_w, 50) * .5)
    final_h = int(round_up_to(max_h, 16) * .5)
    canvas = np.zeros((final_h, final_w),dtype=np.float32)
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

def simple_bresenham_line(x1, y1, x2, y2, thickness=1):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1

    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1

    pixels = []
    half = thickness // 2

    if dx > dy:   
        err = dx // 2
        while x != x2:
            for t in range(-half, half + 1):
                pixels.append((y + t, x))   # (fila, columna)
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        while y != y2:
            for t in range(-half, half + 1):
                pixels.append((y, x + t))   # (fila, columna)
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    # último punto
    for t in range(-half, half + 1):
        if dx > dy:
            pixels.append((y + t, x2))
        else:
            pixels.append((y2, x + t))

    return pixels



def bresenham_line(x1, y1, x2, y2, height, width, thickness=1):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    half = max(1.0, thickness / 2.0)

    pixel_coords = []

    def collect_pixel_block(cx, cy):
        y_start = max(0, math.floor(cy - half))
        y_end   = min(height, math.ceil(cy + half + 1))
        x_start = max(0, math.floor(cx - half))
        x_end   = min(width, math.ceil(cx + half + 1))

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

def crop_black_columns(img):
    """
    Remove columns tah are completely black (all zeros)
    """
    mask = ~(img == 0).all(axis=0)
    no_black_columns = img[:, mask]
    return no_black_columns

def restore_full_width(img, cropped_img):
    """
    Restore the width of the original image by filling the removed columns.
    If fill_value is None, the cropped image will be repeated (tiled) horizontally.
    Otherwise, the remaining columns are filled with the specified value (e.g., 255 for white).
    """
    full_h, full_w = img.shape
    reduced_h, reduced_w = cropped_img.shape
    n_copies = int(np.ceil(full_w / reduced_w))
    tiled = np.tile(cropped_img, (1, n_copies))[:, :full_w]
    return tiled

def clean_and_refill(img):
    """
    Complete pipeline: remove black columns and restore width.
    If fill_value is None, repeats the cropped image to fill the width.
    If fill_value is specified (e.g., 255), fills the removed area with that value.
    """
    cropped_img = crop_black_columns(img)
    final_result = restore_full_width(img, cropped_img)
    return final_result