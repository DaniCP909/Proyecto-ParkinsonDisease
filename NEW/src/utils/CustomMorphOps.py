import numpy as np
import math
import cv2
import random

def fit_into_normalized_canvas(img, max_h, max_w):
    max_h = max_h
    max_w = max_w
    final_w = int(round_up_to(max_w, 50))
    final_h = int(round_up_to(max_h, 16))
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
    Remove columns that are completely black (all zeros).
    Keeps internal spacing structure intact.
    """
    mask = ~(img == 0).all(axis=0)
    no_black_columns = img[:, mask]
    return no_black_columns


def restore_full_width(cropped_img, original_w, fill_value=0):
    """
    Restore the width of the original image using padding instead of tiling.

    The original content is kept on the left side and the remaining
    area is filled with a constant value (default: black).
    """

    reduced_h, reduced_w = cropped_img.shape

    result = np.full(
        (reduced_h, original_w),
        fill_value,
        dtype=cropped_img.dtype
    )

    result[:, :reduced_w] = cropped_img

    return result


def clean_and_refill(img, original_w, remove_black_columns=False, fill_value=0):
    """
    Complete preprocessing pipeline.

    Options:
    - Keep natural spacing between letters/words.
    - Optionally remove fully black columns.
    - Restore original width using padding instead of repetition.
    """

    if remove_black_columns:
        processed_img = crop_black_columns(img)
    else:
        processed_img = img

    final_result = restore_full_width(
        processed_img,
        original_w,
        fill_value=fill_value
    )

    return final_result

def apply_saltpepper(raw_img):
    img = raw_img.copy()

    # Getting the dimensions of the image
    row , col = img.shape
    
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 900)
    for i in range(number_of_pixels):
      
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
        
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
        
        # Color that pixel to white
        img[y_coord][x_coord] = 255
        
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300 , 900)
    for i in range(number_of_pixels):
      
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
        
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
        
        # Color that pixel to black
        img[y_coord][x_coord] = 0
        
    return img

def shear(img, sh_factor):
    if sh_factor == 0.0:
        return img

    h, w = img.shape

    # Desplazamiento máximo
    dx = abs(sh_factor) * h
    new_width = int(w + dx)

    # Compensación si el shear es negativo
    tx = dx if sh_factor < 0 else 0

    matrix = np.float32([
        [1, sh_factor, tx],
        [0, 1, 0]
    ])

    return cv2.warpAffine(
        img,
        matrix,
        (new_width, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def crop_to_content(img, bg_color):
    """
    Recorta la imagen al bounding box del contenido
    """
    # Contenido = todo lo que no sea fondo
    mask = img != bg_color

    # Seguridad: por si no hay contenido
    if not np.any(mask):
        return img

    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    return img[y_min:y_max+1, x_min:x_max+1]

def rotate(img, angle):
    if angle == 0:
        return img
    
    padding = 200

    h, w = img.shape[:2]
    center = ((w) // 2, (h) // 2)

    canvas = np.zeros((h + padding, w + padding), dtype=img.dtype)

    hc, wc = canvas.shape[:2]

    canvas[(padding // 2):(padding//2)+h, (padding // 2):(padding//2)+w] = img


    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(canvas, matrix, (wc, hc))

    cropped = crop_to_content(rotated_img, bg_color=0)

    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)

    return resized