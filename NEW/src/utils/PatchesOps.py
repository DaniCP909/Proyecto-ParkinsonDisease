
def patches_per_image(image_width, patch_width=10, stepsize=2):
    return int((image_width - patch_width)/stepsize + 1)

def patch_generator(image, n_patches=1, patch_height=48, patch_width=10, stepsize=2):
    
    H, W = image.shape
    patches = []
    for p in range(n_patches):
        start_x = p * stepsize
        patch = image[:, start_x: start_x + patch_width]
        patches.append(patch)
    return patches