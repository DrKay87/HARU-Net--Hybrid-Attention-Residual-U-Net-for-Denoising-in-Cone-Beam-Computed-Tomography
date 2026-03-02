# utils.py

import numpy as np
import random
import torch 
import torch.nn.functional as F


def psnr(img1, img2):
    
    assert img1.shape == img2.shape, "Images must have the same dimensions"
    # Compute the Mean Squared Error
    #mse = np.mean((img1 - img2) ** 2)
    mse = F.mse_loss(img1, img2, reduction='mean').item()
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming the images are normalized between 0 and 1
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def batch_psnr(imgs1, imgs2):
    psnr_values = [psnr(imgs1[i], imgs2[i]) for i in range(imgs1.size(0))]
    return np.mean(psnr_values)

def padcrop_resize(image, target_height=1376, target_width=2800):
  
    original_height, original_width = image.shape

    # Crop the image if it is larger than the target dimensions
    if original_height > target_height:
        start_y = (original_height - target_height) // 2
        image = image[start_y:start_y + target_height, :]
    if original_width > target_width:
        start_x = (original_width - target_width) // 2
        image = image[:, start_x:start_x + target_width]

    # Zero pad the image if it is smaller than the target dimensions
    if original_height < target_height or original_width < target_width:
        padded_image = np.zeros((target_height, target_width), dtype=image.dtype)
        start_y = (target_height - image.shape[0]) // 2
        start_x = (target_width - image.shape[1]) // 2
        padded_image[start_y:start_y + image.shape[0], start_x:start_x + image.shape[1]] = image
        image = np.array(padded_image)

    return np.array(image)

def Patch_Positions(image_size, patch_size=256, max_overlap=24):
    step_size = patch_size - max_overlap
    image_height, image_width = image_size
    # Calculating the number of patches ensuring they don't exceed the image size
    x_positions = list(range(0, image_width - max_overlap - 1, step_size))
    y_positions = list(range(0, image_height - max_overlap - 1, step_size))

    # Ensure the last patch doesn't exceed the image size by adding positions for any remaining area
    if x_positions[-1] + patch_size > image_width:
        x_positions[-1] = image_width - patch_size
    if y_positions[-1] + patch_size > image_height:
        y_positions[-1] = image_height - patch_size

    # Final patch positions
    patch_positions = [(x, y) for y in y_positions for x in x_positions]

    # Create and save patches (assuming the image is already loaded)
    # Replace "your_image.png" with the actual image path if needed
    #image = Image.new('RGB', (image_width, image_height))  # Dummy image for this example

    return  patch_positions

def rescale_pixel_range(image, min_val=0.1,max_val=0.9):
    x1 = 0
    x2 = 1
    y1 = min_val
    y2 = max_val
    scale = (y2-y1)/(x2-x1)
    image = scale*(image - x1) + y1

    return image