import cv2
import numpy as np
import random
import os
from scipy.ndimage import gaussian_filter, rotate
from scipy import signal
from skimage.transform import resize

def generate_snow_effect(input_img, std):
    img_shape = input_img.shape

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, dsize=(640, 640), interpolation=cv2.INTER_LINEAR)

    SM_MEAN = 0.5  # mean for gaussian noise
    SM_SD = std  # standard deviation for gaussian noise
    SM_GAUSS_SD = 3  # gaussian blur standard deviation
    SM_SCALE_ARRAY = [.5, 1, 2, 3, 5]  # scale to determine motion blur kernel size

    SM_THRESH_RANGE = (.72, .78)  # threshold for gaussian noise map
    SM_ROTATE_RANGE = 60  # rotate range for motion blur kernel
    SM_NO_BLUR_FRAC = 0  # percent of time with no motion blur

    input_shape = input_img.shape[:2]  # Shape of the input image

    input_img = input_img.astype(np.float64)

    threshold = random.uniform(SM_THRESH_RANGE[0], SM_THRESH_RANGE[1])
    base_angle = random.uniform(-1 * SM_ROTATE_RANGE, SM_ROTATE_RANGE)

    for scale in SM_SCALE_ARRAY:
        inv_scale = 1 / scale
        layer = np.random.normal(SM_MEAN, SM_SD, (int(input_shape[0] * scale), int(input_shape[1] * scale)))
        layer = gaussian_filter(layer, sigma=SM_GAUSS_SD)
        layer = layer > threshold
        layer = resize(layer, input_shape)

        kernel_size = random.randint(10, 15)
        angle = base_angle + random.uniform(-30, 30)
        SM_KERNEL_SIZE = min(max(int(kernel_size * inv_scale), 3), 15)
        kernel_v = np.zeros((SM_KERNEL_SIZE, SM_KERNEL_SIZE))
        kernel_v[int((SM_KERNEL_SIZE - 1) / 2), :] = np.ones(SM_KERNEL_SIZE)
        kernel_v = rotate(kernel_v, 90 - angle)
        if scale > 4:
            kernel_v = gaussian_filter(kernel_v, sigma=1)
        elif scale < 1:
            kernel_v = gaussian_filter(kernel_v, sigma=3)
        else:
            kernel_v = gaussian_filter(kernel_v, sigma=int(4 - scale))
        kernel_v *= 1 / np.sum(kernel_v)
        if random.random() > SM_NO_BLUR_FRAC:
            layer = signal.convolve2d(layer, kernel_v, boundary='symm', mode='same')

        layer = np.expand_dims(layer, axis=2)

        input_img = input_img * (1 - layer) + layer * 255

        input_img = np.clip(input_img, 0, 255)

    output_img = input_img / np.max(input_img)

    output_img = cv2.resize(output_img, dsize=(img_shape[1], img_shape[0]))

    return output_img

def process_images(input_folder, output_folder, std):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv2.imread(img_path)
            snow_img = generate_snow_effect(img, std)
            snow_img = (snow_img * 255).astype(np.uint8)
            snow_img = cv2.cvtColor(snow_img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(output_path, snow_img)

# 使用双反斜杠或原始字符串
input_folder = '/data1/cxy/jiaxue'
output_folder = '/data1/cxy/zhongxue'
std = 1.3  # Adjust the standard deviation for snow effect

process_images(input_folder, output_folder, std)
