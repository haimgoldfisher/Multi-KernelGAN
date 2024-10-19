import numpy as np
from torch.utils.data import Dataset
from imresize import imresize
from util import read_image, create_gradient_map, im2tensor, create_probability_map, nn_interpolation
import cv2
import random
import re
import matplotlib.pyplot as plt



class DataGenerator(Dataset):
    """
    The data generator loads an image once, calculates it's gradient map on initialization and then outputs a cropped version
    of that image whenever called.
    """

    def extract_numbers_from_filename(self, filename):
        # Use regular expression to find numbers in the filename
        match = re.search(r'\d+', filename)

        # If numbers are found, convert them to an integer
        if match:
            return int(match.group())
        else:
            return None

    def __init__(self, conf, gan):
        # Default shapes
        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = gan.G.output_size  # shape entering D downscaled by G
        self.d_output_shape = self.d_input_shape - gan.D.forward_shave

        # Read input image
        self.input_image = read_image(conf.input_image_path) / 255.
        self.shave_edges(scale_factor=conf.scale_factor, real_image=conf.real_image)

        self.in_rows, self.in_cols = self.input_image.shape[0:2]

        # Create prob map for choosing the crop
        self.crop_indices_for_g, self.crop_indices_for_d = self.make_list_of_crop_indices(conf=conf)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """Get a crop for both G and D """
        g_in = self.next_crop(for_g=True, idx=idx)
        d_in = self.next_crop(for_g=False, idx=idx)

        return g_in, d_in

    def compute_frequency(self, image_patches):
        f_transform = np.fft.fft2(image_patches)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
        return np.mean(magnitude_spectrum, axis=(-2, -1))

    def create_frequency_map(self, img):
        img = img * 255
        if img.dtype == 'float64':
            img = cv2.convertScaleAbs(img)  # Convert to 8-bit depth
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.uint8)
        pad_size = 2
        padded_img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
        patch_size = 5
        patches = np.lib.stride_tricks.sliding_window_view(padded_img, (patch_size, patch_size))
        frequency_values = self.compute_frequency(patches)
        
        # Reshape frequency values to match the original image size
        frequency_map = frequency_values.reshape(img.shape[0], img.shape[1])
        
        return frequency_map

    def create_frequency_list(self, img, mask, for_g, scale_factor):
        frequency_map =  self.create_frequency_map(img)
        if for_g:
          frequency_map = nn_interpolation(frequency_map, int(1 / scale_factor))

        frequency_map = frequency_map.astype(np.uint8)
        mask = mask.astype(np.uint8)
        white_part = cv2.bitwise_and(frequency_map, frequency_map, mask=mask)
        min_frequency, max_frequency = np.min(white_part), np.max(white_part)
        bins = np.linspace(min_frequency, max_frequency, 21)  # 20 bins within the specified range
        hist, _ = np.histogram(white_part, bins=bins)
        sorted_pixels = []

        for bin_size in sorted(hist, reverse=True):
            mask = hist == bin_size
            pixels_in_bin = np.argwhere(mask[white_part.astype(int)])
            single_numbers = pixels_in_bin[:, 0] * img.shape[1] + pixels_in_bin[:, 1]
            sorted_pixels.extend(single_numbers)

        return sorted_pixels

    def next_crop(self, for_g, idx):
        """Return a crop according to the pre-determined list of indices. Noise is added to crops for D"""
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_top_left(size, for_g, idx)
        crop_im = self.input_image[top:top + size, left:left + size, :]
        if not for_g:  # Add noise to the image for d
            crop_im += np.random.randn(*crop_im.shape) / 255.0
        return im2tensor(crop_im)

    def make_list_of_crop_indices(self, conf):
        iterations = conf.max_iters
        indexes = [index for index, char in enumerate(conf.input_image_path) if char == '/']
        img_name = conf.input_image_path[indexes[-1] + 1:]
        img_num = self.extract_numbers_from_filename(img_name)
        mask_path = conf.masks_dir_path+"/img"+str(img_num)+"/"
        if "back" in img_name:
          mask_path += "back_lr_mask.png"
        else:
          mask_path += "obj_lr_mask.png"
        mask = cv2.imread(mask_path)
        mask = mask[10:-10, 10:-10, :]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        print("\nmask_path = ", mask_path)

        target_width = int(self.input_image.shape[1] * conf.scale_factor)
        target_height = int(self.input_image.shape[0] * conf.scale_factor)
        s_img = cv2.resize(self.input_image, dsize=(target_width, target_height), interpolation=cv2.INTER_CUBIC)

        freq_list_sml = self.create_frequency_list(s_img, mask, for_g = True, scale_factor = conf.scale_factor)
        freq_list_big = self.create_frequency_list(self.input_image, mask, for_g = False, scale_factor = conf.scale_factor)
        crop_indices_for_g = freq_list_sml[:min(iterations, len(freq_list_sml))]
        crop_indices_for_d = freq_list_big[:min(iterations, len(freq_list_big))]
        return crop_indices_for_g, crop_indices_for_d

    def shave_edges(self, scale_factor, real_image):
        """Shave pixels from edges to avoid code-bugs"""
        # Crop 10 pixels to avoid boundaries effects in synthetically generated examples
        if not real_image:
            self.input_image = self.input_image[10:-10, 10:-10, :]
        # Crop pixels for the shape to be divisible by the scale factor
        sf = int(1 / scale_factor)
        shape = self.input_image.shape
        self.input_image = self.input_image[:-(shape[0] % sf), :, :] if shape[0] % sf > 0 else self.input_image
        self.input_image = self.input_image[:, :-(shape[1] % sf), :] if shape[1] % sf > 0 else self.input_image

    def get_top_left(self, size, for_g, idx):
        """Translate the center of the index of the crop to it's corresponding top-left"""
        center = self.crop_indices_for_g[idx] if for_g else self.crop_indices_for_d[idx]
        row, col = int(center / self.in_cols), center % self.in_cols
        top, left = min(max(0, row - size // 2), self.in_rows - size), min(max(0, col - size // 2), self.in_cols - size)
        # Choose even indices (to avoid misalignment with the loss map for_g)
        return top - top % 2, left - left % 2