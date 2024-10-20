from PIL import Image
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from pypiqe import piqe
from brisque import BRISQUE as brisque_

'''
    BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) evaluates the
    visual quality of an image using Natural Scene Statistics (NSS) - features
    that represent the typical characteristics of natural images, such as
    patterns and textures that are commonly found in high-quality photographs.

    It works by extracting these NSS features from the input image and 
    them to a statistical model derived from pristine images. 
    This comparison allows BRISQUE to determine how "natural" the image looks. 

    The result of the BRISQUE algorithm is a quality score:
    - Lower scores indicate that the image closely resembles natural images and 
      thus is of higher visual quality.
    - Higher scores indicate that the image deviates from natural appearance, 
      suggesting lower visual quality.
'''


def BRISQUE(path):
    img = Image.open(path).convert('RGB')
    brisque = brisque_()
    score = brisque.score(img)
    return score


'''
    PIQE score for the input image A, returned as a nonnegative scalar in the
    range [0, 100]. The PIQE score is the no-reference image quality score and
    it is inversely correlated to the perceptual quality of an image. A low
    score value indicates high perceptual quality and high score value indicates
    low perceptual quality.
'''


def PIQE(path):
    img = Image.open(path).convert('RGB')
    score = piqe(np.array(img))[0]
    return score


path_gt = '/Users/haimgoldfisher/KernelGAN-Masks/imgs/img69/img69_hr.png'
path_kg = '/Users/haimgoldfisher/KernelGAN-Masks/kernelGAN-Exps/kernelgan/ZSSR_im_69.png'
path_ours = '/Users/haimgoldfisher/KernelGAN-Masks/kernelGAN-Exps/our/img69/run2/full.png'

# Load images
ground_truth = Image.open(path_gt).convert('RGB')
img_by_kernelgan = Image.open(path_kg).convert('RGB')
img_ours = Image.open(path_ours).convert('RGB')

# Convert images to numpy arrays
ground_truth_np = np.array(ground_truth)
reconstructed1_np = np.array(img_by_kernelgan)
reconstructed2_np = np.array(img_ours)

# Ensure the images are the same size
assert ground_truth_np.shape == reconstructed1_np.shape == reconstructed2_np.shape, "All images must have the same dimensions"

# Calculate MSE
mse1 = mean_squared_error(ground_truth_np, reconstructed1_np)
mse2 = mean_squared_error(ground_truth_np, reconstructed2_np)

# Calculate PSNR
psnr1 = peak_signal_noise_ratio(ground_truth_np, reconstructed1_np)
psnr2 = peak_signal_noise_ratio(ground_truth_np, reconstructed2_np)

# Calculate SSIM with appropriate window size and channel_axis for color images
win_size = min(ground_truth_np.shape[0], ground_truth_np.shape[1],
               7) // 2 * 2 + 1  # Ensure win_size is odd and <= min dimension
ssim1, _ = structural_similarity(ground_truth_np, reconstructed1_np, full=True, win_size=win_size, channel_axis=-1)
ssim2, _ = structural_similarity(ground_truth_np, reconstructed2_np, full=True, win_size=win_size, channel_axis=-1)

# Print the results
print(f"GT - PIQE: {PIQE(path_gt):.4f}, BRISQUE: {BRISQUE(path_gt):.4f}")
print(
    f"KernelGAN - MSE: {mse1:.2f}, PSNR: {psnr1:.2f} dB, SSIM: {ssim1:.4f}, PIQE:{PIQE(path_kg):.4f}, BRISQUE: {BRISQUE(path_kg):.4f}")
print(
    f"Ours - MSE: {mse2:.2f}, PSNR: {psnr2:.2f} dB, SSIM: {ssim2:.4f}, PIQE:{PIQE(path_ours):.4f}, BRISQUE: {BRISQUE(path_ours):.4f}")
