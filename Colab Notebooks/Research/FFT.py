import cv2
import numpy as np
import matplotlib.pyplot as plt
import re


def compute_frequency(image_patch):
    f_transform = np.fft.fft2(image_patch)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    return np.mean(magnitude_spectrum)


def create_frequency_mask(image_patch, patch_size=50):
    height, width = image_patch.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    frequencies = []

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = image_patch[y:y + patch_size, x:x + patch_size]
            avg_frequency = compute_frequency(patch)
            frequencies.append(avg_frequency)

    average_frequency_threshold = np.mean(frequencies)

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = image_patch[y:y + patch_size, x:x + patch_size]
            avg_frequency = compute_frequency(patch)

            if avg_frequency >= average_frequency_threshold:
                mask[y:y + patch_size, x:x + patch_size] = 255

    # Plot the histogram of the frequencies
    plt.figure()
    plt.hist(frequencies, bins=20, color='blue', alpha=0.7)
    plt.axvline(average_frequency_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold = {average_frequency_threshold:.2f}')
    plt.title('Histogram of Frequency Values')
    plt.xlabel('Average Frequency')
    plt.ylabel('Frequency Count')
    plt.legend()  # Add legend to show the threshold
    plt.show()

    return mask


def remove_small_blobs(mask, min_blob_size):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_blob_size]
    filtered_mask = np.zeros_like(mask)
    cv2.drawContours(filtered_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
    return filtered_mask


def create_mask(image):
    img = create_frequency_mask(image, patch_size=10)

    # Apply binary thresholding
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    min_blob_size = 500
    img = remove_small_blobs(img, min_blob_size)
    img = cv2.bitwise_not(img)
    return img


# Take an image as input
image_path = 'input/im_3.png'  # Change this to your image path
orig_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create mask
mask = create_mask(orig_img)

# Display the original image and the mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(orig_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Mask")
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.show()
