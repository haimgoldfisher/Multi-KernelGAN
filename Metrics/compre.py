import os
from PIL import Image
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
# Function to calculate image quality metrics
def calculate_metrics(ground_truth_np, reconstructed_np):
    mse = mean_squared_error(ground_truth_np, reconstructed_np)
    psnr = peak_signal_noise_ratio(ground_truth_np, reconstructed_np)
    win_size = min(ground_truth_np.shape[0], ground_truth_np.shape[1], 7) // 2 * 2 + 1
    ssim, _ = structural_similarity(ground_truth_np, reconstructed_np, full=True, win_size=win_size, channel_axis=-1)
    return mse, psnr, ssim


# Directory containing the folders
input_dir = 'input'

# Prepare a list to hold results
results = []

# Loop over each subdirectory (e.g., img1, img2, etc.)
for folder in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder}")

        # Load ground truth image
        gt_path = os.path.join(folder_path, 'gt.png')
        ground_truth = Image.open(gt_path).convert('RGB')
        ground_truth_np = np.array(ground_truth)

        # Initialize variables to store the best image and metrics (based on the lowest MSE)
        best_image = None
        best_metrics = {'mse': float('inf'), 'psnr': -float('inf'), 'ssim': -float('inf')}

        # Store ZSSR metrics separately
        zssr_metrics = None

        # Loop over all images in the folder
        for image_name in os.listdir(folder_path):
            if image_name == 'gt.png':
                continue  # Skip ground truth image

            # Check if the image is the ZSSR image
            if image_name.startswith('ZSSR'):
                # Calculate metrics for ZSSR image
                image_path = os.path.join(folder_path, image_name)
                reconstructed = Image.open(image_path).convert('RGB')
                reconstructed_np = np.array(reconstructed)

                if ground_truth_np.shape != reconstructed_np.shape:
                    print(f"Skipping {image_name} due to shape mismatch")
                    continue

                mse, psnr, ssim = calculate_metrics(ground_truth_np, reconstructed_np)
                zssr_metrics = {'image': image_name, 'mse': mse, 'psnr': psnr, 'ssim': ssim}

                continue  # Skip further comparison for ZSSR image

            # Process non-ZSSR images
            image_path = os.path.join(folder_path, image_name)
            reconstructed = Image.open(image_path).convert('RGB')
            reconstructed_np = np.array(reconstructed)

            # Ensure images have the same shape
            if ground_truth_np.shape != reconstructed_np.shape:
                print(f"Skipping {image_name} due to shape mismatch")
                continue

            # Calculate metrics
            mse, psnr, ssim = calculate_metrics(ground_truth_np, reconstructed_np)

            # Update best image if this one has a lower MSE
            if mse < best_metrics['mse']:
                best_metrics = {'image': image_name, 'mse': mse, 'psnr': psnr, 'ssim': ssim}

        # Append best image and ZSSR metrics to results
        if best_metrics['image'] and zssr_metrics:
            results.append({
                'folder': folder,
                'best_image': best_metrics['image'],
                'best_mse': best_metrics['mse'],
                'best_psnr': best_metrics['psnr'],
                'best_ssim': best_metrics['ssim'],
                'zssr_image': zssr_metrics['image'],
                'zssr_mse': zssr_metrics['mse'],
                'zssr_psnr': zssr_metrics['psnr'],
                'zssr_ssim': zssr_metrics['ssim']
            })

# Save results to a CSV file
df = pd.DataFrame(results)
df.to_csv('image_comparison_results.csv', index=False)

print("Results saved to image_comparison_results.csv")

# Load the CSV file
df = pd.read_csv('image_comparison_results.csv')

# Calculate the difference between best image and ZSSR image for each metric
df['mse_diff'] = df['zssr_mse'] - df['best_mse']
df['psnr_diff'] = df['best_psnr'] - df['zssr_psnr']
df['ssim_diff'] = df['best_ssim'] - df['zssr_ssim']

# Separate cases where best image is better or worse than ZSSR
mse_better = df[df['mse_diff'] > 0]['mse_diff']  # Best MSE is smaller (better)
mse_worse = df[df['mse_diff'] <= 0]['mse_diff']  # Best MSE is larger (worse)

psnr_better = df[df['psnr_diff'] > 0]['psnr_diff']  # Best PSNR is larger (better)
psnr_worse = df[df['psnr_diff'] <= 0]['psnr_diff']  # Best PSNR is smaller (worse)

ssim_better = df[df['ssim_diff'] > 0]['ssim_diff']  # Best SSIM is larger (better)
ssim_worse = df[df['ssim_diff'] <= 0]['ssim_diff']  # Best SSIM is smaller (worse)

# Calculate the average differences for each metric
avg_mse_better = mse_better.mean()
avg_mse_worse = mse_worse.mean()
avg_mse_overall = df['mse_diff'].mean()

avg_psnr_better = psnr_better.mean()
avg_psnr_worse = psnr_worse.mean()
avg_psnr_overall = df['psnr_diff'].mean()

avg_ssim_better = ssim_better.mean()
avg_ssim_worse = ssim_worse.mean()
avg_ssim_overall = df['ssim_diff'].mean()

# Display the results
print("Average MSE Difference:")
print(f"Better: {avg_mse_better:.4f}, Worse: {avg_mse_worse:.4f}, Overall: {avg_mse_overall:.4f}\n")

print("Average PSNR Difference:")
print(f"Better: {avg_psnr_better:.4f}, Worse: {avg_psnr_worse:.4f}, Overall: {avg_psnr_overall:.4f}\n")

print("Average SSIM Difference:")
print(f"Better: {avg_ssim_better:.4f}, Worse: {avg_ssim_worse:.4f}, Overall: {avg_ssim_overall:.4f}")
