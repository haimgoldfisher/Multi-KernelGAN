To enhance the README, you can include detailed instructions on how to run each notebook and what they do. Here's an improved version:

---

# "Multi-KernelGAN"
## Multi Kernel Estimation based Object Segmentation

### Asaf Yekutiel, Haim Goldfisher

<img src="Images/template.png" alt="Multi-KernelGAN Model Pipeline">

This work builds upon [KernelGAN](https://github.com/sefibk/KernelGAN) by Sefi Bell-Kligler, Assaf Shocher, and Michal Irani. It also leverages the [SAM - Segment Anything Model](https://github.com/facebookresearch/segment-anything) and [YOLOv8](https://github.com/ultralytics/ultralytics) algorithms for image segmentation and object detection.

---

### Notebooks

#### 1. **[Original KernelGAN+ZSSR in Google Colab](https://colab.research.google.com/github/kuty007/Multi-Kernel-GAN/blob/main/Colab%20Notebooks/KernelGAN.ipynb)**  
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kuty007/Multi-Kernel-GAN/blob/main/Colab%20Notebooks/KernelGAN.ipynb)  

This notebook implements the original KernelGAN+ZSSR pipeline. It estimates the kernel of an input image and applies the Zero-Shot Super-Resolution (ZSSR) algorithm to upscale the image.

**How to run:**
1. Open the notebook by clicking the Colab badge above.
2. create 2 folders one with the name input and one with the name output
3. Upload your low-resolution input image to input folder.
4. Follow the steps in the notebook to estimate the image kernel and perform super-resolution using the ZSSR algorithm.
5. The final output will be an upscaled version of the input image and will be found in the output folder.

#### 2. **[Mask Creation with SAM (Segment Anything Model)](https://colab.research.google.com/github/kuty007/Multi-Kernel-GAN/blob/main/Colab%20Notebooks/Mask_Generator.ipynb)**  
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kuty007/Multi-Kernel-GAN/blob/main/Colab%20Notebooks/Mask_Generator.ipynb)  

This notebook generates object masks using Facebook's SAM (Segment Anything Model), which can be used for object segmentation in later stages of the pipeline.

**How to run:**
1. Open the notebook by clicking the Colab badge above.
2. Upload your input image.
3. The notebook will use SAM to generate segmentation masks for various objects in the image.
4. You can download the mask images and use them in the next stages of the pipeline or for visualization purposes.

#### 3. **[Multi-KernelGAN+ZSSR (Ours)](https://colab.research.google.com/github/kuty007/Multi-Kernel-GAN/blob/main/Colab%20Notebooks/Run_MultiKernelGAN%2BZSSR.ipynb)**  
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kuty007/Multi-Kernel-GAN/blob/main/Colab%20Notebooks/Run_MultiKernelGAN%2BZSSR.ipynb)  

This notebook runs the complete Multi-KernelGAN pipeline with ZSSR. It estimates multiple kernels for different regions of the image based on segmentation and object detection, and then applies ZSSR to each region individually for better super-resolution results.

**How to run:**
1. Open the notebook by clicking the Colab badge above.
2. Upload your input image.
3. The notebook will first run YOLOv8 to detect objects in the image.
4. It will then segment the image using SAM and estimate individual kernels for each segmented object or region.
5. Finally, ZSSR will be applied to each region to produce the super-resolved output.
6. The resulting upscaled image will be saved for download.

---

<img src="Images/good_example.png" alt="Multi-KernelGAN Model Performance">

