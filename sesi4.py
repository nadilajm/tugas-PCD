# Import library yang dibutuhkan
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Fungsi untuk melakukan histogram equalization
def histogram_equalization(image):
    # Hitung histogram citra
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    
    # Hitung CDF
    cdf = histogram.cumsum()  # Fungsi distribusi kumulatif
    cdf_normalized = cdf * histogram.max() / cdf.max()  # Normalisasi

    # Normalisasi CDF untuk menghindari nilai 0
    cdf_m = np.ma.masked_equal(cdf, 0)  # Masking nilai 0
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # Isi nilai 0 kembali

    # Menerapkan hasil CDF pada citra asli
    equalized_image = cdf[image]
    return equalized_image

# Fungsi untuk menerapkan Gaussian blur sebagai bagian dari image enhancement
def apply_gaussian_blur(image, sigma=2):
    return gaussian_filter(image, sigma=sigma)

# Fungsi utama untuk enhancement dan histogram equalization
def enhance_and_equalize_image(image_path):
    # Muat gambar dalam grayscale
    
    image = imageio.imread(image_path).astype(np.uint8)

    # Image Enhancement dengan Gaussian blur
    blurred_image = apply_gaussian_blur(image, sigma=1)

    # Histogram Equalization
    equalized_image = histogram_equalization(blurred_image)
    
    # Tampilkan hasil
    plt.figure(figsize=(15, 5))

    # Gambar asli
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # Gambar hasil Gaussian blur
    plt.subplot(1, 3, 2)
    plt.title("Enhanced Image (Gaussian Blur)")
    plt.imshow(blurred_image, cmap='gray')
    plt.axis('off')

    # Gambar hasil histogram equalization
    plt.subplot(1, 3, 3)
    plt.title("Histogram Equalized Image")
    plt.imshow(equalized_image, cmap='gray')
    plt.axis('off')

    plt.show()

# Jalankan program
if __name__ == "__main__":
    # Ganti 'image.jpg' dengan path gambar yang ingin Anda proses
    image_path = 'daun_singkong.jpg'
    enhance_and_equalize_image(image_path)
