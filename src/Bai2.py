import numpy as np
import cv2
import os
#================
#Convolution
#================

def create_average_kernel(kernel_size):
    #Tạo kernel trung bình kích thước kernel_size x kernel_size
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel = kernel / (kernel_size * kernel_size)
    return kernel

def convolution(image, kernel, padding = 0, stride = 1):
    #Phép tích chập trên ảnh xám
    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)
    kernel_h, kernel_w = kernel.shape

    padded_image = np.pad(
        image, 
        pad_width=padding,
        mode='constant',
        constant_values=0
    )

    padded_h, padded_w = padded_image.shape

    output_h = ((padded_h - kernel_h) // stride) + 1
    output_w = ((padded_w - kernel_w) // stride) + 1

    output = np.zeros((output_h, output_w), dtype=np.float32)

    for i in range(output_h):
        for j in range(output_w):
            start_i = i * stride
            start_j = j * stride

            region = padded_image[
                start_i:start_i + kernel_h,
                start_j:start_j + kernel_w
            ]

            output[i,j] = np.sum(region * kernel)

    output = np.clip(output, 0,255)
    return output.astype(np.uint8)

def create_I1_I2_I3(img_gray):
    #Tạo I1, I2, I3
    #I1: kernel 3x3, padding 1
    #I2: kernel 5x5, padding 2
    #I3: kernel 7x7, padding 3, stride 2

    kernel_3x3 = create_average_kernel(3)
    kernel_5x5 = create_average_kernel(5)
    kernel_7x7 = create_average_kernel(7)

    I1 = convolution(img_gray, kernel_3x3, padding=1, stride=1)
    I2 = convolution(img_gray, kernel_5x5, padding=2, stride=1)
    I3 = convolution(img_gray, kernel_7x7, padding=3, stride=2)

    return I1, I2, I3

def process_bai2(img_gray, output_dir, filename_without_ext):
    #Tạo I1, I2, I3
    folder_name = filename_without_ext.replace("image_", "img")
    bai2_dir = os.path.join(output_dir, folder_name)
    os.makedirs(bai2_dir, exist_ok=True)

    I1, I2, I3 = create_I1_I2_I3(img_gray)

    I1_path = os.path.join(
        bai2_dir,
        f"{filename_without_ext}_I1_conv_3x3_padding1.png"
    )
    I2_path = os.path.join(
        bai2_dir,
        f"{filename_without_ext}_I2_conv_5x5_padding2.png"
    )
    I3_path = os.path.join(
        bai2_dir,
        f"{filename_without_ext}_I3_conv_7x7_padding3_stride2.png"
    )

    cv2.imwrite(I1_path, I1)
    cv2.imwrite(I2_path, I2)
    cv2.imwrite(I3_path, I3)

    print(f"Đã lưu kết quả Bài 2 cho ảnh: {filename_without_ext}")
    print(f"I1 shape: {I1.shape}")
    print(f"I2 shape: {I2.shape}")
    print(f"I3 shape: {I3.shape}")
    print("-" * 60)

    return I1, I2, I3

# =========================
# PHẦN QUỐC - MEDIAN + I6
# =========================
# Quốc sẽ viết tiếp dưới này:
# def median_filter(...)
# def pad_to_same_size(...)
# def create_I4_I5_I6(...)