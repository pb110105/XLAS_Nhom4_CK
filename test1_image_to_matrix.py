import cv2
import numpy as np

img = cv2.imread("data/output/grayscale/image_01_gray.png", cv2.IMREAD_GRAYSCALE)

print("Kích thước:", img.shape)

print("\nMa trận ảnh:")
print(img)