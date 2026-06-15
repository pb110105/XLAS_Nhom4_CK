import cv2
import numpy as np

def rgb_to_gray(image_rgb):
    """
    Chuyển đổi ảnh màu sang ảnh xám.
    Sử dụng công thức luminosity tiêu chuẩn: Y = 0.299*R + 0.587*G + 0.114*B
    """
    # OpenCV đọc ảnh theo thứ tự BGR
    b, g, r = cv2.split(image_rgb)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)