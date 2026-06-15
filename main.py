import os
import cv2
import numpy as np

def rgb_to_gray(image_rgb):
    """
    Chuyển đổi ảnh màu sang ảnh xám.
    Sử dụng công thức luminosity tiêu chuẩn.
    """
    b, g, r = cv2.split(image_rgb)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)

def process_one_image(image_path, img_file, base_output_dir):
    """
    Hàm xử lý cho một ảnh: Đọc ảnh, chuyển xám, và lưu trực tiếp vào gray_images/.
    """
    # 1. Đọc ảnh màu
    img_rgb = cv2.imread(image_path)
    if img_rgb is None:
        print(f"Lỗi: Không thể đọc ảnh tại {image_path}")
        return

    # 2. Tạo thư mục tập trung cho ảnh xám
    gray_dir = os.path.join(base_output_dir, "gray_images")
    os.makedirs(gray_dir, exist_ok=True)

    # 3. Chuyển RGB sang Gray
    img_gray = rgb_to_gray(img_rgb)

    # 4. Lưu ảnh xám vào chung thư mục gray_images/ với tên gốc
    gray_filename = os.path.join(gray_dir, img_file)
    cv2.imwrite(gray_filename, img_gray)
    print(f"Đã xử lý và lưu: {gray_filename}")

    # ==========================================
    # NHƯỜNG PHẦN DƯỚI ĐÂY CHO CÁC THÀNH VIÊN KHÁC
    # Ví dụ cách họ sẽ thêm code vào pipeline của bạn từ utils.py
    # (Lưu thẳng ra thư mục outputs/ thay vì tạo folder riêng):
    # 
    # H2_img = utils.histogram_equalization(img_gray)         # Hải (TV2)
    # cv2.imwrite(os.path.join(base_output_dir, f"H2_{img_file}"), H2_img)
    #
    # I1 = utils.convolution(img_gray, kernel)                # Bảo (TV3)
    # cv2.imwrite(os.path.join(base_output_dir, f"I1_{img_file}"), I1)
    # ==========================================

def process_all_images(input_dir="images", base_output_dir="outputs"):
    """
    Hàm lặp qua 10 ảnh trong thư mục images/ và chạy pipeline.
    """
    if not os.path.exists(input_dir):
        print(f"Không tìm thấy thư mục '{input_dir}'. Hãy đảm bảo cấu trúc thư mục đúng.")
        return

    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)])
    
    print(f"Tìm thấy {len(image_files)} ảnh. Bắt đầu xử lý pipeline...")

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        
        # Truyền img_file để lấy tên file gốc lưu vào thư mục gray_images
        process_one_image(img_path, img_file, base_output_dir)

if __name__ == "__main__":
    process_all_images(input_dir="images", base_output_dir="outputs")