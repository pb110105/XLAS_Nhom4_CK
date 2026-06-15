import os
import cv2

# Import hàm chuyển xám từ utils.py
from src.utils import rgb_to_gray

def process_one_image(img_path, img_file, input_dir, output_dir):
    """
    Hàm xử lý cho một ảnh: Đọc ảnh, chuyển xám, và lưu.
    """
    # 1. Đọc ảnh màu
    img_rgb = cv2.imread(img_path)
    if img_rgb is None:
        print(f"Lỗi: Không thể đọc ảnh tại {img_path}")
        return

    # 2. Tạo thư mục tập trung cho ảnh xám: data/output/grayscale/
    gray_dir = os.path.join(output_dir, "grayscale")
    os.makedirs(gray_dir, exist_ok=True)

    # 3. Chuyển RGB sang Gray
    img_gray = rgb_to_gray(img_rgb)

    # 4. Đổi tên file (VD: img01.jpg -> img01_gray.png) và lưu
    filename_without_ext = os.path.splitext(img_file)[0]
    gray_filename = f"{filename_without_ext}_gray.png"
    gray_filepath = os.path.join(gray_dir, gray_filename)
    
    cv2.imwrite(gray_filepath, img_gray)
    print(f"Đã xử lý và lưu: {gray_filepath}")


def process_all_images(input_dir="data/input", output_dir="data/output"):
    """
    Hàm lặp qua 10 ảnh trong thư mục data/input/ và chạy pipeline.
    """
    if not os.path.exists(input_dir):
        print(f"Không tìm thấy thư mục '{input_dir}'. Hãy đảm bảo cấu trúc thư mục đúng.")
        return

    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)])
    
    print(f"Tìm thấy {len(image_files)} ảnh. Bắt đầu xử lý pipeline...")

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        process_one_image(img_path, img_file, input_dir, output_dir)