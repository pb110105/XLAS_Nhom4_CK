import numpy as np
import cv2
import os


# ================
# Convolution
# ================

def create_average_kernel(kernel_size):
    # Tạo kernel trung bình kích thước kernel_size x kernel_size
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel = kernel / (kernel_size * kernel_size)
    return kernel


def convolution(image, kernel, padding=0, stride=1):
    # Phép tích chập trên ảnh xám
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

            output[i, j] = np.sum(region * kernel)

    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)


def create_I1_I2_I3(img_gray):
    # Tạo I1, I2, I3
    # I1: kernel 3x3, padding 1
    # I2: kernel 5x5, padding 2
    # I3: kernel 7x7, padding 3, stride 2

    kernel_3x3 = create_average_kernel(3)
    kernel_5x5 = create_average_kernel(5)
    kernel_7x7 = create_average_kernel(7)

    I1 = convolution(img_gray, kernel_3x3, padding=1, stride=1)
    I2 = convolution(img_gray, kernel_5x5, padding=2, stride=1)
    I3 = convolution(img_gray, kernel_7x7, padding=3, stride=2)

    return I1, I2, I3


# =========================
# PHẦN QUỐC - MEDIAN + I6
# =========================

def median_filter_manual(image, kernel_size):
    """
    Lọc trung vị thủ công.

    Input:
    - image: ảnh xám đầu vào
    - kernel_size: kích thước vùng lân cận, ví dụ 3 hoặc 5

    Output:
    - ảnh sau khi lọc trung vị

    Trong phần này dùng zero padding.
    """
    image = image.astype(np.uint8)

    # kernel 3x3 -> pad = 1
    # kernel 5x5 -> pad = 2
    pad = kernel_size // 2

    # Zero padding: thêm số 0 quanh ảnh
    padded_image = np.pad(
        image,
        pad_width=pad,
        mode='constant',
        constant_values=0
    )

    rows, cols = image.shape
    output = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            # Lấy vùng lân cận kernel_size x kernel_size
            region = padded_image[
                i:i + kernel_size,
                j:j + kernel_size
            ]

            # Chuyển vùng lân cận thành mảng 1 chiều
            values = region.flatten()

            # Sắp xếp tăng dần
            values_sorted = np.sort(values)

            # Lấy giá trị trung vị
            median_index = len(values_sorted) // 2
            median_value = values_sorted[median_index]

            output[i, j] = median_value

    return output


def pad_image_to_shape(image, target_shape):
    """
    Padding ảnh về kích thước target_shape bằng zero padding.

    Cách padding:
    - Ảnh nhỏ được đặt vào giữa khung ảnh lớn.
    - Phần thiếu xung quanh được thêm số 0.

    Ví dụ:
    image: 128x128
    target_shape: 256x256

    Sau padding:
    image vẫn giữ nguyên nội dung,
    nhưng nằm ở giữa ảnh 256x256.
    """
    current_h, current_w = image.shape
    target_h, target_w = target_shape

    pad_h = target_h - current_h
    pad_w = target_w - current_w

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Ảnh hiện tại lớn hơn target_shape, không thể padding.")

    # Chia padding đều trên/dưới, trái/phải
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_image = np.pad(
        image,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )

    return padded_image


def pad_to_same_size(image_a, image_b):
    """
    Đưa hai ảnh về cùng kích thước bằng zero padding.

    Nếu image_a nhỏ hơn image_b:
        padding image_a

    Nếu image_b nhỏ hơn image_a:
        padding image_b

    Nếu bằng nhau:
        giữ nguyên
    """
    h_a, w_a = image_a.shape
    h_b, w_b = image_b.shape

    target_h = max(h_a, h_b)
    target_w = max(w_a, w_b)

    image_a_padded = pad_image_to_shape(image_a, (target_h, target_w))
    image_b_padded = pad_image_to_shape(image_b, (target_h, target_w))

    return image_a_padded, image_b_padded


def create_I6(I4, I5):
    """
    Tạo ảnh I6 theo công thức đề bài:

    Nếu I4(x,y) > I5(x,y):
        I6(x,y) = 0

    Ngược lại:
        I6(x,y) = I5(x,y)

    Nếu I4 và I5 khác kích thước thì zero padding cho cùng kích thước.

    Đây là bản I6 chính thức theo yêu cầu đề.
    """
    if I4.shape != I5.shape:
        I4, I5 = pad_to_same_size(I4, I5)

    I6 = np.where(
        I4 > I5,
        0,
        I5
    )

    return I6.astype(np.uint8)


def create_I4_I5_I6(I1, I3):
    """
    Tạo I4, I5, I6 từ I1 và I3.

    I4 = median filter 3x3 từ I3
    I5 = median filter 5x5 từ I1
    I6 = bản chính thức dùng padding theo đề

    Trả về:
    - I4
    - I5
    - I6
    """
    # I4: lọc trung vị ảnh I3 với neighbors 3x3
    I4 = median_filter_manual(I3, kernel_size=3)

    # I5: lọc trung vị ảnh I1 với neighbors 5x5
    I5 = median_filter_manual(I1, kernel_size=5)

    # I6 chính thức theo đề: nếu khác kích thước thì dùng zero padding
    I6 = create_I6(I4, I5)

    return I4, I5, I6


def process_bai2(img_gray, output_dir, filename_without_ext):
    # Tạo I1, I2, I3
    folder_name = filename_without_ext.replace("image_", "img")
    bai2_dir = os.path.join(output_dir, folder_name)
    os.makedirs(bai2_dir, exist_ok=True)

    I1, I2, I3 = create_I1_I2_I3(img_gray)

    # Tạo I4, I5, I6 - Phần Quốc
    I4, I5, I6 = create_I4_I5_I6(I1, I3)

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
    I4_path = os.path.join(
        bai2_dir,
        f"{filename_without_ext}_I4_median_3x3_from_I3.png"
    )
    I5_path = os.path.join(
        bai2_dir,
        f"{filename_without_ext}_I5_median_5x5_from_I1.png"
    )
    I6_path = os.path.join(
        bai2_dir,
        f"{filename_without_ext}_I6_threshold.png"
    )

    cv2.imwrite(I1_path, I1)
    cv2.imwrite(I2_path, I2)
    cv2.imwrite(I3_path, I3)
    cv2.imwrite(I4_path, I4)
    cv2.imwrite(I5_path, I5)
    cv2.imwrite(I6_path, I6)

    print(f"Đã lưu kết quả Bài 2 cho ảnh: {filename_without_ext}")
    print(f"I1 shape: {I1.shape}")
    print(f"I2 shape: {I2.shape}")
    print(f"I3 shape: {I3.shape}")
    print(f"I4 shape: {I4.shape}")
    print(f"I5 shape: {I5.shape}")
    print(f"I6 shape: {I6.shape}")
    print("-" * 60)

    return I1, I2, I3, I4, I5, I6


# =========================
# CHẠY BÀI 2 CHO TOÀN BỘ ẢNH XÁM
# =========================

def run_all_bai2():
    """
    Chạy Bài 2 cho toàn bộ ảnh xám trong:
    data/output/grayscale/

    Ví dụ input:
    data/output/grayscale/img01_gray.png

    Output:
    data/output/img01/
    """

    # Lấy thư mục gốc project:
    # XLAS_Nhom4_CK/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Thư mục ảnh xám
    gray_dir = os.path.join(project_root, "data", "output", "grayscale")

    # Thư mục lưu kết quả
    output_dir = os.path.join(project_root, "data", "output")

    if not os.path.exists(gray_dir):
        print(f"[LỖI] Không tìm thấy thư mục ảnh xám: {gray_dir}")
        return

    image_files = sorted([
        f for f in os.listdir(gray_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if len(image_files) == 0:
        print(f"[LỖI] Không có ảnh xám trong thư mục: {gray_dir}")
        return

    print(f"Tìm thấy {len(image_files)} ảnh xám. Bắt đầu chạy Bài 2...")

    for img_file in image_files:
        img_path = os.path.join(gray_dir, img_file)

        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            print(f"[BỎ QUA] Không đọc được ảnh: {img_path}")
            continue

        # Ví dụ:
        # img01_gray.png -> img01
        # image_01_gray.png -> image_01 -> folder img01
        filename_without_ext = os.path.splitext(img_file)[0]
        filename_without_ext = filename_without_ext.replace("_gray", "")

        process_bai2(img_gray, output_dir, filename_without_ext)

    print("Đã xử lý xong toàn bộ Bài 2!")


if __name__ == "__main__":
    run_all_bai2()