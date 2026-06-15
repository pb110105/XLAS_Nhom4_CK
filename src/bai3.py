import numpy as np
import cv2
import os
import math

def circular_lbp_vectorized(img, P, R):
    """
    Hàm tính toán Circular LBP sử dụng nội suy song tuyến tính (Bilinear Interpolation)
    và tối ưu hóa bằng Vectorization của Numpy.
    """
    rows, cols = img.shape
    
    # Khởi tạo ma trận kết quả (dùng uint32 để chứa đủ 24-bit nếu P=24)
    lbp_result = np.zeros((rows, cols), dtype=np.uint32)
    
    # Ép kiểu ảnh gốc về float để nội suy không bị tràn số lượng tử
    img_float = img.astype(np.float32)

    # Lấy tọa độ lưới của toàn bộ ảnh
    y_idx, x_idx = np.indices((rows, cols))

    for p in range(P):
        # 1. Tính toán độ dời tọa độ theo góc
        theta = 2 * math.pi * p / P
        dx = R * math.cos(theta)
        dy = -R * math.sin(theta)

        # 2. Tọa độ thực tế của điểm lân cận thứ p
        x_p = x_idx + dx
        y_p = y_idx + dy

        # 3. Kẹp tọa độ (Clip) để không bị văng ra khỏi biên ảnh
        x_p = np.clip(x_p, 0, cols - 1)
        y_p = np.clip(y_p, 0, rows - 1)

        # 4. Tìm 4 điểm nguyên gần nhất để nội suy song tuyến tính
        x1 = np.floor(x_p).astype(np.int32)
        x2 = np.ceil(x_p).astype(np.int32)
        y1 = np.floor(y_p).astype(np.int32)
        y2 = np.ceil(y_p).astype(np.int32)

        # Tính trọng số nội suy
        tx = x_p - x1
        ty = y_p - y1
        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty

        # 5. Lấy giá trị mức xám của 4 điểm lân cận
        v1 = img_float[y1, x1]
        v2 = img_float[y1, x2]
        v3 = img_float[y2, x1]
        v4 = img_float[y2, x2]

        # Tính giá trị sub-pixel bằng nội suy song tuyến tính
        neighbor_val = w1*v1 + w2*v2 + w3*v3 + w4*v4

        # 6. So sánh Threshold (hàm s(x)) và gán bit
        # Nếu neighbor_val >= center_val (img_float), bit = 1, ngược lại = 0
        bit_matrix = (neighbor_val >= img_float).astype(np.uint32)
        
        # Dịch bit và cộng dồn vào kết quả tổng
        lbp_result |= (bit_matrix << p)

    return lbp_result

def process_and_save_lbp(img, filename, output_dir, P, R):
    """
    Hàm xử lý LBP và tách chuỗi 8-bit lưu thành các ảnh con.
    """
    print(f"  -> Đang xử lý LBP (P={P}, R={R}) cho {filename}...")
    lbp_full = circular_lbp_vectorized(img, P, R)

    # Tính số lượng ảnh 8-bit cần thiết để lưu trữ
    num_bytes = math.ceil(P / 8)
    
    base_name = os.path.splitext(filename)[0]
    
    for i in range(num_bytes):
        # Trích xuất 8-bit tương ứng bằng phép toán thao tác bit (Bitwise AND & Shift)
        # i=0: lấy 8 bit thấp nhất (0-7)
        # i=1: lấy 8 bit tiếp theo (8-15)
        # i=2: lấy 8 bit cao nhất (16-23)
        byte_layer = (lbp_full >> (8 * i)) & 0xFF
        byte_layer = byte_layer.astype(np.uint8)
        
        # Đặt tên file: ví dụ image_01_LBP_P16_R2_part1.png
        out_name = f"{base_name}_LBP_P{P}_R{R}_part{i+1}.png"
        out_path = os.path.join(output_dir, out_name)
        
        cv2.imwrite(out_path, byte_layer)
        # Cố ý không in dòng log save để terminal chạy gọn gàng hơn, bạn có thể thêm nếu cần.

def run_task_3():
    """
    Hàm main pipeline cho TV5 (Bài 3).
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_gray_dir = os.path.join(
        BASE_DIR,
        "data",
        "output",
        "grayscale"
    )
    
    # Các cấu hình LBP theo yêu cầu đề bài
    configs = [
        (8, 1),
        (8, 2),
        (16, 2),
        (16, 3),
        (24, 3)
    ]

    # Quét thư mục lấy ảnh xám do TV1 đã làm
    if not os.path.exists(input_gray_dir):
        print(f"[LỖI] Thư mục {input_gray_dir} không tồn tại. Yêu cầu TV1 chạy script trước!")
        return

    gray_images = [f for f in os.listdir(input_gray_dir) if f.endswith(('.png', '.jpg'))]
    
    for img_name in gray_images:
        print(f"Đang phân tích ảnh: {img_name}")
        img_path = os.path.join(input_gray_dir, img_name)
        
        # Đọc ảnh grayscale
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
            
        # Xác định thư mục output (data/output/imgXX) từ tên file ảnh
        # Giả sử tên file của TV1 có dạng 'image_01_gray.png' -> tách lấy '01'
        try:
            img_num = img_name.split('_')[1] # Lấy số '01', '02'...
            output_dir = os.path.join(BASE_DIR, "data", "output", f"img{img_num}")
            os.makedirs(output_dir, exist_ok=True)
        except IndexError:
            # Fallback nếu tên file không đúng chuẩn
            output_dir = os.path.join(BASE_DIR, "data", "output")

        # Chạy qua toàn bộ các cấu hình LBP
        for P, R in configs:
            process_and_save_lbp(img_gray, img_name, output_dir, P, R)
            
    print("\n[HOÀN THÀNH] Đã trích xuất xong đặc trưng LBP cho toàn bộ ảnh!")

if __name__ == "__main__":
    run_task_3()