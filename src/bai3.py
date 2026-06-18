import numpy as np
import cv2
import os
import math

def circular_lbp_hand_calc(img, P, R):
    """
    Tính Circular LBP bằng các vòng lặp for
    """
    rows, cols = img.shape
    
    # Khởi tạo ma trận kết quả toàn số 0 (chứa đủ 24-bit)
    lbp_result = np.zeros((rows, cols), dtype=np.uint32)
    
    # Ép kiểu ảnh sang float để tính toán nội suy thập phân không bị sai số
    img_float = img.astype(np.float32)

    # 1. Quét qua từng pixel trong ảnh
    for y in range(rows):
        for x in range(cols):
            center_val = img_float[y, x]
            pixel_lbp = 0 # Biến chứa giá trị LBP cho pixel (y, x)
            
            # 2. Quét qua P điểm lân cận xung quanh tâm (x, y)
            for p in range(P):
                # Tính góc xoay (ngược chiều kim đồng hồ)
                theta = 2 * math.pi * p / P
                
                # Tọa độ thực của điểm lân cận (có thể là số thập phân)
                x_p = x + R * math.cos(theta)
                y_p = y - R * math.sin(theta)
                
                # Ép tọa độ không được vượt ra khỏi viền ảnh (padding đơn giản)
                x_p = max(0, min(x_p, cols - 1))
                y_p = max(0, min(y_p, rows - 1))
                
                # 3. Nội suy song tuyến tính (Bilinear Interpolation)
                # Tìm 4 tọa độ nguyên gần nhất xung quanh x_p, y_p
                x1 = int(math.floor(x_p))
                x2 = int(math.ceil(x_p))
                y1 = int(math.floor(y_p))
                y2 = int(math.ceil(y_p))
                
                # Trọng số dời điểm
                tx = x_p - x1
                ty = y_p - y1
                
                # Trọng số của 4 góc
                w1 = (1 - tx) * (1 - ty) # Góc trên trái
                w2 = tx * (1 - ty)       # Góc trên phải
                w3 = (1 - tx) * ty       # Góc dưới trái
                w4 = tx * ty             # Góc dưới phải
                
                # Lấy giá trị thực tế của 4 điểm góc
                v1 = img_float[y1, x1]
                v2 = img_float[y1, x2]
                v3 = img_float[y2, x1]
                v4 = img_float[y2, x2]
                
                # Giá trị tại điểm thập phân (y_p, x_p)
                neighbor_val = w1*v1 + w2*v2 + w3*v3 + w4*v4
                
                # 4. Áp dụng hàm Threshold s(x)
                if neighbor_val >= center_val:
                    bit = 1
                else:
                    bit = 0
                    
                # 5. Dịch bit vào đúng vị trí thứ p và cộng dồn
                pixel_lbp = pixel_lbp | (bit << p)
                
            # Lưu giá trị LBP hoàn chỉnh vào ma trận kết quả
            lbp_result[y, x] = pixel_lbp
            
    return lbp_result

def process_and_save_lbp(img, filename, output_dir, P, R):
    """
    Xử lý LBP: Nếu P=16 hoặc 24, tách chuỗi nhị phân thành các phần 8-bit.
    Chỉ lấy phần có giá trị lớn nhất gán cho pixel đang xét.
    """
    print(f"  -> Đang tính LBP (P={P}, R={R}) cho {filename}....")
    lbp_full = circular_lbp_hand_calc(img, P, R)

    # Tính số lượng phần 8-bit cần cắt (P=8 -> 1 phần, P=16 -> 2 phần, P=24 -> 3 phần)
    num_bytes = math.ceil(P / 8)
    base_name = os.path.splitext(filename)[0]
    
    rows, cols = lbp_full.shape
    # Tạo MỘT ảnh duy nhất để chứa kết quả cuối cùng
    final_lbp_img = np.zeros((rows, cols), dtype=np.uint8)
    
    # Quét qua từng pixel trong ma trận LBP đã tính
    for y in range(rows):
        for x in range(cols):
            val_32bit = lbp_full[y, x]
            max_8bit_val = 0 # Biến tạm để tìm giá trị lớn nhất
            
            # Tiến hành cắt chuỗi 32-bit thành các đoạn 8-bit
            for i in range(num_bytes):
                # Dịch bit và AND với 255 để lấy 8 bit
                current_8bit = (val_32bit >> (8 * i)) & 255
                
                # So sánh để tìm phần có giá trị lớn nhất
                if current_8bit > max_8bit_val:
                    max_8bit_val = current_8bit
            
            # Gán phần có giá trị lớn nhất cho pixel đang xét
            final_lbp_img[y, x] = max_8bit_val
            
    # Lưu ra duy nhất 1 ảnh cho cấu hình này (không còn part1, part2 nữa)
    out_name = f"{base_name}_LBP_P{P}_R{R}.png"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, final_lbp_img)

def run(input_dir="data/output/grayscale", output_dir="data/output"):
    # Cấu hình đề bài
    configs = [
        (8, 1),
        (8, 2),
        (16, 2),
        (16, 3),
        (24, 3)
    ]

    if not os.path.exists(input_dir):
        print(f"[LỖI] Không tìm thấy {input_dir}. Hãy chạy bước tạo ảnh xám trước.")
        return

    gray_images = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    
    for img_name in gray_images:
        img_path = os.path.join(input_dir, img_name)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img_gray is None:
            continue
            
        # Lấy số hàng (rows) và số cột (cols) của ảnh
        rows, cols = img_gray.shape
        total_pixels = rows * cols # Tổng số pixel
        
        # In ra màn hình dòng thông báo như bạn muốn
        print(f"\n[BẮT ĐẦU] Xử lý: {img_name} có số pixel là {total_pixels:,} ({cols}x{rows})")
            
        # Tìm folder lưu trữ theo tên ảnh (image_01_gray -> img01/b3)
        try:
            img_num = img_name.split('_')[1]
            image_output_dir = os.path.join(output_dir, f"img{img_num}", "b3")
        except IndexError:
            image_output_dir = os.path.join(output_dir, "lbp_results", "b3")

        os.makedirs(image_output_dir, exist_ok=True)

        for P, R in configs:
            process_and_save_lbp(img_gray, img_name, image_output_dir, P, R)
            
    print("\n[HOÀN THÀNH] Xong bài 3!")

if __name__ == "__main__":
    run()
