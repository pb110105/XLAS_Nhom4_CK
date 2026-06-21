import numpy as np

from src.bai1 import (
    calculate_histogram,
    build_equalization_table,
    shrink_histogram_range
)

# Ma trận mức xám 4x4
gray = np.array([
    [10, 20, 20, 30],
    [30, 40, 40, 50],
    [50, 60, 60, 70],
    [70, 80, 80, 90]
], dtype=np.uint8)

print("Ảnh xám:")
print(gray)

# Histogram
hist = calculate_histogram(gray)

print("\nCác mức xám xuất hiện:")
for i in range(256):
    if hist[i] > 0:
        print(f"Mức xám {i}: {hist[i]} pixel")

# Histogram Equalization
equalized_img, table = build_equalization_table(gray)

print("\nẢnh sau cân bằng histogram:")
print(equalized_img)

# Thu hẹp về [30,120]
narrowed_img = shrink_histogram_range(equalized_img)

print("\nẢnh sau thu hẹp [30,120]:")
print(narrowed_img)






equalized_img, table = build_equalization_table(gray)

print("rk nk p(rk) sk rk' round(rk')")

for row in table:
    if row[1] > 0:      # chỉ in mức xám xuất hiện
        print(
            row[0],
            row[1],
            round(row[2],4),
            round(row[3],4),
            round(row[4],2),
            row[5]
        )