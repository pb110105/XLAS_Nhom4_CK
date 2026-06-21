import numpy as np
from src.bai3 import circular_lbp_hand_calc

# Ma trận ảnh xám
img = np.array([
    [10, 20, 30, 40, 50],
    [15, 25, 35, 45, 55],
    [20, 30, 40, 50, 60],
    [25, 35, 45, 55, 65],
    [30, 40, 50, 60, 70]
], dtype=np.uint8)

lbp = circular_lbp_hand_calc(img, P=8, R=1)

print("Ảnh gốc:")
print(img)

print("\nLBP:")
print(lbp)